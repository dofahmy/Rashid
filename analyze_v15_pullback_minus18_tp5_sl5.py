#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 Deep Pullback -18% Analyzer
===============================
Strategy:

1) Use only trades that hit the ORIGINAL -3% stop within sessions 1-3.
2) After that early stop, place Buy Limit at 18% below ORIGINAL entry.
3) If filled:
   - Target = 5% above NEW buy price.
   - Stop = 5% below NEW buy price.
4) Limit remains active through session 12 after the original signal.
5) After fill, TP/SL are monitored for 12 sessions.
6) If TP and SL are both touched on the same day, SL is counted first conservatively.
7) Round-trip cost = 0.30 percentage points.

Example:
  Original entry = 100
  Early original stop = 97
  New buy = 82
  New stop = 77.90
  New target = 86.10

Reads:
  /data/us_v15_trades_2025.csv
  /data/us_v15_trades_2026.csv

Requires:
  /app/us_setup_optimizer_v11.py

Outputs:
  /data/us_v15_pullback18_tp5_summary.csv
  /data/us_v15_pullback18_tp5_detail.csv
  /data/us_v15_pullback18_tp5_report.txt
"""

import sys, subprocess
from pathlib import Path

def ensure(pkg, name=None):
    name = name or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(name)
    except Exception:
        print(f"[setup] installing {pkg} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

for pkg, name in [("numpy", None), ("pandas", None)]:
    ensure(pkg, name)

import numpy as np
import pandas as pd

try:
    import us_setup_optimizer_v11 as v11
except Exception as e:
    raise RuntimeError("Put /app/us_setup_optimizer_v11.py beside this analyzer") from e

DATA = Path("/data")
FILES = {
    2025: DATA / "us_v15_trades_2025.csv",
    2026: DATA / "us_v15_trades_2026.csv",
}

OUT_SUMMARY = DATA / "us_v15_pullback18_tp5_summary.csv"
OUT_DETAIL = DATA / "us_v15_pullback18_tp5_detail.csv"
OUT_REPORT = DATA / "us_v15_pullback18_tp5_report.txt"

ORIGINAL_STOP_PCT = 3.0
BUY_DISCOUNT_PCT = 18.0
NEW_STOP_PCT = 5.0
NEW_TARGET_PCT = 5.0
EARLY_STOP_MAX_SESSION = 3
LIMIT_EXPIRY_SESSION = 12
POST_FILL_HOLD = 12
COST = 0.30

def load_early_stops():
    out = []
    for year, p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        t = pd.read_csv(p)
        t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
        t["exit_step"] = pd.to_numeric(t["exit_step"], errors="coerce")
        t["year"] = year
        t = t[
            (t["exit_kind"] == "SL") &
            (t["exit_step"] <= EARLY_STOP_MAX_SESSION)
        ].copy()
        out.append(t)
    return pd.concat(out, ignore_index=True)

def simulate_trade(symbol_df, signal_date, original_entry):
    arr = symbol_df.reset_index(drop=True)
    hits = np.flatnonzero(arr["d"].to_numpy() == np.datetime64(signal_date))
    if not len(hits):
        return {"status": "NO_SIGNAL_ROW"}

    i0 = int(hits[0])

    original_stop = original_entry * (1 - ORIGINAL_STOP_PCT / 100)
    buy_limit = original_entry * (1 - BUY_DISCOUNT_PCT / 100)
    new_stop = buy_limit * (1 - NEW_STOP_PCT / 100)
    target = buy_limit * (1 + NEW_TARGET_PCT / 100)

    # Confirm the original -3% stop was hit within sessions 1-3.
    stop_i = None
    for j in range(i0 + 1, min(len(arr), i0 + EARLY_STOP_MAX_SESSION + 1)):
        if float(arr.iloc[j]["l"]) <= original_stop:
            stop_i = j
            break

    if stop_i is None:
        return {"status": "NO_EARLY_STOP_FOUND"}

    # New deep limit becomes active from the early-stop session onward.
    expiry_i = min(len(arr) - 1, i0 + LIMIT_EXPIRY_SESSION)
    fill_i = None

    for j in range(stop_i, expiry_i + 1):
        if float(arr.iloc[j]["l"]) <= buy_limit:
            fill_i = j
            break

    if fill_i is None:
        return {
            "status": "NOT_FILLED",
            "original_stop": original_stop,
            "buy_limit": buy_limit,
            "new_stop": new_stop,
            "target": target,
        }

    entry = buy_limit
    end_i = min(len(arr) - 1, fill_i + POST_FILL_HOLD)

    exit_kind = "TIME"
    exit_i = end_i
    exit_price = float(arr.iloc[end_i]["c"])

    for j in range(fill_i, end_i + 1):
        high = float(arr.iloc[j]["h"])
        low = float(arr.iloc[j]["l"])

        hit_sl = low <= new_stop
        hit_tp = high >= target

        # Conservative same-day ambiguity.
        if hit_sl:
            exit_kind = "SL"
            exit_i = j
            exit_price = new_stop
            break

        if hit_tp:
            exit_kind = "TP"
            exit_i = j
            exit_price = target
            break

    gross = 100 * (exit_price / entry - 1)
    net = gross - COST

    return {
        "status": "FILLED",
        "original_stop": original_stop,
        "stop_touch_date": pd.Timestamp(arr.iloc[stop_i]["d"]),
        "stop_touch_step": stop_i - i0,
        "buy_limit": buy_limit,
        "fill_date": pd.Timestamp(arr.iloc[fill_i]["d"]),
        "fill_step": fill_i - i0,
        "new_stop": new_stop,
        "target": target,
        "exit_kind": exit_kind,
        "exit_date": pd.Timestamp(arr.iloc[exit_i]["d"]),
        "exit_step_after_fill": exit_i - fill_i,
        "gross_return_pct": gross,
        "net_return_pct": net,
    }

def metrics(df):
    if df.empty:
        return {
            "n_filled": 0,
            "avg_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "payoff": np.nan,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "time_rate": np.nan,
            "avg_fill_step": np.nan,
            "avg_hold_after_fill": np.nan,
        }

    r = df["net_return_pct"].to_numpy(float)
    wins = r[r > 0]
    losses = r[r <= 0]

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0

    payoff = (
        float("inf") if len(losses) == 0
        else (avg_win / avg_loss if avg_loss > 0 else float("inf"))
    )

    return {
        "n_filled": int(len(df)),
        "avg_net": float(r.mean()),
        "median_net": float(np.median(r)),
        "win_rate": float(100 * (r > 0).mean()),
        "payoff": payoff,
        "tp_rate": float(100 * (df["exit_kind"] == "TP").mean()),
        "sl_rate": float(100 * (df["exit_kind"] == "SL").mean()),
        "time_rate": float(100 * (df["exit_kind"] == "TIME").mean()),
        "avg_fill_step": float(df["fill_step"].mean()),
        "avg_hold_after_fill": float(df["exit_step_after_fill"].mean()),
    }

def main():
    print("[load] reading V15 early-stop trades...", flush=True)
    trades = load_early_stops()

    print(
        "[trades] early-stop candidates:",
        trades.groupby("year").size().to_dict(),
        flush=True
    )

    print("[load] loading daily candles...", flush=True)
    full = v11.load_daily().copy()
    full["d"] = pd.to_datetime(full["d"]).dt.normalize()
    full = full.sort_values(["symbol", "d"])

    by_symbol = {
        s: g.copy()
        for s, g in full.groupby("symbol", sort=False)
    }

    rows = []

    for i, r in trades.iterrows():
        sym = str(r["symbol"])
        if sym not in by_symbol:
            continue

        sim = simulate_trade(
            by_symbol[sym],
            pd.Timestamp(r["entry_date"]),
            float(r["entry_price"]),
        )

        rows.append({
            "year": int(r["year"]),
            "symbol": sym,
            "signal_date": pd.Timestamp(r["entry_date"]),
            "original_entry": float(r["entry_price"]),
            "original_exit_kind": str(r["exit_kind"]),
            "original_exit_step": float(r["exit_step"]),
            **sim,
        })

        if (i + 1) % 100 == 0:
            print(f"[progress] {i+1}/{len(trades)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False)

    summaries = []

    for year in [2025, 2026]:
        y = detail[detail["year"] == year].copy()
        total = len(y)
        filled = y[y["status"] == "FILLED"].copy()
        not_filled = int((y["status"] == "NOT_FILLED").sum())

        m = metrics(filled)

        summaries.append({
            "year": year,
            "early_stop_candidates": total,
            "filled_at_minus18": len(filled),
            "not_filled_at_minus18": not_filled,
            "fill_rate_pct": 100 * len(filled) / total if total else np.nan,
            **m,
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_SUMMARY, index=False)

    lines = [
        "V15 DEEP PULLBACK -18% / TP +5% / SL -5% ANALYSIS",
        "=" * 105,
        "",
        "RULES",
        "-" * 105,
        "Only trades that hit the original -3% stop within sessions 1-3 are studied.",
        "After that early stop, Buy Limit = 18% below original entry.",
        "Target = 5% above the new buy price.",
        "Stop = 5% below the new buy price.",
        "Limit stays active through session 12 after original signal.",
        "After fill, target/stop are monitored for 12 sessions.",
        "Same-day TP+SL ambiguity is counted as SL first.",
        f"Round-trip cost deducted: {COST:.2f} percentage points.",
        "",
        "Example if original entry = 100:",
        "  Original early stop = 97",
        "  New buy = 82",
        "  New stop = 77.90",
        "  New target = 86.10",
        "",
        "SUMMARY",
        "-" * 105,
        summary.to_string(index=False),
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(
        f"\nSaved:\n{OUT_SUMMARY}\n{OUT_DETAIL}\n{OUT_REPORT}",
        flush=True
    )

if __name__ == "__main__":
    main()
