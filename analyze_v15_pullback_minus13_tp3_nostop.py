#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 No-Stop Deep Pullback -13% / Rebound +3% Analyzer
=====================================================
Strategy:
1) Use only V15 trades that hit the ORIGINAL -3% stop within sessions 1-3.
2) After that early stop, place a Buy Limit at 13% below ORIGINAL entry.
3) NO stop loss.
4) Exit at +3% above the NEW buy price.
5) Limit remains active through session 12 after the original signal.
6) After fill, monitor target for up to 12 sessions.
7) If target is not hit, close at the final close of the 12-session hold.
8) Round-trip cost = 0.30 percentage points.
9) Report drawdown after entry because there is no stop.

Example:
  Original entry = 100
  New buy = 87
  Target = 89.61

Reads:
  /data/us_v15_trades_2025.csv
  /data/us_v15_trades_2026.csv

Requires:
  /app/us_setup_optimizer_v11.py
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

OUT_SUMMARY = DATA / "us_v15_pullback13_tp3_nostop_summary.csv"
OUT_DETAIL = DATA / "us_v15_pullback13_tp3_nostop_detail.csv"
OUT_REPORT = DATA / "us_v15_pullback13_tp3_nostop_report.txt"

BUY_DISCOUNT_PCT = 13.0
TARGET_PCT = 3.0
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
        t = t[(t["exit_kind"] == "SL") & (t["exit_step"] <= EARLY_STOP_MAX_SESSION)].copy()
        out.append(t)
    return pd.concat(out, ignore_index=True)

def simulate(symbol_df, signal_date, original_entry):
    arr = symbol_df.reset_index(drop=True)
    hits = np.flatnonzero(arr["d"].to_numpy() == np.datetime64(signal_date))
    if not len(hits):
        return {"status":"NO_SIGNAL_ROW"}

    i0 = int(hits[0])

    original_stop = original_entry * 0.97
    buy_price = original_entry * (1 - BUY_DISCOUNT_PCT/100)
    target = buy_price * (1 + TARGET_PCT/100)

    # Confirm early -3% stop hit.
    stop_i = None
    for j in range(i0+1, min(len(arr), i0+EARLY_STOP_MAX_SESSION+1)):
        if float(arr.iloc[j]["l"]) <= original_stop:
            stop_i = j
            break
    if stop_i is None:
        return {"status":"NO_EARLY_STOP_FOUND"}

    # Look for fill at -13% from original entry.
    fill_i = None
    expiry_i = min(len(arr)-1, i0 + LIMIT_EXPIRY_SESSION)
    for j in range(stop_i, expiry_i+1):
        if float(arr.iloc[j]["l"]) <= buy_price:
            fill_i = j
            break

    if fill_i is None:
        return {"status":"NOT_FILLED","buy_price":buy_price,"target":target}

    end_i = min(len(arr)-1, fill_i + POST_FILL_HOLD)

    exit_kind = "TIME"
    exit_i = end_i
    exit_price = float(arr.iloc[end_i]["c"])

    for j in range(fill_i, end_i+1):
        if float(arr.iloc[j]["h"]) >= target:
            exit_kind = "TP"
            exit_i = j
            exit_price = target
            break

    lows = arr.iloc[fill_i:exit_i+1]["l"].astype(float).to_numpy()
    worst_low = float(np.min(lows)) if len(lows) else buy_price
    max_drawdown_pct = 100 * (worst_low / buy_price - 1)

    gross = 100 * (exit_price / buy_price - 1)
    net = gross - COST

    return {
        "status":"FILLED",
        "buy_price":buy_price,
        "target":target,
        "fill_date":pd.Timestamp(arr.iloc[fill_i]["d"]),
        "fill_step":fill_i-i0,
        "exit_kind":exit_kind,
        "exit_date":pd.Timestamp(arr.iloc[exit_i]["d"]),
        "exit_step_after_fill":exit_i-fill_i,
        "gross_return_pct":gross,
        "net_return_pct":net,
        "max_drawdown_pct":max_drawdown_pct,
    }

def metrics(df):
    if df.empty:
        return {
            "n_filled":0,"avg_net":np.nan,"median_net":np.nan,"win_rate":np.nan,
            "tp_rate":np.nan,"time_rate":np.nan,"avg_fill_step":np.nan,
            "avg_hold_after_fill":np.nan,"median_max_drawdown_pct":np.nan,
            "worst_max_drawdown_pct":np.nan,"p25_max_drawdown_pct":np.nan
        }

    r = df["net_return_pct"].to_numpy(float)
    dd = df["max_drawdown_pct"].to_numpy(float)

    return {
        "n_filled":int(len(df)),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "tp_rate":float(100*(df["exit_kind"]=="TP").mean()),
        "time_rate":float(100*(df["exit_kind"]=="TIME").mean()),
        "avg_fill_step":float(df["fill_step"].mean()),
        "avg_hold_after_fill":float(df["exit_step_after_fill"].mean()),
        "median_max_drawdown_pct":float(np.median(dd)),
        "worst_max_drawdown_pct":float(np.min(dd)),
        "p25_max_drawdown_pct":float(np.percentile(dd,25)),
    }

def main():
    print("[load] reading V15 early-stop trades...", flush=True)
    trades = load_early_stops()
    print("[trades]", trades.groupby("year").size().to_dict(), flush=True)

    print("[load] loading US daily candles...", flush=True)
    full = v11.load_daily().copy()
    full["d"] = pd.to_datetime(full["d"]).dt.normalize()
    full = full.sort_values(["symbol","d"])
    by_symbol = {s:g.copy() for s,g in full.groupby("symbol", sort=False)}

    rows = []

    for i, r in trades.iterrows():
        sym = str(r["symbol"])
        if sym not in by_symbol:
            continue

        sim = simulate(
            by_symbol[sym],
            pd.Timestamp(r["entry_date"]),
            float(r["entry_price"]),
        )

        rows.append({
            "year":int(r["year"]),
            "symbol":sym,
            "signal_date":pd.Timestamp(r["entry_date"]),
            "original_entry":float(r["entry_price"]),
            **sim,
        })

        if (i+1) % 100 == 0:
            print(f"[progress] {i+1}/{len(trades)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False)

    summaries = []

    for year in [2025, 2026]:
        y = detail[detail["year"]==year].copy()
        total = len(y)
        filled = y[y["status"]=="FILLED"].copy()
        not_filled = int((y["status"]=="NOT_FILLED").sum())
        m = metrics(filled)

        summaries.append({
            "year":year,
            "early_stop_candidates":total,
            "filled_at_minus13":len(filled),
            "not_filled_at_minus13":not_filled,
            "fill_rate_pct":100*len(filled)/total if total else np.nan,
            **m,
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_SUMMARY, index=False)

    lines = [
        "V15 NO-STOP PULLBACK -13% / REBOUND +3% ANALYSIS",
        "="*105,
        "",
        "RULES",
        "-"*105,
        "Only trades that hit the original -3% stop within sessions 1-3 are studied.",
        "Buy Limit = 13% below original entry.",
        "No stop loss.",
        "Target = 3% above the new buy price.",
        "Limit remains active through session 12 after original signal.",
        "After fill, target is monitored for 12 sessions.",
        "If target is not hit, close at the final close of the hold window.",
        f"Round-trip cost deducted: {COST:.2f} percentage points.",
        "",
        "Example if original entry = 100:",
        "  Buy = 87",
        "  Target = 89.61",
        "",
        "SUMMARY",
        "-"*105,
        summary.to_string(index=False),
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nSaved:\n{OUT_SUMMARY}\n{OUT_DETAIL}\n{OUT_REPORT}", flush=True)

if __name__=="__main__":
    main()
