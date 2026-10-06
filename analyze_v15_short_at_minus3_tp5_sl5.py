#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 Early-Stop Short Analyzer
=============================
Strategy:

1) Use only trades that hit the ORIGINAL -3% stop within sessions 1-3.
2) Enter SHORT at the original -3% stop price.
3) Short target = 5% BELOW the short entry.
4) Short stop = 5% ABOVE the short entry.
5) Monitor for up to 12 sessions from short entry.
6) If target and stop are both touched on the same day, STOP is counted first conservatively.
7) Round-trip cost = 0.30 percentage points.

Example:
  Original long entry = 100
  Short entry = 97
  Short target = 92.15
  Short stop = 101.85

Reads:
  /data/us_v15_trades_2025.csv
  /data/us_v15_trades_2026.csv

Requires:
  /app/us_setup_optimizer_v11.py

Outputs:
  /data/us_v15_short_at_minus3_summary.csv
  /data/us_v15_short_at_minus3_detail.csv
  /data/us_v15_short_at_minus3_report.txt
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

for pkg,name in [("numpy",None),("pandas",None)]:
    ensure(pkg,name)

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

OUT_SUMMARY = DATA / "us_v15_short_at_minus3_summary.csv"
OUT_DETAIL = DATA / "us_v15_short_at_minus3_detail.csv"
OUT_REPORT = DATA / "us_v15_short_at_minus3_report.txt"

SHORT_TRIGGER_PCT = 3.0
SHORT_TARGET_PCT = 5.0
SHORT_STOP_PCT = 5.0
EARLY_STOP_MAX_SESSION = 3
POST_ENTRY_HOLD = 12
COST = 0.30

def load_early_stops():
    out = []
    for year,p in FILES.items():
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

def simulate_short(symbol_df, signal_date, original_entry):
    arr = symbol_df.reset_index(drop=True)

    hits = np.flatnonzero(arr["d"].to_numpy() == np.datetime64(signal_date))
    if not len(hits):
        return {"status":"NO_SIGNAL_ROW"}

    i0 = int(hits[0])

    short_entry = original_entry * (1.0 - SHORT_TRIGGER_PCT/100.0)
    target = short_entry * (1.0 - SHORT_TARGET_PCT/100.0)
    stop = short_entry * (1.0 + SHORT_STOP_PCT/100.0)

    # Find first touch of -3% within sessions 1-3.
    entry_i = None
    for j in range(i0+1, min(len(arr), i0+EARLY_STOP_MAX_SESSION+1)):
        low = float(arr.iloc[j]["l"])
        if low <= short_entry:
            entry_i = j
            break

    if entry_i is None:
        return {"status":"NO_SHORT_TRIGGER"}

    end_i = min(len(arr)-1, entry_i + POST_ENTRY_HOLD)

    exit_kind = "TIME"
    exit_i = end_i
    exit_price = float(arr.iloc[end_i]["c"])

    for j in range(entry_i, end_i+1):
        high = float(arr.iloc[j]["h"])
        low = float(arr.iloc[j]["l"])

        hit_target = low <= target
        hit_stop = high >= stop

        # Conservative ambiguity: stop first.
        if hit_stop:
            exit_kind = "SL"
            exit_i = j
            exit_price = stop
            break

        if hit_target:
            exit_kind = "TP"
            exit_i = j
            exit_price = target
            break

    # Short return: entry/exit direction reversed.
    gross = 100.0 * (short_entry/exit_price - 1.0)
    net = gross - COST

    return {
        "status":"FILLED",
        "short_entry":short_entry,
        "entry_date":pd.Timestamp(arr.iloc[entry_i]["d"]),
        "entry_step":entry_i-i0,
        "target":target,
        "stop":stop,
        "exit_kind":exit_kind,
        "exit_date":pd.Timestamp(arr.iloc[exit_i]["d"]),
        "exit_step_after_entry":exit_i-entry_i,
        "gross_return_pct":gross,
        "net_return_pct":net,
    }

def metrics(df):
    if df.empty:
        return {
            "n":0, "avg_net":np.nan, "median_net":np.nan,
            "win_rate":np.nan, "payoff":np.nan,
            "tp_rate":np.nan, "sl_rate":np.nan, "time_rate":np.nan,
            "avg_entry_step":np.nan, "avg_hold":np.nan
        }

    r = df["net_return_pct"].to_numpy(float)
    wins = r[r>0]
    losses = r[r<=0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    payoff = float("inf") if len(losses)==0 else (avg_win/avg_loss if avg_loss>0 else float("inf"))

    return {
        "n":int(len(df)),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "payoff":payoff,
        "tp_rate":float(100*(df["exit_kind"]=="TP").mean()),
        "sl_rate":float(100*(df["exit_kind"]=="SL").mean()),
        "time_rate":float(100*(df["exit_kind"]=="TIME").mean()),
        "avg_entry_step":float(df["entry_step"].mean()),
        "avg_hold":float(df["exit_step_after_entry"].mean()),
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
    full = full.sort_values(["symbol","d"])

    by_symbol = {s:g.copy() for s,g in full.groupby("symbol", sort=False)}

    rows = []

    for i,r in trades.iterrows():
        sym = str(r["symbol"])
        if sym not in by_symbol:
            continue

        sim = simulate_short(
            by_symbol[sym],
            pd.Timestamp(r["entry_date"]),
            float(r["entry_price"]),
        )

        rows.append({
            "year":int(r["year"]),
            "symbol":sym,
            "signal_date":pd.Timestamp(r["entry_date"]),
            "original_long_entry":float(r["entry_price"]),
            "original_exit_step":float(r["exit_step"]),
            **sim,
        })

        if (i+1) % 100 == 0:
            print(f"[progress] {i+1}/{len(trades)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False)

    summaries = []

    for year in [2025,2026]:
        y = detail[(detail["year"]==year) & (detail["status"]=="FILLED")].copy()
        m = metrics(y)

        summaries.append({
            "year":year,
            "early_stop_candidates":int((detail["year"]==year).sum()),
            **m
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_SUMMARY, index=False)

    lines = [
        "V15 SHORT AT ORIGINAL -3% STOP ANALYSIS",
        "="*100,
        "",
        "RULES",
        "-"*100,
        "Only trades that hit the original -3% stop within sessions 1-3 are studied.",
        "Enter SHORT exactly at the original -3% stop price.",
        "Target = 5% below short entry.",
        "Stop = 5% above short entry.",
        "Hold up to 12 sessions after short entry.",
        "Same-day TP+SL ambiguity is counted as SL first.",
        f"Round-trip cost deducted: {COST:.2f} percentage points.",
        "",
        "Example if original long entry = 100:",
        "  Short entry = 97",
        "  Short target = 92.15",
        "  Short stop = 101.85",
        "",
        "SUMMARY",
        "-"*100,
        summary.to_string(index=False),
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(
        f"\nSaved:\n{OUT_SUMMARY}\n{OUT_DETAIL}\n{OUT_REPORT}",
        flush=True
    )

if __name__=="__main__":
    main()
