#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze V15 TP timing and test delayed entry after a signal survives early SL.

Reads:
  /data/us_v15_trades_2025.csv
  /data/us_v15_trades_2026.csv

Uses:
  /app/us_setup_optimizer_v11.py
  DATABASE_URL via V11 loader

Reports:
1) When TP winners hit: sessions 1-3, 4-7, 8+
2) Delayed-entry simulations for:
   - enter on session 4 (after surviving sessions 1-3)
   - enter on session 5 (after surviving sessions 1-4)

For delayed simulations:
- eligibility: the ORIGINAL signal has not touched its original -3% stop before delayed entry.
- new entry price: delayed day's CLOSE.
- new TP/SL: recalculated from delayed entry price.
- hold window: 12 sessions from delayed entry for B_18_3_12.
- if TP and SL both hit same day, SL is counted first conservatively.
- cost: 0.30 percentage points deducted from realized gross return.
"""

import os, sys, subprocess, math
from pathlib import Path

def ensure(pkg, name=None):
    name = name or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

for pkg,name in [("numpy",None),("pandas",None)]:
    ensure(pkg,name)

import numpy as np
import pandas as pd

try:
    import us_setup_optimizer_v11 as v11
except Exception as e:
    raise RuntimeError("Put us_setup_optimizer_v11.py in /app beside this script") from e

DATA = Path("/data")
FILES = {
    2025: DATA / "us_v15_trades_2025.csv",
    2026: DATA / "us_v15_trades_2026.csv",
}
OUT_TIMING = DATA / "us_v15_tp_timing_analysis.csv"
OUT_DELAY = DATA / "us_v15_delayed_entry_analysis.csv"
OUT_REPORT = DATA / "us_v15_delayed_entry_report.txt"

TP_PCT = 18.0
SL_PCT = 3.0
HOLD = 12
COST = 0.30

def metrics(df):
    if df.empty:
        return dict(n=0, avg_net=np.nan, median_net=np.nan, win_rate=np.nan,
                    payoff=np.nan, tp_rate=np.nan, sl_rate=np.nan, time_rate=np.nan)
    r = df["net_return_pct"].to_numpy(float)
    w = r[r > 0]
    l = r[r <= 0]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(abs(l.mean())) if len(l) else 0.0
    payoff = float("inf") if len(l)==0 else (aw/al if al>0 else float("inf"))
    return {
        "n": int(len(df)),
        "avg_net": float(r.mean()),
        "median_net": float(np.median(r)),
        "win_rate": float(100*(r>0).mean()),
        "payoff": payoff,
        "tp_rate": float(100*(df["exit_kind"]=="TP").mean()),
        "sl_rate": float(100*(df["exit_kind"]=="SL").mean()),
        "time_rate": float(100*(df["exit_kind"]=="TIME").mean()),
    }

def load_selected():
    all_trades = []
    for year,p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        t = pd.read_csv(p)
        t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
        t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.normalize()
        t["year"] = year
        t["exit_step"] = pd.to_numeric(t["exit_step"], errors="coerce")
        all_trades.append(t)
    return pd.concat(all_trades, ignore_index=True)

def tp_timing(trades):
    rows = []
    for year in [2025, 2026]:
        y = trades[trades["year"]==year]
        tp = y[y["exit_kind"]=="TP"].copy()
        total = len(tp)
        buckets = [
            ("TP sessions 1-3", tp["exit_step"].between(1,3, inclusive="both")),
            ("TP sessions 4-7", tp["exit_step"].between(4,7, inclusive="both")),
            ("TP sessions 8-12", tp["exit_step"].between(8,12, inclusive="both")),
        ]
        for name,mask in buckets:
            n = int(mask.sum())
            rows.append({
                "year": year,
                "bucket": name,
                "count": n,
                "share_of_tp_pct": (100*n/total if total else np.nan),
                "total_tp": total,
                "total_trades": len(y),
            })
        if total:
            rows.append({
                "year":year, "bucket":"TP median session",
                "count":float(tp["exit_step"].median()),
                "share_of_tp_pct":np.nan,
                "total_tp":total, "total_trades":len(y),
            })
    return pd.DataFrame(rows)

def normalize_daily(full):
    f = full.copy()
    f["d"] = pd.to_datetime(f["d"]).dt.normalize()
    return f.sort_values(["symbol","d"])

def simulate_one(symbol_df, signal_date, original_entry, delay_session):
    # signal row = session 0. delay_session=4 means enter on 4th session AFTER signal,
    # i.e. after observing the first 3 following sessions.
    arr = symbol_df.reset_index(drop=True)
    hits = np.flatnonzero(arr["d"].to_numpy() == np.datetime64(signal_date))
    if not len(hits):
        return None
    i0 = int(hits[0])

    # Need enough bars to reach delayed entry.
    idelay = i0 + delay_session
    if idelay >= len(arr):
        return None

    original_stop = float(original_entry) * (1.0 - SL_PCT/100.0)

    # Eligibility: original signal must survive all bars BEFORE delayed entry.
    # Conservative: touching original stop at any prior bar disqualifies it.
    pre = arr.iloc[i0+1:idelay]
    if len(pre) and (pre["l"].astype(float) <= original_stop).any():
        return {
            "eligible": False,
            "reason": "original_stop_hit_before_delay",
        }

    entry_date = pd.Timestamp(arr.iloc[idelay]["d"])
    entry = float(arr.iloc[idelay]["c"])
    tp_px = entry * (1.0 + TP_PCT/100.0)
    sl_px = entry * (1.0 - SL_PCT/100.0)

    end = min(len(arr)-1, idelay + HOLD)
    exit_kind = "TIME"
    exit_step = end - idelay
    exit_px = float(arr.iloc[end]["c"])

    for j in range(idelay+1, end+1):
        high = float(arr.iloc[j]["h"])
        low = float(arr.iloc[j]["l"])

        hit_sl = low <= sl_px
        hit_tp = high >= tp_px

        # Conservative same-day ambiguity: SL first.
        if hit_sl:
            exit_kind = "SL"
            exit_step = j-idelay
            exit_px = sl_px
            break
        if hit_tp:
            exit_kind = "TP"
            exit_step = j-idelay
            exit_px = tp_px
            break

    gross = 100.0 * (exit_px/entry - 1.0)
    net = gross - COST

    return {
        "eligible": True,
        "reason": "",
        "delayed_entry_date": entry_date,
        "delayed_entry_price": entry,
        "exit_kind": exit_kind,
        "exit_step": exit_step,
        "net_return_pct": net,
    }

def delayed_test(trades, full):
    by_symbol = {s:g.copy() for s,g in full.groupby("symbol", sort=False)}
    detail = []

    for _,r in trades.iterrows():
        sym = str(r["symbol"])
        if sym not in by_symbol:
            continue
        for delay in [4,5]:
            sim = simulate_one(
                by_symbol[sym],
                pd.Timestamp(r["entry_date"]),
                float(r["entry_price"]),
                delay,
            )
            if sim is None:
                continue
            row = {
                "year": int(r["year"]),
                "symbol": sym,
                "original_entry_date": pd.Timestamp(r["entry_date"]),
                "original_exit_kind": str(r["exit_kind"]),
                "original_exit_step": float(r["exit_step"]) if pd.notna(r["exit_step"]) else np.nan,
                "delay_session": delay,
                **sim,
            }
            detail.append(row)

    d = pd.DataFrame(detail)
    summaries = []

    for year in [2025,2026]:
        for delay in [4,5]:
            x = d[(d["year"]==year)&(d["delay_session"]==delay)]
            elig = x[x["eligible"]==True].copy()
            m = metrics(elig) if len(elig) else metrics(pd.DataFrame())
            summaries.append({
                "year":year,
                "delay_session":delay,
                "original_selected_trades":int((trades["year"]==year).sum()),
                "eligible_after_survival":len(elig),
                "eligibility_pct":100*len(elig)/max(1,int((trades["year"]==year).sum())),
                **m,
            })

    return pd.DataFrame(summaries), d

def main():
    print("[load] reading V15 selected trades...", flush=True)
    trades = load_selected()

    timing = tp_timing(trades)
    timing.to_csv(OUT_TIMING, index=False)

    print("[load] loading daily candles for delayed-entry simulation...", flush=True)
    full = normalize_daily(v11.load_daily())

    summary, detail = delayed_test(trades, full)
    summary.to_csv(OUT_DELAY, index=False)
    detail.to_csv(DATA/"us_v15_delayed_entry_detail.csv", index=False)

    lines = [
        "V15 TP TIMING + DELAYED ENTRY ANALYSIS",
        "="*90,
        "",
        "WHEN DID WINNERS HIT TP?",
        "-"*90,
        timing.to_string(index=False),
        "",
        "DELAYED ENTRY RESULTS",
        "-"*90,
        summary.to_string(index=False),
        "",
        "Interpretation:",
        "- delay_session=4: wait through the first 3 sessions; enter at session 4 close if original -3% stop was never touched.",
        "- delay_session=5: wait through the first 4 sessions; enter at session 5 close if original -3% stop was never touched.",
        "- TP/SL are recalculated from the delayed entry price.",
        "- Hold is 12 sessions from delayed entry.",
        "- Same-day TP+SL ambiguity is counted as SL first (conservative).",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nSaved:\n{OUT_TIMING}\n{OUT_DELAY}\n{OUT_REPORT}", flush=True)

if __name__ == "__main__":
    main()
