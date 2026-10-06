#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze what happened AFTER trades hit the original stop within first 3 sessions.

Basis:
- V15 selected trades:
    /data/us_v15_trades_2025.csv
    /data/us_v15_trades_2026.csv
- Original selected setup B_18_3_12
- Original stop = -3% from original entry
- We only study trades whose original stop was hit within sessions 1-3.

For each stopped trade, from the STOP-HIT session through session 12 after the original signal:
1) Lowest low reached, relative to original entry.
2) Highest high reached, relative to original entry.
3) Maximum further drop BELOW the -3% stop.
4) Maximum rebound from the post-stop low.
5) Whether price recovered to:
      - original entry
      - +3%
      - +5%
      - +10%
      - +18%
6) Day of post-stop low and day of best rebound.

Outputs:
  /data/v15_early_stop_aftermath_detail.csv
  /data/v15_early_stop_aftermath_summary.csv
  /data/v15_early_stop_aftermath_report.txt
"""

import sys, subprocess
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
    raise RuntimeError("Put /app/us_setup_optimizer_v11.py beside this analyzer") from e

DATA = Path("/data")
FILES = {
    2025: DATA / "us_v15_trades_2025.csv",
    2026: DATA / "us_v15_trades_2026.csv",
}
DETAIL = DATA / "v15_early_stop_aftermath_detail.csv"
SUMMARY = DATA / "v15_early_stop_aftermath_summary.csv"
REPORT = DATA / "v15_early_stop_aftermath_report.txt"

ORIG_STOP_PCT = 3.0
ORIG_HOLD = 12

def load_trades():
    out=[]
    for year,p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        t=pd.read_csv(p)
        t["entry_date"]=pd.to_datetime(t["entry_date"]).dt.normalize()
        t["year"]=year
        t["exit_step"]=pd.to_numeric(t["exit_step"], errors="coerce")
        # early stop = SL within first 3 sessions
        t=t[(t["exit_kind"]=="SL") & (t["exit_step"]<=3)].copy()
        out.append(t)
    return pd.concat(out, ignore_index=True)

def analyze_one(symdf, signal_date, entry):
    arr=symdf.reset_index(drop=True)
    hits=np.flatnonzero(arr["d"].to_numpy()==np.datetime64(signal_date))
    if not len(hits):
        return None
    i0=int(hits[0])
    stop_px=entry*(1-ORIG_STOP_PCT/100)

    # Find first stop touch in sessions 1..3
    stop_i=None
    for j in range(i0+1, min(len(arr), i0+4)):
        if float(arr.iloc[j]["l"]) <= stop_px:
            stop_i=j
            break
    if stop_i is None:
        return None

    end_i=min(len(arr)-1, i0+ORIG_HOLD)
    window=arr.iloc[stop_i:end_i+1].copy()
    if window.empty:
        return None

    lows=window["l"].astype(float).to_numpy()
    highs=window["h"].astype(float).to_numpy()

    low_rel=100*(lows/entry-1)
    high_rel=100*(highs/entry-1)

    min_idx=int(np.argmin(low_rel))
    max_idx=int(np.argmax(high_rel))

    lowest_pct=float(low_rel[min_idx])
    highest_pct=float(high_rel[max_idx])

    # Further drop below original -3% stop, expressed in percentage points.
    further_below_stop=max(0.0, -(lowest_pct + ORIG_STOP_PCT))

    # Rebound from the post-stop low to the best subsequent high.
    low_abs=float(lows[min_idx])
    subsequent_highs=highs[min_idx:]
    best_after_low=float(np.max(subsequent_highs))
    rebound_from_low_pct=100*(best_after_low/low_abs-1)

    # Max rebound versus original entry after stop.
    max_recovery_vs_entry_pct=highest_pct

    recovered_entry=bool(np.max(highs) >= entry)
    rec3=bool(np.max(highs) >= entry*1.03)
    rec5=bool(np.max(highs) >= entry*1.05)
    rec10=bool(np.max(highs) >= entry*1.10)
    rec18=bool(np.max(highs) >= entry*1.18)

    return {
        "stop_step": stop_i-i0,
        "stop_date": pd.Timestamp(arr.iloc[stop_i]["d"]),
        "lowest_after_stop_pct_vs_entry": lowest_pct,
        "further_drop_below_stop_pp": further_below_stop,
        "lowest_day_from_signal": (stop_i-i0)+min_idx,
        "highest_after_stop_pct_vs_entry": highest_pct,
        "highest_day_from_signal": (stop_i-i0)+max_idx,
        "rebound_from_poststop_low_pct": rebound_from_low_pct,
        "recovered_original_entry": recovered_entry,
        "reached_plus3": rec3,
        "reached_plus5": rec5,
        "reached_plus10": rec10,
        "reached_plus18": rec18,
    }

def q(x,p):
    return float(np.nanpercentile(x,p)) if len(x) else np.nan

def main():
    print("[load] reading early-stop V15 trades...", flush=True)
    trades=load_trades()
    print(f"[trades] early stops total={len(trades)}", flush=True)

    print("[load] loading daily candles...", flush=True)
    full=v11.load_daily().copy()
    full["d"]=pd.to_datetime(full["d"]).dt.normalize()
    full=full.sort_values(["symbol","d"])
    bysym={s:g.copy() for s,g in full.groupby("symbol",sort=False)}

    rows=[]
    for i,r in trades.iterrows():
        sym=str(r["symbol"])
        if sym not in bysym:
            continue
        a=analyze_one(bysym[sym], pd.Timestamp(r["entry_date"]), float(r["entry_price"]))
        if a is None:
            continue
        rows.append({
            "year":int(r["year"]),
            "symbol":sym,
            "signal_date":pd.Timestamp(r["entry_date"]),
            "original_entry":float(r["entry_price"]),
            **a
        })
        if (i+1)%100==0:
            print(f"[progress] {i+1}/{len(trades)}", flush=True)

    d=pd.DataFrame(rows)
    d.to_csv(DETAIL,index=False)

    sums=[]
    for year in [2025,2026]:
        y=d[d["year"]==year].copy()
        if y.empty:
            continue

        lo=y["lowest_after_stop_pct_vs_entry"].to_numpy(float)
        hi=y["highest_after_stop_pct_vs_entry"].to_numpy(float)
        reb=y["rebound_from_poststop_low_pct"].to_numpy(float)
        further=y["further_drop_below_stop_pp"].to_numpy(float)

        sums.append({
            "year":year,
            "early_stop_trades":len(y),

            "worst_fall_pct_vs_entry":float(np.min(lo)),
            "median_low_pct_vs_entry":float(np.median(lo)),
            "p25_low_pct_vs_entry":q(lo,25),
            "least_bad_low_pct_vs_entry":float(np.max(lo)),

            "max_further_drop_below_minus3_pp":float(np.max(further)),
            "median_further_drop_below_minus3_pp":float(np.median(further)),

            "best_recovery_pct_vs_entry":float(np.max(hi)),
            "median_best_recovery_pct_vs_entry":float(np.median(hi)),
            "least_recovery_pct_vs_entry":float(np.min(hi)),

            "max_rebound_from_low_pct":float(np.max(reb)),
            "median_rebound_from_low_pct":float(np.median(reb)),
            "min_rebound_from_low_pct":float(np.min(reb)),

            "recovered_entry_pct":100*y["recovered_original_entry"].mean(),
            "reached_plus3_pct":100*y["reached_plus3"].mean(),
            "reached_plus5_pct":100*y["reached_plus5"].mean(),
            "reached_plus10_pct":100*y["reached_plus10"].mean(),
            "reached_plus18_pct":100*y["reached_plus18"].mean(),
        })

    s=pd.DataFrame(sums)
    s.to_csv(SUMMARY,index=False)

    lines=[
        "V15 EARLY-STOP AFTERMATH ANALYSIS",
        "="*105,
        "",
        "Window: from first -3% stop touch (sessions 1-3) through session 12 after original signal.",
        "",
        "SUMMARY",
        "-"*105,
        s.to_string(index=False),
        "",
        "Definitions:",
        "- worst_fall_pct_vs_entry: deepest low after the stop, relative to original entry.",
        "- max_further_drop_below_minus3_pp: how many extra percentage points it fell below the -3% stop.",
        "- best_recovery_pct_vs_entry: highest high after stop, relative to original entry.",
        "- least_recovery_pct_vs_entry: weakest 'best bounce' among all stopped trades.",
        "- rebound_from_low_pct: bounce from the post-stop low to the best later high.",
    ]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{DETAIL}\n{SUMMARY}\n{REPORT}",flush=True)

if __name__=="__main__":
    main()
