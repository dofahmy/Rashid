#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V15 Pullback -13% / TP +3% / MAX HOLD 2 SESSIONS
=================================================
Rules:
1) Use only V15 trades that hit the ORIGINAL -3% stop within sessions 1-3.
2) Buy Limit = 13% below ORIGINAL entry.
3) No stop loss.
4) Target = +3% above the NEW buy price.
5) Buy limit remains active through session 12 after the original signal.
6) After fill:
   - if +3% target is hit on fill day / next session / second session, exit at target.
   - otherwise force-close at the CLOSE of session 2 after fill.
7) Round-trip cost = 0.30 percentage points.

Example:
  Original entry = 100
  Buy = 87
  Target = 89.61
  If target is not hit within max 2 sessions after fill -> close at end of session 2.
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

DATA=Path("/data")
FILES={2025:DATA/"us_v15_trades_2025.csv", 2026:DATA/"us_v15_trades_2026.csv"}

OUT_SUMMARY=DATA/"us_v15_pullback13_tp3_hold2_summary.csv"
OUT_DETAIL=DATA/"us_v15_pullback13_tp3_hold2_detail.csv"
OUT_REPORT=DATA/"us_v15_pullback13_tp3_hold2_report.txt"

BUY_DISCOUNT_PCT=13.0
TARGET_PCT=3.0
EARLY_STOP_MAX_SESSION=3
LIMIT_EXPIRY_SESSION=12
MAX_HOLD_AFTER_FILL=2
COST=0.30

def load_early_stops():
    out=[]
    for year,p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        t=pd.read_csv(p)
        t["entry_date"]=pd.to_datetime(t["entry_date"]).dt.normalize()
        t["exit_step"]=pd.to_numeric(t["exit_step"],errors="coerce")
        t["year"]=year
        t=t[(t["exit_kind"]=="SL") & (t["exit_step"]<=EARLY_STOP_MAX_SESSION)].copy()
        out.append(t)
    return pd.concat(out,ignore_index=True)

def simulate(symdf, signal_date, original_entry):
    arr=symdf.reset_index(drop=True)
    hits=np.flatnonzero(arr["d"].to_numpy()==np.datetime64(signal_date))
    if not len(hits):
        return {"status":"NO_SIGNAL_ROW"}
    i0=int(hits[0])

    original_stop=original_entry*0.97
    buy_price=original_entry*(1-BUY_DISCOUNT_PCT/100)
    target=buy_price*(1+TARGET_PCT/100)

    stop_i=None
    for j in range(i0+1,min(len(arr),i0+EARLY_STOP_MAX_SESSION+1)):
        if float(arr.iloc[j]["l"])<=original_stop:
            stop_i=j
            break
    if stop_i is None:
        return {"status":"NO_EARLY_STOP_FOUND"}

    fill_i=None
    expiry_i=min(len(arr)-1,i0+LIMIT_EXPIRY_SESSION)
    for j in range(stop_i,expiry_i+1):
        if float(arr.iloc[j]["l"])<=buy_price:
            fill_i=j
            break

    if fill_i is None:
        return {"status":"NOT_FILLED","buy_price":buy_price,"target":target}

    end_i=min(len(arr)-1, fill_i+MAX_HOLD_AFTER_FILL)

    exit_kind="TIME2"
    exit_i=end_i
    exit_price=float(arr.iloc[end_i]["c"])

    for j in range(fill_i,end_i+1):
        if float(arr.iloc[j]["h"])>=target:
            exit_kind="TP"
            exit_i=j
            exit_price=target
            break

    lows=arr.iloc[fill_i:exit_i+1]["l"].astype(float).to_numpy()
    worst_low=float(np.min(lows)) if len(lows) else buy_price
    max_drawdown_pct=100*(worst_low/buy_price-1)

    gross=100*(exit_price/buy_price-1)
    net=gross-COST

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
        return {}
    r=df["net_return_pct"].to_numpy(float)
    dd=df["max_drawdown_pct"].to_numpy(float)
    tp=(df["exit_kind"]=="TP")
    return {
        "n_filled":len(df),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "tp_rate":float(100*tp.mean()),
        "forced_close_rate":float(100*(df["exit_kind"]=="TIME2").mean()),
        "avg_hold_after_fill":float(df["exit_step_after_fill"].mean()),
        "median_max_drawdown_pct":float(np.median(dd)),
        "worst_max_drawdown_pct":float(np.min(dd)),
        "p25_max_drawdown_pct":float(np.percentile(dd,25)),
        "avg_forced_close_net":float(df.loc[df["exit_kind"]=="TIME2","net_return_pct"].mean()) if (~tp).any() else np.nan,
        "median_forced_close_net":float(df.loc[df["exit_kind"]=="TIME2","net_return_pct"].median()) if (~tp).any() else np.nan,
    }

def main():
    print("[load] reading V15 early-stop cohort...",flush=True)
    trades=load_early_stops()
    print("[trades]",trades.groupby("year").size().to_dict(),flush=True)

    print("[load] loading daily candles...",flush=True)
    full=v11.load_daily().copy()
    full["d"]=pd.to_datetime(full["d"]).dt.normalize()
    full=full.sort_values(["symbol","d"])
    bysym={s:g.copy() for s,g in full.groupby("symbol",sort=False)}

    rows=[]
    for i,r in trades.iterrows():
        sym=str(r["symbol"])
        if sym not in bysym:
            continue
        sim=simulate(bysym[sym],pd.Timestamp(r["entry_date"]),float(r["entry_price"]))
        rows.append({
            "year":int(r["year"]),
            "symbol":sym,
            "signal_date":pd.Timestamp(r["entry_date"]),
            "original_entry":float(r["entry_price"]),
            **sim
        })
        if (i+1)%100==0:
            print(f"[progress] {i+1}/{len(trades)}",flush=True)

    detail=pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL,index=False)

    summaries=[]
    for year in [2025,2026]:
        y=detail[detail["year"]==year].copy()
        total=len(y)
        filled=y[y["status"]=="FILLED"].copy()
        not_filled=int((y["status"]=="NOT_FILLED").sum())
        m=metrics(filled)
        summaries.append({
            "year":year,
            "early_stop_candidates":total,
            "filled_at_minus13":len(filled),
            "not_filled_at_minus13":not_filled,
            "fill_rate_pct":100*len(filled)/total if total else np.nan,
            **m
        })

    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT_SUMMARY,index=False)

    lines=[
        "V15 PULLBACK -13% / TP +3% / MAX HOLD 2 SESSIONS",
        "="*105,
        "",
        "RULES",
        "-"*105,
        "Buy at -13% from original entry after early -3% stop.",
        "No stop loss.",
        "Target = +3% above new buy.",
        "Maximum hold = 2 sessions after fill.",
        "If target is not reached, force-close at session-2 close.",
        f"Round-trip cost deducted: {COST:.2f} percentage points.",
        "",
        "SUMMARY",
        "-"*105,
        summary.to_string(index=False),
    ]

    OUT_REPORT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{OUT_SUMMARY}\n{OUT_DETAIL}\n{OUT_REPORT}",flush=True)

if __name__=="__main__":
    main()
