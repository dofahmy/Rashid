#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
FILES={2025:DATA/"us_v15_trades_2025.csv",2026:DATA/"us_v15_trades_2026.csv"}
OUT_SUMMARY=DATA/"us_v15_pullback5_summary.csv"
OUT_DETAIL=DATA/"us_v15_pullback5_detail.csv"
OUT_REPORT=DATA/"us_v15_pullback5_report.txt"

ORIG_SL_PCT=3.0
BUY_DISCOUNT_PCT=5.0
NEW_SL_PCT=5.0
WAIT_SESSIONS=3
LIMIT_EXPIRY_SESSION=12
POST_FILL_HOLD=12
COST=0.30

def load_selected():
    out=[]
    for year,p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        t=pd.read_csv(p)
        t["entry_date"]=pd.to_datetime(t["entry_date"]).dt.normalize()
        t["year"]=year
        out.append(t)
    return pd.concat(out,ignore_index=True)

def metrics(df):
    if df.empty:
        return {"n_filled":0,"avg_net":np.nan,"median_net":np.nan,"win_rate":np.nan,
                "payoff":np.nan,"tp_rate":np.nan,"sl_rate":np.nan,"time_rate":np.nan,
                "avg_days_to_fill":np.nan,"avg_hold_after_fill":np.nan}
    r=df["net_return_pct"].to_numpy(float)
    w=r[r>0]; l=r[r<=0]
    aw=float(w.mean()) if len(w) else 0.0
    al=float(abs(l.mean())) if len(l) else 0.0
    payoff=float("inf") if len(l)==0 else (aw/al if al>0 else float("inf"))
    return {
        "n_filled":len(df),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "payoff":payoff,
        "tp_rate":float(100*(df["exit_kind"]=="TP").mean()),
        "sl_rate":float(100*(df["exit_kind"]=="SL").mean()),
        "time_rate":float(100*(df["exit_kind"]=="TIME").mean()),
        "avg_days_to_fill":float(df["fill_step"].mean()),
        "avg_hold_after_fill":float(df["exit_step_after_fill"].mean()),
    }

def simulate(symbol_df, signal_date, original_entry):
    arr=symbol_df.reset_index(drop=True)
    hits=np.flatnonzero(arr["d"].to_numpy()==np.datetime64(signal_date))
    if not len(hits):
        return {"status":"NO_SIGNAL_ROW"}
    i0=int(hits[0])

    orig_stop=original_entry*(1-ORIG_SL_PCT/100)
    buy_limit=original_entry*(1-BUY_DISCOUNT_PCT/100)
    new_stop=buy_limit*(1-NEW_SL_PCT/100)
    target=original_entry

    survive_end=i0+WAIT_SESSIONS
    if survive_end>=len(arr):
        return {"status":"INSUFFICIENT_DATA"}

    pre=arr.iloc[i0+1:survive_end+1]
    if len(pre) and (pre["l"].astype(float)<=orig_stop).any():
        return {"status":"FAILED_BEFORE_ELIGIBLE","buy_limit":buy_limit,"new_stop":new_stop,"target":target}

    expiry=min(len(arr)-1,i0+LIMIT_EXPIRY_SESSION)
    fill_i=None
    for j in range(i0+WAIT_SESSIONS+1,expiry+1):
        if float(arr.iloc[j]["l"])<=buy_limit:
            fill_i=j
            break
    if fill_i is None:
        return {"status":"NOT_FILLED","buy_limit":buy_limit,"new_stop":new_stop,"target":target}

    entry=buy_limit
    end=min(len(arr)-1,fill_i+POST_FILL_HOLD)
    exit_kind="TIME"; exit_i=end; exit_price=float(arr.iloc[end]["c"])

    for j in range(fill_i,end+1):
        high=float(arr.iloc[j]["h"]); low=float(arr.iloc[j]["l"])
        hit_sl=low<=new_stop
        hit_tp=high>=target
        if hit_sl:
            exit_kind="SL"; exit_i=j; exit_price=new_stop; break
        if hit_tp:
            exit_kind="TP"; exit_i=j; exit_price=target; break

    gross=100*(exit_price/entry-1)
    net=gross-COST
    return {
        "status":"FILLED",
        "fill_date":pd.Timestamp(arr.iloc[fill_i]["d"]),
        "fill_step":fill_i-i0,
        "buy_limit":buy_limit,
        "new_stop":new_stop,
        "target":target,
        "exit_kind":exit_kind,
        "exit_date":pd.Timestamp(arr.iloc[exit_i]["d"]),
        "exit_step_after_fill":exit_i-fill_i,
        "gross_return_pct":gross,
        "net_return_pct":net,
    }

def main():
    print("[load] reading V15 selected trades...",flush=True)
    trades=load_selected()

    print("[load] loading US daily candles...",flush=True)
    full=v11.load_daily().copy()
    full["d"]=pd.to_datetime(full["d"]).dt.normalize()
    full=full.sort_values(["symbol","d"])
    by_symbol={s:g.copy() for s,g in full.groupby("symbol",sort=False)}

    rows=[]
    for i,r in trades.iterrows():
        sym=str(r["symbol"])
        if sym not in by_symbol:
            continue
        sim=simulate(by_symbol[sym],pd.Timestamp(r["entry_date"]),float(r["entry_price"]))
        rows.append({
            "year":int(r["year"]),
            "symbol":sym,
            "signal_date":pd.Timestamp(r["entry_date"]),
            "original_entry":float(r["entry_price"]),
            "original_exit_kind":str(r.get("exit_kind","")),
            "original_exit_step":float(r.get("exit_step",np.nan)) if pd.notna(r.get("exit_step",np.nan)) else np.nan,
            **sim
        })
        if (i+1)%100==0:
            print(f"[progress] {i+1}/{len(trades)}",flush=True)

    detail=pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL,index=False)

    sums=[]
    for year in [2025,2026]:
        y=detail[detail["year"]==year].copy()
        total=len(y)
        failed=int((y["status"]=="FAILED_BEFORE_ELIGIBLE").sum())
        survived=total-failed
        not_filled=int((y["status"]=="NOT_FILLED").sum())
        filled=y[y["status"]=="FILLED"].copy()
        m=metrics(filled)
        sums.append({
            "year":year,
            "original_selected_trades":total,
            "failed_original_stop_first_3":failed,
            "survived_first_3":survived,
            "not_filled_at_minus5":not_filled,
            "filled_at_minus5":len(filled),
            "fill_rate_of_all_pct":100*len(filled)/total if total else np.nan,
            "fill_rate_of_survivors_pct":100*len(filled)/survived if survived else np.nan,
            **m
        })

    summary=pd.DataFrame(sums)
    summary.to_csv(OUT_SUMMARY,index=False)

    lines=[
        "V15 PULLBACK -5% ENTRY ANALYSIS","="*95,"","RULES","-"*95,
        "Wait 3 sessions after original signal.",
        "If original -3% stop was not touched, place Buy Limit at original entry -5%.",
        "After fill: Stop = 5% below new buy price.",
        "Target = original entry price.",
        "Limit remains active through session 12 after the signal.",
        "After fill, target/stop are monitored for 12 sessions.",
        "Same-day TP+SL ambiguity is counted as SL first.",
        f"Round-trip cost deducted: {COST:.2f} percentage points.",
        "","SUMMARY","-"*95,summary.to_string(index=False)
    ]
    OUT_REPORT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{OUT_SUMMARY}\n{OUT_DETAIL}\n{OUT_REPORT}",flush=True)

if __name__=="__main__":
    main()
