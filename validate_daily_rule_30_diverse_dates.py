#!/usr/bin/env python3
"""
Rajih — validation v2: 30 NEW setups across 30 DISTINCT dates.

Fixes the main flaw in v1:
- v1 chose the first chronological 30 matches, which all landed on 2024-08-19.
- v2 enforces one setup per calendar trading date and spreads the 30 samples
  across the full available history.

Selection uses ONLY information known at the signal date.
Future returns are calculated only AFTER the 30 samples are locked.

Locked rule
-----------
- $1 <= raw close < $20
- 20d avg dollar volume >= $1M
- worst opening gap over prior 20 sessions <= -9%
- average daily range over last 5 sessions >= 9.42%
- no >100% adjusted-close discontinuity in prior 20 sessions
- exclude original discovery 30 symbols (hardcoded fallback included)
- one selected setup per distinct signal date
- dates spread approximately evenly across all qualifying dates
- within each chosen date, select the highest 20d dollar-volume candidate
  (pre-signal information only)

Run:
    python validate_daily_rule_30_diverse_dates.py
"""

from __future__ import annotations

import os
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
from statistics import mean, median
from sqlalchemy import MetaData, Table, select, func
from core import database

DISCOVERY_SYMBOLS = {
    "SMCI","PATH","OPEN","RIVN","HIMS","MARA","BMNR","WBD","F","HL",
    "RGTI","CLF","AG","SOFI","SOUN","JOBY","WULF","RDW","DJT","GAP",
    "PSKY","SBET","LYFT","SRPT","VFC","BBAI","AEO","U","AIFF","SNAP"
}

def finite(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except: return None

def fmt(x,n=4):
    return None if x is None or not math.isfinite(float(x)) else round(float(x),n)

def ema_all(vals,n):
    if not vals: return []
    out=[None]*len(vals); a=2/(n+1); e=float(vals[0]); out[0]=e
    for i in range(1,len(vals)):
        e=a*float(vals[i])+(1-a)*e; out[i]=e
    return out

def atr_all(h,l,c,n=14):
    out=[None]*len(c)
    if len(c)<=n: return out
    tr=[None]*len(c)
    for i in range(1,len(c)):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=sum(tr[1:n+1])/n; out[n]=a
    for i in range(n+1,len(c)):
        a=((n-1)*a+tr[i])/n; out[i]=a
    return out

def rsi_all(c,n=14):
    out=[None]*len(c)
    if len(c)<=n: return out
    gains=[]; losses=[]
    for i in range(1,len(c)):
        d=c[i]-c[i-1]; gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains[:n])/n; al=sum(losses[:n])/n
    out[n]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    for i in range(n+1,len(c)):
        ag=((n-1)*ag+gains[i-1])/n; al=((n-1)*al+losses[i-1])/n
        out[i]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    return out

def adx_all(h,l,c,n=14):
    m=len(c); out=[None]*m
    if m<2*n+1: return out
    tr=[0.0]*m; pdm=[0.0]*m; mdm=[0.0]*m
    for i in range(1,m):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0.0
        mdm[i]=dn if dn>up and dn>0 else 0.0
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    atrs=sum(tr[1:n+1]); ps=sum(pdm[1:n+1]); ms=sum(mdm[1:n+1])
    dx=[None]*m
    for i in range(n,m):
        if i>n:
            atrs=atrs-atrs/n+tr[i]; ps=ps-ps/n+pdm[i]; ms=ms-ms/n+mdm[i]
        if atrs<=0: continue
        pdi=100*ps/atrs; mdi=100*ms/atrs; den=pdi+mdi
        dx[i]=0 if den==0 else 100*abs(pdi-mdi)/den
    seed=[x for x in dx[n:2*n] if x is not None]
    if len(seed)<n: return out
    a=sum(seed)/n; out[2*n-1]=a
    for i in range(2*n,m):
        if dx[i] is not None:
            a=((n-1)*a+dx[i])/n; out[i]=a
    return out

def stdev(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs)<2: return None
    mu=sum(xs)/len(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))

def load_symbol(DB,daily,sym):
    with DB() as s:
        raw=list(s.execute(select(
            daily.c.session_date,daily.c.o,daily.c.h,daily.c.l,
            daily.c.c,daily.c.v,daily.c.adj_c
        ).where(daily.c.symbol==sym).order_by(daily.c.session_date)).all())
    rows=[]
    for x in raw:
        ac=finite(x[6]); rc=finite(x[4])
        if ac is None or rc is None or ac<=0 or rc<=0: continue
        rows.append({"date":str(x[0]),"ro":float(x[1]),"rh":float(x[2]),
                     "rl":float(x[3]),"rc":rc,"v":float(x[5] or 0),"c":ac})
    return rows

def build(rows):
    ro=[r["ro"] for r in rows]; rh=[r["rh"] for r in rows]
    rl=[r["rl"] for r in rows]; rc=[r["rc"] for r in rows]
    v=[r["v"] for r in rows]; c=[r["c"] for r in rows]
    fac=[c[i]/rc[i] for i in range(len(c))]
    o=[ro[i]*fac[i] for i in range(len(c))]
    h=[rh[i]*fac[i] for i in range(len(c))]
    l=[rl[i]*fac[i] for i in range(len(c))]
    atr=atr_all(h,l,c,14); rsi=rsi_all(c,14); adx=adx_all(h,l,c,14)
    e20=ema_all(c,20); e50=ema_all(c,50)

    avgdol=[None]*len(c); r5=[None]*len(c); r20=[None]*len(c)
    mingap=[None]*len(c); rv=[None]*len(c); hi252=[None]*len(c)
    ret=[None]*len(c); gap=[None]*len(c)
    dollar=[rc[i]*v[i] for i in range(len(c))]
    for i in range(1,len(c)):
        ret[i]=c[i]/c[i-1]-1
        gap[i]=100*(o[i]/c[i-1]-1)
    for i in range(len(c)):
        if i>=4:
            r5[i]=mean(100*(h[j]-l[j])/c[j] for j in range(i-4,i+1))
        if i>=19:
            avgdol[i]=mean(dollar[i-19:i+1])
            r20[i]=mean(100*(h[j]-l[j])/c[j] for j in range(i-19,i+1))
            gg=[gap[j] for j in range(i-19,i+1) if gap[j] is not None]
            mingap[i]=min(gg) if gg else None
            rr=[ret[j] for j in range(i-19,i+1) if ret[j] is not None]
            rv[i]=stdev(rr)
        if i>=251:
            hi252[i]=max(h[i-251:i+1])
    return {"date":[r["date"] for r in rows],"rc":rc,"c":c,"h":h,"l":l,
            "atr":atr,"rsi":rsi,"adx":adx,"e20":e20,"e50":e50,
            "avgdol":avgdol,"r5":r5,"r20":r20,"mingap":mingap,"rv":rv,
            "hi252":hi252}

def discontinuity_ok(f,i,max_abs):
    if i<20: return False
    for j in range(i-19,i+1):
        if j<=0: continue
        if abs(100*(f["c"][j]/f["c"][j-1]-1))>max_abs: return False
    return True

def outcome(f,i,lookahead):
    j2=min(len(f["c"])-1,i+lookahead)
    if j2<=i: return None
    start=f["c"][i]; fut=range(i+1,j2+1)
    def maxh(n):
        jj=min(j2,i+n)
        return max(100*(f["h"][j]/start-1) for j in range(i+1,jj+1))
    def mae(n):
        jj=min(j2,i+n)
        return min(100*(f["l"][j]/start-1) for j in range(i+1,jj+1))
    peak=max(fut,key=lambda j:f["h"][j])
    def dto(th):
        for j in fut:
            if 100*(f["h"][j]/start-1)>=th: return j-i
        return None
    return {"max_high_5d_pct":fmt(maxh(5)),"max_high_10d_pct":fmt(maxh(10)),
            "max_high_21d_pct":fmt(maxh(21)),"mae_21d_pct":fmt(mae(21)),
            "days_to_20":dto(20),"days_to_30":dto(30),"days_to_50":dto(50),
            "days_to_100":dto(100),"days_to_peak":peak-i}

def evenly_spaced_indices(n,k):
    if k>=n: return list(range(n))
    if k==1: return [n//2]
    idx=[]
    for j in range(k):
        x=round(j*(n-1)/(k-1))
        if x not in idx: idx.append(x)
    return idx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample-size",type=int,default=30)
    ap.add_argument("--min-price",type=float,default=1.0)
    ap.add_argument("--max-price",type=float,default=20.0)
    ap.add_argument("--min-dollar-volume",type=float,default=1_000_000)
    ap.add_argument("--gap-threshold",type=float,default=-9.0)
    ap.add_argument("--range5-threshold",type=float,default=9.42)
    ap.add_argument("--max-prior-jump",type=float,default=100.0)
    ap.add_argument("--lookahead",type=int,default=21)
    ap.add_argument("--output",default=str(DATA_DIR / "daily_rule_validation30_diverse_dates.csv"))
    ap.add_argument("--summary-output",default=str(DATA_DIR / "daily_rule_validation30_diverse_dates_summary.json"))
    args=ap.parse_args()

    DB=database()
    with DB() as s:
        md=MetaData()
        daily=Table("market_candles_1d",md,autoload_with=s.get_bind())
        symbols=list(s.execute(
            select(daily.c.symbol).group_by(daily.c.symbol)
            .having(func.count()>=220).order_by(daily.c.symbol)
        ).scalars().all())

    by_date=defaultdict(list)
    print("\nRajih — DIVERSE-DATE VALIDATION")
    print(f"Universe: {len(symbols)}")
    print(f"Excluded discovery symbols: {len(DISCOVERY_SYMBOLS)}")
    print("Collecting all rule-matched historical setups using only pre-signal data...\n")

    for n,sym in enumerate(symbols,1):
        if sym in DISCOVERY_SYMBOLS: continue
        rows=load_symbol(DB,daily,sym)
        if len(rows)<220: continue
        f=build(rows)

        last_match=-999
        for i in range(219,len(rows)-args.lookahead):
            # 21-session cooldown avoids a long qualifying regime from dominating.
            if i-last_match < 21: continue
            if not (args.min_price <= f["rc"][i] < args.max_price): continue
            if f["avgdol"][i] is None or f["avgdol"][i] < args.min_dollar_volume: continue
            if f["mingap"][i] is None or f["mingap"][i] > args.gap_threshold: continue
            if f["r5"][i] is None or f["r5"][i] < args.range5_threshold: continue
            if not discontinuity_ok(f,i,args.max_prior_jump): continue

            row={
                "symbol":sym,"signal_date":f["date"][i],"signal_raw_close":fmt(f["rc"][i]),
                "avg_dollar_volume20":fmt(f["avgdol"][i],0),
                "min_gap20_pct":fmt(f["mingap"][i]),"avg_range5_pct":fmt(f["r5"][i]),
                "avg_range20_pct":fmt(f["r20"][i]),
                "atr_pct":fmt(100*f["atr"][i]/f["c"][i] if f["atr"][i] else None),
                "rsi14":fmt(f["rsi"][i]),"adx14":fmt(f["adx"][i]),
                "ema20_vs_ema50_pct":fmt(100*(f["e20"][i]/f["e50"][i]-1) if f["e20"][i] and f["e50"][i] else None),
                "realized_vol20_pct":fmt(100*f["rv"][i] if f["rv"][i] is not None else None),
                "distance_to_252d_high_pct":fmt(100*(f["c"][i]/f["hi252"][i]-1) if f["hi252"][i] else None),
                "_i":i,"_f":f,
            }
            by_date[row["signal_date"]].append(row)
            last_match=i

        if n%250==0 or n==len(symbols):
            print(f"Scanned {n}/{len(symbols)} | distinct qualifying dates={len(by_date)}")

    dates=sorted(by_date)
    if not dates:
        raise SystemExit("No matching dates found.")
    chosen_dates=[dates[i] for i in evenly_spaced_indices(len(dates),min(args.sample_size,len(dates)))]

    # lock one candidate per chosen date using highest liquidity, before future outcome.
    selected=[]
    used_symbols=set()
    for d in chosen_dates:
        pool=sorted(by_date[d],key=lambda r:(r["avg_dollar_volume20"] or 0),reverse=True)
        pick=next((r for r in pool if r["symbol"] not in used_symbols),None)
        if pick is None: pick=pool[0]
        selected.append(pick); used_symbols.add(pick["symbol"])

    out=[]
    for rank,r in enumerate(selected,1):
        f=r.pop("_f"); i=r.pop("_i")
        x={"rank":rank,**r}; x.update(outcome(f,i,args.lookahead)); out.append(x)

    fields=list(out[0].keys())
    with Path(args.output).open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(out)

    def count(th):
        return sum((r.get("max_high_21d_pct") or -999)>=th for r in out)
    gains=[r["max_high_21d_pct"] for r in out]
    maes=[r["mae_21d_pct"] for r in out]
    summary={
        "sample_size":len(out),
        "distinct_dates":len(set(r["signal_date"] for r in out)),
        "available_qualifying_dates":len(dates),
        "results":{
            "hit_20pct_21d":count(20),"hit_30pct_21d":count(30),
            "hit_50pct_21d":count(50),"hit_100pct_21d":count(100),
            "hit_50pct_rate":fmt(100*count(50)/len(out)),
            "mean_max_high_21d_pct":fmt(mean(gains)),
            "median_max_high_21d_pct":fmt(median(gains)),
            "median_mae_21d_pct":fmt(median(maes)),
        }
    }
    Path(args.summary_output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\n=== DIVERSE VALIDATION 30 ===")
    print("rank | symbol | signal | price | gap20 | range5 | max21d | MAE21 | +50?")
    for r in out:
        yes="YES" if r["max_high_21d_pct"]>=50 else "NO"
        print(f"{r['rank']:>4} | {r['symbol']:<7} | {r['signal_date']} | "
              f"${r['signal_raw_close']:>6.2f} | {r['min_gap20_pct']:>7.2f}% | "
              f"{r['avg_range5_pct']:>6.2f}% | {r['max_high_21d_pct']:>7.2f}% | "
              f"{r['mae_21d_pct']:>7.2f}% | {yes}")
    z=summary["results"]
    print("\n=== RESULTS ===")
    print(f"Distinct dates: {summary['distinct_dates']}/{len(out)}")
    print(f"+20% within 21d: {z['hit_20pct_21d']}/{len(out)}")
    print(f"+30% within 21d: {z['hit_30pct_21d']}/{len(out)}")
    print(f"+50% within 21d: {z['hit_50pct_21d']}/{len(out)} = {z['hit_50pct_rate']:.2f}%")
    print(f"+100% within 21d: {z['hit_100pct_21d']}/{len(out)}")
    print(f"Mean max high 21d: {z['mean_max_high_21d_pct']:.2f}%")
    print(f"Median max high 21d: {z['median_max_high_21d_pct']:.2f}%")
    print(f"Median MAE 21d: {z['median_mae_21d_pct']:.2f}%")
    print(f"\nCSV: {args.output}")
    print(f"Summary: {args.summary_output}")

if __name__=="__main__":
    main()
