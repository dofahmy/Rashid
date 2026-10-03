#!/usr/bin/env python3
"""
Rajih — validation sample for the discovered daily rule.

Goal
----
Select 30 NEW historical setups based ONLY on the discovered conditions,
excluding the original 30 discovery winners, then inspect what happened next.

IMPORTANT
---------
This script does NOT pre-select stocks because they later gained +50%.
It scans the whole stored daily universe and selects condition-matching setups
without looking at the future outcome first.

Locked rule from discovery
--------------------------
- Actual stock price at signal: $1 <= close < $20
- Prior 20d average dollar volume >= $1,000,000
- Worst opening gap in prior 20 sessions <= -9.0%
- Average daily range over last 5 sessions >= 9.42%
- No >100% one-day adjusted-close discontinuity in prior 20 sessions
- At least 220 daily bars of history
- Exclude the original 30 discovery symbols

Sampling
--------
- One setup per symbol
- Deterministic chronological sample:
  take the first 30 qualifying setups after sorting by signal date, then symbol.
- Future outcome is calculated only AFTER the 30 setups are selected.

Outputs
-------
daily_rule_validation30.csv
daily_rule_validation30_summary.json

Run:
    python validate_daily_rule_30_new.py

Optional:
    python validate_daily_rule_30_new.py --sample-size 30
    python validate_daily_rule_30_new.py --gap-threshold -9.0 --range5-threshold 9.42
"""

from __future__ import annotations

import os

import argparse
import csv
import json
import math
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
from statistics import mean, median

from sqlalchemy import MetaData, Table, select, func

from core import database


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def fmt(x, n=4):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), n)


def stdev(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs)<2:
        return None
    mu=mean(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))


def sma(vals,i,n):
    if i+1<n:
        return None
    xs=vals[i+1-n:i+1]
    return sum(xs)/n


def ema_all(vals,n):
    if not vals:
        return []
    out=[None]*len(vals)
    a=2/(n+1)
    e=float(vals[0])
    out[0]=e
    for i in range(1,len(vals)):
        e=a*float(vals[i])+(1-a)*e
        out[i]=e
    return out


def atr_all(h,l,c,n=14):
    out=[None]*len(c)
    if len(c)<=n:
        return out
    tr=[None]*len(c)
    for i in range(1,len(c)):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=sum(tr[1:n+1])/n
    out[n]=a
    for i in range(n+1,len(c)):
        a=((n-1)*a+tr[i])/n
        out[i]=a
    return out


def rsi_all(c,n=14):
    out=[None]*len(c)
    if len(c)<=n:
        return out
    gains=[]; losses=[]
    for i in range(1,len(c)):
        d=c[i]-c[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains[:n])/n
    al=sum(losses[:n])/n
    out[n]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    for i in range(n+1,len(c)):
        ag=((n-1)*ag+gains[i-1])/n
        al=((n-1)*al+losses[i-1])/n
        out[i]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    return out


def adx_all(h,l,c,n=14):
    m=len(c); out=[None]*m
    if m<2*n+1:
        return out
    tr=[0.0]*m; pdm=[0.0]*m; mdm=[0.0]*m
    for i in range(1,m):
        up=h[i]-h[i-1]
        dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0.0
        mdm[i]=dn if dn>up and dn>0 else 0.0
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    atrs=sum(tr[1:n+1]); ps=sum(pdm[1:n+1]); ms=sum(mdm[1:n+1])
    dx=[None]*m
    for i in range(n,m):
        if i>n:
            atrs=atrs-atrs/n+tr[i]
            ps=ps-ps/n+pdm[i]
            ms=ms-ms/n+mdm[i]
        if atrs<=0:
            continue
        pdi=100*ps/atrs
        mdi=100*ms/atrs
        den=pdi+mdi
        dx[i]=0 if den==0 else 100*abs(pdi-mdi)/den
    seed=[x for x in dx[n:2*n] if x is not None]
    if len(seed)<n:
        return out
    a=sum(seed)/n
    out[2*n-1]=a
    for i in range(2*n,m):
        if dx[i] is not None:
            a=((n-1)*a+dx[i])/n
            out[i]=a
    return out


def load_symbol(DB,daily,sym):
    with DB() as s:
        raw=list(s.execute(
            select(
                daily.c.session_date,daily.c.o,daily.c.h,daily.c.l,
                daily.c.c,daily.c.v,daily.c.adj_c
            ).where(daily.c.symbol==sym).order_by(daily.c.session_date)
        ).all())

    rows=[]
    for x in raw:
        ac=finite(x[6]); rc=finite(x[4])
        if ac is None or rc is None or ac<=0 or rc<=0:
            continue
        rows.append({
            "date":str(x[0]),
            "raw_o":float(x[1]),"raw_h":float(x[2]),"raw_l":float(x[3]),
            "raw_c":rc,"v":float(x[5] or 0),"adj_c":ac,
        })
    return rows


def build_features(rows):
    raw_o=[r["raw_o"] for r in rows]
    raw_h=[r["raw_h"] for r in rows]
    raw_l=[r["raw_l"] for r in rows]
    raw_c=[r["raw_c"] for r in rows]
    v=[r["v"] for r in rows]
    c=[r["adj_c"] for r in rows]

    factor=[c[i]/raw_c[i] for i in range(len(c))]
    o=[raw_o[i]*factor[i] for i in range(len(c))]
    h=[raw_h[i]*factor[i] for i in range(len(c))]
    l=[raw_l[i]*factor[i] for i in range(len(c))]

    atr=atr_all(h,l,c,14)
    rsi=rsi_all(c,14)
    adx=adx_all(h,l,c,14)
    e20=ema_all(c,20)
    e50=ema_all(c,50)

    avg_dol20=[None]*len(c)
    range5=[None]*len(c)
    range20=[None]*len(c)
    min_gap20=[None]*len(c)
    rv20=[None]*len(c)

    ret=[None]*len(c)
    gap=[None]*len(c)
    dollar=[raw_c[i]*v[i] for i in range(len(c))]
    for i in range(1,len(c)):
        ret[i]=c[i]/c[i-1]-1
        gap[i]=100*(o[i]/c[i-1]-1)

    for i in range(len(c)):
        if i>=19:
            avg_dol20[i]=mean(dollar[i-19:i+1])
            range20[i]=mean(100*(h[j]-l[j])/c[j] for j in range(i-19,i+1))
            gg=[gap[j] for j in range(i-19,i+1) if gap[j] is not None]
            min_gap20[i]=min(gg) if gg else None
            rr=[ret[j] for j in range(i-19,i+1) if ret[j] is not None]
            rv20[i]=stdev(rr)
        if i>=4:
            range5[i]=mean(100*(h[j]-l[j])/c[j] for j in range(i-4,i+1))

    return {
        "date":[r["date"] for r in rows],
        "raw_c":raw_c,"c":c,"h":h,"l":l,"v":v,
        "atr":atr,"rsi":rsi,"adx":adx,"e20":e20,"e50":e50,
        "avg_dol20":avg_dol20,"range5":range5,"range20":range20,
        "min_gap20":min_gap20,"rv20":rv20,"ret":ret,
    }


def discontinuity_ok(f,i,max_abs=100):
    if i<20:
        return False
    for j in range(i-19,i+1):
        if j<=0:
            continue
        if abs(100*(f["c"][j]/f["c"][j-1]-1)) > max_abs:
            return False
    return True


def forward_outcome(f,i,lookahead=21):
    j2=min(len(f["c"])-1,i+lookahead)
    if j2<=i:
        return None

    start=f["c"][i]
    future=range(i+1,j2+1)

    def max_high(n):
        jj=min(j2,i+n)
        return max(100*(f["h"][j]/start-1) for j in range(i+1,jj+1))

    def max_close(n):
        jj=min(j2,i+n)
        return max(100*(f["c"][j]/start-1) for j in range(i+1,jj+1))

    def mae(n):
        jj=min(j2,i+n)
        return min(100*(f["l"][j]/start-1) for j in range(i+1,jj+1))

    def days_to(th):
        for j in future:
            if 100*(f["h"][j]/start-1) >= th:
                return j-i
        return None

    peak_j=max(future,key=lambda j:f["h"][j])

    return {
        "max_high_5d_pct":fmt(max_high(5)),
        "max_high_10d_pct":fmt(max_high(10)),
        "max_high_21d_pct":fmt(max_high(21)),
        "max_close_5d_pct":fmt(max_close(5)),
        "max_close_10d_pct":fmt(max_close(10)),
        "max_close_21d_pct":fmt(max_close(21)),
        "mae_5d_pct":fmt(mae(5)),
        "mae_10d_pct":fmt(mae(10)),
        "mae_21d_pct":fmt(mae(21)),
        "days_to_20":days_to(20),
        "days_to_30":days_to(30),
        "days_to_50":days_to(50),
        "days_to_100":days_to(100),
        "days_to_peak":peak_j-i,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample-size",type=int,default=30)
    ap.add_argument("--min-price",type=float,default=1.0)
    ap.add_argument("--max-price",type=float,default=20.0)
    ap.add_argument("--min-dollar-volume",type=float,default=1_000_000.0)
    ap.add_argument("--gap-threshold",type=float,default=-9.0)
    ap.add_argument("--range5-threshold",type=float,default=9.42)
    ap.add_argument("--max-prior-jump",type=float,default=100.0)
    ap.add_argument("--lookahead",type=int,default=21)
    ap.add_argument("--discovery-file",default=str(DATA_DIR / "daily_50pct_tradable30.csv"))
    ap.add_argument("--output",default=str(DATA_DIR / "daily_rule_validation30.csv"))
    ap.add_argument("--summary-output",default=str(DATA_DIR / "daily_rule_validation30_summary.json"))
    args=ap.parse_args()

    excluded=set()
    p=Path(args.discovery_file)
    if p.exists():
        with p.open(encoding="utf-8-sig",newline="") as fh:
            for r in csv.DictReader(fh):
                excluded.add(r["symbol"])
    else:
        print(f"WARNING: {p} not found; no discovery symbols excluded.")

    DB=database()
    with DB() as s:
        md=MetaData()
        daily=Table("market_candles_1d",md,autoload_with=s.get_bind())
        symbols=list(s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count()>=220)
            .order_by(daily.c.symbol)
        ).scalars().all())

    print("\nRajih — 30 NEW RULE-MATCHED SETUPS")
    print(f"Universe: {len(symbols)} symbols")
    print(f"Excluded discovery symbols: {len(excluded)}")
    print(
        f"Locked rule: price ${args.min_price:g}-${args.max_price:g}, "
        f"avg$vol20 >= ${args.min_dollar_volume:,.0f}, "
        f"min_gap20 <= {args.gap_threshold:g}%, "
        f"avg_range5 >= {args.range5_threshold:g}%"
    )
    print("Selection is based ONLY on pre-signal data.\n")

    candidates=[]

    for n,sym in enumerate(symbols,1):
        if sym in excluded:
            continue

        rows=load_symbol(DB,daily,sym)
        if len(rows)<220:
            continue
        f=build_features(rows)

        first_match=None
        for i in range(219,len(rows)-args.lookahead):
            price=f["raw_c"][i]
            if not (args.min_price <= price < args.max_price):
                continue
            if f["avg_dol20"][i] is None or f["avg_dol20"][i] < args.min_dollar_volume:
                continue
            if f["min_gap20"][i] is None or f["min_gap20"][i] > args.gap_threshold:
                continue
            if f["range5"][i] is None or f["range5"][i] < args.range5_threshold:
                continue
            if not discontinuity_ok(f,i,args.max_prior_jump):
                continue

            first_match=(i,{
                "symbol":sym,
                "signal_date":f["date"][i],
                "signal_raw_close":fmt(price),
                "avg_dollar_volume20":fmt(f["avg_dol20"][i],0),
                "min_gap20_pct":fmt(f["min_gap20"][i]),
                "avg_range5_pct":fmt(f["range5"][i]),
                "avg_range20_pct":fmt(f["range20"][i]),
                "atr_pct":fmt(100*f["atr"][i]/f["c"][i] if f["atr"][i] else None),
                "rsi14":fmt(f["rsi"][i]),
                "adx14":fmt(f["adx"][i]),
                "ema20_vs_ema50_pct":fmt(
                    100*(f["e20"][i]/f["e50"][i]-1)
                    if f["e20"][i] and f["e50"][i] else None
                ),
                "realized_vol20_pct":fmt(100*f["rv20"][i] if f["rv20"][i] is not None else None),
            })
            break

        if first_match:
            i,row=first_match
            row["_i"]=i
            row["_f"]=f
            candidates.append(row)

        if n%250==0 or n==len(symbols):
            print(f"Scanned {n}/{len(symbols)} | rule-matched symbols={len(candidates)}")

    # Lock sample before looking at outcomes.
    candidates.sort(key=lambda r:(r["signal_date"],r["symbol"]))
    selected=candidates[:args.sample_size]

    # Only now calculate future outcomes.
    out=[]
    for rank,r in enumerate(selected,1):
        f=r.pop("_f")
        i=r.pop("_i")
        x={"rank":rank,**r}
        x.update(forward_outcome(f,i,args.lookahead))
        out.append(x)

    if not out:
        print("No qualifying setups found.")
        return 2

    fields=list(out[0].keys())
    with Path(args.output).open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    def hit(th,n=21):
        col=f"max_high_{n}d_pct"
        return sum(1 for r in out if (r.get(col) or -999)>=th)

    gains=[r["max_high_21d_pct"] for r in out if r.get("max_high_21d_pct") is not None]
    maes=[r["mae_21d_pct"] for r in out if r.get("mae_21d_pct") is not None]

    summary={
        "sample_size":len(out),
        "selection_rule":{
            "price":[args.min_price,args.max_price],
            "avg_dollar_volume20_min":args.min_dollar_volume,
            "min_gap20_max_pct":args.gap_threshold,
            "avg_range5_min_pct":args.range5_threshold,
            "max_prior_one_day_abs_return_pct":args.max_prior_jump,
            "excluded_discovery_symbols":len(excluded),
            "sampling":"first chronological match per symbol; first 30 by date+symbol",
        },
        "results":{
            "hit_20pct_21d":hit(20),
            "hit_30pct_21d":hit(30),
            "hit_50pct_21d":hit(50),
            "hit_100pct_21d":hit(100),
            "hit_50pct_rate":fmt(100*hit(50)/len(out)),
            "mean_max_high_21d_pct":fmt(mean(gains)),
            "median_max_high_21d_pct":fmt(median(gains)),
            "mean_mae_21d_pct":fmt(mean(maes)),
            "median_mae_21d_pct":fmt(median(maes)),
        }
    }

    Path(args.summary_output).write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\n=== VALIDATION 30 ===")
    print("rank | symbol | signal | price | gap20 | range5 | max21d | MAE21 | +50?")
    for r in out:
        yes="YES" if (r["max_high_21d_pct"] or -999)>=50 else "NO"
        print(
            f"{r['rank']:>4} | {r['symbol']:<7} | {r['signal_date']} | "
            f"${r['signal_raw_close']:>6.2f} | {r['min_gap20_pct']:>7.2f}% | "
            f"{r['avg_range5_pct']:>6.2f}% | {r['max_high_21d_pct']:>7.2f}% | "
            f"{r['mae_21d_pct']:>7.2f}% | {yes}"
        )

    z=summary["results"]
    print("\n=== RESULTS ===")
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
