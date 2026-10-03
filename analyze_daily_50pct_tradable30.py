#!/usr/bin/env python3
"""
Rajih — tradable sample of 30 sub-$20 US stocks with genuine +50% to +200% monthly moves
within ~1 trading month, using split-adjusted prices to avoid false moves from
reverse splits / stock splits.

Reads only: market_candles_1d
Writes only CSV/JSON output files.
Does NOT modify the database or bot.

Key correction vs first analyzer
--------------------------------
- Price eligibility (<$20) uses the ACTUAL raw close at the event start.
- Return measurement uses split-adjusted prices:
    adjusted_close = adj_c
    adjusted_high  = raw_high * (adj_c/raw_close)
  This keeps the economic move comparable through stock splits/reverse splits.
- Requires usable adj_c at start and throughout the forward window.
- One strongest qualifying event per symbol.
- Top 30 distinct symbols.
- Same pre-move snapshots at T-20, T-5, T-1.
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


def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return 100.0 * (float(a) / float(b) - 1.0)


def sma(vals, i, n):
    if i + 1 < n:
        return None
    xs = vals[i+1-n:i+1]
    if any(x is None for x in xs):
        return None
    return sum(xs) / n


def ema_all(vals, n):
    out = [None] * len(vals)
    if not vals:
        return out
    e = float(vals[0])
    a = 2/(n+1)
    out[0] = e
    for i in range(1, len(vals)):
        e = a*float(vals[i]) + (1-a)*e
        out[i] = e
    return out


def atr_all(h, l, c, n=14):
    out = [None]*len(c)
    if len(c) <= n:
        return out
    tr = [None]*len(c)
    for i in range(1, len(c)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = sum(tr[1:n+1])/n
    out[n] = a
    for i in range(n+1, len(c)):
        a = ((n-1)*a + tr[i])/n
        out[i] = a
    return out


def rsi_all(c, n=14):
    out = [None]*len(c)
    if len(c) <= n:
        return out
    gains=[]; losses=[]
    for i in range(1,len(c)):
        d=c[i]-c[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains[:n])/n; al=sum(losses[:n])/n
    out[n]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    for i in range(n+1,len(c)):
        ag=((n-1)*ag+gains[i-1])/n
        al=((n-1)*al+losses[i-1])/n
        out[i]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    return out


def adx_all(h,l,c,n=14):
    m=len(c); out=[None]*m
    if m < 2*n+1: return out
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


def q(vals,p):
    vals=sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
    if not vals: return None
    if len(vals)==1: return vals[0]
    k=(len(vals)-1)*p; lo=math.floor(k); hi=math.ceil(k)
    if lo==hi: return vals[lo]
    return vals[lo]*(hi-k)+vals[hi]*(k-lo)


def fmt(x,n=4):
    return None if x is None else round(float(x),n)


def build_features(rows):
    d=[r["session_date"] for r in rows]
    o=[r["o"] for r in rows]; h=[r["h"] for r in rows]
    l=[r["l"] for r in rows]; c=[r["c"] for r in rows]
    v=[r["v"] for r in rows]; ac=[r["adj_c"] for r in rows]

    # Adjustment factor per date. Adjust OHLC into a split/dividend comparable series.
    factor=[(ac[i]/c[i] if ac[i] and c[i] and c[i]>0 else None) for i in range(len(c))]
    ao=[o[i]*factor[i] if factor[i] else None for i in range(len(c))]
    ah=[h[i]*factor[i] if factor[i] else None for i in range(len(c))]
    al=[l[i]*factor[i] if factor[i] else None for i in range(len(c))]

    e12=ema_all(ac,12); e20=ema_all(ac,20); e26=ema_all(ac,26); e50=ema_all(ac,50)
    macd=[e12[i]-e26[i] for i in range(len(ac))]
    msig=ema_all(macd,9)
    atr=atr_all(ah,al,ac,14)
    rsi=rsi_all(ac,14); adx=adx_all(ah,al,ac,14)
    s20=[sma(ac,i,20) for i in range(len(ac))]
    s50=[sma(ac,i,50) for i in range(len(ac))]
    s200=[sma(ac,i,200) for i in range(len(ac))]
    av20=[sma(v,i,20) for i in range(len(v))]
    dollar=[c[i]*v[i] for i in range(len(c))]
    adol20=[sma(dollar,i,20) for i in range(len(c))]

    ret=[None]*len(ac)
    for i in range(1,len(ac)):
        if ac[i-1] and ac[i]:
            ret[i]=ac[i]/ac[i-1]-1

    rv20=[None]*len(ac); ar20=[None]*len(ac); hi20=[None]*len(ac); lo20=[None]*len(ac)
    for i in range(len(ac)):
        if i>=19:
            rv20[i]=stdev(ret[i-19:i+1])
            ar20[i]=mean(100*(ah[j]-al[j])/ac[j] for j in range(i-19,i+1) if ac[j]>0)
            hi20[i]=max(ah[i-19:i+1]); lo20[i]=min(al[i-19:i+1])

    return {
        "date":d,"o":o,"h":h,"l":l,"c":c,"v":v,"adj_c":ac,"adj_h":ah,"adj_l":al,
        "ema20":e20,"ema50":e50,"macd":macd,"macd_signal":msig,
        "atr14":atr,"rsi14":rsi,"adx14":adx,
        "sma20":s20,"sma50":s50,"sma200":s200,
        "avgvol20":av20,"avgdol20":adol20,
        "retvol20":rv20,"avg_range20":ar20,"high20":hi20,"low20":lo20,
    }


def snapshot(f,i):
    if i<0 or i>=len(f["c"]): return {}
    c=f["adj_c"][i]
    atr=f["atr14"][i]; av=f["avgvol20"][i]
    s20=f["sma20"][i]; s50=f["sma50"][i]; s200=f["sma200"][i]
    e20=f["ema20"][i]; e50=f["ema50"][i]
    macd=f["macd"][i]; ms=f["macd_signal"][i]
    hi20=f["high20"][i]; lo20=f["low20"][i]
    pos=None
    if hi20 is not None and lo20 is not None and hi20>lo20:
        pos=100*(c-lo20)/(hi20-lo20)
    return {
        "date":f["date"][i],
        "raw_close":fmt(f["c"][i]),
        "adj_close":fmt(c),
        "ret_5d_pct":fmt(pct(c,f["adj_c"][i-5]) if i>=5 else None),
        "ret_20d_pct":fmt(pct(c,f["adj_c"][i-20]) if i>=20 else None),
        "atr14_adj":fmt(atr),
        "atr14_pct":fmt(100*atr/c if atr is not None and c>0 else None),
        "rsi14":fmt(f["rsi14"][i]),"adx14":fmt(f["adx14"][i]),
        "dist_sma20_pct":fmt(pct(c,s20)),"dist_sma50_pct":fmt(pct(c,s50)),
        "dist_sma200_pct":fmt(pct(c,s200)),
        "ema20_vs_ema50_pct":fmt(pct(e20,e50)),
        "macd_gap":fmt(macd-ms if macd is not None and ms is not None else None),
        "volume":fmt(f["v"][i],0),"avg_volume20":fmt(av,0),
        "volume_ratio20":fmt(f["v"][i]/av if av and av>0 else None),
        "dollar_volume":fmt(f["c"][i]*f["v"][i],0),
        "avg_dollar_volume20":fmt(f["avgdol20"][i],0),
        "realized_vol20_pct":fmt(100*f["retvol20"][i] if f["retvol20"][i] is not None else None),
        "avg_range20_pct":fmt(f["avg_range20"][i]),
        "position_in_20d_range_pct":fmt(pos),
        "distance_to_20d_high_pct":fmt(pct(c,hi20)),
    }


def flatten(prefix,d,out):
    for k,v in d.items(): out[f"{prefix}_{k}"]=v


def summarize(rows, fields):
    ans=[]
    for col in fields:
        vals=[]
        for r in rows:
            try:
                x=float(r.get(col))
                if math.isfinite(x): vals.append(x)
            except: pass
        if vals:
            ans.append({
                "metric":col,"n":len(vals),"mean":fmt(mean(vals)),"median":fmt(median(vals)),
                "p25":fmt(q(vals,.25)),"p75":fmt(q(vals,.75)),
                "min":fmt(min(vals)),"max":fmt(max(vals))
            })
    return ans


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--top",type=int,default=30)
    ap.add_argument("--min-price",type=float,default=1.0)
    ap.add_argument("--max-price",type=float,default=20.0)
    ap.add_argument("--min-gain",type=float,default=50.0)
    ap.add_argument("--max-gain",type=float,default=200.0)
    ap.add_argument("--min-avg-dollar-volume20",type=float,default=1_000_000.0)
    ap.add_argument("--max-prior-1d-abs-return",type=float,default=100.0,
                    help="Reject events with >100%% absolute adjusted close jump in the prior 20 sessions.")
    ap.add_argument("--lookahead",type=int,default=21)
    ap.add_argument("--min-history",type=int,default=220)
    ap.add_argument("--output",default=str(DATA_DIR / "daily_50pct_tradable30.csv"))
    ap.add_argument("--commonality-output",default=str(DATA_DIR / "daily_50pct_tradable30_commonality.csv"))
    ap.add_argument("--summary-output",default=str(DATA_DIR / "daily_50pct_tradable30_summary.json"))
    args=ap.parse_args()

    DB=database()
    with DB() as s:
        md=MetaData()
        daily=Table("market_candles_1d",md,autoload_with=s.get_bind())

    with DB() as s:
        syms=list(s.execute(
            select(daily.c.symbol).group_by(daily.c.symbol)
            .having(func.count()>=args.min_history).order_by(daily.c.symbol)
        ).scalars().all())

    print("\nRajih — CLEAN Daily +50% discovery (split-adjusted returns)")
    print(f"Eligible symbols: {len(syms)}")
    print(f"Actual start price: ${args.min_price:g} <= price < ${args.max_price:g}")
    print(f"Adjusted forward HIGH gain: {args.min_gain:g}% to {args.max_gain:g}% within {args.lookahead} sessions")
    print(f"Prior 20d avg dollar volume >= ${args.min_avg_dollar_volume20:,.0f}")
    print(f"Reject prior 20d one-day abs return > {args.max_prior_1d_abs_return:g}%\n")

    candidates=[]
    for n,sym in enumerate(syms,1):
        with DB() as s:
            raw=list(s.execute(
                select(daily.c.session_date,daily.c.o,daily.c.h,daily.c.l,daily.c.c,daily.c.v,daily.c.adj_c)
                .where(daily.c.symbol==sym).order_by(daily.c.session_date)
            ).all())
        rows=[]
        for x in raw:
            ac=finite(x[6]); c=finite(x[4])
            if ac is None or c is None or ac<=0 or c<=0: continue
            rows.append({
                "session_date":str(x[0]),"o":float(x[1]),"h":float(x[2]),"l":float(x[3]),
                "c":c,"v":float(x[5] or 0),"adj_c":ac
            })
        if len(rows)<args.min_history: continue
        f=build_features(rows)
        best=None
        first=max(args.min_history-1,200)
        for i in range(first,len(rows)-1):
            if not (args.min_price <= f["c"][i] < args.max_price): continue
            # Require meaningful tradability before the move.
            avg_dol20 = f["avgdol20"][i-1] if i >= 1 else None
            if avg_dol20 is None or avg_dol20 < args.min_avg_dollar_volume20: continue
            # Reject obvious pre-event history discontinuities/ticker-reuse artifacts.
            if i < 20: continue
            prior_abs = []
            for jj in range(i-19, i+1):
                if jj <= 0 or not f["adj_c"][jj-1]:
                    continue
                prior_abs.append(abs(100*(f["adj_c"][jj]/f["adj_c"][jj-1]-1)))
            if prior_abs and max(prior_abs) > args.max_prior_1d_abs_return: continue
            j2=min(len(rows)-1,i+args.lookahead)
            fut=[(j,f["adj_h"][j]) for j in range(i+1,j2+1) if f["adj_h"][j] is not None]
            if not fut: continue
            peak_i,peak=max(fut,key=lambda z:z[1])
            gain=100*(peak/f["adj_c"][i]-1)
            if gain < args.min_gain or gain > args.max_gain: continue
            max_close=max(f["adj_c"][i+1:j2+1])
            ev={
                "symbol":sym,"start_i":i,"peak_i":peak_i,
                "start_date":f["date"][i],"peak_date":f["date"][peak_i],
                "start_raw_close":fmt(f["c"][i]),"start_adj_close":fmt(f["adj_c"][i]),
                "peak_raw_high":fmt(f["h"][peak_i]),"peak_adj_high":fmt(peak),
                "forward_adjusted_high_gain_pct":fmt(gain),
                "forward_adjusted_close_gain_pct":fmt(100*(max_close/f["adj_c"][i]-1)),
                "trading_days_to_peak":peak_i-i,
                "prior20_avg_dollar_volume":fmt(avg_dol20,0),
                "_f":f,
            }
            if best is None or gain>best["forward_adjusted_high_gain_pct"]:
                best=ev
        if best: candidates.append(best)
        if n%250==0 or n==len(syms):
            print(f"Scanned {n}/{len(syms)} | qualifying distinct symbols={len(candidates)}")

    # Select the most tradable 30, not the most extreme gainers.
    candidates.sort(key=lambda x:(x.get("prior20_avg_dollar_volume") or 0),reverse=True)
    selected=candidates[:args.top]

    out=[]
    for rank,ev in enumerate(selected,1):
        f=ev.pop("_f"); i=ev["start_i"]
        r={k:v for k,v in ev.items() if k not in ("start_i","peak_i")}
        r["rank"]=rank
        r={"rank":rank,**r}
        flatten("t_minus_20",snapshot(f,i-20),r)
        flatten("t_minus_5",snapshot(f,i-5),r)
        flatten("t_minus_1",snapshot(f,i-1),r)
        out.append(r)

    if not out:
        print("No qualifying clean events found.")
        return 2

    fields=list(out[0].keys())
    with Path(args.output).open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(out)

    common=summarize(out,[x for x in fields if x not in {
        "rank","symbol","start_date","peak_date",
        "t_minus_20_date","t_minus_5_date","t_minus_1_date"
    }])
    with Path(args.commonality_output).open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=["metric","n","mean","median","p25","p75","min","max"])
        w.writeheader(); w.writerows(common)

    def prop(fn):
        return round(100*sum(1 for r in out if fn(r))/len(out),2)

    patterns={
        "pct_t1_above_sma20":prop(lambda r:(r.get("t_minus_1_dist_sma20_pct") or -999)>0),
        "pct_t1_above_sma50":prop(lambda r:(r.get("t_minus_1_dist_sma50_pct") or -999)>0),
        "pct_t1_above_sma200":prop(lambda r:(r.get("t_minus_1_dist_sma200_pct") or -999)>0),
        "pct_t1_ema20_above_ema50":prop(lambda r:(r.get("t_minus_1_ema20_vs_ema50_pct") or -999)>0),
        "pct_t1_macd_above_signal":prop(lambda r:(r.get("t_minus_1_macd_gap") or -999)>0),
        "pct_t1_rsi_above_50":prop(lambda r:(r.get("t_minus_1_rsi14") or -999)>50),
        "pct_t1_adx_above_20":prop(lambda r:(r.get("t_minus_1_adx14") or -999)>=20),
        "pct_t1_volume_ratio_above_1":prop(lambda r:(r.get("t_minus_1_volume_ratio20") or -999)>=1),
        "pct_t1_volume_ratio_above_1_5":prop(lambda r:(r.get("t_minus_1_volume_ratio20") or -999)>=1.5),
        "pct_t1_atr_pct_above_5":prop(lambda r:(r.get("t_minus_1_atr14_pct") or -999)>=5),
        "pct_t1_within_10pct_of_20d_high":prop(lambda r:(r.get("t_minus_1_distance_to_20d_high_pct") or -999)>=-10),
    }

    summary={
        "method":"split-adjusted return; start price $1-$20; +50% to +200%; liquid/tradable sample",
        "filters": {
            "min_price": args.min_price,
            "max_price": args.max_price,
            "min_gain_pct": args.min_gain,
            "max_gain_pct": args.max_gain,
            "min_prior20_avg_dollar_volume": args.min_avg_dollar_volume20,
            "max_prior20_1d_abs_return_pct": args.max_prior_1d_abs_return,
        },
        "qualifying_distinct_symbols_total":len(candidates),
        "selected_count":len(out),
        "mean_adjusted_gain_pct":fmt(mean(r["forward_adjusted_high_gain_pct"] for r in out)),
        "median_adjusted_gain_pct":fmt(median(r["forward_adjusted_high_gain_pct"] for r in out)),
        "patterns_t_minus_1":patterns,
        "files":{"events":args.output,"commonality":args.commonality_output}
    }
    Path(args.summary_output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\n=== CLEAN TRADABLE 30 ===")
    print("rank | symbol | start | raw$ | gain% | days | prior20 avg $vol")
    for r in out:
        print(f"{r['rank']:>4} | {r['symbol']:<7} | {r['start_date']} | "
              f"{r['start_raw_close']:>6.2f} | "
              f"{r['forward_adjusted_high_gain_pct']:>7.2f}% | {r['trading_days_to_peak']:>4} | "
              f"${r['prior20_avg_dollar_volume']:,.0f}")
    print("\n=== COMMON PATTERNS T-1 ===")
    for k,v in patterns.items(): print(f"{k}: {v:.2f}%")
    print(f"\nCSV: {args.output}")
    print(f"Commonality: {args.commonality_output}")
    print(f"Summary: {args.summary_output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
