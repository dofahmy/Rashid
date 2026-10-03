#!/usr/bin/env python3
"""
Rajih — DEEP RESEARCH on the 30 liquid US stocks (<$20) that gained +50% to +200%
within 21 trading sessions.

This script is the "research phase" before choosing any strategy.

It does ALL of the following:
1) Rebuilds daily indicators from market_candles_1d.
2) Studies the 30 winners at T-20, T-10, T-5, T-3, T-1.
3) Measures changes/acceleration between those checkpoints.
4) Builds a full event-study path from T-20 through T+21.
5) Measures outcome quality:
   - max gain in 5/10/21 sessions
   - max close gain
   - max adverse excursion
   - days to +20/+30/+50/+100
   - days to peak
6) Adds broader technical/volatility/volume features:
   ATR%, RSI, ADX, SMA/EMA distances, MACD, Bollinger width/z,
   Stochastic, Williams %R, CCI, MFI, realized volatility,
   volume ratios, dollar-volume acceleration, range compression,
   up-day ratios, 52-week position, gaps, etc.
7) Builds MATCHED CONTROLS:
   - same event date
   - price $1-$20
   - liquid (20d avg dollar volume >= $1M)
   - no >100% one-day discontinuity in previous 20 sessions
   - did NOT gain +50% in the next 21 sessions
   - matched to each winner by price and liquidity
8) Compares winners vs controls using:
   - means/medians
   - winner/control ratio where valid
   - AUC-style separation probability
   - direction of useful separation
9) Scans single-feature thresholds for possible predictive filters.
10) Scans two-feature rule combinations for stronger separation.
11) Calculates correlations with:
   - eventual max gain
   - days to peak
12) Produces a plain-text research report plus CSV/JSON outputs.

IMPORTANT
---------
- READ ONLY. Does not modify the database or bot.
- Uses the existing selected events file:
      daily_50pct_tradable30.csv
- Uses stored daily data:
      market_candles_1d
- This is exploratory research, not a trading rule yet.

Run:
    python analyze_daily_50pct_deep_research.py

Useful:
    python analyze_daily_50pct_deep_research.py --controls-per-winner 10
    python analyze_daily_50pct_deep_research.py --events daily_50pct_tradable30.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from sqlalchemy import MetaData, Table, select, func

from core import database


OFFSETS = (-20, -10, -5, -3, -1)
EVENT_PATH = tuple(range(-20, 22))


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def fmt(x, n=4):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), n)


def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return 100.0 * (float(a) / float(b) - 1.0)


def safe_mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return mean(xs) if xs else None


def safe_median(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return median(xs) if xs else None


def stdev(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs) < 2:
        return None
    mu = mean(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))


def quantile(xs, p):
    xs = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs)-1)*p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo]*(hi-k) + xs[hi]*(k-lo)


def corr(x, y):
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None
        and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((a-mx)**2 for a in xs))
    sy = math.sqrt(sum((b-my)**2 for b in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((a-mx)*(b-my) for a,b in pairs)/(sx*sy)


def auc_separation(winners, controls):
    """
    P(winner > control) with ties=0.5.
    0.5 = no separation. Values near 1 mean higher is winner-like,
    near 0 mean lower is winner-like.
    """
    w = [float(x) for x in winners if x is not None and math.isfinite(float(x))]
    c = [float(x) for x in controls if x is not None and math.isfinite(float(x))]
    if not w or not c:
        return None
    wins = ties = 0
    total = len(w)*len(c)
    for a in w:
        for b in c:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5*ties)/total


def sma_series(vals, n):
    out = [None]*len(vals)
    s = 0.0
    q = []
    for i, x in enumerate(vals):
        q.append(float(x))
        s += float(x)
        if len(q) > n:
            s -= q.pop(0)
        if len(q) == n:
            out[i] = s/n
    return out


def ema_series(vals, n):
    if not vals:
        return []
    out = [None]*len(vals)
    a = 2/(n+1)
    e = float(vals[0])
    out[0] = e
    for i in range(1, len(vals)):
        e = a*float(vals[i]) + (1-a)*e
        out[i] = e
    return out


def atr_series(h, l, c, n=14):
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


def rsi_series(c, n=14):
    out = [None]*len(c)
    if len(c) <= n:
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


def adx_series(h,l,c,n=14):
    m=len(c); out=[None]*m
    if m < 2*n+1:
        return out
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
            atrs=atrs-atrs/n+tr[i]
            ps=ps-ps/n+pdm[i]
            ms=ms-ms/n+mdm[i]
        if atrs<=0:
            continue
        pdi=100*ps/atrs; mdi=100*ms/atrs
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


def mfi_series(h,l,c,v,n=14):
    out=[None]*len(c)
    tp=[(h[i]+l[i]+c[i])/3 for i in range(len(c))]
    flow=[tp[i]*v[i] for i in range(len(c))]
    for i in range(n, len(c)):
        pos=neg=0.0
        for j in range(i-n+1, i+1):
            if j<=0:
                continue
            if tp[j] > tp[j-1]:
                pos += flow[j]
            elif tp[j] < tp[j-1]:
                neg += flow[j]
        if neg == 0:
            out[i] = 100.0 if pos > 0 else 50.0
        else:
            mr=pos/neg
            out[i]=100-100/(1+mr)
    return out


def cci_series(h,l,c,n=20):
    out=[None]*len(c)
    tp=[(h[i]+l[i]+c[i])/3 for i in range(len(c))]
    ma=sma_series(tp,n)
    for i in range(n-1,len(c)):
        window=tp[i-n+1:i+1]
        md=mean(abs(x-ma[i]) for x in window)
        out[i]=0.0 if md==0 else (tp[i]-ma[i])/(0.015*md)
    return out


def stochastic_series(h,l,c,n=14):
    out=[None]*len(c)
    for i in range(n-1,len(c)):
        hh=max(h[i-n+1:i+1]); ll=min(l[i-n+1:i+1])
        out[i]=50.0 if hh==ll else 100*(c[i]-ll)/(hh-ll)
    return out


def williams_r_series(h,l,c,n=14):
    out=[None]*len(c)
    for i in range(n-1,len(c)):
        hh=max(h[i-n+1:i+1]); ll=min(l[i-n+1:i+1])
        out[i]=-50.0 if hh==ll else -100*(hh-c[i])/(hh-ll)
    return out


def build_features(rows):
    d=[r["session_date"] for r in rows]
    o=[r["o"] for r in rows]; h0=[r["h"] for r in rows]
    l0=[r["l"] for r in rows]; c0=[r["c"] for r in rows]
    v=[r["v"] for r in rows]; ac=[r["adj_c"] for r in rows]

    factor=[ac[i]/c0[i] if c0[i] and ac[i] else None for i in range(len(c0))]
    o=[o[i]*factor[i] for i in range(len(o))]
    h=[h0[i]*factor[i] for i in range(len(h0))]
    l=[l0[i]*factor[i] for i in range(len(l0))]
    c=ac

    sma20=sma_series(c,20); sma50=sma_series(c,50); sma200=sma_series(c,200)
    ema12=ema_series(c,12); ema20=ema_series(c,20); ema26=ema_series(c,26); ema50=ema_series(c,50)
    macd=[ema12[i]-ema26[i] for i in range(len(c))]
    macd_sig=ema_series(macd,9)
    atr14=atr_series(h,l,c,14)
    rsi14=rsi_series(c,14)
    adx14=adx_series(h,l,c,14)
    mfi14=mfi_series(h,l,c,v,14)
    cci20=cci_series(h,l,c,20)
    stoch14=stochastic_series(h,l,c,14)
    will14=williams_r_series(h,l,c,14)

    vol20=sma_series(v,20)
    vol5=sma_series(v,5)
    dollar=[c0[i]*v[i] for i in range(len(c))]
    dol20=sma_series(dollar,20); dol5=sma_series(dollar,5)

    ret=[None]*len(c)
    gap=[None]*len(c)
    for i in range(1,len(c)):
        ret[i]=c[i]/c[i-1]-1
        gap[i]=100*(o[i]/c[i-1]-1) if c[i-1] else None

    realized20=[None]*len(c)
    range5=[None]*len(c); range20=[None]*len(c)
    bb_width=[None]*len(c); bb_z=[None]*len(c)
    high20=[None]*len(c); low20=[None]*len(c)
    high126=[None]*len(c); low126=[None]*len(c)
    high252=[None]*len(c); low252=[None]*len(c)
    up10=[None]*len(c); up20=[None]*len(c)
    max_gap20=[None]*len(c); min_gap20=[None]*len(c)
    days_since_high20=[None]*len(c); days_since_low20=[None]*len(c)

    for i in range(len(c)):
        if i>=4:
            range5[i]=mean(100*(h[j]-l[j])/c[j] for j in range(i-4,i+1))
        if i>=19:
            rr=[ret[j] for j in range(i-19,i+1) if ret[j] is not None]
            realized20[i]=stdev(rr)
            range20[i]=mean(100*(h[j]-l[j])/c[j] for j in range(i-19,i+1))
            high20[i]=max(h[i-19:i+1]); low20[i]=min(l[i-19:i+1])
            closes=c[i-19:i+1]
            sd=stdev(closes)
            bb_width[i]=None if sma20[i] in (None,0) or sd is None else 100*(4*sd)/sma20[i]
            bb_z[i]=None if sd in (None,0) else (c[i]-sma20[i])/sd
            up20[i]=100*sum(1 for j in range(i-19,i+1) if j>0 and c[j]>c[j-1])/20
            gg=[gap[j] for j in range(i-19,i+1) if gap[j] is not None]
            max_gap20[i]=max(gg) if gg else None
            min_gap20[i]=min(gg) if gg else None
            wh=h[i-19:i+1]; wl=l[i-19:i+1]
            days_since_high20[i]=(len(wh)-1)-max(range(len(wh)), key=lambda k: wh[k])
            days_since_low20[i]=(len(wl)-1)-min(range(len(wl)), key=lambda k: wl[k])
        if i>=9:
            up10[i]=100*sum(1 for j in range(i-9,i+1) if j>0 and c[j]>c[j-1])/10
        if i>=125:
            high126[i]=max(h[i-125:i+1]); low126[i]=min(l[i-125:i+1])
        if i>=251:
            high252[i]=max(h[i-251:i+1]); low252[i]=min(l[i-251:i+1])

    return {
        "date":d,"raw_o":[r["o"] for r in rows],"raw_h":h0,"raw_l":l0,"raw_c":c0,
        "o":o,"h":h,"l":l,"c":c,"v":v,
        "sma20":sma20,"sma50":sma50,"sma200":sma200,
        "ema20":ema20,"ema50":ema50,
        "macd":macd,"macd_signal":macd_sig,
        "atr14":atr14,"rsi14":rsi14,"adx14":adx14,
        "mfi14":mfi14,"cci20":cci20,"stoch14":stoch14,"williams_r14":will14,
        "vol20":vol20,"vol5":vol5,"dol20":dol20,"dol5":dol5,
        "ret":ret,"gap":gap,"realized20":realized20,
        "range5":range5,"range20":range20,
        "bb_width20":bb_width,"bb_z20":bb_z,
        "high20":high20,"low20":low20,
        "high126":high126,"low126":low126,
        "high252":high252,"low252":low252,
        "up10":up10,"up20":up20,
        "max_gap20":max_gap20,"min_gap20":min_gap20,
        "days_since_high20":days_since_high20,"days_since_low20":days_since_low20,
    }


def snap(f,i):
    if i<0 or i>=len(f["c"]):
        return {}
    c=f["c"][i]; atr=f["atr14"][i]
    h20=f["high20"][i]; l20=f["low20"][i]
    h126=f["high126"][i]; l126=f["low126"][i]
    h252=f["high252"][i]; l252=f["low252"][i]
    pos20=None
    if h20 is not None and l20 is not None and h20>l20:
        pos20=100*(c-l20)/(h20-l20)

    atr_pct = 100*atr/c if atr is not None and c>0 else None
    return {
        "raw_close":fmt(f["raw_c"][i]),
        "adj_close":fmt(c),
        "ret1_pct":fmt(100*f["ret"][i] if f["ret"][i] is not None else None),
        "ret3_pct":fmt(pct(c,f["c"][i-3]) if i>=3 else None),
        "ret5_pct":fmt(pct(c,f["c"][i-5]) if i>=5 else None),
        "ret10_pct":fmt(pct(c,f["c"][i-10]) if i>=10 else None),
        "ret20_pct":fmt(pct(c,f["c"][i-20]) if i>=20 else None),
        "atr14":fmt(atr),
        "atr_pct":fmt(atr_pct),
        "rsi14":fmt(f["rsi14"][i]),
        "adx14":fmt(f["adx14"][i]),
        "mfi14":fmt(f["mfi14"][i]),
        "cci20":fmt(f["cci20"][i]),
        "stoch14":fmt(f["stoch14"][i]),
        "williams_r14":fmt(f["williams_r14"][i]),
        "dist_sma20_pct":fmt(pct(c,f["sma20"][i])),
        "dist_sma50_pct":fmt(pct(c,f["sma50"][i])),
        "dist_sma200_pct":fmt(pct(c,f["sma200"][i])),
        "ema20_vs_ema50_pct":fmt(pct(f["ema20"][i],f["ema50"][i])),
        "macd_gap":fmt(
            f["macd"][i]-f["macd_signal"][i]
            if f["macd"][i] is not None and f["macd_signal"][i] is not None else None
        ),
        "macd_gap_atr":fmt(
            (f["macd"][i]-f["macd_signal"][i])/atr
            if atr not in (None,0) and f["macd"][i] is not None and f["macd_signal"][i] is not None else None
        ),
        "volume":fmt(f["v"][i],0),
        "avg_volume20":fmt(f["vol20"][i],0),
        "volume_ratio20":fmt(f["v"][i]/f["vol20"][i] if f["vol20"][i] else None),
        "volume_accel_5v20":fmt(f["vol5"][i]/f["vol20"][i] if f["vol5"][i] and f["vol20"][i] else None),
        "dollar_volume":fmt(f["raw_c"][i]*f["v"][i],0),
        "avg_dollar_volume20":fmt(f["dol20"][i],0),
        "dollar_volume_accel_5v20":fmt(f["dol5"][i]/f["dol20"][i] if f["dol5"][i] and f["dol20"][i] else None),
        "realized_vol20_pct":fmt(100*f["realized20"][i] if f["realized20"][i] is not None else None),
        "avg_range5_pct":fmt(f["range5"][i]),
        "avg_range20_pct":fmt(f["range20"][i]),
        "range_compression_5v20":fmt(
            f["range5"][i]/f["range20"][i]
            if f["range5"][i] and f["range20"][i] else None
        ),
        "bb_width20_pct":fmt(f["bb_width20"][i]),
        "bb_z20":fmt(f["bb_z20"][i]),
        "position_20d_range_pct":fmt(pos20),
        "distance_to_20d_high_pct":fmt(pct(c,h20)),
        "distance_to_126d_high_pct":fmt(pct(c,h126)),
        "distance_to_252d_high_pct":fmt(pct(c,h252)),
        "distance_from_20d_low_pct":fmt(pct(c,l20)),
        "distance_from_126d_low_pct":fmt(pct(c,l126)),
        "distance_from_252d_low_pct":fmt(pct(c,l252)),
        "up_days_10_pct":fmt(f["up10"][i]),
        "up_days_20_pct":fmt(f["up20"][i]),
        "gap_pct":fmt(f["gap"][i]),
        "max_gap20_pct":fmt(f["max_gap20"][i]),
        "min_gap20_pct":fmt(f["min_gap20"][i]),
        "days_since_20d_high":fmt(f["days_since_high20"][i],0),
        "days_since_20d_low":fmt(f["days_since_low20"][i],0),
    }


def outcome(f,i,lookahead=21):
    start=f["c"][i]
    j2=min(len(f["c"])-1,i+lookahead)
    future=range(i+1,j2+1)
    if i+1>j2:
        return {}

    highs=[f["h"][j] for j in future]
    lows=[f["l"][j] for j in future]
    closes=[f["c"][j] for j in future]
    peak_j=max(future,key=lambda j:f["h"][j])

    def maxgain(n):
        jj=min(j2,i+n)
        if jj<=i: return None
        return max(100*(f["h"][j]/start-1) for j in range(i+1,jj+1))

    def maxclose(n):
        jj=min(j2,i+n)
        if jj<=i: return None
        return max(100*(f["c"][j]/start-1) for j in range(i+1,jj+1))

    def mae(n):
        jj=min(j2,i+n)
        if jj<=i: return None
        return min(100*(f["l"][j]/start-1) for j in range(i+1,jj+1))

    def days_to(threshold):
        for j in future:
            if 100*(f["h"][j]/start-1) >= threshold:
                return j-i
        return None

    return {
        "max_gain_high_5d":fmt(maxgain(5)),
        "max_gain_high_10d":fmt(maxgain(10)),
        "max_gain_high_21d":fmt(maxgain(21)),
        "max_gain_close_5d":fmt(maxclose(5)),
        "max_gain_close_10d":fmt(maxclose(10)),
        "max_gain_close_21d":fmt(maxclose(21)),
        "mae_5d_pct":fmt(mae(5)),
        "mae_10d_pct":fmt(mae(10)),
        "mae_21d_pct":fmt(mae(21)),
        "days_to_20":days_to(20),
        "days_to_30":days_to(30),
        "days_to_50":days_to(50),
        "days_to_100":days_to(100),
        "days_to_peak":peak_j-i,
        "peak_gain_pct":fmt(100*(f["h"][peak_j]/start-1)),
    }


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
            "session_date":str(x[0]),"o":float(x[1]),"h":float(x[2]),
            "l":float(x[3]),"c":rc,"v":float(x[5] or 0),"adj_c":ac,
        })
    return rows


def discontinuity_ok(f,i,max_abs=100):
    if i<20:
        return False
    for j in range(i-19,i+1):
        if j<=0:
            continue
        r=abs(100*(f["c"][j]/f["c"][j-1]-1))
        if r>max_abs:
            return False
    return True


def matched_distance(winner_snap, ctrl_snap):
    wp=winner_snap.get("raw_close"); cp=ctrl_snap.get("raw_close")
    wd=winner_snap.get("avg_dollar_volume20"); cd=ctrl_snap.get("avg_dollar_volume20")
    if not wp or not cp or not wd or not cd or min(wp,cp,wd,cd)<=0:
        return 999
    return abs(math.log(cp/wp)) + 0.65*abs(math.log(cd/wd))


def flatten(prefix,d,row):
    for k,v in d.items():
        row[f"{prefix}_{k}"]=v


def write_csv(path,rows):
    if not rows:
        return
    fields=[]
    seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with Path(path).open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def feature_names_from_rows(rows,prefix="t-1_"):
    if not rows:
        return []
    bad={"symbol","start_date","group","winner_symbol","match_rank"}
    return [
        k for k in rows[0]
        if k.startswith(prefix) and k not in bad
    ]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",default="daily_50pct_tradable30.csv")
    ap.add_argument("--controls-per-winner",type=int,default=10)
    ap.add_argument("--min-price",type=float,default=1.0)
    ap.add_argument("--max-price",type=float,default=20.0)
    ap.add_argument("--min-dollar-volume",type=float,default=1_000_000)
    ap.add_argument("--max-prior-jump",type=float,default=100.0)
    ap.add_argument("--control-max-forward-gain",type=float,default=50.0)
    args=ap.parse_args()

    events_path=Path(args.events)
    if not events_path.exists():
        raise SystemExit(f"Missing {events_path}. Run analyze_daily_50pct_tradable30.py first.")

    with events_path.open(encoding="utf-8-sig",newline="") as fh:
        events=list(csv.DictReader(fh))
    if not events:
        raise SystemExit("Events CSV is empty.")

    winners=[{"symbol":r["symbol"],"start_date":r.get("start_date") or r.get("move_start_date")} for r in events]
    winner_dates={x["start_date"] for x in winners}
    winner_symbols={x["symbol"] for x in winners}

    DB=database()
    with DB() as s:
        md=MetaData()
        daily=Table("market_candles_1d",md,autoload_with=s.get_bind())
        symbols=list(s.execute(
            select(daily.c.symbol).group_by(daily.c.symbol)
            .having(func.count()>=220).order_by(daily.c.symbol)
        ).scalars().all())

    print("\nRajih — DEEP DAILY WINNER RESEARCH")
    print(f"Winners: {len(winners)}")
    print(f"Control matches per winner: {args.controls_per_winner}")
    print(f"Universe with >=220 bars: {len(symbols)}")
    print("Phase 1: winner feature reconstruction...\n")

    feature_cache={}
    winner_rows=[]
    event_paths=[]

    for n,w in enumerate(winners,1):
        rows=load_symbol(DB,daily,w["symbol"])
        f=build_features(rows)
        feature_cache[w["symbol"]]=f
        try:
            i=f["date"].index(w["start_date"])
        except ValueError:
            print(f"WARNING winner date not found: {w}")
            continue

        row={"group":"WINNER","symbol":w["symbol"],"start_date":w["start_date"]}
        for off in OFFSETS:
            flatten(f"t{off}",snap(f,i+off),row)

        # Changes/acceleration
        for metric in (
            "atr_pct","rsi14","adx14","volume_ratio20","volume_accel_5v20",
            "dollar_volume_accel_5v20","ema20_vs_ema50_pct","macd_gap_atr",
            "bb_width20_pct","bb_z20","distance_to_20d_high_pct",
            "position_20d_range_pct","realized_vol20_pct","range_compression_5v20",
        ):
            for a,b,label in ((-20,-5,"d20_to_d5"),(-5,-1,"d5_to_d1"),(-20,-1,"d20_to_d1")):
                va=row.get(f"t{a}_{metric}")
                vb=row.get(f"t{b}_{metric}")
                row[f"{label}_{metric}"]=fmt(vb-va if va is not None and vb is not None else None)

        row.update(outcome(f,i,21))
        winner_rows.append(row)

        start=f["c"][i]
        for off in EVENT_PATH:
            j=i+off
            if 0<=j<len(f["c"]):
                event_paths.append({
                    "symbol":w["symbol"],"start_date":w["start_date"],
                    "offset":off,
                    "close_return_pct":fmt(100*(f["c"][j]/start-1)),
                    "high_return_pct":fmt(100*(f["h"][j]/start-1)),
                    "low_return_pct":fmt(100*(f["l"][j]/start-1)),
                    "volume_ratio20":snap(f,j).get("volume_ratio20"),
                    "atr_pct":snap(f,j).get("atr_pct"),
                    "rsi14":snap(f,j).get("rsi14"),
                    "adx14":snap(f,j).get("adx14"),
                })
        print(f"Winner {n}/{len(winners)} {w['symbol']} {w['start_date']}")

    print("\nPhase 2: matched-control search across the universe...")
    controls_by_date=defaultdict(list)

    # Process each symbol once and only inspect winner dates it contains.
    for n,sym in enumerate(symbols,1):
        if sym in winner_symbols:
            continue
        rows=load_symbol(DB,daily,sym)
        if len(rows)<220:
            continue
        dates=[r["session_date"] for r in rows]
        relevant=[d for d in winner_dates if d in set(dates)]
        if not relevant:
            continue
        f=build_features(rows)

        for d in relevant:
            i=bisect.bisect_left(f["date"],d)
            if i>=len(f["date"]) or f["date"][i]!=d or i<220:
                continue
            raw_price=f["raw_c"][i]
            if not (args.min_price <= raw_price < args.max_price):
                continue
            s1=snap(f,i-1)
            adv=s1.get("avg_dollar_volume20")
            if adv is None or adv<args.min_dollar_volume:
                continue
            if not discontinuity_ok(f,i,args.max_prior_jump):
                continue
            oc=outcome(f,i,21)
            if (oc.get("max_gain_high_21d") or -999) >= args.control_max_forward_gain:
                continue

            base={"group":"CONTROL","symbol":sym,"start_date":d}
            for off in OFFSETS:
                flatten(f"t{off}",snap(f,i+off),base)
            for metric in (
                "atr_pct","rsi14","adx14","volume_ratio20","volume_accel_5v20",
                "dollar_volume_accel_5v20","ema20_vs_ema50_pct","macd_gap_atr",
                "bb_width20_pct","bb_z20","distance_to_20d_high_pct",
                "position_20d_range_pct","realized_vol20_pct","range_compression_5v20",
            ):
                for a,b,label in ((-20,-5,"d20_to_d5"),(-5,-1,"d5_to_d1"),(-20,-1,"d20_to_d1")):
                    va=base.get(f"t{a}_{metric}")
                    vb=base.get(f"t{b}_{metric}")
                    base[f"{label}_{metric}"]=fmt(vb-va if va is not None and vb is not None else None)
            base.update(oc)
            controls_by_date[d].append(base)

        if n%250==0 or n==len(symbols):
            found=sum(len(v) for v in controls_by_date.values())
            print(f"Scanned {n}/{len(symbols)} | eligible control candidates={found}")

    # Match each winner by date, price and liquidity.
    matched=[]
    winner_map={(r["symbol"],r["start_date"]):r for r in winner_rows}
    for w in winner_rows:
        ws=w.get("t-1_raw_close")
        wd=w.get("t-1_avg_dollar_volume20")
        winner_snap={"raw_close":ws,"avg_dollar_volume20":wd}
        pool=[]
        for c in controls_by_date.get(w["start_date"],[]):
            cs={"raw_close":c.get("t-1_raw_close"),"avg_dollar_volume20":c.get("t-1_avg_dollar_volume20")}
            dist=matched_distance(winner_snap,cs)
            pool.append((dist,c))
        pool.sort(key=lambda z:z[0])
        for rank,(dist,c) in enumerate(pool[:args.controls_per_winner],1):
            cc=dict(c)
            cc["winner_symbol"]=w["symbol"]
            cc["match_rank"]=rank
            cc["match_distance"]=fmt(dist)
            matched.append(cc)

    print(f"\nMatched controls: {len(matched)}")

    # Winner-vs-control feature comparison.
    comparison=[]
    feature_cols=[
        k for k in winner_rows[0].keys()
        if (
            k.startswith("t-1_") or k.startswith("t-5_") or
            k.startswith("d5_to_d1_") or k.startswith("d20_to_d1_")
        )
        and k not in {"t-1_raw_close","t-5_raw_close"}
    ]

    for col in feature_cols:
        wv=[r.get(col) for r in winner_rows]
        cv=[r.get(col) for r in matched]
        wclean=[float(x) for x in wv if x is not None and finite(x) is not None]
        cclean=[float(x) for x in cv if x is not None and finite(x) is not None]
        if len(wclean)<10 or len(cclean)<20:
            continue
        auc=auc_separation(wclean,cclean)
        sep=None if auc is None else max(auc,1-auc)
        direction="HIGHER" if auc is not None and auc>=0.5 else "LOWER"
        comparison.append({
            "feature":col,
            "winner_n":len(wclean),"control_n":len(cclean),
            "winner_mean":fmt(mean(wclean)),"winner_median":fmt(median(wclean)),
            "control_mean":fmt(mean(cclean)),"control_median":fmt(median(cclean)),
            "auc_winner_gt_control":fmt(auc),
            "separation":fmt(sep),
            "winner_like_direction":direction,
            "median_difference":fmt(median(wclean)-median(cclean)),
        })
    comparison.sort(key=lambda r:(r["separation"] or 0),reverse=True)

    # Correlations inside winners.
    correlations=[]
    for col in feature_cols:
        vals=[r.get(col) for r in winner_rows]
        correlations.append({
            "feature":col,
            "corr_with_peak_gain":fmt(corr(vals,[r.get("peak_gain_pct") for r in winner_rows])),
            "corr_with_days_to_peak":fmt(corr(vals,[r.get("days_to_peak") for r in winner_rows])),
            "corr_with_mae21":fmt(corr(vals,[r.get("mae_21d_pct") for r in winner_rows])),
        })
    correlations.sort(
        key=lambda r:max(
            abs(r["corr_with_peak_gain"] or 0),
            abs(r["corr_with_days_to_peak"] or 0),
            abs(r["corr_with_mae21"] or 0),
        ),
        reverse=True
    )

    # Threshold scan using top-separated features.
    thresholds=[]
    top_features=[r["feature"] for r in comparison[:20]]
    for col in top_features:
        w=[float(r[col]) for r in winner_rows if finite(r.get(col)) is not None]
        c=[float(r[col]) for r in matched if finite(r.get(col)) is not None]
        if len(w)<10 or len(c)<20:
            continue
        combined=w+c
        cuts=sorted(set(fmt(quantile(combined,p),6) for p in (.1,.2,.3,.4,.5,.6,.7,.8,.9)))
        for cut in cuts:
            if cut is None:
                continue
            for op in (">=","<="):
                if op==">=":
                    wp=sum(x>=cut for x in w)/len(w)
                    cp=sum(x>=cut for x in c)/len(c)
                else:
                    wp=sum(x<=cut for x in w)/len(w)
                    cp=sum(x<=cut for x in c)/len(c)
                if wp<0.30:
                    continue
                lift=wp/cp if cp>0 else 99.0
                precision=(wp*len(w))/(wp*len(w)+cp*len(c)) if (wp*len(w)+cp*len(c))>0 else 0
                thresholds.append({
                    "feature":col,"operator":op,"cut":cut,
                    "winner_coverage_pct":fmt(100*wp),
                    "control_pass_pct":fmt(100*cp),
                    "lift":fmt(lift),
                    "sample_precision_pct":fmt(100*precision),
                })
    thresholds.sort(key=lambda r:(r["lift"],r["winner_coverage_pct"]),reverse=True)

    # Pair rules from top single rules, using only unique feature pairs.
    pairs=[]
    singles=thresholds[:30]
    for a_i in range(len(singles)):
        a=singles[a_i]
        for b_i in range(a_i+1,len(singles)):
            b=singles[b_i]
            if a["feature"]==b["feature"]:
                continue

            def passes(r,rule):
                x=finite(r.get(rule["feature"]))
                if x is None: return False
                return x>=rule["cut"] if rule["operator"]==">=" else x<=rule["cut"]

            wp=sum(passes(r,a) and passes(r,b) for r in winner_rows)/len(winner_rows)
            cp=sum(passes(r,a) and passes(r,b) for r in matched)/len(matched) if matched else 0
            if wp<0.30:
                continue
            lift=wp/cp if cp>0 else 99.0
            pairs.append({
                "rule1":f"{a['feature']} {a['operator']} {a['cut']}",
                "rule2":f"{b['feature']} {b['operator']} {b['cut']}",
                "winner_coverage_pct":fmt(100*wp),
                "control_pass_pct":fmt(100*cp),
                "lift":fmt(lift),
            })
    pairs.sort(key=lambda r:(r["lift"],r["winner_coverage_pct"]),reverse=True)

    # Event-study aggregate.
    trajectory=[]
    for off in EVENT_PATH:
        rr=[x for x in event_paths if x["offset"]==off]
        for metric in ("close_return_pct","high_return_pct","low_return_pct","volume_ratio20","atr_pct","rsi14","adx14"):
            vals=[x[metric] for x in rr if finite(x.get(metric)) is not None]
            if not vals:
                continue
            trajectory.append({
                "offset":off,"metric":metric,"n":len(vals),
                "mean":fmt(mean(vals)),"median":fmt(median(vals)),
                "p25":fmt(quantile(vals,.25)),"p75":fmt(quantile(vals,.75)),
            })

    # Summary stats by checkpoint.
    checkpoints=[]
    snapshot_metrics=[
        "atr_pct","rsi14","adx14","volume_ratio20","volume_accel_5v20",
        "dollar_volume_accel_5v20","ema20_vs_ema50_pct","macd_gap_atr",
        "bb_width20_pct","bb_z20","distance_to_20d_high_pct",
        "position_20d_range_pct","realized_vol20_pct","range_compression_5v20",
        "ret5_pct","ret20_pct","mfi14","cci20","stoch14","williams_r14",
    ]
    for off in OFFSETS:
        for m in snapshot_metrics:
            vals=[r.get(f"t{off}_{m}") for r in winner_rows]
            vals=[float(x) for x in vals if finite(x) is not None]
            if vals:
                checkpoints.append({
                    "offset":off,"metric":m,"n":len(vals),
                    "mean":fmt(mean(vals)),"median":fmt(median(vals)),
                    "p25":fmt(quantile(vals,.25)),"p75":fmt(quantile(vals,.75)),
                })

    # Write outputs.
    write_csv("daily_50pct_deep_winners.csv",winner_rows)
    write_csv("daily_50pct_deep_matched_controls.csv",matched)
    write_csv("daily_50pct_deep_winner_vs_control.csv",comparison)
    write_csv("daily_50pct_deep_correlations.csv",correlations)
    write_csv("daily_50pct_deep_threshold_rules.csv",thresholds)
    write_csv("daily_50pct_deep_pair_rules.csv",pairs)
    write_csv("daily_50pct_deep_trajectory.csv",trajectory)
    write_csv("daily_50pct_deep_checkpoints.csv",checkpoints)

    # Readable report.
    lines=[]
    lines.append("RAJIH — DAILY +50% DEEP RESEARCH")
    lines.append("="*72)
    lines.append(f"Winners: {len(winner_rows)}")
    lines.append(f"Matched controls: {len(matched)}")
    lines.append("")
    lines.append("TOP WINNER-vs-CONTROL SEPARATORS")
    lines.append("-"*72)
    for r in comparison[:20]:
        lines.append(
            f"{r['feature']}: winner median={r['winner_median']} | "
            f"control median={r['control_median']} | "
            f"separation={r['separation']} | {r['winner_like_direction']}"
        )
    lines.append("")
    lines.append("TOP SINGLE THRESHOLD RULES")
    lines.append("-"*72)
    for r in thresholds[:20]:
        lines.append(
            f"{r['feature']} {r['operator']} {r['cut']} | "
            f"winner coverage={r['winner_coverage_pct']}% | "
            f"control pass={r['control_pass_pct']}% | lift={r['lift']}"
        )
    lines.append("")
    lines.append("TOP TWO-FEATURE RULES")
    lines.append("-"*72)
    for r in pairs[:20]:
        lines.append(
            f"{r['rule1']} AND {r['rule2']} | "
            f"winner coverage={r['winner_coverage_pct']}% | "
            f"control pass={r['control_pass_pct']}% | lift={r['lift']}"
        )
    lines.append("")
    lines.append("STRONGEST CORRELATIONS INSIDE WINNERS")
    lines.append("-"*72)
    for r in correlations[:20]:
        lines.append(
            f"{r['feature']} | gain={r['corr_with_peak_gain']} | "
            f"days_to_peak={r['corr_with_days_to_peak']} | MAE21={r['corr_with_mae21']}"
        )

    Path("daily_50pct_deep_report.txt").write_text("\n".join(lines),encoding="utf-8")

    summary={
        "winners":len(winner_rows),
        "matched_controls":len(matched),
        "top_separators":comparison[:20],
        "top_single_rules":thresholds[:20],
        "top_pair_rules":pairs[:20],
        "top_correlations":correlations[:20],
        "output_files":[
            "daily_50pct_deep_winners.csv",
            "daily_50pct_deep_matched_controls.csv",
            "daily_50pct_deep_winner_vs_control.csv",
            "daily_50pct_deep_correlations.csv",
            "daily_50pct_deep_threshold_rules.csv",
            "daily_50pct_deep_pair_rules.csv",
            "daily_50pct_deep_trajectory.csv",
            "daily_50pct_deep_checkpoints.csv",
            "daily_50pct_deep_report.txt",
        ],
    }
    Path("daily_50pct_deep_report.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\n=== DEEP RESEARCH COMPLETE ===")
    print(f"Winners: {len(winner_rows)}")
    print(f"Matched controls: {len(matched)}")
    print("\nTOP 10 SEPARATORS:")
    for r in comparison[:10]:
        print(
            f"{r['feature']}: W med={r['winner_median']} | "
            f"C med={r['control_median']} | sep={r['separation']} | "
            f"{r['winner_like_direction']}"
        )

    print("\nTOP 10 SINGLE RULES:")
    for r in thresholds[:10]:
        print(
            f"{r['feature']} {r['operator']} {r['cut']} | "
            f"W={r['winner_coverage_pct']}% C={r['control_pass_pct']}% lift={r['lift']}"
        )

    print("\nTOP 10 PAIR RULES:")
    for r in pairs[:10]:
        print(
            f"{r['rule1']} AND {r['rule2']} | "
            f"W={r['winner_coverage_pct']}% C={r['control_pass_pct']}% lift={r['lift']}"
        )

    print("\nCreated:")
    for x in summary["output_files"]:
        print(" ",x)
    print("  daily_50pct_deep_report.json")


if __name__=="__main__":
    main()
