#!/usr/bin/env python3
"""
Rajih — CLEAN tradable sample:
Stocks that gained +200% within ~3 months, but did NOT reach +200% in month 1.

Cleaner filters:
- Start raw close >= $1
- 20d average dollar volume >= $1,000,000
- First +200% touch is trading day 22..63
- Reject any adjusted-close one-day jump >100% in the 20 sessions BEFORE start
- Reject any adjusted-close one-day jump >100% DURING the 63-session forward window
- One event per symbol: strongest qualifying event
- Rank TOP 30 by 20d dollar liquidity, not by extreme % gain

Reads:
  market_candles_1d

Outputs:
  /data/daily_200pct_3months_clean.csv
  /data/daily_200pct_3months_clean_summary.json
"""

from __future__ import annotations
import os, csv, json, math
from pathlib import Path
from statistics import mean, median
from sqlalchemy import MetaData, Table, select, func
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = DATA_DIR / "daily_200pct_3months_clean.csv"
OUT_JSON = DATA_DIR / "daily_200pct_3months_clean_summary.json"

TARGET = 200.0
M1, M2, M3 = 21, 42, 63
MIN_HISTORY = 220
MIN_PRICE = 1.0
MIN_DOLLAR_VOL20 = 1_000_000.0
MAX_ABS_DAILY_ADJ_RETURN = 100.0

def fnum(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except:
        return None

def pct(a,b):
    return 100*(a/b-1)

def load_symbol(DB,daily,sym):
    with DB() as s:
        raw=list(s.execute(
            select(
                daily.c.session_date,daily.c.o,daily.c.h,daily.c.l,
                daily.c.c,daily.c.v,daily.c.adj_c
            )
            .where(daily.c.symbol==sym)
            .order_by(daily.c.session_date)
        ).all())
    rows=[]
    for d,o,h,l,c,v,ac in raw:
        ro,rh,rl,rc,adj=map(fnum,(o,h,l,c,ac))
        if None in (ro,rh,rl,rc,adj) or min(ro,rh,rl,rc,adj)<=0:
            continue
        fac=adj/rc
        rows.append({
            "date":str(d),"raw_close":rc,"close":adj,
            "high":rh*fac,"low":rl*fac,"volume":float(v or 0)
        })
    return rows

def avg_dollar_vol20(rows,i):
    if i<19: return None
    vals=[rows[j]["raw_close"]*rows[j]["volume"] for j in range(i-19,i+1)]
    return sum(vals)/20

def clean_window(rows,a,b):
    a=max(1,a)
    b=min(len(rows)-1,b)
    for j in range(a,b+1):
        r=abs(pct(rows[j]["close"],rows[j-1]["close"]))
        if r>MAX_ABS_DAILY_ADJ_RETURN:
            return False
    return True

DB=database()
with DB() as s:
    md=MetaData()
    daily=Table("market_candles_1d",md,autoload_with=s.get_bind())
    symbols=list(s.execute(
        select(daily.c.symbol)
        .group_by(daily.c.symbol)
        .having(func.count()>=MIN_HISTORY+M3)
        .order_by(daily.c.symbol)
    ).scalars().all())

print("\nRajih — CLEAN +200% / 3-MONTH SAMPLE")
print(f"Universe: {len(symbols)}")
print("Filters: start >= $1, $vol20 >= $1M, no >100% adjusted 1-day discontinuity, first +200 day 22..63\n")

best={}
events=0

for n,sym in enumerate(symbols,1):
    rows=load_symbol(DB,daily,sym)
    if len(rows)<MIN_HISTORY+M3:
        continue

    for i in range(MIN_HISTORY-1,len(rows)-M3):
        if rows[i]["raw_close"]<MIN_PRICE:
            continue

        adv=avg_dollar_vol20(rows,i)
        if adv is None or adv<MIN_DOLLAR_VOL20:
            continue

        if not clean_window(rows,i-19,i):
            continue
        if not clean_window(rows,i+1,i+M3):
            continue

        start=rows[i]["close"]
        m1=max(pct(rows[j]["high"],start) for j in range(i+1,i+M1+1))
        if m1>=TARGET:
            continue

        full=[]
        first200=None
        peak_gain=-1e99
        peak_j=None
        for j in range(i+1,i+M3+1):
            g=pct(rows[j]["high"],start)
            full.append(g)
            if first200 is None and g>=TARGET:
                first200=j-i
            if g>peak_gain:
                peak_gain=g
                peak_j=j

        if first200 is None or first200<=M1:
            continue

        events+=1
        m2=max(pct(rows[j]["high"],start) for j in range(i+M1+1,i+M2+1))
        m3=max(pct(rows[j]["high"],start) for j in range(i+M2+1,i+M3+1))

        ev={
            "symbol":sym,
            "start_date":rows[i]["date"],
            "start_raw_close":round(rows[i]["raw_close"],6),
            "avg_dollar_volume20":round(adv,2),
            "month1_max_gain_pct":round(m1,4),
            "month2_block_max_gain_pct":round(m2,4),
            "month3_block_max_gain_pct":round(m3,4),
            "first_200_touch_day":first200,
            "first_200_touch_date":rows[i+first200]["date"],
            "max_gain_63d_pct":round(peak_gain,4),
            "peak_day_63":peak_j-i,
            "peak_date_63":rows[peak_j]["date"],
            "close_return_day21_pct":round(pct(rows[i+M1]["close"],start),4),
            "close_return_day42_pct":round(pct(rows[i+M2]["close"],start),4),
            "close_return_day63_pct":round(pct(rows[i+M3]["close"],start),4),
            "gain_added_after_month1_pct_points":round(peak_gain-m1,4),
        }

        prev=best.get(sym)
        if prev is None or ev["max_gain_63d_pct"]>prev["max_gain_63d_pct"]:
            best[sym]=ev

    if n%250==0 or n==len(symbols):
        print(f"Scanned {n}/{len(symbols)} | events={events} | distinct={len(best)}")

results=sorted(best.values(),key=lambda r:(r["avg_dollar_volume20"],r["max_gain_63d_pct"]),reverse=True)

if not results:
    raise SystemExit("No qualifying clean events found.")

with OUT_CSV.open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)

summary={
    "events_before_dedup":events,
    "distinct_symbols":len(results),
    "median_first_200_touch_day":round(median(r["first_200_touch_day"] for r in results),2),
    "median_month1_max_gain_pct":round(median(r["month1_max_gain_pct"] for r in results),2),
    "median_63d_max_gain_pct":round(median(r["max_gain_63d_pct"] for r in results),2),
    "filters":{
        "min_start_price":MIN_PRICE,
        "min_avg_dollar_volume20":MIN_DOLLAR_VOL20,
        "max_abs_adjusted_daily_return":MAX_ABS_DAILY_ADJ_RETURN,
    }
}
OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

print("\n=== CLEAN RESULTS ===")
print(f"Qualifying events: {events}")
print(f"Distinct symbols: {len(results)}")
print(f"Median first +200 touch: day {summary['median_first_200_touch_day']}")
print(f"Median month-1 max gain: {summary['median_month1_max_gain_pct']:.2f}%")
print(f"Median 3-month max gain: {summary['median_63d_max_gain_pct']:.2f}%")

print("\nTOP 30 BY LIQUIDITY:")
print("rank | symbol | start | price | $vol20 | M1 max | first+200 | 63d max")
for k,r in enumerate(results[:30],1):
    print(
        f"{k:>4} | {r['symbol']:<7} | {r['start_date']} | "
        f"${r['start_raw_close']:>7.2f} | "
        f"${r['avg_dollar_volume20']/1e6:>8.1f}M | "
        f"{r['month1_max_gain_pct']:>7.1f}% | "
        f"{r['first_200_touch_day']:>9} | "
        f"{r['max_gain_63d_pct']:>7.1f}%"
    )

print(f"\nCreated:\n {OUT_CSV}\n {OUT_JSON}")
