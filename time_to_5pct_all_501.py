#!/usr/bin/env python3
"""
Rajih — timing to +5% across the 501 all-date setups.

Reads:
  /data/daily_rule_all_dates.csv
  market_candles_1d

Outputs:
  /data/daily_501_time_to_5pct.csv
  /data/daily_501_time_to_5pct_summary.json
"""

import os, csv, json, math
from pathlib import Path
from statistics import mean, median
from sqlalchemy import MetaData, Table, select
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_rule_all_dates.csv"

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run test_daily_rule_all_501_dates_PERSISTENT.py first.")

def fnum(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except:
        return None

with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    setups=list(csv.DictReader(fh))

DB=database()
with DB() as s:
    md=MetaData()
    daily=Table("market_candles_1d", md, autoload_with=s.get_bind())

rows=[]
for idx,r in enumerate(setups,1):
    sym=r["symbol"]
    dt=r["signal_date"]

    with DB() as s:
        raw=list(s.execute(
            select(daily.c.session_date, daily.c.h, daily.c.c, daily.c.adj_c)
            .where(daily.c.symbol==sym)
            .order_by(daily.c.session_date)
        ).all())

    bars=[]
    for d,h,c,ac in raw:
        rc=fnum(c); adj=fnum(ac); hh=fnum(h)
        if None in (rc,adj,hh) or min(rc,adj,hh)<=0:
            continue
        fac=adj/rc
        bars.append((str(d), adj, hh*fac))

    dates=[x[0] for x in bars]
    try:
        i=dates.index(dt)
    except ValueError:
        continue
    if i+1>=len(bars):
        continue

    j2=min(len(bars)-1,i+21)
    start=bars[i][1]
    day_to_5=None
    max_gain=-1e99
    for j in range(i+1,j2+1):
        g=100*(bars[j][2]/start-1)
        max_gain=max(max_gain,g)
        if day_to_5 is None and g>=5:
            day_to_5=j-i

    rows.append({
        "symbol":sym,
        "signal_date":dt,
        "days_to_5":day_to_5 if day_to_5 is not None else "",
        "hit_5":1 if day_to_5 is not None else 0,
        "max_high_21d_pct":round(max_gain,4),
    })

    if idx%100==0 or idx==len(setups):
        print(f"Processed {idx}/{len(setups)}")

hit=[r for r in rows if r["hit_5"]]
days=[int(r["days_to_5"]) for r in hit]

def quantile(vals,p):
    vals=sorted(vals)
    if not vals: return None
    k=(len(vals)-1)*p
    lo=math.floor(k); hi=math.ceil(k)
    if lo==hi: return vals[lo]
    return vals[lo]*(hi-k)+vals[hi]*(k-lo)

checkpoints=[1,2,3,5,7,10,15,21]
cum=[]
for d in checkpoints:
    n=sum(x<=d for x in days)
    cum.append({
        "day":d,
        "hit_n":n,
        "pct_of_all":round(100*n/len(rows),2),
        "pct_of_5pct_hitters":round(100*n/len(hit),2) if hit else None,
    })

summary={
    "total":len(rows),
    "hit_5_n":len(hit),
    "hit_5_rate_pct":round(100*len(hit)/len(rows),2),
    "median_days_to_5":round(median(days),2) if days else None,
    "mean_days_to_5":round(mean(days),2) if days else None,
    "p25_days_to_5":round(quantile(days,.25),2) if days else None,
    "p75_days_to_5":round(quantile(days,.75),2) if days else None,
    "cumulative":cum,
}

out_csv=DATA_DIR/"daily_501_time_to_5pct.csv"
with out_csv.open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

out_json=DATA_DIR/"daily_501_time_to_5pct_summary.json"
out_json.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

print("\n=== TIME TO +5% — 501 SETUPS ===")
print(f"Reached +5%: {len(hit)}/{len(rows)} = {summary['hit_5_rate_pct']:.2f}%")
print(f"Median days to +5%: {summary['median_days_to_5']}")
print(f"Mean days to +5%: {summary['mean_days_to_5']}")
print(f"P25-P75: {summary['p25_days_to_5']} to {summary['p75_days_to_5']} days")
print("\nCumulative:")
for x in cum:
    print(
        f"By day {x['day']:>2}: {x['hit_n']}/{len(rows)} = "
        f"{x['pct_of_all']:.2f}% of all setups "
        f"({x['pct_of_5pct_hitters']:.2f}% of eventual +5% hitters)"
    )
print(f"\nCreated:\n {out_csv}\n {out_json}")
