#!/usr/bin/env python3
import os
import csv, math
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
from statistics import median, mean
from sqlalchemy import MetaData, Table, select
from core import database

INPUT = DATA_DIR / "daily_rule_validation30_diverse_dates.csv"

def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except:
        return None

p = Path(INPUT)
if not p.exists():
    raise SystemExit(f"Missing {INPUT}. Run validate_daily_rule_30_diverse_dates.py first.")

with p.open(encoding="utf-8-sig", newline="") as fh:
    sample = list(csv.DictReader(fh))

DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

def load(sym):
    with DB() as s:
        raw = list(s.execute(
            select(daily.c.session_date, daily.c.h, daily.c.l, daily.c.c, daily.c.adj_c)
            .where(daily.c.symbol == sym)
            .order_by(daily.c.session_date)
        ).all())
    out = []
    for d,h,l,c,ac in raw:
        rc=fnum(c); adj=fnum(ac); hh=fnum(h); ll=fnum(l)
        if None in (rc,adj,hh,ll) or min(rc,adj,hh,ll) <= 0:
            continue
        fac=adj/rc
        out.append({"date":str(d),"c":adj,"h":hh*fac,"l":ll*fac})
    return out

def first_touch(bars, i, j2, start, th):
    for j in range(i+1, j2+1):
        if 100*(bars[j]["h"]/start-1) >= th:
            return j-i
    return None

rows=[]
for r in sample:
    sym=r["symbol"]; dt=r["signal_date"]
    bars=load(sym)
    dates=[x["date"] for x in bars]
    try:
        i=dates.index(dt)
    except ValueError:
        continue
    j2=min(len(bars)-1, i+21)
    if j2<=i:
        continue
    start=bars[i]["c"]
    peak_j=max(range(i+1,j2+1), key=lambda j:bars[j]["h"])
    final_ret=100*(bars[j2]["c"]/start-1)
    x={
        "symbol":sym,
        "signal_date":dt,
        "peak_gain_pct":100*(bars[peak_j]["h"]/start-1),
        "days_to_peak":peak_j-i,
        "final_return_pct":final_ret,
    }
    for th in (20,30,50,100):
        x[f"days_to_{th}"]=first_touch(bars,i,j2,start,th)
    rows.append(x)

print("\n=== 30-SAMPLE TIMING ===")
print(f"Usable: {len(rows)}")
print(f"Median days to peak (all 30): {median(r['days_to_peak'] for r in rows):.1f}")
print(f"Mean days to peak (all 30): {mean(r['days_to_peak'] for r in rows):.1f}")

for th in (20,30,50,100):
    touched=[r for r in rows if r[f"days_to_{th}"] is not None]
    if not touched:
        continue
    days=[r[f"days_to_{th}"] for r in touched]
    peakdays=[r["days_to_peak"] for r in touched]
    ended_down=[r for r in touched if r["final_return_pct"] < 0]
    print(f"\nTouched +{th}%: {len(touched)}/{len(rows)} = {100*len(touched)/len(rows):.2f}%")
    print(f"Median days to +{th}: {median(days):.1f}")
    print(f"Median days to peak: {median(peakdays):.1f}")
    print(f"Finished DOWN after touch: {len(ended_down)}/{len(touched)} = {100*len(ended_down)/len(touched):.2f}%")

with (DATA_DIR / "daily_validation30_timing.csv").open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"\nCreated: {DATA_DIR / 'daily_validation30_timing.csv'}")
