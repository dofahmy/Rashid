#!/usr/bin/env python3
"""READ ONLY: evaluate the exact S&P 500 price-only bottom rule for US stocks on 18 specified dates.

Uses local PostgreSQL daily price cache; NO Time-Price candidates, pivots or filters.
Includes all US market_stocks with cached daily bars, and reports unassessable coverage.
"""
import argparse
import csv
from pathlib import Path
from sqlalchemy import text

DATES = (
    '2018-02-08','2018-03-23','2018-10-11','2018-11-20','2018-12-07',
    '2019-06-03','2020-02-25','2022-01-21','2022-03-07','2022-04-26',
    '2022-06-13','2022-09-20','2023-03-10','2023-09-26','2023-10-26',
    '2024-08-05','2025-03-10','2026-03-20',
)

# Same four conditions as sp500_price_only_bottom.py:
# current close / close 20 daily BARS earlier - 1 <= -3.5%;
# 5 bars <= -1.5%; current close / highest HIGH of 60 bars - 1 <= -7%;
# (current close - lowest LOW of 60 bars) / (highest HIGH-lowest LOW) <=25%.
SQL = """
WITH ranked AS (
 SELECT b.symbol, b.day, b.c AS close, b.h, b.l,
        LAG(b.c,5) OVER (PARTITION BY b.symbol ORDER BY b.day) AS c5,
        LAG(b.c,20) OVER (PARTITION BY b.symbol ORDER BY b.day) AS c20,
        MAX(b.h) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS high60,
        MIN(b.l) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS low60,
        COUNT(*) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS n60
 FROM us_tp_daily_bars b
 JOIN market_stocks s ON s.symbol=b.symbol AND s.market='US'
 WHERE b.day <= :lastday
), evaluated AS (
 SELECT r.symbol, r.day, s.company, r.close, r.c5,r.c20,r.high60,r.low60,r.n60,
        (100.0*(r.close/r.c5-1)) AS prior5_pct,
        (100.0*(r.close/r.c20-1)) AS prior20_pct,
        (100.0*(r.close/r.high60-1)) AS fromhigh60_pct,
        (CASE WHEN r.high60>r.low60 THEN 100.0*(r.close-r.low60)/(r.high60-r.low60) ELSE 50.0 END) AS pos60_pct
 FROM ranked r
 JOIN market_stocks s ON s.symbol=r.symbol
 WHERE r.day IN (""" + ','.join(':d'+str(i) for i in range(len(DATES))) + """))
SELECT *, CASE WHEN n60>=60 AND c5>0 AND c20>0 AND high60>0 AND low60>0 THEN TRUE ELSE FALSE END AS enough,
       CASE WHEN n60>=60 AND c5>0 AND c20>0 AND high60>0 AND low60>0
                  AND prior5_pct <= -1.5 AND prior20_pct <= -3.5
                  AND fromhigh60_pct <= -7.0 AND pos60_pct <= 25.0
            THEN TRUE ELSE FALSE END AS matched
FROM evaluated
ORDER BY day, symbol
"""


def save(path, cols, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/tmp/us_price_only_18')
    args=p.parse_args()
    from core import database
    db=database()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    binds={'lastday': max(DATES), **{'d'+str(i):d for i,d in enumerate(DATES)}}
    summary={d:{'date':d,'matched_stocks':0,'assessed_stocks':0,'incomplete_history':0,'daily_bars_present':0,'symbols':''} for d in DATES}
    matching=[]
    with db() as s:
        total_us=s.execute(text("SELECT COUNT(*) FROM market_stocks WHERE market='US'")).scalar()
        minmax=s.execute(text("SELECT MIN(b.day), MAX(b.day), COUNT(DISTINCT b.symbol) FROM us_tp_daily_bars b JOIN market_stocks s ON s.symbol=b.symbol WHERE s.market='US'")).one()
        print('US UNIVERSE:',total_us, '| CACHED PRICE SYMBOLS:',minmax[2], '| CACHE RANGE:',minmax[0],minmax[1],flush=True)
        # Streaming avoids loading all historical rows into Python; SQL windows evaluate all needed past bars.
        res=s.execute(text(SQL),binds).mappings()
        matched_by_date={d:[] for d in DATES}
        for r in res:
            d=r['day']; row=summary[d]; row['daily_bars_present']+=1
            if not r['enough']:
                row['incomplete_history']+=1;continue
            row['assessed_stocks']+=1
            if r['matched']:
                row['matched_stocks']+=1
                matched_by_date[d].append(r['symbol'])
                matching.append({'date':d,'symbol':r['symbol'],'company':r['company'],
                    **{k:round(float(r[k]),4) for k in ('prior5_pct','prior20_pct','fromhigh60_pct','pos60_pct')},
                    'close':round(float(r['close']),4)})
    results=[]
    print('RULE: prior20<=-3.5 prior5<=-1.5 fromhigh60<=-7 pos60<=25; all conditions simultaneously',flush=True)
    print('DATES:',len(DATES),'| NO Time-Price | read-only',flush=True)
    for d in DATES:
        r=summary[d]
        r['symbols']=';'.join(matched_by_date[d])
        r['no_bar_universe']=total_us-r['daily_bars_present']
        r['status']='NO_CACHED_BARS' if r['daily_bars_present']==0 else ('PARTIAL_COVERAGE' if r['assessed_stocks']<total_us else 'COVERED')
        results.append(r)
        print('DATE',d,'MATCHING:',r['matched_stocks'],'ASSESSABLE:',r['assessed_stocks'],
              'HAS_BAR_BUT_LESS_THAN_60:',r['incomplete_history'], 'NO_BAR:',r['no_bar_universe'],
              'STATUS:',r['status'],flush=True)
    save(str(out)+'_counts.csv',list(results[0]),results)
    save(str(out)+'_stocks.csv',['date','symbol','company','close','prior5_pct','prior20_pct','fromhigh60_pct','pos60_pct'],matching)
    print('FILES:',str(out)+'_counts.csv',str(out)+'_stocks.csv')
    print('NOTE: Historical DB coverage varies; 0 matches with zero assessable symbols means UNKNOWN, not zero US stocks.')
    print('NOTE: stocks with fewer than 60 available daily bars are unassessable. Delisted historical symbols may be missing (survivorship bias).')

if __name__=='__main__':main()
