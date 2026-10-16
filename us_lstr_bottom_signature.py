#!/usr/bin/env python3
"""Find US shares whose four price features match the fixed LSTR 2022-09-20 fingerprint.

No Time-Price, no S&P comparison on scan dates. Read-only database query.
"""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from sqlalchemy import text

REF = {
    'prior5_pct': -2.01805,
    'prior20_pct': -6.74533,
    'fromhigh60_pct': -10.84791,
    'pos60_pct': 21.53217,
}
METRICS = tuple(REF)

# limit history before a bounded scan while retaining at least 60 sessions
# 180 calendar days is ample for the rolling 60-session lookback.
CANDIDATES_SQL = """
WITH rolling AS (
 SELECT b.symbol, b.day, b.c AS close, b.h, b.l,
 LAG(b.c,5) OVER (PARTITION BY b.symbol ORDER BY b.day) AS c5,
 LAG(b.c,20) OVER (PARTITION BY b.symbol ORDER BY b.day) AS c20,
 MAX(b.h) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS high60,
 MIN(b.l) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS low60,
 COUNT(*) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS n60
 FROM us_tp_daily_bars b
 JOIN market_stocks s ON s.symbol=b.symbol AND s.market='US'
 WHERE b.day >= :lookback AND b.day <= :finish
), metrics AS (
 SELECT r.symbol, r.day, r.close,
 100.0*(r.close/NULLIF(r.c5,0)-1) AS prior5_pct,
 100.0*(r.close/NULLIF(r.c20,0)-1) AS prior20_pct,
 100.0*(r.close/NULLIF(r.high60,0)-1) AS fromhigh60_pct,
 CASE WHEN r.high60>r.low60 THEN 100.0*(r.close-r.low60)/(r.high60-r.low60) END AS pos60_pct
 FROM rolling r WHERE r.day >= :start AND r.day <= :finish AND r.n60=60
), accepted AS (
 SELECT * FROM metrics WHERE
 prior5_pct BETWEEN :p5lo AND :p5hi AND
 prior20_pct BETWEEN :p20lo AND :p20hi AND
 fromhigh60_pct BETWEEN :fhlo AND :fhhi AND
 pos60_pct BETWEEN :poslo AND :poshi
)
SELECT a.*, s.company
FROM accepted a JOIN market_stocks s ON s.symbol=a.symbol AND s.market='US'
ORDER BY a.day,a.symbol
"""
LATEST_SQL = """
SELECT DISTINCT ON (b.symbol) b.symbol, b.day AS latest_day, b.c AS latest_close
FROM us_tp_daily_bars b
WHERE b.symbol IN :symbols
ORDER BY b.symbol,b.day DESC
"""
# Use temp inline aggregate for latest bars if IN expansion not available.
LATEST_BY_SYMBOL_SQL = """
SELECT b.day, b.c FROM us_tp_daily_bars b
WHERE b.symbol=:sym AND b.day >= :asof
ORDER BY b.day DESC LIMIT 1
"""


def write_csv(path, rows, cols):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--date', default=None, help='YYYY-MM-DD, e.g. 2026-10-12')
    g.add_argument('--all-history', action='store_true', help='scan every cached trading date, in calendar-month batches')
    p.add_argument('--deviation', type=float, default=10, help='relative percentage tolerance per one of 4 features, default 10')
    p.add_argument('--out', default='/tmp/us_lstr_bottom_signature')
    args=p.parse_args()
    if not (0 <= args.deviation <=100):p.error('--deviation must be 0..100')
    from core import database
    db=database()
    with db() as session:
        lohi=session.execute(text("SELECT MIN(b.day), MAX(b.day) FROM us_tp_daily_bars b JOIN market_stocks s ON s.symbol=b.symbol AND s.market='US' ")).one()
    db_start, db_end = lohi
    if not db_start or not db_end:raise RuntimeError('No US daily bars present')
    if isinstance(db_start,str):db_start=date.fromisoformat(db_start)
    if isinstance(db_end,str):db_end=date.fromisoformat(db_end)
    if args.all_history:
        chunks=[]
        cursor=date(db_start.year,db_start.month,1)
        while cursor<=db_end:
            nextmonth=date(cursor.year+1,1,1) if cursor.month==12 else date(cursor.year,cursor.month+1,1)
            chunks.append((max(cursor,db_start),min(nextmonth-timedelta(days=1),db_end)))
            cursor=nextmonth
    else:
        target=date.fromisoformat(args.date or '2026-10-12')
        chunks=[(target,target)]
    tol=args.deviation/100
    bands={k:(min(v*(1-tol),v*(1+tol)),max(v*(1-tol),v*(1+tol))) for k,v in REF.items()}
    queryargs={
        'p5lo':bands['prior5_pct'][0], 'p5hi':bands['prior5_pct'][1],
        'p20lo':bands['prior20_pct'][0], 'p20hi':bands['prior20_pct'][1],
        'fhlo':bands['fromhigh60_pct'][0], 'fhhi':bands['fromhigh60_pct'][1],
        'poslo':bands['pos60_pct'][0], 'poshi':bands['pos60_pct'][1],
    }
    print('FIXED REFERENCE: LSTR on 2022-09-20, NOT S&P readings on each scan day')
    print('REFERENCE:',REF,'RELATIVE DEVIATION +/-',args.deviation,'% EACH')
    print('CACHED US PRICES:',db_start,'to',db_end,'CHUNKS:',len(chunks),flush=True)
    found=[]
    with db() as session:
        for i,(start,finish) in enumerate(chunks,1):
            params={**queryargs,'lookback':start-timedelta(days=180),'start':start,'finish':finish}
            for raw in session.execute(text(CANDIDATES_SQL),params).mappings():
                row=dict(raw)
                row['date']=str(row.pop('day'))
                row['signal_close']=float(row.pop('close'))
                for feature in REF:
                    row[feature]=float(row[feature])
                    row['deviation_'+feature]=100*abs(row[feature]-REF[feature])/abs(REF[feature])
                row['max_deviation_pct']=max(row['deviation_'+k] for k in REF)
                found.append(row)
            if args.all_history and (i%6==0 or i==len(chunks)):
                print('SCANNED MONTHS:',i,'/',len(chunks),'MATCHES SO FAR:',len(found),flush=True)
        # Latest cached close (at/after each signal date) from existing US cache.
        # Group symbols so the latest bar is only looked up once per unique symbol.
        latest={}
        for symbol in sorted({r['symbol'] for r in found}):
            result=session.execute(text("SELECT day,c FROM us_tp_daily_bars WHERE symbol=:symbol ORDER BY day DESC LIMIT 1"), {'symbol':symbol}).first()
            latest[symbol]=(str(result[0]),float(result[1])) if result else ('',None)
    for r in found:
        ld,lc=latest[r['symbol']]
        r['latest_day']=ld
        r['latest_close']=lc
        r['change_to_latest_pct']=round(100*(lc/r['signal_close']-1),4) if lc is not None and ld>=r['date'] and r['signal_close'] else None
        r['signal_close']=round(r['signal_close'],4)
        r['max_deviation_pct']=round(r['max_deviation_pct'],3)
        for key in REF:
            r[key]=round(r[key],5)
            r['deviation_'+key]=round(r['deviation_'+key],3)
    counts={}
    for r in found:counts[r['date']]=counts.get(r['date'],0)+1
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    cols=['date','symbol','company','signal_close','latest_day','latest_close','change_to_latest_pct']+list(REF)+['deviation_'+k for k in REF]+['max_deviation_pct']
    write_csv(str(output)+'_matches.csv',found,cols)
    write_csv(str(output)+'_counts.csv',[{'date':d,'count':v} for d,v in sorted(counts.items())],['date','count'])
    print('TOTAL MATCHING ROWS:',len(found),'DATES WITH MATCH:',len(counts))
    for d,n in sorted(counts.items()):print('DATE:',d,'MATCHES:',n)
    if not args.all_history:
        for r in found:
            print('STOCK',r['symbol'],r['company'],'SIGNAL',r['date'],r['signal_close'],'LATEST',r['latest_day'],r['latest_close'],'CHANGE %',r['change_to_latest_pct'],'MAX DEV %',r['max_deviation_pct'])
        if not found:print('NO MATCHES for requested date; check cached bars and try --all-history for wider search.')
    print('FILES:',str(output)+'_matches.csv',str(output)+'_counts.csv')
    print('READ ONLY. No trading, Time-Price, or index-relative comparison. 2021+ cached coverage is incomplete and has survivorship bias.')

if __name__=='__main__':main()
