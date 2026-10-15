#!/usr/bin/env python3
"""Sector lookup and dated breadth report for Rajih US Time-Price.

Fetches current Yahoo Finance sector/industry metadata for relevant symbols,
with a persistent PostgreSQL cache. Does NOT alter trading tables or publish alerts.
Current sectors are not guaranteed to match historical company classifications.
"""
import argparse
import collections
import datetime as dt
import json
import sys
import time
from sqlalchemy import text
from core import database

SQL_GROUPS = text('''
WITH eligible AS (
 SELECT day,signal_type,anchor_type,status,COUNT(DISTINCT symbol) AS n
 FROM us_tp_daily_candidates
 WHERE signal_type='SAME_ANGLE_DIFF_ANCHORS' AND anchor_type='LOW'
   AND status IN ('CROSS_DOWN_CLOSE','TOUCH_ABOVE')
 GROUP BY day,signal_type,anchor_type,status
 HAVING COUNT(DISTINCT symbol)>=:minimum
)
SELECT DISTINCT c.day,c.status,c.symbol
FROM us_tp_daily_candidates c JOIN eligible g
 ON g.day=c.day AND g.signal_type=c.signal_type
 AND g.anchor_type=c.anchor_type AND g.status=c.status
WHERE c.signal_type='SAME_ANGLE_DIFF_ANCHORS' AND c.anchor_type='LOW'
  AND c.status IN ('CROSS_DOWN_CLOSE','TOUCH_ABOVE')
ORDER BY c.day,c.status,c.symbol
''')
SQL_CREATE = text('''CREATE TABLE IF NOT EXISTS us_tp_sector_lookup (
  symbol VARCHAR(40) PRIMARY KEY,
  sector TEXT,
  industry TEXT,
  source TEXT NOT NULL DEFAULT 'yfinance',
  status TEXT NOT NULL DEFAULT 'UNKNOWN',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)''')


def get_universe(db, minimum):
    with db() as s:
        rows=[dict(r) for r in s.execute(SQL_GROUPS,{'minimum':minimum}).mappings()]
        ticker_map=dict(s.execute(text("SELECT symbol,COALESCE(NULLIF(feed_symbol,''),symbol) FROM market_stocks WHERE market='US'")).all())
    return rows,ticker_map


def fetch(db, rows, ticker_map, limit, sleep_s, refresh, retry_failed):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit('Missing yfinance. Install with: pip install yfinance')
    symbols=sorted({r['symbol'] for r in rows})
    with db.begin() as s:
        s.execute(SQL_CREATE)
    with db() as s:
        existing={r['symbol']:r['status'] for r in s.execute(text('SELECT symbol,status FROM us_tp_sector_lookup')).mappings()}
    pending=[sym for sym in symbols if refresh or sym not in existing or (retry_failed and existing[sym]!='OK')]
    if limit > 0: pending=pending[:limit]
    print(f'Qualified unique stocks: {len(symbols)} | Cached: {len(set(symbols)&set(existing))} | To fetch now: {len(pending)}',flush=True)
    for i,sym in enumerate(pending,1):
        feed=ticker_map.get(sym,sym)
        tries=[feed]
        alt=feed.replace('.','-')
        if alt not in tries: tries.append(alt)
        sector=industry=None; status='ERROR'
        try:
            for ticker in tries:
                info=yf.Ticker(ticker).info or {}
                sector=info.get('sector') or None
                industry=info.get('industry') or None
                if sector or industry:
                    status='OK';break
            if status!='OK': status='UNCLASSIFIED'
        except Exception as e:
            status='ERROR'
            if i<=3 or i%25==0: print(f'{sym} fetch failed: {type(e).__name__}: {str(e)[:120]}',flush=True)
        with db.begin() as s:
            s.execute(text('''INSERT INTO us_tp_sector_lookup(symbol,sector,industry,source,status,updated_at)
                VALUES(:symbol,:sector,:industry,'yfinance',:status,NOW())
                ON CONFLICT(symbol) DO UPDATE SET sector=excluded.sector,industry=excluded.industry,
                source=excluded.source,status=excluded.status,updated_at=NOW()'''),
                {'symbol':sym,'sector':sector,'industry':industry,'status':status})
        if i%25==0 or i==len(pending): print(f'Fetched {i}/{len(pending)}; latest {sym}: {sector or status}',flush=True)
        if sleep_s>0: time.sleep(sleep_s)


def report(db, rows, top_n, date_filter):
    with db.begin() as s: s.execute(SQL_CREATE)
    with db() as s:
        lut={r['symbol']:dict(r) for r in s.execute(text('SELECT symbol,sector,industry,status FROM us_tp_sector_lookup')).mappings()}
    stock_groups=collections.defaultdict(set)
    for r in rows:
        if date_filter and r['day']!=date_filter: continue
        stock_groups[(r['day'],r['status'])].add(r['symbol'])
    print('DATE | SIGNAL | STOCKS | CLASSIFIED | SECTORS | TOP_SECTORS | UNKNOWN',flush=True)
    for (day,status),symbols in sorted(stock_groups.items(),reverse=True):
        counts=collections.Counter()
        unknown=0
        for sym in symbols:
            sector=lut.get(sym,{}).get('sector')
            if sector: counts[sector]+=1
            else: unknown+=1
        ranked=', '.join(f'{sector}: {count} ({count/len(symbols)*100:.1f}%)' for sector,count in counts.most_common(top_n))
        print(f'{day} | {status} | {len(symbols)} | {len(symbols)-unknown} | {len(counts)} | {ranked or "—"} | {unknown}')
    print('NOTE: sectors are CURRENT classifications; unknown companies remain UNKNOWN, not inferred.')
    print('NOTE: 100% of a group is not 100% of the market. A sector-weighted baseline is required for market breadth claims.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['fetch','report','both'],default='both')
    p.add_argument('--min-stocks',type=int,default=5)
    p.add_argument('--limit',type=int,default=100,help='Max symbols to fetch in this invocation; 0=all')
    p.add_argument('--sleep',type=float,default=0.8,help='Delay between requests in seconds')
    p.add_argument('--refresh',action='store_true',help='Refresh all cached symbols')
    p.add_argument('--retry-failed',action='store_true',help='Retry cached ERROR or UNCLASSIFIED symbols')
    p.add_argument('--date',help='Limit report to YYYY-MM-DD')
    p.add_argument('--top',type=int,default=5)
    a=p.parse_args()
    if a.min_stocks<2 or a.limit<0 or a.sleep<0: p.error('invalid parameters')
    db=database()
    rows,ticker_map=get_universe(db,a.min_stocks)
    print('Qualified signal rows:',len(rows),'Unique symbols:',len({r['symbol'] for r in rows}),flush=True)
    if a.mode in ('fetch','both'): fetch(db,rows,ticker_map,a.limit,a.sleep,a.refresh,a.retry_failed)
    if a.mode in ('report','both'): report(db,rows,a.top,a.date)

if __name__=='__main__':main()
