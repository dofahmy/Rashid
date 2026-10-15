#!/usr/bin/env python3
"""Read-only test: S&P 500 responses after qualified multi-sector US Time-Price dates.

Default is the 86-date snapshot provided on 2026-10-15. --live recomputes
qualified dates from the current database (which may have more symbols now).
No database writes, recommendations, or notifications.
"""
import argparse
import csv
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SNAPSHOT = '''
2026-10-14 10 7
2026-10-01 9 3
2026-09-14 11 5
2026-05-15 10 6
2026-03-20 10 5
2026-03-19 9 5
2026-03-13 9 6
2026-03-09 17 8
2026-03-06 13 6
2026-03-03 9 5
2026-02-23 14 6
2026-02-12 10 6
2026-02-05 13 5
2026-02-04 21 7
2026-02-03 12 6
2026-01-29 11 5
2025-11-21 9 5
2025-11-17 13 6
2025-11-14 11 4
2025-11-07 17 8
2025-11-04 17 7
2025-10-30 14 6
2025-10-22 9 6
2025-10-16 16 4
2025-10-10 11 5
2025-08-01 12 6
2025-07-31 12 7
2025-07-30 9 4
2025-04-09 11 5
2025-04-08 10 6
2025-04-07 48 11
2025-04-04 82 11
2025-04-03 26 7
2025-03-31 12 5
2025-03-28 13 6
2025-03-13 13 7
2025-03-11 11 7
2025-03-10 19 7
2025-03-04 29 9
2025-03-03 13 6
2025-02-28 11 7
2025-02-25 10 5
2025-02-24 12 6
2025-02-21 15 7
2025-02-20 9 4
2025-02-03 9 6
2025-01-27 13 6
2025-01-13 20 5
2025-01-10 19 9
2024-12-19 12 7
2024-12-18 42 10
2024-12-17 12 6
2024-12-13 16 9
2024-11-15 11 4
2024-11-06 9 4
2024-10-31 14 6
2024-09-10 12 8
2024-09-09 9 5
2024-09-05 10 7
2024-08-05 56 9
2024-08-02 18 9
2024-08-01 13 8
2024-04-12 10 7
2024-02-13 14 8
2024-02-05 9 8
2024-01-17 9 5
2023-10-20 10 6
2023-10-19 12 8
2023-10-03 10 4
2023-10-02 11 7
2023-09-27 11 6
2023-09-21 13 6
2023-09-07 12 5
2023-08-17 11 6
2023-07-06 9 5
2023-03-15 13 7
2023-03-13 18 8
2023-03-10 18 7
2022-10-13 10 5
2022-09-26 10 7
2022-09-23 22 10
2022-09-22 11 7
2022-09-20 9 6
2022-09-19 10 5
2022-09-01 12 8
2022-08-29 9 6
'''

LIVE_SQL = '''
WITH groups AS (
 SELECT day, signal_type, anchor_type, status
 FROM us_tp_daily_candidates
 WHERE signal_type='SAME_ANGLE_DIFF_ANCHORS' AND anchor_type='LOW'
   AND status IN ('CROSS_DOWN_CLOSE','TOUCH_ABOVE')
 GROUP BY day,signal_type,anchor_type,status
 HAVING COUNT(DISTINCT symbol)>=5
), signals AS (
 SELECT DISTINCT c.day,c.symbol
 FROM us_tp_daily_candidates c JOIN groups g
 ON (c.day,c.signal_type,c.anchor_type,c.status)=(g.day,g.signal_type,g.anchor_type,g.status)
), daily AS (
 SELECT signals.day, COUNT(DISTINCT signals.symbol) stocks,
        COUNT(DISTINCT NULLIF(TRIM(l.sector),'')) sectors
 FROM signals LEFT JOIN us_tp_sector_lookup l ON l.symbol=signals.symbol
 GROUP BY signals.day
)
SELECT day,stocks,sectors FROM daily WHERE stocks>=9 AND sectors>=3 ORDER BY day
'''


def load_dates(live):
    if not live:
        out = []
        for line in SNAPSHOT.splitlines():
            if line.strip():
                d, n, s = line.split()
                out.append(dict(day=d, stocks=int(n), sectors=int(s)))
        assert len(out) == 86, f'Expected 86 dates; got {len(out)}'
        return sorted(out, key=lambda x: x['day'])
    from sqlalchemy import text
    from core import database
    DB = database()
    with DB() as session:
        return [dict(r) for r in session.execute(text(LIVE_SQL)).mappings()]


def fetch_index(ticker):
    # Public Yahoo chart data; end-of-day OHLC of the index itself, not a tradable fill.
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + quote(ticker, safe='')
    url += '?' + urlencode({'interval':'1d','range':'10y','includePrePost':'false'})
    err = None
    for k in range(4):
        try:
            req = Request(url, headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'})
            with urlopen(req,timeout=30) as f:
                data = json.load(f)
            result = (data.get('chart',{}).get('result') or [None])[0]
            if not result:
                raise ValueError('Yahoo returned no index data: '+str(data.get('chart',{}).get('error')))
            q = result['indicators']['quote'][0]
            bars = []
            for i, ts in enumerate(result.get('timestamp',[])):
                c = q['close'][i]
                h = q['high'][i]
                l = q['low'][i]
                if any(v is None for v in (c,h,l)) or c<=0:
                    continue
                # NY exchange-day date using exchange timezone, not UTC midnight.
                from zoneinfo import ZoneInfo
                d = datetime.fromtimestamp(ts, timezone.utc).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
                bars.append((d,float(c),float(h),float(l)))
            return sorted({r[0]:r for r in bars}.values())
        except (HTTPError,URLError,TimeoutError,ValueError,KeyError) as exc:
            err = exc
            time.sleep(2*(k+1))
    raise RuntimeError(f'Yahoo index fetch failed: {err}. Try running again later.')


def summarize(rows, horizons):
    ans = []
    for grp in sorted(set(x['group'] for x in rows)):
        for h in horizons:
            ar = [x for x in rows if x['group']==grp and x['horizon']==h and x['complete']]
            if not ar: continue
            ret = [x['return_pct'] for x in ar]
            ans.append(dict(group=grp,horizon=h,n=len(ar),average_pct=round(statistics.mean(ret),4),
                            median_pct=round(statistics.median(ret),4),positive_pct=round(100*sum(v>0 for v in ret)/len(ret),2),
                            avg_max_up_pct=round(statistics.mean(x['max_up_pct'] for x in ar),4),
                            avg_max_down_pct=round(statistics.mean(x['max_down_pct'] for x in ar),4)))
    return ans


def write_csv(path, rows, cols):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/us_tp_sp500')
    p.add_argument('--ticker',default='^GSPC',help='S&P 500 index symbol, default ^GSPC')
    p.add_argument('--live',action='store_true',help='Rebuild date list from current DB instead of frozen 86-date snapshot')
    a=p.parse_args()
    events=load_dates(a.live)
    dates={r['day']:r for r in events}
    print(f'SIGNAL DATES: {len(dates)}; source={"live DB" if a.live else "86-date frozen snapshot"}',flush=True)
    bars=fetch_index(a.ticker)
    pos={b[0]:i for i,b in enumerate(bars)}
    print(f'INDEX BARS: {len(bars)} {bars[0][0]}..{bars[-1][0]} ({a.ticker})',flush=True)
    horizons=[5,10,20]
    starts=[d for d in dates if d in pos]
    missing=sorted(set(dates)-set(starts))
    if missing: print('WARNING: dates without matching index bar:',missing)
    first,last=min(starts),max(starts)
    # Compare against ordinary trading dates within same historical signal span;
    # exclude ALL signal days, regardless of signal maturity.
    rows=[]
    for i,(date,c,hi,lo) in enumerate(bars):
        if date<first or date>last: continue
        event=dates.get(date)
        for h in horizons:
            future=bars[i+1:i+h+1]
            complete=len(future)==h
            if not future:
                continue
            rows.append(dict(day=date,group=('signal' if event else 'ordinary_day'),
                stocks=(event['stocks'] if event else ''),sectors=(event['sectors'] if event else ''),
                horizon=h,complete=int(complete),base_close=round(c,5),end_day=future[-1][0],
                return_pct=round((future[-1][1]/c-1)*100,5) if complete else '',
                max_up_pct=round((max(x[2] for x in future)/c-1)*100,5),
                max_down_pct=round((min(x[3] for x in future)/c-1)*100,5)))
    # Explicitly don't interpret incomplete horizons as failures or returns.
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    eventrows=[r for r in rows if r['group']=='signal']
    summary=summarize(rows,horizons)
    complete_dates=sorted(set(r['day'] for r in eventrows if r['horizon']==20 and r['complete']))
    pair_rows=[]
    for r in eventrows:
        if not r['complete']: continue
        r2=dict(r)
        r2['breadth_band']='9-14' if int(r['stocks'])<15 else ('15-29' if int(r['stocks'])<30 else '30+')
        r2['sector_band']='3-4' if int(r['sectors'])<5 else ('5-6' if int(r['sectors'])<7 else '7+')
        pair_rows.append(r2)
    cols=['day','group','stocks','sectors','horizon','complete','base_close','end_day','return_pct','max_up_pct','max_down_pct']
    write_csv(Path(str(out)+'_dates.csv'),eventrows,cols)
    write_csv(Path(str(out)+'_summary.csv'),summary,['group','horizon','n','average_pct','median_pct','positive_pct','avg_max_up_pct','avg_max_down_pct'])
    write_csv(Path(str(out)+'_bands.csv',str(out)+'_band_summary.csv'),pair_rows,cols+['breadth_band','sector_band'])
    band_summary=[]
    for field in ('breadth_band','sector_band'):
        for band in sorted({r[field] for r in pair_rows}):
            for h in horizons:
                ar=[r for r in pair_rows if r[field]==band and r['horizon']==h]
                if not ar: continue
                ret=[r['return_pct'] for r in ar]
                band_summary.append(dict(classification=field,band=band,horizon=h,n=len(ar),
                    avg_return_pct=round(statistics.mean(ret),4),median_return_pct=round(statistics.median(ret),4),
                    positive_pct=round(100*sum(v>0 for v in ret)/len(ret),2),
                    avg_max_up_pct=round(statistics.mean(r['max_up_pct'] for r in ar),4),
                    avg_max_down_pct=round(statistics.mean(r['max_down_pct'] for r in ar),4)))
    write_csv(Path(str(out)+'_band_summary.csv'),band_summary,
        ['classification','band','horizon','n','avg_return_pct','median_return_pct','positive_pct','avg_max_up_pct','avg_max_down_pct'])
    print('BREADTH AND SECTOR SUBGROUPS:')
    for r in band_summary:print(r)
    print('SUMMARY (close-to-close after signal-date close):')
    for r in summary:print(r)
    print(f'COMPLETE 20-SESSION SIGNAL DATES: {len(complete_dates)} / {len(dates)}')
    print('FILES:',str(out)+'_dates.csv',str(out)+'_summary.csv',str(out)+'_bands.csv',str(out)+'_band_summary.csv')
    print('READ-ONLY. WARNING: clustered/overlapping dates are not independent; ordinary-day baseline is descriptive, not a causal control.')
    print('Classification snapshot reflects CURRENT sectors; survivorship and look-ahead in signal generation require separate audit.')

if __name__=='__main__':main()
