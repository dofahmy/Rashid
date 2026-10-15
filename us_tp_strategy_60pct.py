#!/usr/bin/env python3
"""Read-only two-branch S&P 500 signal backtest, live DB dates minus 2024-12-18.

RISING (prior 20-session close return > 0): place 3%-below-signal-close
buy limit starting next session; target original signal close.
FALLING (prior return <= -3%): buy next session open and target
entry * (1 + 0.6*abs(prior20 return)). This applies the 60%-of-drop
percentage to executable next-session entry rather than the signal close.
OTHER: ignored. Uses daily candles; conservatively does not permit a limit
entry and profit exit on the same daily candle because intraday order is unknown.
All trades still open at the end of the chosen horizon are marked to
market at the final available close (not forced exited in reality).
"""
import argparse
import csv
import json
import statistics
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.request import urlopen, Request
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError
import time


EXCLUDED = '2024-12-18'


def get_bars(ticker):
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/'+quote(ticker, safe='')+'?'+urlencode({'interval':'1d','range':'10y','includePrePost':'false'})
    last_error=None
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'}), timeout=35) as f:
                payload=json.load(f)
            result=(payload.get('chart',{}).get('result') or [None])[0]
            if not result: raise ValueError(str(payload.get('chart',{}).get('error')))
            q=result['indicators']['quote'][0]
            bars=[]
            for k,ts in enumerate(result.get('timestamp',[])):
                o,h,l,c=(q[x][k] for x in ('open','high','low','close'))
                if any(x is None or x<=0 for x in (o,h,l,c)): continue
                d=datetime.fromtimestamp(ts,timezone.utc).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
                bars.append(dict(date=d,open=float(o),high=float(h),low=float(l),close=float(c)))
            bars=list({b['date']:b for b in bars}.values())
            return sorted(bars,key=lambda x:x['date'])
        except (HTTPError,URLError,TimeoutError,ValueError,KeyError) as e:
            last_error=e; time.sleep(2*(attempt+1))
    raise RuntimeError(f'Could not fetch {ticker}: {last_error}')


def trial(bars,index,event,horizon):
    base=bars[index]['close']
    prev=bars[index-20]['close']
    prior=100*(base/prev-1)
    if prior>0:
        branch='RISE_LIMIT_MINUS3'; buy_level=base*.97; target=base
    elif prior<=-3:
        branch='FALL_GE3_60PCT_RECOVERY'; buy_level=None; target=None
    else:
        branch='SKIP'; buy_level=None; target=None
    r=dict(date=event['day'],stocks=event['stocks'],sectors=event['sectors'],
           prior20_pct=round(prior,4),branch=branch,signal_close=round(base,4),
           horizon=horizon,limit_price=round(buy_level,4) if buy_level else '',
           entry_date='',entry_session='',entry_price='',target_price='',
           exit_date='',exit_session='',exit_price='',status='SKIPPED',
           return_pct='',max_drawdown_pct_after_entry='',mark_to_market_pct='',
           complete_horizon=int(len(bars)>index+horizon),
           notes='')
    if branch=='SKIP':
        r['notes']='prior20 between -3% and 0% inclusive'
        return r
    if index+1>=len(bars):
        r['status']='NO_FUTURE_BARS';return r
    future=bars[index+1:index+horizon+1]
    if len(future)<horizon:
        r['status']='INCOMPLETE'  # analysis horizon not fully elapsed; can still report a realized exit
    buyj=None;entry=None
    for j,b in enumerate(future,1):
        if branch=='RISE_LIMIT_MINUS3':
            # A buy limit at 97% fills at opening price if market gaps below limit;
            # otherwise at the limit if the daily low touches it.
            if buyj is None and b['low']<=buy_level:
                buyj=j;entry=min(b['open'],buy_level)
        else:
            if j==1:
                buyj=1;entry=b['open'];target=entry*(1+abs(prior)*.006)
        if buyj is not None: break
    if buyj is None:
        r['status']='NOT_FILLED' if len(future)==horizon else 'INCOMPLETE_NOT_FILLED'
        return r
    r.update(entry_date=future[buyj-1]['date'],entry_session=buyj,
             entry_price=round(entry,4),target_price=round(target,4))
    # Conservative daily-bar rule: only exits from the NEXT candle onward; entry-day target may already be hit but order unknown.
    exitj=None
    for j in range(buyj+1,len(future)+1):
        b=future[j-1]
        if b['high']>=target:
            exitj=j
            exitprice=max(target,b['open'])  # take-profit marketable on gap up, otherwise limit target
            r.update(exit_date=b['date'],exit_session=j,exit_price=round(exitprice,4),
                     status='TARGET_HIT',return_pct=round((exitprice/entry-1)*100,4))
            break
    upto=future[buyj-1:exitj] if exitj else future[buyj-1:]
    if upto:
        r['max_drawdown_pct_after_entry']=round(100*(min(x['low'] for x in upto)/entry-1),4)
        r['mark_to_market_pct']=round(100*(upto[-1]['close']/entry-1),4)
    if exitj is None:
        r['status']='OPEN_AT_HORIZON' if len(future)==horizon else 'CENSORED_OPEN'
        r['notes']='No time-stop assumed; mark-to-market only. Entry-day exit not counted.'
    return r


def write(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summary(rows):
    result=[]
    for h in sorted(set(r['horizon'] for r in rows)):
        for branch in ['RISE_LIMIT_MINUS3','FALL_GE3_60PCT_RECOVERY','SKIP']:
            rr=[r for r in rows if r['horizon']==h and r['branch']==branch]
            tradable=[r for r in rr if r['status'] not in ('SKIPPED','NOT_FILLED','INCOMPLETE_NOT_FILLED','INCOMPLETE','NO_FUTURE_BARS')]
            wins=[r for r in tradable if r['status']=='TARGET_HIT']
            completed=[r for r in rr if r['complete_horizon']]
            closedreturns=[r['return_pct'] for r in wins]
            result.append(dict(horizon=h,branch=branch,signals=len(rr),complete_horizon=len(completed),
                filled=len(tradable),target_hits=len(wins),not_filled=sum(r['status']=='NOT_FILLED' for r in rr),
                still_open=sum(r['status'] in ('OPEN_AT_HORIZON','CENSORED_OPEN') for r in rr),
                target_hit_pct_of_filled=round(100*len(wins)/len(tradable),2) if tradable else None,
                median_entry_session=statistics.median(r['entry_session'] for r in tradable) if tradable else None,
                median_exit_session=statistics.median(r['exit_session'] for r in wins) if wins else None,
                avg_hit_return_pct=round(statistics.mean(closedreturns),3) if closedreturns else None,
                worst_entry_drawdown_pct=min((r['max_drawdown_pct_after_entry'] for r in tradable if r['max_drawdown_pct_after_entry']!=''),default=None)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/sp500_two_way')
    p.add_argument('--ticker',default='^GSPC')
    p.add_argument('--horizons',type=int,nargs='+',default=[60])
    p.add_argument('--min-stocks',type=int,default=5,help='Minimum distinct symbols in one day, default 5')
    p.add_argument('--min-sectors',type=int,default=2,help='Minimum distinct sectors in one day, default 2')
    p.add_argument('--compare-old',action='store_true',help='Also report the original 9 stocks / 3 sectors filter on the same database snapshot')
    a=p.parse_args()
    if a.min_stocks < 1 or a.min_sectors < 1: p.error('Minimums must be positive')
    ev=[e for e in load_dates_live_threshold(a.min_stocks,a.min_sectors) if e['day']!=EXCLUDED]
    bars=get_bars(a.ticker)
    index={b['date']:i for i,b in enumerate(bars)}
    rows=[];missing=[]
    for event in ev:
        i=index.get(event['day'])
        if i is None or i<20:
            missing.append(event['day']);continue
        for h in a.horizons:
            rows.append(trial(bars,i,event,h))
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    write(str(path)+'_trades.csv',rows)
    agg=summary(rows);write(str(path)+'_summary.csv',agg)
    print('NEW FILTER: distinct stocks >=',a.min_stocks,'distinct sectors >=',a.min_sectors,'; SAME_ANGLE_DIFF_ANCHORS / LOW / valid statuses; original 5-stock subgroup requirement preserved')
    print('SIGNALS:',len(ev),'EXCLUDED:',EXCLUDED,'MISSING:',missing)
    if a.compare_old:
        older=[e for e in load_dates_live_threshold(9,3) if e['day']!=EXCLUDED]
        older_rows=[trial(bars,index[e['day']],e,h) for e in older for h in a.horizons if e['day'] in index and index[e['day']]>=20]
        write(str(path)+'_old_9x3_trades.csv',older_rows)
        oldagg=summary(older_rows);write(str(path)+'_old_9x3_summary.csv',oldagg)
        print('OLD FILTER SIGNALS:',len(older))
        for old in oldagg: print('OLD SUMMARY',old)
    print('INDEX BARS:',len(bars),bars[0]['date'],'to',bars[-1]['date'],a.ticker)
    print('TRADING RULES: prior20>0 => -3% buy limit, exit at signal close; prior20<=-3 => next session OPEN, target +60% of abs(prior20%); others SKIP.')
    print('EXIT ENTRY CANDLE: NOT counted. No stoploss or forced liquidation.')
    for r in agg:print('SUMMARY',r)
    print('FALLING >=3% SIGNALS (60 sessions):')
    for r in rows:
        if r['horizon']==60 and r['branch']=='FALL_GE3_60PCT_RECOVERY':
            print(r['date'],'stocks',r['stocks'],'prior20',r['prior20_pct'],'entry',r['entry_price'],'target',r['target_price'],'status',r['status'],'exit session',r['exit_session'],'worst drawdown',r['max_drawdown_pct_after_entry'])
    print('FILES:',str(path)+'_trades.csv',str(path)+'_summary.csv')
    print('READ ONLY; daily index proxy is not tradable; no fees, slippage or financing. Overlapping signal dates are correlated.')
    print('WARNING: historical signal pivot confirmation and current sector labels may introduce look-ahead/survivorship bias.')

def load_dates_live_threshold(min_stocks=5, min_sectors=2):
    """Daily market breadth built from original eligible candidate groups.

    Requires >=5 distinct stocks inside each (day, signal_type, anchor_type,
    status) group, then combines eligible groups into unique day-symbol rows;
    sector classification is one row per symbol. Does not modify the DB.
    """
    from core import database
    from sqlalchemy import text
    sql = text("""
    WITH qualified_groups AS (
      SELECT day, signal_type, anchor_type, status
      FROM us_tp_daily_candidates
      WHERE signal_type='SAME_ANGLE_DIFF_ANCHORS'
        AND anchor_type='LOW'
        AND status IN ('CROSS_DOWN_CLOSE','TOUCH_ABOVE')
      GROUP BY day, signal_type, anchor_type, status
      HAVING COUNT(DISTINCT symbol) >= 5
    ), unique_signals AS (
      SELECT DISTINCT c.day, c.symbol
      FROM us_tp_daily_candidates c
      JOIN qualified_groups g ON
        (c.day,c.signal_type,c.anchor_type,c.status) =
        (g.day,g.signal_type,g.anchor_type,g.status)
    ), daily AS (
      SELECT u.day,
             COUNT(DISTINCT u.symbol) AS stocks,
             COUNT(DISTINCT NULLIF(TRIM(s.sector),'')) AS sectors
      FROM unique_signals u
      LEFT JOIN us_tp_sector_lookup s ON s.symbol=u.symbol
      GROUP BY u.day
    )
    SELECT day, stocks, sectors
    FROM daily
    WHERE stocks >= :min_stocks AND sectors >= :min_sectors
    ORDER BY day
    """)
    maker = database()
    with maker() as session:
        rows = session.execute(sql, {'min_stocks':min_stocks,'min_sectors':min_sectors}).mappings().all()
    return [dict(day=str(r['day'])[:10], stocks=int(r['stocks']), sectors=int(r['sectors'])) for r in rows]


if __name__=='__main__':main()
