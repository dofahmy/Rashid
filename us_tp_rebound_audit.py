#!/usr/bin/env python3
"""READ ONLY. Analyze pre/post S&P 500 (^GSPC) action around frozen 86 US Time-Price breadth dates.
Dependencies: Python standard library + us_tp_sp500_market_test.py bundled alongside.
Outputs per-day audit plus 20-session forward and 20-session non-overlap exploratory summaries.
"""
import argparse, csv, statistics
from pathlib import Path
from us_tp_sp500_market_test import load_dates, fetch_index, write_csv

HORIZONS=(5,10,20)

def percent(end,start):
    return round(100*(end/start-1),4) if start and end is not None else None

def avg(items):
    clean=[v for v in items if v is not None]
    return round(statistics.mean(clean),4) if clean else None

def median(items):
    clean=[v for v in items if v is not None]
    return round(statistics.median(clean),4) if clean else None

def label(row):
    # Descriptive only. NOT a predictive or validated market-regime label.
    if row['pre20_pct'] is None or row['dd_from_high20_pct'] is None: return 'insufficient_history'
    if row['dd_from_high20_pct']<=-5 and row['pre5_pct'] is not None and row['pre5_pct']>0:
        return 'rebounding_below_20d_high'
    if row['dd_from_high20_pct']<=-5 and row['pre5_pct'] is not None and row['pre5_pct']<=0:
        return 'falling_below_20d_high'
    if row['pre20_pct']<0: return 'mild_or_mixed_decline'
    return 'rising_or_range'

def compute(bars,events):
    ix={b[0]:i for i,b in enumerate(bars)}
    rows=[]; missing=[]
    for e in sorted(events,key=lambda r:r['day']):
        d=e['day']
        if d not in ix: missing.append(d); continue
        i=ix[d]; close=bars[i][1]
        r=dict(date=d,stocks=e['stocks'],sectors=e['sectors'],index_close=round(close,5),
               breadth_band=('30+' if e['stocks']>=30 else ('15-29' if e['stocks']>=15 else '9-14')),
               sector_band=('7+' if e['sectors']>=7 else ('5-6' if e['sectors']>=5 else '3-4')))
        for w in HORIZONS:
            r[f'pre{w}_pct']=percent(close,bars[i-w][1]) if i>=w else None
        for w in (20,60):
            past=bars[max(0,i-w+1):i+1]
            if len(past)<w: r[f'dd_from_high{w}_pct']=None;continue
            highest=max(x[2] for x in past)
            r[f'dd_from_high{w}_pct']=percent(close,highest)
        r['pre_regime']=label(r)
        for h in HORIZONS:
            future=bars[i+1:i+h+1]
            r[f'complete{h}']=len(future)==h
            r[f'forward{h}_pct']=percent(future[-1][1],close) if len(future)==h else None
            r[f'max_up{h}_pct']=percent(max(v[2] for v in future),close) if len(future)==h else None
            r[f'max_down{h}_pct']=percent(min(v[3] for v in future),close) if len(future)==h else None
        future20=bars[i+1:i+21]
        r['future20_complete']=len(future20)==20
        # Do not treat partial latest windows as final outcomes.
        valid=future20 if len(future20)==20 else []
        if valid:
            low_idx=min(range(len(valid)),key=lambda j:valid[j][3])
            high_idx=max(range(len(valid)),key=lambda j:valid[j][2])
            r['trough_date']=valid[low_idx][0];r['trough_sessions']=low_idx+1
            r['trough_pct']=percent(valid[low_idx][3],close)
            r['peak_date']=valid[high_idx][0];r['peak_sessions']=high_idx+1
            r['peak_pct']=percent(valid[high_idx][2],close)
            # Recovery above reference close, AFTER the future trough using future daily closing prices.
            rec=next((k for k in range(low_idx+1,len(valid)) if valid[k][1]>=close),None)
            r['recover_close_date']=valid[rec][0] if rec is not None else ''
            r['recover_close_sessions']=rec+1 if rec is not None else None
            r['recovered_after_trough_20']=int(rec is not None)
            # If future close after trough >= trough-low + 3%: indicative rebound, NOT tradable TP.
            rebound=max((v[2]/valid[low_idx][3]-1)*100 for v in valid[low_idx:])
            r['rebound_from_trough_max_pct']=round(rebound,4)
        else:
            for k in ('trough_date','trough_sessions','trough_pct','peak_date','peak_sessions','peak_pct',
                      'recover_close_date','recover_close_sessions','recovered_after_trough_20','rebound_from_trough_max_pct'):
                r[k]=None
        rows.append(r)
    return rows,missing,ix

def describe(rows,name):
    print(f'\n{name}: {len(rows)} signal dates')
    for h in HORIZONS:
        valid=[r for r in rows if r[f'complete{h}']]
        if not valid: continue
        arr=[r[f'forward{h}_pct'] for r in valid]
        print(f'  {h} sessions: n={len(valid)} avg={avg(arr):+.3f}% median={median(arr):+.3f}% '
              f'positive={sum(x>0 for x in arr)}/{len(arr)} '
              f'avg_max_up={avg([r[f"max_up{h}_pct"] for r in valid]):+.3f}% '
              f'avg_max_down={avg([r[f"max_down{h}_pct"] for r in valid]):+.3f}%')
    valid=[r for r in rows if r['future20_complete']]
    if valid:
        print(f'  prior20 avg={avg([r["pre20_pct"] for r in valid]):+.3f}% '
              f'avg drawdown-from-20d-high={avg([r["dd_from_high20_pct"] for r in valid]):+.3f}%')
        print(f'  future trough: mean session={avg([r["trough_sessions"] for r in valid])} '
              f'median session={median([r["trough_sessions"] for r in valid])}; '
              f'close recovered after future trough={sum(r["recovered_after_trough_20"] for r in valid)}/{len(valid)}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/sp500_rebound_audit')
    p.add_argument('--ticker',default='^GSPC')
    p.add_argument('--live',action='store_true',help='Use latest DB dates instead of fixed 86 dates')
    a=p.parse_args()
    events=load_dates(a.live)
    print('LOADING INDEX:',a.ticker,'EVENTS:',len(events),'MODE:','LIVE' if a.live else 'FROZEN 86',flush=True)
    bars=fetch_index(a.ticker)
    print('INDEX DATA:',len(bars),bars[0][0],bars[-1][0],flush=True)
    rows,missing,positions=compute(bars,events)
    if missing:print('WARNING missing signal days:',missing)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    write_csv(Path(str(out)+'_dates.csv'),rows,list(rows[0]) if rows else ['date'])
    print('\nFOUR EXTREME BREADTH DATES (30+ distinct stocks):')
    for r in rows:
        if r['stocks']>=30:
            print(f"{r['date']} stocks={r['stocks']} sectors={r['sectors']} prior20={r['pre20_pct']}% "
                  f"from20high={r['dd_from_high20_pct']}% regime={r['pre_regime']} "
                  f"future20={r['forward20_pct']}% trough={r['trough_date']} "
                  f"trough_day={r['trough_sessions']} trough%={r['trough_pct']} "
                  f"peak={r['peak_date']} recovery_after_trough={r['recover_close_date']}")
    describe(rows,'ALL QUALIFIED SIGNALS')
    describe([r for r in rows if r['stocks']>=30],'EXTREME 30+ STOCKS')
    describe([r for r in rows if r['stocks']<30],'OTHER 9-29 STOCKS')
    for regime in sorted({r['pre_regime'] for r in rows}):
        describe([r for r in rows if r['pre_regime']==regime], 'PRIOR MARKET STATE: '+regime)
    # Non-overlapping first occurrence selection (not statistically independent market samples).
    selected=[];last_i=-10000
    for r in rows:
        i=positions[r['date']]
        if i-last_i>20:
            selected.append(r);last_i=i
    describe(selected,'NON-OVERLAPPING SIGNALS (at least 21 index sessions apart)')
    print('\nDATES BY REGIME:')
    for r in rows:
        if r['stocks']>=30:
            print(r['date'],r['pre_regime'],'20d change',r['forward20_pct'])
    print('FILES:',str(out)+'_dates.csv')
    print('READ ONLY. Same-day close is a reference, not an executable signal fill. Future OHLC trough/peak order within a day is unknown.')
    print('CAUTION: hindsight pivots, clustering, current sector classifications and regime selection can bias historical results.')

if __name__=='__main__':main()
