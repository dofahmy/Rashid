#!/usr/bin/env python3
"""Read-only comparison of user-selected S&P 500 turning-point examples vs other LOW signals.

Inputs from existing Railway project: core.database(), us_tp_sector_lookup,
us_tp_daily_candidates, us_tp_strategy_60pct.get_bars().

IMPORTANT: This is descriptive/retrospective research, NOT a predictive model.
OHLC features reference signal day or past ONLY; past breadth looks only backwards.
Stock-sector cache contains CURRENT sector classifications, not historical snapshots.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from us_tp_bottom_to_peak import events, BOTTOM_STATUSES
from us_tp_strategy_60pct import get_bars

# Representative dates of the five hand-marked episodes (not independent of history).
EXAMPLES = ['2022-10-13', '2023-10-26', '2024-08-05', '2025-04-08', '2026-03-30']
ALT_EXAMPLES = {'2025-04-04': '2025-04-08'}

def pct(a,b):
    return 100 * (a / b - 1) if b else None

def rnd(v):
    return round(v,4) if v is not None else ''

def features(bars, idx, event, event_history, days_ahead):
    row = bars[idx]
    if idx < 60:
        return None
    c = row['close']; hist5=bars[idx-4:idx+1]; hist20=bars[idx-19:idx+1]; hist60=bars[idx-59:idx+1]
    top20=max(x['high'] for x in hist20)
    low20=min(x['low'] for x in hist20)
    top60=max(x['high'] for x in hist60)
    low60=min(x['low'] for x in hist60)
    rnge=row['high']-row['low']
    prior20=pct(c,bars[idx-20]['close'])
    obs=[x for x in event_history if x['day'] <= event['day']]
    recent5=[x for x in obs if idx-5 <= x['bar_index'] < idx]
    recent20=[x for x in obs if idx-20 <= x['bar_index'] < idx]
    stocks=event['stocks']; sectors=event['sectors']
    prev5avg=mean([x['stocks'] for x in recent5]) if recent5 else 0
    r = dict(date=event['day'],stocks=stocks,sectors=sectors,
       prior3_pct=rnd(pct(c,bars[idx-3]['close'])),
       prior5_pct=rnd(pct(c,bars[idx-5]['close'])),
       prior10_pct=rnd(pct(c,bars[idx-10]['close'])),
       prior20_pct=rnd(prior20),
       from_high20_pct=rnd(pct(c,top20)),
       from_high60_pct=rnd(pct(c,top60)),
       position20_pct=rnd(100*(c-low20)/(top20-low20) if top20>low20 else None),
       position60_pct=rnd(100*(c-low60)/(top60-low60) if top60>low60 else None),
       daily_close_position_pct=rnd(100*(c-row['low'])/rnge if rnge>0 else None),
       upper_wick_pct_range=rnd(100*(row['high']-max(c,row['open']))/rnge if rnge>0 else None),
       lower_wick_pct_range=rnd(100*(min(c,row['open'])-row['low'])/rnge if rnge>0 else None),
       day_body_pct_range=rnd(100*(c-row['open'])/rnge if rnge>0 else None),
       prior5_event_days=len(recent5),
       prior20_event_days=len(recent20),
       prior5_max_stocks=max((x['stocks'] for x in recent5),default=0),
       stocks_over_prior5_avg=rnd(stocks/prev5avg) if prev5avg else '',
       chosen_example=int(event['day'] in EXAMPLES),
       episode='selected' if event['day'] in EXAMPLES else 'other',
       full_forward20=int(idx+20<len(bars)),
    )
    future=bars[idx+1:idx+days_ahead+1]
    r['observed_future_sessions']=len(future)
    if len(future)==days_ahead:
        r['future_max_high_pct']=rnd(pct(max(x['high'] for x in future),c))
        r['future_max_low_pct']=rnd(pct(min(x['low'] for x in future),c))
        r['future_final_close_pct']=rnd(pct(future[-1]['close'],c))
        r['hit_plus3_in20']=int(max(x['high'] for x in future)>=c*1.03) if days_ahead==20 else ''
        r['fell_minus5_in20']=int(min(x['low'] for x in future)<=c*.95) if days_ahead==20 else ''
    else:
        r.update(future_max_high_pct='',future_max_low_pct='',future_final_close_pct='',hit_plus3_in20='',fell_minus5_in20='')
    return r

def save(path, data, fields=None):
    if not fields:
        fields=list(data[0]) if data else ['date']
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)

def summary(rows, columns, label):
    out=[]
    for group in ('selected','other'):
        g=[r for r in rows if r['episode']==group]
        d={'class':label,'group':group,'n':len(g)}
        for col in columns:
            values=[float(r[col]) for r in g if isinstance(r.get(col),(int,float)) and r[col]!='']
            d[col+'_mean']=rnd(mean(values)) if values else ''
            d[col+'_median']=rnd(median(values)) if values else ''
        out.append(d)
    return out

def run(args):
    evs=events('LOW',BOTTOM_STATUSES,args.min_stocks,args.min_sectors,args.group_min)
    evs=[{**e,'day':str(e['day'])[:10]} for e in evs if str(e['day'])[:10] not in args.exclude]
    bars=get_bars(args.ticker)
    bar_index={b['date']:i for i,b in enumerate(bars)}
    history=[dict(**e,bar_index=bar_index[e['day']]) for e in evs if e['day'] in bar_index]
    rows=[]; omitted=[]
    for e in history:
        r=features(bars,e['bar_index'],e,history,args.forward)
        if r: rows.append(r)
        else: omitted.append(e['day'])
    known={r['date'] for r in rows}
    absent=[d for d in EXAMPLES if d not in known]
    # IMPORTANT: Never silently substitute another date for a selected episode.
    if absent:
        print('WARNING SELECTED DATES NOT IN ELIGIBLE EVENTS:',absent)
    cols=['stocks','sectors','prior3_pct','prior5_pct','prior10_pct','prior20_pct',
          'from_high20_pct','from_high60_pct','position20_pct','position60_pct',
          'daily_close_position_pct','upper_wick_pct_range','lower_wick_pct_range',
          'day_body_pct_range','prior5_event_days','prior20_event_days',
          'prior5_max_stocks','stocks_over_prior5_avg']
    stats=summary(rows,cols,'all_eligible')
    # Apples-to-apples subset after decline >=3%, since all chosen examples are intended as rebound bottoms.
    decline=[r for r in rows if r['prior20_pct']!='' and r['prior20_pct']<=-3]
    stats+=summary(decline,cols,'only_prior20_down3plus')
    # Descriptive percentile ranks compared to other events; no supervised model fitting.
    for r in rows:
        if r['episode'] != 'selected':continue
        comparators=[x for x in decline if x['date']!=r['date']] if r['prior20_pct']!='' and r['prior20_pct']<=-3 else [x for x in rows if x['date']!=r['date']]
        for k in ['position20_pct','daily_close_position_pct','stocks','sectors','prior5_pct','from_high20_pct']:
            nums=[float(x[k]) for x in comparators if x[k]!='']
            r[k+'_percentile_vs_others']=rnd(100*sum(x<=float(r[k]) for x in nums)/len(nums)) if nums and r[k]!='' else ''
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    save(str(out)+'_all_events.csv',rows)
    save(str(out)+'_chosen_five.csv',[x for x in rows if x['episode']=='selected'])
    save(str(out)+'_comparison.csv',stats)
    print('INDEX:',args.ticker,'BARS:',len(bars),'DATES:',bars[0]['date'],'to',bars[-1]['date'])
    print('ELIGIBLE BOTTOM DATES:',len(evs),'USABLE:',len(rows),'OMITTED NO 60 PRIOR BARS:',omitted)
    print('CHOSEN:',sum(x['episode']=='selected' for x in rows),'OTHERS:',sum(x['episode']=='other' for x in rows))
    print('IMPORTANT: User-selected cases are handpicked by hindsight. Descriptive NOT statistically validated.')
    print('CHOSEN EXAMPLES:')
    for r in rows:
        if r['episode']=='selected':
            print({k:r[k] for k in ['date','stocks','sectors','prior5_pct','prior20_pct','from_high20_pct','position20_pct','daily_close_position_pct','prior5_event_days','future_max_high_pct','future_max_low_pct']})
    print('COMPARISON:')
    for r in stats:print(r)
    print('FILES:',str(out)+'_all_events.csv',str(out)+'_chosen_five.csv',str(out)+'_comparison.csv')
    print('READ ONLY; no writes, trades, or alerts. All indicator features use current/past OHLC only.')
    print('CAVEATS: Pivot look-ahead must be audited, sector tags are current, chosen examples have hindsight bias, signals overlap.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/rajih_bottom_fingerprint')
    p.add_argument('--ticker',default='^GSPC')
    p.add_argument('--min-stocks',type=int,default=5)
    p.add_argument('--min-sectors',type=int,default=2)
    p.add_argument('--group-min',type=int,default=5)
    p.add_argument('--forward',type=int,default=20,choices=[20],help='Future outcome diagnostics: 20 complete sessions; NOT model features')
    p.add_argument('--exclude',action='append',default=['2024-12-18'])
    x=p.parse_args()
    run(x)
