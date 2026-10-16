#!/usr/bin/env python3
"""Read-only backtest of saved fixed-LSTR-fingerprint signal CSV.

For each stock/date signal, baseline is SIGNAL DAY CLOSE. Starting the NEXT
trading session, +3% target is touched if daily HIGH >= signal_close*1.03.
Separate mark-to-market returns are computed at 10/20/30/60 subsequent trading
session closes. A target-first strategy is independently simulated for each
horizon, selling at exactly +3% when HIGH touches; otherwise valuing at the
horizon's closing price. No stop loss, fees, dividends, gap improvement, or
simultaneous-position restrictions. Overlapping signals are NOT independent.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from sqlalchemy import text
from core import database

HORIZONS = (10, 20, 30, 60)
SQL = text("""
SELECT b.day, b.h, b.l, b.c
FROM us_tp_daily_bars b
WHERE b.symbol=:symbol AND b.day>:day
ORDER BY b.day ASC LIMIT 60
""")

def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def rnd(x):
    return round(x,4) if x is not None else ''

def get_num(row, k):
    value=row.get(k, '')
    return float(value) if value not in (None,'') else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matches',default='/tmp/lstr_dev20_matches.csv',help='matches CSV from --all-history --deviation 20')
    p.add_argument('--out',default='/tmp/lstr_dev20_target3',help='file prefix for outputs')
    p.add_argument('--target',type=float,default=3.0,help='take profit percent, default 3')
    p.add_argument('--year',type=int,default=None,help='optional one signal year filter')
    args=p.parse_args()
    if args.target <= 0 or args.target > 100: p.error('--target must be >0 and <=100')
    if not Path(args.matches).exists(): p.error('CSV not found: '+args.matches+' ; rerun us_lstr_bottom_signature.py --all-history --deviation 20 --out /tmp/lstr_dev20')
    with open(args.matches,encoding='utf-8-sig',newline='') as f:
        signals=list(csv.DictReader(f))
    if args.year: signals=[r for r in signals if str(r['date'])[:4]==str(args.year)]
    rows=[]
    session=database()()
    try:
        for i,sig in enumerate(signals,1):
            date=str(sig['date'])[:10]
            sym=sig['symbol']
            base=float(sig['signal_close'])
            if base<=0: continue
            bars=list(session.execute(SQL,{'symbol':sym,'day':date}).mappings())
            observed=len(bars)
            target_price=base*(1+args.target/100)
            first_hit=next((j for j,b in enumerate(bars,1) if b['h'] is not None and float(b['h'])>=target_price),None)
            row={'date':date,'year':date[:4],'symbol':sym,'company':sig.get('company',''),
                 'signal_close':rnd(base), 'target_price':rnd(target_price),
                 'max_deviation_pct':sig.get('max_deviation_pct',''),
                 'observed_sessions':observed,
                 'first_target_session':first_hit if first_hit else '',
                 'first_target_date':str(bars[first_hit-1]['day'])[:10] if first_hit else '',
                 'target_hit_within_60':1 if first_hit else (0 if observed>=60 else ''),
                 'latest_available_day':str(bars[-1]['day'])[:10] if bars else ''}
            for horizon in HORIZONS:
                tag=str(horizon)
                completed=observed>=horizon
                hit=first_hit is not None and first_hit<=horizon
                row['complete_'+tag]=int(completed)
                # Hit is provable even before the horizon is complete.
                row['hit_'+tag]=1 if hit else (0 if completed else '')
                if completed:
                    sub=bars[:horizon]
                    row['day_'+tag]=str(sub[-1]['day'])[:10]
                    row['close_'+tag]=rnd(float(sub[-1]['c']))
                    row['return_'+tag+'_pct']=rnd((float(sub[-1]['c'])/base-1)*100)
                    row['max_rise_'+tag+'_pct']=rnd((max(float(b['h']) for b in sub)/base-1)*100)
                    row['max_fall_'+tag+'_pct']=rnd((min(float(b['l']) for b in sub)/base-1)*100)
                    # Target reached earlier -> realize exactly specified %.
                    # Else unrealized P/L at horizon (modelled forced close).
                    row['strategy_'+tag+'_pct']=rnd(args.target if hit else (float(sub[-1]['c'])/base-1)*100)
                    row['strategy_'+tag+'_status']='TARGET' if hit else 'HORIZON_CLOSE'
                else:
                    for key in ('day_','close_','return_','max_rise_','max_fall_','strategy_','strategy_'):
                        row[key+tag+('_pct' if key in ('return_','max_rise_','max_fall_','strategy_') else '')]=''
                    row['strategy_'+tag+'_status']='INCOMPLETE'
            rows.append(row)
            if i%200==0 or i==len(signals):
                print('PROCESSED:',i,'/',len(signals),flush=True)
    finally: session.close()
    prefix=Path(args.out);prefix.parent.mkdir(parents=True,exist_ok=True)
    columns=['date','year','symbol','company','signal_close','target_price','max_deviation_pct',
             'observed_sessions','first_target_session','first_target_date','target_hit_within_60','latest_available_day']
    for h in HORIZONS:
        t=str(h)
        columns += ['complete_'+t,'hit_'+t,'day_'+t,'close_'+t,'return_'+t+'_pct',
                    'max_rise_'+t+'_pct','max_fall_'+t+'_pct','strategy_'+t+'_pct','strategy_'+t+'_status']
    write_csv(str(prefix)+'_details.csv',rows,columns)
    summaries=[]
    for y in ['ALL']+sorted({r['year'] for r in rows}):
        group=rows if y=='ALL' else [r for r in rows if r['year']==y]
        for h in HORIZONS:
            t=str(h)
            complete=[r for r in group if r['complete_'+t]]
            target_hits=sum(r['hit_'+t]==1 for r in complete)
            # Known hits from incomplete horizons are reported separately,
            # but not mixed into complete-horizon success rates.
            hit_early=sum(r['hit_'+t]==1 for r in group if not r['complete_'+t])
            endrets=[get_num(r,'return_'+t+'_pct') for r in complete]
            strat=[get_num(r,'strategy_'+t+'_pct') for r in complete]
            summary={
                'year':y,'horizon_sessions':h,'signals':len(group),
                'complete':len(complete),'incomplete':len(group)-len(complete),
                'target_hits_of_complete':target_hits,
                'distinct_stocks_with_target_hit':len({r['symbol'] for r in complete if r['hit_'+t]==1}),
                'target_hit_pct_of_complete':rnd(100*target_hits/len(complete)) if complete else '',
                'target_hits_in_incomplete':hit_early,
                'avg_close_return_pct':rnd(mean(endrets)) if endrets else '',
                'median_close_return_pct':rnd(median(endrets)) if endrets else '',
                'close_positive':sum(v>0 for v in endrets),
                'avg_target_first_return_pct':rnd(mean(strat)) if strat else '',
                'median_target_first_return_pct':rnd(median(strat)) if strat else '',
                'target_first_positive':sum(v>0 for v in strat),
                'worst_close_return_pct':rnd(min(endrets)) if endrets else '',
                'worst_target_first_return_pct':rnd(min(strat)) if strat else '',
            }
            summaries.append(summary)
    scol=list(summaries[0]) if summaries else ['year','horizon_sessions','signals']
    write_csv(str(prefix)+'_summary.csv',summaries,scol)
    print('\nMETHOD: signal CLOSE -> subsequent daily HIGH touches +'+str(args.target)+'% to take profit.')
    print('CLOSE PERFORMANCE: signal CLOSE -> CLOSE after exact horizon.')
    print('TARGET-FIRST PERFORMANCE: +'+str(args.target)+'% if reached by horizon, else horizon closing return.')
    print('Only COMPLETE horizon observations enter comparative percentages. INCOMPLETE hits listed separately.')
    print('YEAR | DAYS | SIGNALS | COMPLETE | INCOMPLETE | TARGET_HITS | HIT_RATE | AVG_CLOSE_RETURN | AVG_TARGET_FIRST')
    for s in summaries:
        print(s['year'],'|',s['horizon_sessions'],'|',s['signals'],'|',s['complete'],'|',s['incomplete'],
              '|',s['target_hits_of_complete'],'|',s['target_hit_pct_of_complete'],
              '|',s['avg_close_return_pct'],'|',s['avg_target_first_return_pct'])
    print('FILES:',str(prefix)+'_details.csv',str(prefix)+'_summary.csv')
    print('READ ONLY. No broker execution. Multiple signals may overlap; incomplete history/survivorship bias and unadjusted corporate actions can distort returns.')

if __name__=='__main__': main()
