#!/usr/bin/env python3
"""Read-only: when did initially unsuccessful 50%-recovery signals finally reach target?
Requires us_tp_sp500_market_test.py next to this file. Uses frozen 86-date snapshot,
excludes 2024-12-18. No DB writes. Not a fill/backtest.
"""
import argparse
import statistics
from pathlib import Path
from us_tp_sp500_market_test import load_dates, fetch_index, write_csv

EXCLUDED = '2024-12-18'

def pct(v, ref): return 100.0*(v/ref-1.0)
def rnd(v): return round(v,4) if v is not None else None

def first_hit(bars, i, begin, stop, price, field):
    return next((j for j in range(begin, stop+1) if bars[i+j][field] >= price), None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/sp500_late_recovery')
    p.add_argument('--ticker',default='^GSPC')
    p.add_argument('--max-sessions',type=int,default=0,help='0=all sessions available through latest index date, otherwise cap days after signal (>=21)')
    p.add_argument('--live',action='store_true',help='Use live DB instead of frozen 86-date sample')
    args=p.parse_args()
    if args.max_sessions and args.max_sessions<21: p.error('--max-sessions must be 0 or >=21')
    events=[e for e in load_dates(args.live) if str(e['day']) != EXCLUDED]
    bars=fetch_index(args.ticker)
    ix={b[0]:i for i,b in enumerate(bars)}
    allrows=[]; failures=[]; declining_qualified=0; incomplete=0
    for e in events:
        date=str(e['day']);i=ix.get(date)
        if i is None or i<20: continue
        initial=bars[i][1]
        prior=pct(initial,bars[i-20][1])
        if prior>=0: continue
        if i+20>=len(bars):
            incomplete+=1;continue
        declining_qualified+=1
        # Existing definition: +50% of prior20 close-to-close percentage drop,
        # from signal-date close; not midpoint between actual swing high and low.
        target_pct=(-prior/2.0)
        target_price=initial*(1+target_pct/100)
        early_first=first_hit(bars,i,1,20,target_price,2)
        post20_pct=pct(bars[i+20][1],initial)
        first20_min_low=min(b[3] for b in bars[i+1:i+21])
        max_avail=len(bars)-i-1
        end=min(max_avail,args.max_sessions) if args.max_sessions else max_avail
        later_high=first_hit(bars,i,21,end,target_price,2) if end>=21 else None
        later_close=first_hit(bars,i,21,end,target_price,1) if end>=21 else None
        hit_close_any=first_hit(bars,i,1,end,target_price,1) if end>=1 else None
        row={
            'date':date,'stocks':e['stocks'],'sectors':e['sectors'],
            'prior20_pct':rnd(prior),'target_gain_pct':rnd(target_pct),
            'signal_close':rnd(initial),'target_index_level':rnd(target_price),
            'post20_close_pct':rnd(post20_pct),
            'post20_continued_down':int(post20_pct<0),
            'post20_min_low_pct':rnd(pct(first20_min_low,initial)),
            'hit50_in_first20':int(early_first is not None),
            'late_hit_high_session':later_high,'late_hit_high_date':bars[i+later_high][0] if later_high else None,
            'late_hit_close_session':later_close,'late_hit_close_date':bars[i+later_close][0] if later_close else None,
            'first_close_hit_session_anytime':hit_close_any,
            'observed_sessions':end,'last_observed_date':bars[i+end][0],
            'max_sessions_limit':args.max_sessions or 'all_available',
            'late_hit_status': ('hit_after20' if later_high else 'not_hit_yet'),
        }
        allrows.append(row)
        if early_first is None: failures.append(row)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    cols=list(allrows[0]) if allrows else ['date']
    write_csv(Path(str(out)+'_all_declining.csv'),allrows,cols)
    write_csv(Path(str(out)+'_failed_first20.csv'),failures,cols)
    continued=[r for r in failures if r['post20_continued_down']]
    print('S&P500',args.ticker,'LATEST INDEX DATE',bars[-1][0])
    print('SIGNALS',len(events),'EXCLUDED',EXCLUDED,'DECLINING WITH COMPLETE FIRST20',declining_qualified,'INCOMPLETE',incomplete)
    print('FAILED TARGET IN FIRST20',len(failures),'| OF WHICH CLOSED LOWER AT DAY20',len(continued))
    print('DATE | STOCKS | SECTORS | PRE20% | DAY20% | TARGET+% | FIRST HIGH HIT SESSION | HIT DATE | FIRST CLOSE HIT SESSION AFTER20 | STATUS')
    for r in sorted(failures,key=lambda z:z['date']):
        print(' | '.join(map(str,[r['date'],r['stocks'],r['sectors'],r['prior20_pct'],r['post20_close_pct'],
            r['target_gain_pct'],r['late_hit_high_session'] or '-',r['late_hit_high_date'] or '-',
            r['late_hit_close_session'] or '-',r['late_hit_status']])))
    for label,arr in [('ALL FIRST20 FAILURES',failures),('CONTINUED DOWN BY DAY20',continued),('OTHER FIRST20 FAILURES',[r for r in failures if not r['post20_continued_down']])]:
        hits=[r['late_hit_high_session'] for r in arr if r['late_hit_high_session'] is not None]
        closes=[r['late_hit_close_session'] for r in arr if r['late_hit_close_session'] is not None]
        print(f'{label}: n={len(arr)} later_high_hits={len(hits)} never_hit_or_censored={len(arr)-len(hits)}'
              f' median_session={round(statistics.median(hits),2) if hits else None}'
              f' mean_session={round(statistics.mean(hits),2) if hits else None}'
              f' latest_session={max(hits) if hits else None}'
              f' later_close_hits={len(closes)}')
    print('FILES:',str(out)+'_failed_first20.csv',str(out)+'_all_declining.csv')
    print('NOTES: First20 failure is defined using HIGH (same as prior test). The subsequent first HIGH hit can occur after 20 sessions; close hit is reported separately. Not hit means not hit by last available market date, NOT a permanent failure.')
    print('Important: overlapping signals are not independent; current sector labels, hindsight pivots, and same-day reference close can bias interpretation. READ ONLY.')

if __name__=='__main__':main()
