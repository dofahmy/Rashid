#!/usr/bin/env python3
"""Read-only exploratory study: Does ^GSPC recover >=50% of its prior 20-session decline?
Uses frozen 86-date sample by default. Excludes 2024-12-18 on user request.
Needs us_tp_rebound_audit.py and us_tp_sp500_market_test.py in same directory.
"""
import argparse, csv, statistics
from pathlib import Path
from us_tp_sp500_market_test import load_dates, fetch_index, write_csv
from us_tp_rebound_audit import compute

EXCLUDE='2024-12-18'

def summ(vals):
    return round(statistics.mean(vals),3) if vals else None

def fmt(x):
    return 'N/A' if x is None else f'{x:.2f}'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/tmp/sp500_half_recovery')
    p.add_argument('--ticker',default='^GSPC')
    p.add_argument('--live',action='store_true',help='Use live DB signal dates instead of frozen snapshot')
    a=p.parse_args()
    events=[e for e in load_dates(a.live) if e['day']!=EXCLUDE]
    bars=fetch_index(a.ticker)
    rows,missing,ix=compute(bars,events)
    data=[]
    for r in rows:
        day=r['date']; i=ix[day]; pre=r['pre20_pct']; closed=r['forward20_pct']; complete=r['future20_complete']
        falling=pre is not None and pre<0
        target_pct=(-pre*.5) if falling else None
        future=bars[i+1:i+21] if complete else []
        first_hit=next((j+1 for j,b in enumerate(future) if target_pct is not None and 100*(b[2]/bars[i][1]-1)>=target_pct),None)
        # Compare close and intraperiod-high to 50% of pre20 close-to-close drawdown.
        recovered_close=(closed is not None and target_pct is not None and closed>=target_pct)
        recovered_intraperiod=(first_hit is not None)
        drop_before_target=None
        if target_pct is not None and future:
            # inclusive to first hit; daily OHLC cannot establish intra-session ordering.
            upto=future[:first_hit] if first_hit else future
            drop_before_target=round(100*(min(b[3] for b in upto)/bars[i][1]-1),4)
        row={'date':day,'stocks':r['stocks'],'sectors':r['sectors'],
             'breadth_band':r['breadth_band'],'sector_band':r['sector_band'],
             'pre20_pct':pre,'declined_prior20':int(falling),
             'target_50pct_recovery_from_signal_pct':round(target_pct,4) if target_pct is not None else None,
             'post20_close_pct':closed,'post20_max_high_pct':r['max_up20_pct'],
             'post20_max_low_pct':r['max_down20_pct'],
             'recovery_ratio_close_pct':round(100*closed/(-pre),2) if falling and closed is not None else None,
             'recovery_ratio_high_pct':round(100*r['max_up20_pct']/(-pre),2) if falling and r['max_up20_pct'] is not None else None,
             'hit_50pct_intraperiod':int(recovered_intraperiod) if falling and complete else None,
             'hit_50pct_by_close20':int(recovered_close) if falling and complete else None,
             'first_target_hit_session':first_hit if falling and complete else None,
             'min_low_before_target_pct':drop_before_target,
             'future20_complete':int(complete),'trough_session':r['trough_sessions']}
        data.append(row)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    write_csv(Path(str(path)+'_dates.csv'),data,list(data[0]) if data else ['date'])
    print('EXCLUDED:',EXCLUDE,'| SIGNAL DATES:',len(data),'| MISSING:',missing)
    print('NOTE: Percentage recovered = future % gain / absolute prior20 % decline, measured from signal-close. Not a fraction of a price high-to-low swing.')
    print('NOTE: Only dates with negative prior20 performance qualify for a recovery-of-decline metric.')
    print('DATE | STOCKS | SECTORS | PRE20% | POST20 CLOSE% | MAX HIGH20% | RECOUP CLOSE% | RECOUP HIGH% | HIT HIGH? | FIRST DAY | WORST LOW%')
    for r in sorted(data,key=lambda x:x['date'],reverse=True):
        print(' | '.join(map(str,[r['date'],r['stocks'],r['sectors'],fmt(r['pre20_pct']),fmt(r['post20_close_pct']),fmt(r['post20_max_high_pct']),fmt(r['recovery_ratio_close_pct']),fmt(r['recovery_ratio_high_pct']),r['hit_50pct_intraperiod'] if r['hit_50pct_intraperiod'] is not None else '-',r['first_target_hit_session'] or '-',fmt(r['post20_max_low_pct'])])))
    summary=[]
    for classification,bands in [('breadth_band',['9-14','15-29','30+']),('sector_band',['3-4','5-6','7+'])]:
        for band in bands:
            subset=[r for r in data if r[classification]==band]
            eligible=[r for r in subset if r['declined_prior20'] and r['future20_complete']]
            summary.append({'classification':classification,'band':band,'n_all':len(subset),'n_declining_complete':len(eligible),
                'n_hit_high50':sum(r['hit_50pct_intraperiod'] for r in eligible),
                'pct_hit_high50':round(100*sum(r['hit_50pct_intraperiod'] for r in eligible)/len(eligible),2) if eligible else None,
                'n_hit_close50':sum(r['hit_50pct_by_close20'] for r in eligible),
                'pct_hit_close50':round(100*sum(r['hit_50pct_by_close20'] for r in eligible)/len(eligible),2) if eligible else None,
                'avg_pre20_pct':summ([r['pre20_pct'] for r in eligible]),
                'avg_post20_pct':summ([r['post20_close_pct'] for r in eligible]),
                'avg_recovered_close_pct':summ([r['recovery_ratio_close_pct'] for r in eligible]),
                'avg_recovered_high_pct':summ([r['recovery_ratio_high_pct'] for r in eligible])})
    write_csv(Path(str(path)+'_summary.csv'),summary,list(summary[0]))
    print('\nSUMMARY:')
    for r in summary: print(r)
    print('\nFILES:',str(path)+'_dates.csv',str(path)+'_summary.csv')
    print('READ ONLY. Target hit uses future daily HIGH, which does not imply an executable fill; future window must contain 20 complete sessions.')
    print('CAVEATS: Same-day close entry may not be executable; overlapping dates, historical pivot lookahead, current sector labels, and in-sample threshold selection may bias results.')

if __name__=='__main__': main()
