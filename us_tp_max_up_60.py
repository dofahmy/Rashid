#!/usr/bin/env python3
"""Maximum S&P 500 intraday high in the 60 exchange sessions after each frozen Time-Price signal.
Read-only; historical index OHLC, not a traded ETF execution price.
"""
import argparse
import csv
from pathlib import Path
from statistics import mean, median
from us_tp_sp500_market_test import load_dates, fetch_index

EXCLUDED = '2024-12-18'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='/tmp/sp500_max_up_60')
    parser.add_argument('--sessions',type=int,default=60)
    parser.add_argument('--ticker',default='^GSPC')
    parser.add_argument('--live',action='store_true',help='Use current DB instead of frozen 86-date snapshot')
    args=parser.parse_args()
    if args.sessions<1: parser.error('--sessions must be > 0')
    events=[e for e in load_dates(args.live) if e['day']!=EXCLUDED]
    bars=fetch_index(args.ticker)
    idx={b[0]:i for i,b in enumerate(bars)}
    result=[]
    for e in sorted(events,key=lambda r:r['day'],reverse=True):
        day=e['day']
        r=dict(date=day,stocks=e['stocks'],sectors=e['sectors'],reference_close='',max_rise_pct='',peak_date='',peak_session='',max_fall_pct='',trough_date='',observed_sessions=0,complete_60='NO',status='NO_INDEX_DAY')
        if day in idx:
            i=idx[day];c=bars[i][1];f=bars[i+1:i+1+args.sessions]
            r['reference_close']=round(c,4)
            r['observed_sessions']=len(f)
            r['complete_60']='YES' if len(f)==args.sessions else 'NO'
            r['status']='COMPLETE' if len(f)==args.sessions else ('PARTIAL' if f else 'NO_FUTURE_DATA')
            if f:
                peak_i=max(range(len(f)),key=lambda k:f[k][2]);low_i=min(range(len(f)),key=lambda k:f[k][3])
                r.update(max_rise_pct=round((f[peak_i][2]/c-1)*100,4),peak_date=f[peak_i][0],peak_session=peak_i+1,
                         max_fall_pct=round((f[low_i][3]/c-1)*100,4),trough_date=f[low_i][0])
        result.append(r)
    p=Path(args.out+'_dates.csv');p.parent.mkdir(parents=True,exist_ok=True)
    fields=['date','stocks','sectors','reference_close','max_rise_pct','peak_date','peak_session','max_fall_pct','trough_date','observed_sessions','complete_60','status']
    with p.open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader();writer.writerows(result)
    print(f'SIGNAL DATES: {len(result)} | excluded {EXCLUDED} | sessions={args.sessions} | index={args.ticker}')
    print(f'INDEX DATA THROUGH: {bars[-1][0]}')
    print('DATE | STOCKS | SECTORS | MAX RISE % | PEAK DATE | PEAK SESSION | MAX FALL % | OBSERVED | STATUS')
    for r in result:
        def val(x): return str(x) if x!='' else 'N/A'
        print(' | '.join([str(r['date']),str(r['stocks']),str(r['sectors']),val(r['max_rise_pct']),val(r['peak_date']),val(r['peak_session']),val(r['max_fall_pct']),str(r['observed_sessions']),r['status']]))
    completed=[r for r in result if r['status']=='COMPLETE']
    if completed:
        pct=[r['max_rise_pct'] for r in completed]
        print('COMPLETE:',len(completed),'PARTIAL:',sum(r['status']=='PARTIAL' for r in result),'NO FUTURE:',sum(r['status']=='NO_FUTURE_DATA' for r in result))
        print('AVERAGE MAX RISE: %.3f%% | MEDIAN: %.3f%% | MAX: %.3f%% | MIN: %.3f%%' %(mean(pct),median(pct),max(pct),min(pct)))
    print('CSV:',p)
    print('READ-ONLY. Max rise means highest daily HIGH over future sessions relative to signal-date CLOSE. It is not necessarily executable or a closing return; partial windows must not be compared with complete ones.')

if __name__=='__main__':main()
