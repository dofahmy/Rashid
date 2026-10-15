#!/usr/bin/env python3
"""Research-only S&P 500 bottom fingerprint without Time-Price data.

Uses just past/current daily OHLC. Fits interpretable filters to five handpicked
historical bottoms; displays *all* other matches, never claims a unique rule.
No database, Telegram bot, order submission or future data in signals.
"""
import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path
from datetime import datetime

# Reuse existing Yahoo chart reader, NOT its Time-Price scan or DB access.
from us_tp_strategy_60pct import get_bars

CHOSEN = ('2022-10-13', '2023-10-26', '2024-08-05', '2025-04-08', '2026-03-30')


def pct(a, b):
    return (a / b - 1.0) * 100.0 if b else None


def build_rows(bars):
    data = []
    for i in range(60, len(bars)):
        b = bars[i]
        w20 = bars[i-19:i+1]
        w60 = bars[i-59:i+1]
        high20 = max(x['high'] for x in w20)
        low20 = min(x['low'] for x in w20)
        high60 = max(x['high'] for x in w60)
        low60 = min(x['low'] for x in w60)
        daily_range = b['high'] - b['low']
        data.append(dict(date=b['date'], idx=i,
            open=b['open'], close=b['close'],
            prior3=pct(b['close'],bars[i-3]['close']),
            prior5=pct(b['close'],bars[i-5]['close']),
            prior10=pct(b['close'],bars[i-10]['close']),
            prior20=pct(b['close'],bars[i-20]['close']),
            fromhigh20=pct(b['close'],high20),
            fromhigh60=pct(b['close'],high60),
            pos20=100*(b['close']-low20)/(high20-low20) if high20>low20 else 50,
            pos60=100*(b['close']-low60)/(high60-low60) if high60>low60 else 50,
            candleclose=100*(b['close']-b['low'])/daily_range if daily_range>0 else 50))
    return data


def matches(r, rule):
    return (r['prior20'] <= -rule['drop20']
            and r['prior5'] <= -rule['drop5']
            and r['fromhigh60'] <= -rule['fromhigh60']
            and r['pos60'] <= rule['pos60'])


def find_rule(rows, chosen, gap):
    # Explicit finite grid, calibrated on only five user-selected dates.
    # Minimize distinct extra *events*, not just daily signals.
    universe = [r for r in rows if r['date'] in chosen]
    if len(universe) != len(chosen):
        missing = sorted(set(chosen) - {r['date'] for r in universe})
        raise ValueError(f'Missing chosen dates from historical bars: {missing}')
    best = None
    for a, b, c, d in itertools.product(
        (1.5,2.0,2.5,3.0,3.5),
        (0.5,1.0,1.5,1.8),
        (4.0,5.0,6.0,7.0),
        (15.0,20.0,25.0,35.0,50.0,65.0)):
        rule=dict(drop20=a,drop5=b,fromhigh60=c,pos60=d)
        matched=[r for r in rows if matches(r,rule)]
        captured=sum(matches(r,rule) for r in universe)
        events=cluster(matched,gap)
        extra=sum(not any(x['date'] in chosen for x in e) for e in events)
        # No dishonest 'five only' promise. Prefer recall, then fewer extra events,
        # then fewer matched days, then less complexity. Display all other hits.
        key=(-captured,extra,len(matched),a+b+c+d/100)
        if best is None or key<best[0]:best=(key,rule,matched,events,captured)
    return best[1:]


def cluster(rows,gap):
    groups=[]
    for r in sorted(rows,key=lambda x:x['idx']):
        if groups and r['idx']-groups[-1][-1]['idx']<=gap:
            groups[-1].append(r)
        else: groups.append([r])
    return groups


def diagnostics(bars,r,horizons=(20,60)):
    i=r['idx']; result={}
    for h in horizons:
        future=bars[i+1:min(len(bars),i+h+1)]
        result[f'observed_{h}']=len(future)
        result[f'complete_{h}']=int(len(future)==h)
        if future:
            result[f'maxrise_{h}_pct']=round(pct(max(x['high'] for x in future),r['close']),4)
            result[f'maxfall_{h}_pct']=round(pct(min(x['low'] for x in future),r['close']),4)
            result[f'endclose_{h}_pct']=round(pct(future[-1]['close'],r['close']),4)
        else:
            for k in ('maxrise','maxfall','endclose'):result[f'{k}_{h}_pct']=''
    return result


def csv_out(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='/tmp/sp500_price_only_bottom')
    parser.add_argument('--ticker',default='^GSPC')
    parser.add_argument('--gap',type=int,default=10,help='Group matching dates separated by at most N trading sessions')
    parser.add_argument('--exclude',action='append',default=['2024-12-18'])
    parser.add_argument('--chosen',nargs='*',default=list(CHOSEN))
    args=parser.parse_args()
    if args.gap<1:parser.error('--gap must be >=1')
    chosen=set(args.chosen);excluded=set(args.exclude)
    bars=get_bars(args.ticker)
    rows=build_rows(bars)
    fitrows=[r for r in rows if r['date'] not in excluded]
    rule,matched,events,captured=find_rule(fitrows,chosen,args.gap)
    outfile=Path(args.out);outfile.parent.mkdir(parents=True,exist_ok=True)
    output=[]
    for r in matched:
        output.append({k:round(v,4) if isinstance(v,float) else v for k,v in r.items() if k!='idx'}
                      | {'selected_example':int(r['date'] in chosen)} | diagnostics(bars,r))
    eventout=[]
    for n,e in enumerate(events,1):
        # representative: most deeply declined from past 60-session high
        rep=min(e,key=lambda r:(r['fromhigh60'],r['date']))
        eventout.append(dict(event=n,start=e[0]['date'],end=e[-1]['date'],
             representative=rep['date'],days=len(e),
             matched_selected_examples=','.join(r['date'] for r in e if r['date'] in chosen),
             is_selected_event=int(any(r['date'] in chosen for r in e)),
             prior20_pct=round(rep['prior20'],4),fromhigh60_pct=round(rep['fromhigh60'],4),
             pos60_pct=round(rep['pos60'],4), **diagnostics(bars,rep)))
    chosenout=[]
    for r in rows:
        if r['date'] in chosen:
            chosenout.append(dict(date=r['date'],matched=int(matches(r,rule)),
                prior5=round(r['prior5'],4),prior20=round(r['prior20'],4),
                fromhigh60=round(r['fromhigh60'],4),pos60=round(r['pos60'],4)))
    csv_out(str(outfile)+'_days.csv',output)
    csv_out(str(outfile)+'_events.csv',eventout)
    csv_out(str(outfile)+'_five.csv',chosenout)
    summary=dict(ticker=args.ticker,period=[bars[0]['date'],bars[-1]['date']],
        price_only=True,rule=rule,selected_captured=captured,selected_total=len(chosen),
        matched_days=len(matched),matched_events=len(events),
        other_events=sum(not x['is_selected_event'] for x in eventout),
        other_days=sum(not x['selected_example'] for x in output),
        clustering_gap_sessions=args.gap,excluded=list(excluded),
        caveats=['five dates manually selected in hindsight', 'same period used to choose thresholds and evaluate them',
                 'other matching events do not imply good bottoms',
                 '60 session future prices used in reports only, never conditions',
                 'daily close signal earliest actionable next session',
                 'no Time-Price, sector or stock data queried'])
    with open(str(outfile)+'_summary.json','w',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print('INDEX',args.ticker,'BARS',len(bars),'RANGE',bars[0]['date'],bars[-1]['date'])
    print('RULE (price-only; tuned in hindsight):',rule)
    print('SELECTED FIVE CAPTURED:',captured,'/',len(chosen))
    for r in chosenout:print('FIVE:',r)
    print('MATCHED DAYS:',len(matched),'MATCHED EVENTS:',len(events),
          'ADDITIONAL EVENTS:',summary['other_events'])
    print('ALL EVENTS (including false-positive candidates):')
    for r in eventout:print('EVENT:',r)
    print('FILES:',str(outfile)+'_days.csv',str(outfile)+'_events.csv',
          str(outfile)+'_five.csv',str(outfile)+'_summary.json')
    print('READ ONLY. NO TIME-PRICE. IN-SAMPLE FIVE EXAMPLES ARE NOT VALIDATION.')

if __name__=='__main__':main()
