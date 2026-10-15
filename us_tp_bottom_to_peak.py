#!/usr/bin/env python3
"""Read-only S&P500 market-bottom -> market-peak backtest.

BOTTOM: SAME_ANGLE_DIFF_ANCHORS/LOW, CROSS_DOWN_CLOSE or TOUCH_ABOVE.
PEAK: SAME_ANGLE_DIFF_ANCHORS/HIGH, matching inverse statuses chosen from
actually observed DB statuses (or specified with --peak-status).
Each (day, signal_type, anchor_type, status) group must have >=5 distinct stocks
before the day's eligible groups are combined. Day requires >=5 distinct stocks
and >=2 distinct classified sectors (configurable).

One position at a time: enter at NEXT trading session OPEN after bottom; exit
at NEXT trading session OPEN after peak, not at its same-day close. Never use a
future-confirmed pivot as if known in real time. Intended as historical research.
"""
import argparse
import csv
from pathlib import Path
from collections import Counter
from datetime import date

from us_tp_strategy_60pct import get_bars

BOTTOM_STATUSES = ('CROSS_DOWN_CLOSE', 'TOUCH_ABOVE')
PEAK_INVERSE_PRIORITY = ('CROSS_UP_CLOSE', 'TOUCH_BELOW')


def sql_rows(query, params=None):
    from core import database
    from sqlalchemy import text
    maker = database()
    with maker() as session:
        return [dict(r) for r in session.execute(text(query), params or {}).mappings().all()]


def inspect_statuses():
    return sql_rows("""
        SELECT anchor_type, status, COUNT(*) AS rows,
               COUNT(DISTINCT symbol) AS symbols,
               COUNT(DISTINCT day) AS dates
        FROM us_tp_daily_candidates
        WHERE signal_type='SAME_ANGLE_DIFF_ANCHORS'
          AND anchor_type IN ('LOW','HIGH')
        GROUP BY anchor_type, status ORDER BY anchor_type, rows DESC
    """)


def events(anchor, statuses, min_stocks, min_sectors, subgroup_min=5):
    return sql_rows("""
    WITH qualified_groups AS (
        SELECT day, signal_type, anchor_type, status
        FROM us_tp_daily_candidates
        WHERE signal_type='SAME_ANGLE_DIFF_ANCHORS'
          AND anchor_type=:anchor
          AND status IN :statuses
        GROUP BY day, signal_type, anchor_type, status
        HAVING COUNT(DISTINCT symbol)>=:groupmin
    ), eligible AS (
        SELECT DISTINCT c.day, c.symbol
        FROM us_tp_daily_candidates c
        JOIN qualified_groups g
          ON c.day=g.day AND c.signal_type=g.signal_type
         AND c.anchor_type=g.anchor_type AND c.status=g.status
    ), daily AS (
        SELECT e.day, COUNT(DISTINCT e.symbol) AS stocks,
               COUNT(DISTINCT NULLIF(TRIM(s.sector),'')) AS sectors
        FROM eligible e
        LEFT JOIN us_tp_sector_lookup s ON s.symbol=e.symbol
        GROUP BY e.day
    )
    SELECT day,stocks,sectors FROM daily
    WHERE stocks>=:stockmin AND sectors>=:sectormin
    ORDER BY day
    """.replace('status IN :statuses', 'status IN (' + ','.join(f':status{i}' for i in range(len(statuses))) + ')'),
    dict(anchor=anchor, groupmin=subgroup_min, stockmin=min_stocks, sectormin=min_sectors,
         **{f'status{i}':s for i,s in enumerate(statuses)})) if statuses else []


def write_csv(path, rows, cols=None):
    if cols is None: cols=list(rows[0]) if rows else ['date']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)


def datekey(v):return str(v)[:10]


def simulate(bars, bottoms, peaks, exclude):
    ix={x['date']:i for i,x in enumerate(bars)}
    bottom_map={datekey(x['day']):x for x in bottoms if datekey(x['day']) not in exclude}
    peak_map={datekey(x['day']):x for x in peaks if datekey(x['day']) not in exclude}
    timeline=sorted(set(bottom_map)|set(peak_map))
    signals=[];missing=[]
    for day in timeline:
        if day not in ix:
            missing.append(day);continue
        signals.append({'day':day,'index':ix[day],
                        'bottom_stocks':bottom_map.get(day,{}).get('stocks',0),
                        'bottom_sectors':bottom_map.get(day,{}).get('sectors',0),
                        'peak_stocks':peak_map.get(day,{}).get('stocks',0),
                        'peak_sectors':peak_map.get(day,{}).get('sectors',0)})
    # Execute only from next session open, and limit to one open long.
    # Same-day simultaneous bottom/peak: peaks have no effect if flat;
    # if already long, exit peak has precedence over re-entry.
    actions_by_execution= {}
    for e in signals:
        j=e['index']+1
        if j>=len(bars):continue
        actions_by_execution.setdefault(j,[]).append(e)
    position=None; trades=[]; ignored_bottom=0; ignored_peak=0
    for j,b in enumerate(bars):
        for sig in actions_by_execution.get(j,[]):
            is_peak=sig['peak_stocks']>0
            is_bottom=sig['bottom_stocks']>0
            if position is not None and is_peak:
                entry_price=position['entry_price']
                trade={**position,
                       'peak_signal_date':sig['day'],'peak_stocks':sig['peak_stocks'],
                       'peak_sectors':sig['peak_sectors'],'exit_date':b['date'],
                       'exit_price':round(b['open'],4),
                       'holding_sessions':j-position['entry_index'],
                       'return_pct':round(100*(b['open']/entry_price-1),4),
                       'status':'PEAK_EXIT'}
                lo=min(x['low'] for x in bars[position['entry_index']:j]) if j>position['entry_index'] else entry_price
                trade['max_drawdown_pct']=round(100*(lo/entry_price-1),4)
                trade.pop('entry_index',None)
                trades.append(trade);position=None
                if is_bottom: ignored_bottom+=1
            elif position is None and is_bottom and not is_peak:
                position={'bottom_signal_date':sig['day'],
                          'bottom_stocks':sig['bottom_stocks'],'bottom_sectors':sig['bottom_sectors'],
                          'entry_date':b['date'],'entry_index':j,
                          'entry_price':round(b['open'],4)}
            elif position is None and is_bottom and is_peak:
                # Ambiguous competing simultaneous signals; do not enter.
                ignored_bottom+=1
            elif position is not None and is_bottom:
                ignored_bottom+=1
            elif is_peak:
                ignored_peak+=1
    if position is not None:
        last=bars[-1];entry=position['entry_price']
        lo=min(x['low'] for x in bars[position['entry_index']:])
        trade={**position,'peak_signal_date':'','peak_stocks':'','peak_sectors':'',
               'exit_date':'','exit_price':'',
               'holding_sessions':len(bars)-1-position['entry_index'],
               'return_pct':'','status':'OPEN_NO_PEAK',
               'mark_to_market_pct':round(100*(last['close']/entry-1),4),
               'max_drawdown_pct':round(100*(lo/entry-1),4)}
        trade.pop('entry_index',None);trades.append(trade)
    return trades, signals, missing, ignored_bottom, ignored_peak


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/rajih_bottom_peak')
    p.add_argument('--ticker',default='^GSPC')
    p.add_argument('--min-stocks',type=int,default=5)
    p.add_argument('--min-sectors',type=int,default=2)
    p.add_argument('--group-min',type=int,default=5)
    p.add_argument('--peak-status',action='append',help='Explicit HIGH status (repeatable); else auto-detect inverse CROSS_UP_CLOSE,TOUCH_BELOW')
    p.add_argument('--inspect-peaks',action='store_true',help='Show actual HIGH/LOW statuses and available signal counts; no backtest')
    p.add_argument('--exclude',action='append',default=['2024-12-18'],help='Exclude a signal date; default excludes 2024-12-18')
    args=p.parse_args()
    if min(args.min_stocks,args.min_sectors,args.group_min)<1:p.error('Counts must be >= 1')
    observed=inspect_statuses()
    print('OBSERVED SAME_ANGLE_DIFF_ANCHORS STATUS COUNTS:')
    for row in observed:print(row)
    if args.inspect_peaks:return
    observed_high={x['status'] for x in observed if x['anchor_type']=='HIGH'}
    peak_statuses = list(dict.fromkeys(args.peak_status)) if args.peak_status else [s for s in PEAK_INVERSE_PRIORITY if s in observed_high]
    if args.peak_status:
        unknown=set(peak_statuses)-observed_high
        if unknown:
            p.error(f'Unsupported HIGH statuses: {sorted(unknown)}. Run --inspect-peaks and choose from actual DB values.')
    if not peak_statuses:
        print('NO MATCHING HIGH INVERSE STATUS IN DATABASE. Backtest not run.')
        print('Choose appropriate HIGH status from the list above and rerun with --peak-status ACTUAL_STATUS.')
        return
    bottoms=events('LOW',BOTTOM_STATUSES,args.min_stocks,args.min_sectors,args.group_min)
    peaks=events('HIGH',peak_statuses,args.min_stocks,args.min_sectors,args.group_min)
    print('ENTRY LOW statuses:',BOTTOM_STATUSES,'| EXIT HIGH statuses:',peak_statuses)
    print('BOTTOM EVENTS:',len(bottoms),'PEAK EVENTS:',len(peaks),'EXCLUDED:',args.exclude)
    if not peaks:
        print('NO QUALIFYING MARKET PEAK EVENTS. Nothing to backtest; inspect HIGH candidates and coverage.');return
    bars=get_bars(args.ticker)
    if not bars:raise SystemExit('No index candles')
    trades, signals, missing, ignored_bottom, ignored_peak=simulate(bars,bottoms,peaks,set(args.exclude))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    columns=['bottom_signal_date','bottom_stocks','bottom_sectors','entry_date','entry_price',
             'peak_signal_date','peak_stocks','peak_sectors','exit_date','exit_price',
             'holding_sessions','return_pct','max_drawdown_pct','status','mark_to_market_pct']
    write_csv(str(out)+'_trades.csv',trades,columns)
    write_csv(str(out)+'_signals.csv',signals)
    closed=[t for t in trades if t['status']=='PEAK_EXIT']
    wins=[t for t in closed if t['return_pct']>0]
    from statistics import mean,median
    print('INDEX BARS:',len(bars),bars[0]['date'],bars[-1]['date'],args.ticker)
    print('CLOSED TRADES:',len(closed),'WINNING:',len(wins),'LOSING:',len(closed)-len(wins),
          'WIN RATE %:',round(100*len(wins)/len(closed),2) if closed else 'N/A')
    print('AVG CLOSED RETURN %:',round(mean(t['return_pct'] for t in closed),4) if closed else 'N/A',
          'MEDIAN:',round(median(t['return_pct'] for t in closed),4) if closed else 'N/A')
    print('AVG HOLD SESSIONS:',round(mean(t['holding_sessions'] for t in closed),2) if closed else 'N/A',
          'WORST DRAWDOWN %:',min((t['max_drawdown_pct'] for t in trades),default='N/A'))
    print('OPEN POSITIONS:',sum(t['status']=='OPEN_NO_PEAK' for t in trades),
          'IGNORED BOTTOMS WHILE INVESTED OR AMBIGUOUS:',ignored_bottom,
          'UNUSED PEAKS WHILE FLAT:',ignored_peak,
          'UNMATCHED CALENDAR DAYS:',missing)
    for t in trades:
        print('TRADE', t['bottom_signal_date'],'BUY',t['entry_date'],t['entry_price'],
              'PEAK',t['peak_signal_date'],'SELL',t['exit_date'],t['exit_price'],
              'RETURN %',t['return_pct'],'STATUS',t['status'])
    print('FILES:',str(out)+'_trades.csv',str(out)+'_signals.csv')
    print('READ ONLY; no DB writes, bot messages or trading orders.')
    print('IMPORTANT: HISTORICAL PIVOTS MAY HAVE LOOK-AHEAD; verify signal AVAILABLE-AT timestamps before live deployment.')
    print('No fees/slippage; ^GSPC is not directly tradeable. Peak signal is executed NEXT session OPEN.')

if __name__=='__main__':main()
