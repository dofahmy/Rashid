#!/usr/bin/env python3
"""Read-only: exit at +3% intraday (next session onward), or at Nth session close.
Uses exactly the trades selected by the user's existing backtest trades.csv.
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from sqlalchemy import create_engine, text

DEFAULT_WINDOWS='1,2,3,5,10,15,20,30,40,60,90,120,180,250,350'

def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarise(rows, window, segment, value):
    n=len(rows)
    if not n:return None
    wins=sum(x['return_pct']>0 for x in rows)
    hits=sum(x['exit_reason']=='TP' for x in rows)
    gains=sum(x['return_pct'] for x in rows)
    ar=sorted(x['return_pct'] for x in rows)
    return dict(window_sessions=window,segment=segment,value=value,trades=n,
                tp_hits=hits,tp_hit_pct=round(100*hits/n,2),
                wins=wins,win_pct=round(100*wins/n,2),
                average_return_pct=round(gains/n,4),sum_return_pct=round(gains,4),
                median_return_pct=round((ar[(n-1)//2]+ar[n//2])/2,4),
                worst_return_pct=round(ar[0],4),
                loss_count=sum(x['return_pct']<0 for x in rows),
                forced_exits=n-hits,
                average_holding_sessions=round(sum(x['holding_sessions'] for x in rows)/n,2))

def main():
    p=argparse.ArgumentParser(description='Evaluate +3% take-profit OR force exit after N EGX trading sessions')
    p.add_argument('--trades',default='egx_tp3d_backtest_results/trades.csv')
    p.add_argument('--windows',default=DEFAULT_WINDOWS)
    p.add_argument('--output',default='egx_tp3d_exit_window_results')
    p.add_argument('--cost-pct',type=float,default=0,help='Subtract total round-trip trading costs in percent points, default 0')
    args=p.parse_args()
    windows=sorted(set(int(x) for x in args.windows.split(',')))
    if not windows or min(windows)<1:raise SystemExit('Windows must be positive integers')
    if args.cost_pct<0:raise SystemExit('--cost-pct must be >= 0')
    with open(args.trades,encoding='utf-8-sig',newline='') as f:trades=list(csv.DictReader(f))
    if not trades:raise SystemExit('No trades in file')
    url=os.getenv('DATABASE_URL')
    if not url:raise SystemExit('DATABASE_URL not configured')
    by_sym=defaultdict(list)
    for row in trades:by_sym[row['symbol']].append(row)
    eng=create_engine(url,pool_pre_ping=True,pool_size=1,max_overflow=0)
    outcomes=[];missing=[];windows_max=max(windows)
    try:
        with eng.connect() as con:
            for i,(symbol,records) in enumerate(by_sym.items(),1):
                start=min(r['touch_date'] for r in records)
                bars=[]
                for dt,high,close in con.execute(text('''SELECT session_date,h,c FROM market_candles_1d
                  WHERE symbol=:s AND session_date>=:d ORDER BY session_date'''),{'s':symbol,'d':start}):
                    try:
                        h=float(high);c=float(close);ds=str(dt)[:10]
                        if math.isfinite(h) and math.isfinite(c) and h>0 and c>0 and h+0.00001>=c:bars.append((ds,h,c))
                    except (ValueError,TypeError):continue
                by_date={b[0]:j for j,b in enumerate(bars)}
                for trade in records:
                    index=by_date.get(trade['touch_date'])
                    if index is None:
                        missing.append((symbol,trade['touch_date'],'no candle'));continue
                    entry=float(trade['entry_close']);tp=entry*1.03
                    if entry<=0:continue
                    remaining=len(bars)-index-1
                    for window in windows:
                        # Avoid counting immature recent trades as failures or closing them prematurely.
                        if remaining<window:continue
                        slice_=bars[index+1:index+window+1]
                        hit_index=next((j for j,b in enumerate(slice_,1) if b[1]+1e-9>=tp),None)
                        if hit_index is not None:
                            exit_date=bars[index+hit_index][0];exit_price=tp;hold=hit_index;reason='TP'
                        else:
                            exit_date=bars[index+window][0];exit_price=bars[index+window][2];hold=window;reason='TIME'
                        result={k:trade[k] for k in ('symbol','touch_date','signal_type','anchor_type','status','angle_combo','group_stocks')}
                        result.update(window_sessions=window,entry_price=round(entry,4),exit_date=exit_date,
                              exit_price=round(exit_price,4),exit_reason=reason,holding_sessions=hold,
                              return_pct=round((exit_price/entry-1)*100-args.cost_pct,5))
                        outcomes.append(result)
                if i%25==0:print(f'Processed {i}/{len(by_sym)} stocks...',flush=True)
    finally:eng.dispose()
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    write_csv(target/'trades_by_exit_window.csv',outcomes)
    all_summaries=[]
    for w in windows:
        candidates=[r for r in outcomes if r['window_sessions']==w]
        o=summarise(candidates,w,'ALL','ALL')
        if o:all_summaries.append(o)
    write_csv(target/'compare_windows.csv',all_summaries)
    # Like-for-like: only original trades whose candle history covers every window.
    # Their identities are unique in the source trades list; don't compare cohorts
    # that vary merely because one horizon requires more historical follow-up.
    common_keys={(r['symbol'],r['touch_date'],r['signal_type'],r['anchor_type'],r['status'])
                 for r in outcomes if r['window_sessions']==windows_max}
    common=[]
    for w in windows:
        subset=[r for r in outcomes if r['window_sessions']==w and
                (r['symbol'],r['touch_date'],r['signal_type'],r['anchor_type'],r['status']) in common_keys]
        summary=summarise(subset,w,'MATCHED','ALL')
        if summary:common.append(summary)
    write_csv(target/'compare_windows_same_trades.csv',common)
    detailed=[]
    dimensions=['signal_type','anchor_type','status','angle_combo']
    for w in windows:
        ws=[r for r in outcomes if r['window_sessions']==w]
        for dimension in dimensions:
            vals=sorted(set(r[dimension] for r in ws))
            for val in vals:
                subset=[r for r in ws if r[dimension]==val]
                d=summarise(subset,w,dimension,val)
                if d:detailed.append(d)
    write_csv(target/'compare_by_signal_side_status_angles.csv',detailed)
    print('\n+3% TP OR exit at Nth trading-session close (net cost=%g percentage points)'%args.cost_pct)
    print('Note: each window excludes recent entries without enough follow-up sessions; samples differ!')
    print(f'Original file: {len(trades)} signals; missing entry candles: {len(missing)}')
    print(' DAYS   ELIGIBLE   TP_HITS   TP%   AVG_RET%   MEDIAN%   WORST%  LOSSES')
    for r in all_summaries:
        print(f'{r["window_sessions"]:>5} {r["trades"]:>10} {r["tp_hits"]:>9} {r["tp_hit_pct"]:>5.1f} {r["average_return_pct"]:>10.2f} {r["median_return_pct"]:>9.2f} {r["worst_return_pct"]:>8.2f} {r["loss_count"]:>7}')
    if common:
        print('\nSAME-TRADES COMPARISON (exactly the same mature signals for all windows):')
        print(' DAYS   ELIGIBLE   TP_HITS   TP%   AVG_RET%   MEDIAN%   WORST%  LOSSES')
        for r in common:
            print(f'{r["window_sessions"]:>5} {r["trades"]:>10} {r["tp_hits"]:>9} {r["tp_hit_pct"]:>5.1f} {r["average_return_pct"]:>10.2f} {r["median_return_pct"]:>9.2f} {r["worst_return_pct"]:>8.2f} {r["loss_count"]:>7}')
    if all_summaries:
        highest=max(all_summaries,key=lambda x:x['average_return_pct'])
        print(f'Highest observed mean (NOT causal or recommended): {highest["window_sessions"]} sessions, {highest["average_return_pct"]:.2f}%')
    print(f'Output: {target.resolve()}')

if __name__=='__main__':main()
