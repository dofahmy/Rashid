#!/usr/bin/env python3
"""Read-only backtest of completed EGX TP3D signal groups with >=N unique stocks.

Entry at signal-session close; take profit from the NEXT trading session's HIGH.
One position per symbol/day/type/side/status, irrespective of geometric line count.
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import create_engine, text

SIGNAL_NAMES = {
    'DIFFERENT_ANGLES': 'زاويتان مختلفتان',
    'SAME_ANGLE_DIFF_ANCHORS': 'نفس الزاوية من ارتكازين',
    'SAME_LINE_TWICE': 'نفس الخط مرتين',
}
STATUS_NAMES = {
    'CROSS_UP_CLOSE': 'اختراق صاعد',
    'CROSS_DOWN_CLOSE': 'كسر هابط',
    'TOUCH_ABOVE': 'لمس وإغلاق فوق',
    'TOUCH_BELOW': 'لمس وإغلاق تحت',
}
FIELD_NAMES = {
    'signal_type':'نوع الإشارة', 'anchor_type':'قمة أو قاع', 'status':'حالة الإشارة',
    'angle_combo':'تركيبة الزوايا', 'angle':'الزاوية', 'signal_year':'سنة الإشارة',
}


def args_parser():
    p=argparse.ArgumentParser(description='EGX 3D TP3% console backtest (read-only)')
    p.add_argument('--min-stocks',type=int,default=5,help='Distinct stocks in each signal group (default 5)')
    p.add_argument('--tp',type=float,default=3.,help='Take profit percent, default 3')
    p.add_argument('--start',help='First signal date YYYY-MM-DD (inclusive)')
    p.add_argument('--end',help='Last signal date YYYY-MM-DD (inclusive)')
    p.add_argument('--run-id',type=int,help='Use specific completed V11 run; otherwise latest')
    p.add_argument('--output',default='egx_tp3d_backtest_results',help='Directory for CSV files')
    return p.parse_args()


def selected_run(con, run_id):
    if run_id:
        row=con.execute(text("SELECT run_id,market_day FROM egx_tp3d_runs WHERE run_id=:id AND status='complete'"),{'id':run_id}).mappings().first()
    else:
        row=con.execute(text("SELECT run_id,market_day FROM egx_tp3d_runs WHERE build_version='v11' AND status='complete' ORDER BY run_id DESC LIMIT 1")).mappings().first()
    if row is None: raise RuntimeError('لا توجد نسخة تاريخية مكتملة؛ راجعي run_id وbuild_version')
    return dict(row)


def sql_signals(con, run_id, opts):
    # Group identity EXACTLY matches the dashboard: day/type/pivot-side/status.
    # Restrict period BEFORE HAVING; never apply TP outcome before group selection.
    parts=['run_id=:run_id']
    params={'run_id':run_id,'minimum':opts.min_stocks}
    if opts.start:
        parts.append('touch_date>=:start');params['start']=opts.start
    if opts.end:
        parts.append('touch_date<=:end');params['end']=opts.end
    where=' AND '.join(parts)
    query=f"""WITH qualifying_groups AS (
       SELECT touch_date,signal_type,anchor_type,status,COUNT(DISTINCT symbol) AS group_stocks
       FROM egx_tp3d_signals_v10
       WHERE {where}
       GROUP BY touch_date,signal_type,anchor_type,status
       HAVING COUNT(DISTINCT symbol)>=:minimum
    ), lines AS (
       SELECT s.*,g.group_stocks
       FROM egx_tp3d_signals_v10 s JOIN qualifying_groups g
         ON s.touch_date=g.touch_date AND s.signal_type=g.signal_type
        AND s.anchor_type=g.anchor_type AND s.status=g.status
       WHERE s.run_id=:run_id
    )
    SELECT symbol,touch_date,signal_type,anchor_type,status,
           MAX(group_stocks) AS group_stocks,
           MIN(touch_close) AS entry_price,
           STRING_AGG(DISTINCT angle, '+' ORDER BY angle) AS angle_combo,
           COUNT(*) AS component_lines,
           MAX(latest_price_date) AS stored_last_price_date
    FROM lines
    GROUP BY symbol,touch_date,signal_type,anchor_type,status
    ORDER BY symbol,touch_date,signal_type,anchor_type,status"""
    return [dict(row) for row in con.execute(text(query),params).mappings()]


def candles_for_symbol(con,symbol,first_date):
    q=text('''SELECT session_date,h,c FROM market_candles_1d
              WHERE symbol=:sym AND session_date>=:d ORDER BY session_date''')
    result=[]
    for d,h,c in con.execute(q,{'sym':symbol,'d':first_date}):
        try:
            ds=str(d)[:10]; high=float(h); close=float(c)
            date.fromisoformat(ds)
            if math.isfinite(high) and math.isfinite(close) and high>0 and close>0 and high+0.00001>=close:
                result.append((ds,high,close))
        except (ValueError,TypeError):
            continue
    return result


def round_or_none(v):
    return round(v,4) if v is not None else None


def backtest(con,signals,tp):
    grouped=defaultdict(list)
    for item in signals:grouped[item['symbol']].append(item)
    out=[];skipped=[]
    for counter,(sym,records) in enumerate(grouped.items(),1):
        candles=candles_for_symbol(con,sym,min(x['touch_date'] for x in records))
        by_day={d:i for i,(d,_,_) in enumerate(candles)}
        for s in records:
            idx=by_day.get(s['touch_date'])
            if idx is None:
                skipped.append((sym,s['touch_date'],'missing entry candle'));continue
            entry=float(s['entry_price'])
            if not math.isfinite(entry) or entry<=0:
                skipped.append((sym,s['touch_date'],'invalid entry price'));continue
            target=entry*(1+tp/100)
            # Don't count same-day HIGH: entry is the CLOSE and that high may predate entry.
            after=candles[idx+1:]
            exit_index=next((j for j,bar in enumerate(after) if bar[1]+1e-9>=target),None)
            last=candles[-1]
            realized=exit_index is not None
            end_index=idx+1+exit_index if realized else len(candles)-1
            end_price=target if realized else last[2]
            net_pct=(end_price/entry-1)*100
            max_high=max((b[1] for b in after),default=None)
            row={
                'symbol':sym,'touch_date':s['touch_date'],'signal_year':s['touch_date'][:4],
                'signal_type':s['signal_type'],'anchor_type':s['anchor_type'],'status':s['status'],
                'angle_combo':s['angle_combo'],'component_lines':s['component_lines'],
                'group_stocks':int(s['group_stocks']),
                'entry_close':round_or_none(entry),'target_price':round_or_none(target),
                'trade_status':'CLOSED_TP' if realized else 'OPEN',
                'exit_date':after[exit_index][0] if realized else '',
                'exit_price':round_or_none(target) if realized else '',
                'holding_sessions':end_index-idx,
                'last_close':round_or_none(last[2]), 'last_price_date':last[0],
                'open_return_pct':round_or_none(net_pct) if not realized else '',
                'trade_return_pct':round_or_none(net_pct),
                'max_high_after_entry':round_or_none(max_high),
                'max_return_pct':round_or_none((max_high/entry-1)*100) if max_high is not None else '',
                'entry_to_latest_pct':round_or_none((last[2]/entry-1)*100),
            }
            out.append(row)
        if counter%25==0:print(f'Processed {counter}/{len(grouped)} symbols...',flush=True)
    return out,skipped


def summary_rows(trades, fields):
    buckets=defaultdict(list)
    for row in trades:buckets[tuple(row[f] for f in fields)].append(row)
    result=[]
    for key,items in buckets.items():
        closed=[r for r in items if r['trade_status']=='CLOSED_TP']
        opened=[r for r in items if r['trade_status']=='OPEN']
        open_returns=[r['open_return_pct'] for r in opened]
        profits=[r['trade_return_pct'] for r in items]
        res={f:v for f,v in zip(fields,key)}
        res.update(trades=len(items),closed_tp=len(closed),open_count=len(opened),
                   tp_hit_rate_pct=round(100*len(closed)/len(items),2),
                   open_winning=sum(v>0 for v in open_returns),
                   open_losing=sum(v<0 for v in open_returns),
                   open_flat=sum(v==0 for v in open_returns),
                   avg_open_return_pct=round_or_none(sum(open_returns)/len(open_returns)) if opened else '',
                   total_open_return_pct=round_or_none(sum(open_returns)) if opened else '',
                   avg_trade_return_pct=round_or_none(sum(profits)/len(profits)),
                   total_trade_return_pct=round_or_none(sum(profits)),
                   avg_holding_closed_sessions=round_or_none(sum(r['holding_sessions'] for r in closed)/len(closed)) if closed else '',
                   unique_stocks=len({r['symbol'] for r in items}))
        result.append(res)
    return sorted(result,key=lambda x:(-x['trades'],str(tuple(x[f] for f in fields))))


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def print_report(header,rows, fields, limit=30):
    print('\n'+header)
    print('-'*115)
    for item in rows[:limit]:
        label=' | '.join(f'{FIELD_NAMES.get(f,f)}={item[f]}' for f in fields) if fields else 'الإجمالي'
        print(f'{label}: صفقات={item["trades"]:,} | +3% مغلقة={item["closed_tp"]:,} '
              f'| مفتوحة={item["open_count"]:,} | نسبة تحقيق الهدف={item["tp_hit_rate_pct"]:.1f}% '
              f'| متوسط أداء المفتوح={item["avg_open_return_pct"]}% | متوسط كل الصفقات={item["avg_trade_return_pct"]}%')
    if len(rows)>limit:print(f'... +{len(rows)-limit} rows in CSV')


def main():
    opts=args_parser()
    if opts.min_stocks<1:raise SystemExit('--min-stocks must be >=1')
    if opts.tp<=0:raise SystemExit('--tp must be positive')
    for value in (opts.start,opts.end):
        if value:date.fromisoformat(value)
    if opts.start and opts.end and opts.start>opts.end:raise SystemExit('--start > --end')
    url=os.environ.get('DATABASE_URL')
    if not url:raise SystemExit('Missing DATABASE_URL in Railway Console')
    engine=create_engine(url,pool_pre_ping=True,pool_size=1,max_overflow=0)
    try:
        with engine.connect() as con:
            run=selected_run(con,opts.run_id)
            print(f'Run {run["run_id"]}, market_day={run["market_day"]}, TP={opts.tp}%, min distinct stocks/group={opts.min_stocks}')
            print('Loading qualifying signals (READ ONLY)...',flush=True)
            signals=sql_signals(con,run['run_id'],opts)
            print(f'Qualified stock signals: {len(signals):,}',flush=True)
            trades,skipped=backtest(con,signals,opts.tp)
    finally:engine.dispose()
    outdir=Path(opts.output).expanduser().resolve();outdir.mkdir(parents=True,exist_ok=True)
    if not trades:
        print('No valid trades found.');return
    full_cols=list(trades[0]);write_csv(outdir/'trades.csv',trades,full_cols)
    dimensions=[
        ('overall.csv',()),
        ('by_signal_type.csv',('signal_type',)),
        ('by_low_high.csv',('anchor_type',)),
        ('by_status.csv',('status',)),
        ('by_angle_combo.csv',('angle_combo',)),
        ('by_signal_side_status_angles.csv',('signal_type','anchor_type','status','angle_combo')),
        ('by_day_and_signal.csv',('touch_date','signal_type','anchor_type','status')),
        ('by_year.csv',('signal_year',)),
    ]
    for filename,fields in dimensions:
        rows=summary_rows(trades,fields)
        cols=list(fields)+[k for k in rows[0] if k not in fields]
        write_csv(outdir/filename,rows,cols)
        if filename in ('overall.csv','by_signal_type.csv','by_low_high.csv','by_status.csv','by_angle_combo.csv','by_signal_side_status_angles.csv'):
            print_report(filename,rows,fields,limit=40 if filename=='by_signal_side_status_angles.csv' else 30)
    # Individual angle exposure, not unique position totals: a paired-line position
    # appears under each of its angles and can contribute to more than one angle group.
    expanded=[]
    for item in trades:
        for angle in item['angle_combo'].split('+'): expanded.append({**item,'angle':angle})
    rows=summary_rows(expanded,('angle',))
    write_csv(outdir/'by_individual_angle_nonexclusive.csv',rows,list(rows[0]))
    print_report('by_individual_angle_nonexclusive.csv (OVERLAPPING)',rows,('angle',))
    if skipped:
        write_csv(outdir/'skipped.csv',[{'symbol':s,'touch_date':d,'reason':r} for s,d,r in skipped],['symbol','touch_date','reason'])
    print(f'\nOutput: {outdir}')
    print(f'Valid trades={len(trades):,}, skipped={len(skipped):,}; no DB changes made.')

if __name__=='__main__':main()
