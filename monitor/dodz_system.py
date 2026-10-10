"""DODZ all-US hourly study: BUY, 19:30 UTC, gap 1..12, factors 2..19.

Entry: Repeat Price at a confirmed repeat signal (assumed executable; hypothetical).
Exit: first post-signal 1H High reaching +3%, otherwise close of bar 3*factor.
If neither event is observable in saved bars, position remains OPEN.
One position per symbol across all squaring factors; simultaneous signals are merged.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
from core import database
from monitor.seven_system import _single
from monitor.sp500_seven_hourly import _hourly_clusters, _rule
from monitor.sp500_seven_system import _repeat_direction_from_candle
from monitor.us_all_hourly_buy_study import TABLE as BARS, ensure_tables

TRADES = 'dodz_hourly_trades'

def ensure_table():
    with database().begin() as s:
        s.execute(text(f'''CREATE TABLE IF NOT EXISTS {TRADES} (
           symbol TEXT NOT NULL, signal_utc TIMESTAMP NOT NULL,
           squaring_factor INTEGER NOT NULL, factors TEXT NOT NULL,
           entry_price DOUBLE PRECISION NOT NULL, gap_1h INTEGER NOT NULL,
           touches INTEGER NOT NULL, deadline_bars INTEGER NOT NULL,
           status TEXT NOT NULL, exit_reason TEXT,
           exit_utc TIMESTAMP, exit_price DOUBLE PRECISION,
           bars_to_exit INTEGER, observed_bars INTEGER NOT NULL,
           last_price DOUBLE PRECISION, realized_return_pct DOUBLE PRECISION,
           floating_return_pct DOUBLE PRECISION, combined_return_pct DOUBLE PRECISION,
           max_adverse_pct DOUBLE PRECISION, data_end_utc TIMESTAMP,
           PRIMARY KEY(symbol, signal_utc))'''))

def signals_for_symbol(sym,df):
    if len(df)<61:return []
    df=df.sort_values('date').reset_index(drop=True)
    raw=[]
    for factor in range(2,20):
        scored,_=_single(df,True,system_number=factor)
        for cluster in _hourly_clusters(scored,1.0,12):
            idx=cluster[-1][0]
            gap,touches,higher,match=_rule(df,cluster)
            # The shared MATCH includes a hard-coded 11..20 gap.
            # DODZ overrides only that range; preserves touches=2 and rising 60H lows.
            if not (1<=int(gap)<=12 and touches==2 and higher is True):continue
            stamp=pd.Timestamp(df.iloc[idx].date)
            if stamp.strftime('%H:%M')!='19:30':continue
            candle=df.iloc[idx].to_dict();candle['raw_close']=candle['close']
            direction,_=_repeat_direction_from_candle(candle)
            if direction!='BUY':continue
            entry=float(np.mean([x[2] for x in cluster]))
            if not np.isfinite(entry) or entry<=0:continue
            raw.append(dict(idx=idx,factor=factor,entry=entry,gap=int(gap),touches=int(touches)))
    # Different factors can produce same symbol/time repeat; execute one trade only.
    by_time={}
    for s in raw:by_time.setdefault(s['idx'],[]).append(s)
    events=[]; blocked_until=-1
    for idx, candidates in sorted(by_time.items()):
        if idx<=blocked_until: continue
        primary=min(candidates,key=lambda e:e['factor'])
        factor=primary['factor']; entry=primary['entry']; limit=factor*3
        post=df.iloc[idx+1:]
        deadline=idx+limit
        exit_idx=None;exit_price=None;reason=None
        for j in range(idx+1,min(len(df),deadline+1)):
            row=df.iloc[j]
            if pd.notna(row.high) and float(row.high)>=entry*1.03:
                exit_idx=j;exit_price=entry*1.03;reason='TARGET_3';break
            if j==deadline and pd.notna(row.close):
                exit_idx=j;exit_price=float(row.close);reason='TIME';break
        if exit_idx is not None:blocked_until=exit_idx
        else:blocked_until=len(df)
        end=exit_idx if exit_idx is not None else len(df)-1
        observed=df.iloc[idx+1:end+1]
        minlow=pd.to_numeric(observed.low,errors='coerce').min() if len(observed) else np.nan
        last=float(df.iloc[-1].close)
        actual_ret=(exit_price/entry-1)*100 if exit_idx is not None else None
        floating=(last/entry-1)*100 if exit_idx is None else None
        events.append(dict(symbol=sym,signal_utc=pd.Timestamp(df.iloc[idx].date).to_pydatetime(),
           squaring_factor=factor,factors=','.join(str(c['factor']) for c in sorted(candidates,key=lambda c:c['factor'])),
           entry_price=round(entry,6),gap_1h=primary['gap'],touches=primary['touches'],deadline_bars=limit,
           status='EXIT' if exit_idx is not None else 'OPEN',exit_reason=reason,
           exit_utc=pd.Timestamp(df.iloc[exit_idx].date).to_pydatetime() if exit_idx is not None else None,
           exit_price=round(exit_price,6) if exit_price is not None else None,
           bars_to_exit=exit_idx-idx if exit_idx is not None else None,
           observed_bars=len(df)-idx-1,last_price=last,
           realized_return_pct=round(actual_ret,5) if actual_ret is not None else None,
           floating_return_pct=round(floating,5) if floating is not None else None,
           combined_return_pct=round(actual_ret if actual_ret is not None else floating,5),
           max_adverse_pct=round((float(minlow)/entry-1)*100,5) if np.isfinite(minlow) else None,
           data_end_utc=pd.Timestamp(df.iloc[-1].date).to_pydatetime()))
    return events

def publish_existing(csv_path):
    """Publish completed CSV from a prior scan, skipping the heavy market rescan."""
    path=Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f'نتائج دودز غير موجودة: {path}. شغّلي الفحص الكامل أولًا.')
    df=pd.read_csv(path, keep_default_na=True)
    fields=('symbol','signal_utc','squaring_factor','factors','entry_price','gap_1h','touches','deadline_bars','status','exit_reason','exit_utc','exit_price','bars_to_exit','observed_bars','last_price','realized_return_pct','floating_return_pct','combined_return_pct','max_adverse_pct','data_end_utc')
    if len(df) and (missing:=set(fields)-set(df.columns)):
        raise ValueError(f'CSV missing columns: {sorted(missing)}')
    date_fields={'signal_utc','exit_utc','data_end_utc'}
    integer_fields={'squaring_factor','gap_1h','touches','deadline_bars','bars_to_exit','observed_bars'}
    rows=[]
    for raw in df.to_dict('records'):
        item={}
        for key in fields:
            value=raw.get(key)
            if pd.isna(value): value=None
            elif key in date_fields: value=pd.Timestamp(value).to_pydatetime()
            elif key in integer_fields: value=int(value)
            elif key in {'symbol','factors','status','exit_reason'}: value=str(value)
            else: value=float(value)
            item[key]=value
        rows.append(item)
    ensure_table()
    with database().begin() as s:
        # Atomic replacement: either the full CSV is published or the old data remains.
        s.execute(text(f'DELETE FROM {TRADES}'))
        stmt=text(f'INSERT INTO {TRADES} ({",".join(fields)}) VALUES ({",".join(":"+f for f in fields)})')
        for i in range(0,len(rows),500):
            s.execute(stmt,rows[i:i+500])
    return {'published_trades':len(rows),'source':str(path)}

def run(output_dir, publish=False):
    ensure_tables(); out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    trades=[];last_sym=None;bars=[];count=0
    def flush(sym,buf):
        nonlocal count
        if not buf:return
        d=pd.DataFrame(buf,columns=['date','open','high','low','close','volume'])
        d['date']=pd.to_datetime(d.date,utc=True).dt.tz_localize(None)
        for c in ('open','high','low','close','volume'):d[c]=pd.to_numeric(d[c],errors='coerce')
        try:trades.extend(signals_for_symbol(sym,d))
        except Exception as e:print(f'DODZ error {sym}: {e}',flush=True)
        count+=1
        if count%250==0:print(f'DODZ scanned {count} symbols; trades={len(trades)}',flush=True)
    with database()() as s:
        for r in s.execute(text(f'SELECT symbol,bar_time,open,high,low,close,volume FROM {BARS} ORDER BY symbol,bar_time')):
            if last_sym is not None and r[0]!=last_sym:flush(last_sym,bars);bars=[]
            last_sym=r[0];bars.append(r[1:])
        if last_sym is not None:flush(last_sym,bars)
    df=pd.DataFrame(trades)
    df.to_csv(out/'dodz_all_trades.csv',index=False,encoding='utf-8-sig')
    summary=dict(symbols_scanned=count,total=len(trades),target=sum(t['exit_reason']=='TARGET_3' for t in trades),
        timed=sum(t['exit_reason']=='TIME' for t in trades),open=sum(t['status']=='OPEN' for t in trades),
        average_pct=round(float(df.combined_return_pct.mean()),4) if len(df) else None,
        note='Backtest uses historical prices and hypothetical fill at Repeat Price; fees/spreads/slippage excluded. Multiple simultaneous factors deduplicated using lowest factor.')
    (out/'dodz_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if publish:
        ensure_table()
        fields=('symbol','signal_utc','squaring_factor','factors','entry_price','gap_1h','touches','deadline_bars','status','exit_reason','exit_utc','exit_price','bars_to_exit','observed_bars','last_price','realized_return_pct','floating_return_pct','combined_return_pct','max_adverse_pct','data_end_utc')
        with database().begin() as s:
            s.execute(text(f'DELETE FROM {TRADES}'))
            stmt=text(f'INSERT INTO {TRADES} ({",".join(fields)}) VALUES ({",".join(":"+f for f in fields)})')
            for k in range(0,len(trades),500):s.execute(stmt,[{f:t.get(f) for f in fields} for t in trades[k:k+500]])
    return summary

def dashboard(status='ALL',factor_from=2,factor_to=19,days=None,page=1,symbol=''):
    ensure_table()
    factor_from=int(factor_from);factor_to=int(factor_to)
    if not 2<=factor_from<=factor_to<=19:raise ValueError('عوامل دودز من 2 إلى 19 فقط')
    if status not in ('ALL','OPEN','EXIT','TARGET_3','TIME'):status='ALL'
    page=max(1,int(page)); page_size=100
    wh="squaring_factor BETWEEN :f1 AND :f2 AND UPPER(symbol) LIKE :pattern"
    params=dict(f1=factor_from,f2=factor_to,pattern='%'+symbol.strip().upper()[:25]+'%',limit=page_size,offset=(page-1)*page_size)
    if status in ('OPEN','EXIT'):wh+=' AND status=:status';params['status']=status
    if status in ('TARGET_3','TIME'):wh+=' AND exit_reason=:status';params['status']=status
    if days:
        from datetime import timedelta
        with database()() as s:last=s.execute(text(f'SELECT MAX(data_end_utc) FROM {TRADES}')).scalar()
        if last:wh+=' AND signal_utc >= :cutoff';params['cutoff']=last-timedelta(days=int(days))
    with database()() as s:
        stats=s.execute(text(f'''SELECT COUNT(*) AS total,COUNT(*) FILTER(WHERE exit_reason='TARGET_3') AS target,
          COUNT(*) FILTER(WHERE exit_reason='TIME') AS timed,COUNT(*) FILTER(WHERE status='OPEN') AS opened,
          AVG(combined_return_pct) AS avg_return,AVG(combined_return_pct) FILTER(WHERE status='EXIT') AS avg_closed
          FROM {TRADES} WHERE {wh}'''),params).mappings().one()
        rows=s.execute(text(f'''SELECT * FROM {TRADES} WHERE {wh} ORDER BY signal_utc DESC,symbol
                         LIMIT :limit OFFSET :offset'''),params).mappings().all()
    return dict(rows=rows,stats=dict(stats),page=page,pages=max(1,(stats['total']+page_size-1)//page_size),
       status=status,factor_from=factor_from,factor_to=factor_to,days=days,symbol=symbol)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='/tmp/dodz-study');p.add_argument('--publish-db',action='store_true')
    p.add_argument('--publish-existing',action='store_true',help='Publish saved dodz_all_trades.csv without rescanning')
    args=p.parse_args()
    result=(publish_existing(Path(args.output_dir)/'dodz_all_trades.csv') if args.publish_existing
            else run(args.output_dir,args.publish_db))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
