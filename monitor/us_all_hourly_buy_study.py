"""Independent all-US hourly Double-7/MATCH/BUY-only 3% study.

CLI: python -m monitor.us_all_hourly_buy_study --period 60d --output-dir /data/us-buy-study
Requires existing application's SQLAlchemy config and yfinance dependencies.
No S&P 500 tables, UI or daily data are changed.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
import time
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from sqlalchemy import text
from core import database
from monitor.seven_system import _single
from monitor.sp500_seven_hourly import _hourly_clusters, _rule
from monitor.sp500_seven_system import _repeat_direction_from_candle

TABLE = 'us_all_seven_hourly_bars'
UNIVERSE_TABLE = 'us_all_seven_hourly_universe'

def ensure_tables():
    with database().begin() as s:
        s.execute(text(f'''CREATE TABLE IF NOT EXISTS {TABLE} (
          symbol TEXT NOT NULL, bar_time TIMESTAMP NOT NULL, open DOUBLE PRECISION,
          high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION NOT NULL,
          volume BIGINT, PRIMARY KEY(symbol,bar_time))'''))
        s.execute(text(f'''CREATE TABLE IF NOT EXISTS {UNIVERSE_TABLE} (
          symbol TEXT PRIMARY KEY, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'''))

def universe(symbols_file=None):
    if symbols_file:
        p=Path(symbols_file)
        if p.suffix.lower()=='.csv':
            x=pd.read_csv(p)
            col=next((c for c in x.columns if c.lower() in ('symbol','ticker','stock_symbol')),x.columns[0])
            symbols=x[col].astype(str).tolist()
        else: symbols=p.read_text(encoding='utf-8-sig').splitlines()
    else:
        headers={'User-Agent':'Mozilla/5.0 (compatible; US-stock-research/1.0)'}
        sources=[('nasdaqlisted.txt','Symbol'),('otherlisted.txt','ACT Symbol')]
        symbols=[]
        for path,col in sources:
            url='https://www.nasdaqtrader.com/dynamic/SymDir/'+path
            r=requests.get(url,headers=headers,timeout=40);r.raise_for_status()
            t=pd.read_csv(io.StringIO(r.text),sep='|',dtype=str)
            if col not in t:raise ValueError(f'{path}: expected {col}; got {list(t.columns)}')
            for _,row in t.iterrows():
                if row.get('ETF')!='N' or row.get('Test Issue')!='N':continue
                symbols.append(str(row[col]))
    # US-listed common ticker shapes, ETFs/tests/warrants/units excluded by exchange feed above.
    out=[]
    for v in symbols:
        s=str(v).strip().upper()
        if re.fullmatch(r'[A-Z]{1,7}(?:[.][A-Z]{1,2})?',s):out.append(s)
    return sorted(set(out))

def _persist(symbols):
    with database().begin() as session:
        q=text(f'INSERT INTO {UNIVERSE_TABLE}(symbol) VALUES (:s) ON CONFLICT(symbol) DO NOTHING')
        session.execute(q,[{'s':s} for s in symbols])

def refresh(symbols, period='60d', batch_size=15, pause=0.2):
    import yfinance as yf
    from sqlalchemy import MetaData,Table
    from sqlalchemy.dialects.postgresql import insert
    ensure_tables();_persist(symbols)
    done=0;total_bars=0;errors=[]
    for start in range(0,len(symbols),batch_size):
        batch=symbols[start:start+batch_size]
        try:
            raw=yf.download(' '.join(s.replace('.','-') for s in batch),period=period,interval='1h',
                            auto_adjust=True,group_by='ticker',threads=True,progress=False)
        except Exception as e:
            errors.append({'batch':start,'error':str(e)});continue
        records=[]
        for sym in batch:
            feed=sym.replace('.','-')
            try:
                x=raw[feed] if isinstance(raw.columns,pd.MultiIndex) else (raw if len(batch)==1 else pd.DataFrame())
                if x is None or x.empty:continue
                x=x.rename(columns=str.lower)
                for ts,row in x.iterrows():
                    if pd.isna(row.get('close')):continue
                    when=pd.Timestamp(ts)
                    if when.tzinfo:when=when.tz_convert('UTC').tz_localize(None)
                    records.append({'symbol':sym,'bar_time':when.to_pydatetime(),
                      'open':float(row.open) if pd.notna(row.get('open')) else None,
                      'high':float(row.high) if pd.notna(row.get('high')) else None,
                      'low':float(row.low) if pd.notna(row.get('low')) else None,
                      'close':float(row.close),'volume':int(row.volume) if pd.notna(row.get('volume')) else None})
                done+=1
            except Exception as e:errors.append({'symbol':sym,'error':str(e)})
        if records:
            with database().begin() as sess:
                tb=Table(TABLE,MetaData(),autoload_with=sess.get_bind())
                for k in range(0,len(records),500):
                    q=insert(tb).values(records[k:k+500]);q=q.on_conflict_do_update(
                      index_elements=['symbol','bar_time'],set_={c:getattr(q.excluded,c) for c in ('open','high','low','close','volume')})
                    sess.execute(q)
            total_bars+=len(records)
        print(f'US hourly {min(start+len(batch),len(symbols))}/{len(symbols)} symbols; available={done}; saved={total_bars}; errors={len(errors)}',flush=True)
        if pause:time.sleep(pause)
    return {'requested_symbols':len(symbols),'symbols_with_data':done,'bars_saved':total_bars,'errors':errors[:100]}

def evaluate_symbol(sym,df,tolerance=1.0,max_gap=20,max_factor=7):
    df=df.sort_values('date').reset_index(drop=True)
    if len(df)<61:return []
    events=[]
    for factor in range(2, max_factor + 1):
        scored,_=_single(df,True,system_number=factor)
        clusters=_hourly_clusters(scored,tolerance,max_gap)
        blocked_until=-1
        # One open position per symbol: do not stack new entries while position is open.
        for cluster in clusters:
            lastidx=cluster[-1][0]
            gap,touches,higher,status=_rule(df,cluster)
            if status!='MATCH' or lastidx<=blocked_until:continue
            candle=df.iloc[lastidx].to_dict();candle['raw_close']=candle['close']
            direction,_=_repeat_direction_from_candle(candle)
            if direction!='BUY':continue
            entry=float(np.mean([x[2] for x in cluster]))
            if not np.isfinite(entry) or entry<=0:continue
            post=df.iloc[lastidx+1:]
            highs=pd.to_numeric(post.high,errors='coerce');valid=highs.notna() & (highs>0)
            if not valid.all():continue
            hits=np.flatnonzero((highs>=entry*1.03).to_numpy())
            exitidx=lastidx+1+int(hits[0]) if len(hits) else None
            last=float(df.iloc[-1].close)
            if exitidx is not None:blocked_until=exitidx
            else:blocked_until=len(df) # keep it open, ignore later repeats for this stock
            until=df.iloc[lastidx+1:exitidx+1] if exitidx is not None else post
            minlow=pd.to_numeric(until.low,errors='coerce').min() if len(until) else np.nan
            events.append({
                'symbol':sym,'squaring_factor':factor,'signal_utc':pd.Timestamp(df.iloc[lastidx].date).strftime('%Y-%m-%d %H:%M'),
                'entry_price':round(entry,5),'signal_direction':'BUY','gap_1h':gap,'touches':touches,
                'higher_lows_60h':higher,'status':'EXIT' if exitidx is not None else 'OPEN',
                'exit_utc':pd.Timestamp(df.iloc[exitidx].date).strftime('%Y-%m-%d %H:%M') if exitidx is not None else '',
                'bars_to_exit':exitidx-lastidx if exitidx is not None else '',
                'bars_observed':len(df)-lastidx-1,'last_price':round(last,5),
                'realized_return_pct':3.0 if exitidx is not None else 0.0,
                'floating_return_pct':round((last/entry-1)*100,4) if exitidx is None else 0.0,
                'combined_return_pct':3.0 if exitidx is not None else round((last/entry-1)*100,4),
                'max_adverse_pct':round((float(minlow)/entry-1)*100,4) if np.isfinite(minlow) else None,
                'data_end_utc':pd.Timestamp(df.iloc[-1].date).strftime('%Y-%m-%d %H:%M'),
                'repeat_peak_high':float(pd.to_numeric(post.high,errors='coerce').max()) if not post.empty else None,
                'repeat_peak_bars':int(np.nanargmax(pd.to_numeric(post.high,errors='coerce').to_numpy()))+1 if len(post) and pd.to_numeric(post.high,errors='coerce').notna().any() else None})
    return events

def analyze(output_dir,tolerance=1.0,max_gap=20,max_factor=7):
    ensure_tables();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];symbols_scanned=0;last_sym=None;bars=[]
    def drain(sym,rows):
        nonlocal symbols_scanned
        if not rows:return
        frame=pd.DataFrame(rows,columns=['date','open','high','low','close','volume'])
        for c in ('open','high','low','close','volume'):frame[c]=pd.to_numeric(frame[c],errors='coerce')
        frame['date']=pd.to_datetime(frame['date'],utc=True).dt.tz_localize(None)
        try:records.extend(evaluate_symbol(sym,frame,tolerance,max_gap,max_factor))
        except Exception as e:print(f'analysis error {sym}: {e}',flush=True)
        symbols_scanned+=1
        if symbols_scanned%250==0:print(f'Analyzed {symbols_scanned} symbols; trades={len(records)}',flush=True)
    with database()() as session:
        for r in session.execute(text(f'SELECT symbol,bar_time,open,high,low,close,volume FROM {TABLE} ORDER BY symbol,bar_time')):
            sym=r[0]
            if last_sym is not None and sym!=last_sym:drain(last_sym,bars);bars=[]
            last_sym=sym;bars.append(r[1:])
        if last_sym is not None:drain(last_sym,bars)
    cols=['symbol','squaring_factor','repeat_peak_high','repeat_peak_bars','signal_utc','entry_price','signal_direction','gap_1h','touches','higher_lows_60h','status',
          'exit_utc','bars_to_exit','bars_observed','last_price','realized_return_pct','floating_return_pct',
          'combined_return_pct','max_adverse_pct','data_end_utc']
    df=pd.DataFrame(records,columns=cols)
    df.to_csv(out/'us_buy_only_all_trades.csv',index=False,encoding='utf-8-sig')
    closed=df[df.status=='EXIT'] if not df.empty else df
    opened=df[df.status=='OPEN'] if not df.empty else df
    opened.to_csv(out/'us_buy_only_open.csv',index=False,encoding='utf-8-sig')
    closed.to_csv(out/'us_buy_only_closed.csv',index=False,encoding='utf-8-sig')
    summary={
      'universe':'US-listed Nasdaq/NYSE/NYSE American equities excluding ETFs and tests; or user symbols file',
      'factor_range':f'2..{max_factor}',
      'strategy':'Squaring factors 2..N hourly MATCH and BUY only, entry at Repeat Price, exit at first later High >= 1.03*entry; no stops; one position per stock at a time',
      'symbols_analyzed_with_bars':symbols_scanned,'total_signals':len(df),'closed_count':len(closed),
      'open_count':len(opened),'closed_rate_pct':round(100*len(closed)/len(df),2) if len(df) else None,
      'closed_total_realized_pct_sum':round(float(closed.realized_return_pct.sum()),3) if len(df) else 0,
      'open_floating_pct_sum':round(float(opened.floating_return_pct.sum()),3) if len(df) else 0,
      'all_trades_equal_weight_mean_pct':round(float(df.combined_return_pct.mean()),3) if len(df) else None,
      'avg_bars_to_close':round(float(pd.to_numeric(closed.bars_to_exit).mean()),2) if len(closed) else None,
      'median_bars_to_close':float(pd.to_numeric(closed.bars_to_exit).median()) if len(closed) else None,
      'worst_open_return_pct':round(float(opened.floating_return_pct.min()),2) if len(opened) else None,
      'note':'Hypothetical equal-size signal returns, not a funded/equity-curve backtest. Hourly Yahoo data period is limited; open results are right-censored. Executions, fees, spreads, borrowing and capital allocation are not modeled. Historical universe is not point-in-time.'}
    (out/'us_buy_only_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--symbols-file',help='Optional CSV/TXT of desired US tickers; otherwise NasdaqTrader live universe')
    p.add_argument('--period',default='60d');p.add_argument('--batch-size',type=int,default=15)
    p.add_argument('--limit',type=int,default=0,help='Optional small smoke-test limit; 0 means all')
    p.add_argument('--output-dir',default='/tmp/us-buy-study')
    p.add_argument('--skip-refresh',action='store_true')
    p.add_argument('--max-factor',type=int,default=7,help='Inclusive maximum squaring factor, 2..N')
    p.add_argument('--publish-db',action='store_true',help='Save report to database for web dashboard')
    p.add_argument('--tolerance',type=float,default=1.0);p.add_argument('--max-gap',type=int,default=20)
    args=p.parse_args()
    if not 2 <= args.max_factor <= 50: p.error('--max-factor must be 2..50 (higher factors are computationally expensive)')
    if not args.skip_refresh:
        syms=universe(args.symbols_file)
        if args.limit>0:syms=syms[:args.limit]
        print(f'Universe total={len(syms)}',flush=True)
        print('Download:',refresh(syms,args.period,args.batch_size),flush=True)
    print('Study:',json.dumps(analyze(args.output_dir,args.tolerance,args.max_gap,args.max_factor),ensure_ascii=False,indent=2),flush=True)
    print('CSV/JSON files saved under',args.output_dir,flush=True)
    if args.publish_db:
        from monitor.us_all_hourly_dashboard import publish_trades
        print('Published trades:',publish_trades(Path(args.output_dir)/'us_buy_only_all_trades.csv'),flush=True)
if __name__=='__main__':main()
