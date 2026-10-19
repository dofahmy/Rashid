#!/usr/bin/env python3
"""Audit every saved YDKG compound signal through the newest DB candle, plus TP-specific nonoverlap.
Read-only database. Does not recompute/replace the original historical signal set.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine,text,bindparam

def p(x):return round(float(x),4) if np.isfinite(x) else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    ap.add_argument('--signals',default='/data/ydkg_market_validation/compound_passed_all.csv')
    ap.add_argument('--out',default='/data/ydkg_full_performance')
    ap.add_argument('--horizon',type=int,default=30)
    ap.add_argument('--targets',default='10,20,30,50,75,100,137.66')
    a=ap.parse_args()
    if not a.database_url:ap.error('DATABASE_URL required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sig=pd.read_csv(a.signals,dtype={'symbol':str,'signal_date':str})
    for col in ('symbol','signal_date','entry_close'):
        if col not in sig:ap.error('Missing input column '+col)
    sig['symbol']=sig.symbol.str.upper().str.strip()
    sig['signal_date']=sig.signal_date.str[:10]
    sig=sig.sort_values(['symbol','signal_date']).reset_index(drop=True)
    symbols=sorted(sig.symbol.unique().tolist())
    engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
    sql=text('SELECT symbol, session_date AS date, o,h,l,c,v FROM market_candles_1d WHERE symbol IN :symbols ORDER BY symbol,session_date').bindparams(bindparam('symbols',expanding=True))
    with engine.connect() as conn:
        candles=pd.read_sql(sql,conn,params={'symbols':symbols})
    engine.dispose()
    if len(candles)==0:raise RuntimeError('No DB candles returned for saved symbols')
    candles['date']=pd.to_datetime(candles.date).dt.strftime('%Y-%m-%d')
    for col in ('o','h','l','c','v'):candles[col]=pd.to_numeric(candles[col],errors='coerce')
    candles=candles.dropna(subset=['date','o','h','l','c'])
    candles=candles[(candles.o>0)&(candles.h>0)&(candles.l>0)&(candles.c>0)&(candles.h>=candles[['o','c','l']].max(axis=1))&(candles.l<=candles[['o','c','h']].min(axis=1))]
    candles=candles.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')
    groups={s:g.reset_index(drop=True) for s,g in candles.groupby('symbol')}
    rows=[]; missing=[]
    for r in sig.itertuples(index=False):
        sym=r.symbol; g=groups.get(sym)
        if g is None:missing.append({'symbol':sym,'signal_date':r.signal_date,'reason':'no_candles'});continue
        ix=g.index[g.date.eq(r.signal_date)]
        if len(ix)==0:missing.append({'symbol':sym,'signal_date':r.signal_date,'reason':'signal_date_not_found'});continue
        i=int(ix[0]); entry=float(g.c.iloc[i]); latest=g.iloc[-1]; follow=g.iloc[i+1:]
        if not len(follow):missing.append({'symbol':sym,'signal_date':r.signal_date,'reason':'no_followup'});continue
        first30=follow.iloc[:a.horizon]
        high30=float(first30.h.max()); worst30=float(first30.l.min())
        highall=float(follow.h.max()); lowall=float(follow.l.min())
        row={'symbol':sym,'signal_date':r.signal_date,'entry_close':p(entry),
             'db_latest_date':str(latest.date),'db_latest_close':p(latest.c),
             'sessions_to_latest':len(follow),'pct_latest_close_from_signal':p((latest.c/entry-1)*100),
             'pct_max_high_first30':p((high30/entry-1)*100),'pct_worst_low_first30':p((worst30/entry-1)*100),
             'pct_close_session30':p((first30.c.iloc[-1]/entry-1)*100) if len(first30)==a.horizon else None,
             'pct_max_high_through_latest':p((highall/entry-1)*100),
             'date_max_high_through_latest':str(follow.iloc[int(np.argmax(follow.h.to_numpy()))].date),
             'pct_worst_low_through_latest':p((lowall/entry-1)*100),
             'sessions_observed_first30':len(first30),
             'historical_30_complete':len(first30)==a.horizon,
             'historical_100_hit_first30':bool(np.any(first30.h.to_numpy()>=entry*2)),
             'branch1':getattr(r,'branch1',None),'branch2':getattr(r,'branch2',None),'branch3':getattr(r,'branch3',None)}
        for n in (1,3,5,10,15,20,30):
            v=follow.iloc[n-1] if len(follow)>=n else None
            row[f'close_day{n}_return_pct']=p((float(v.c)/entry-1)*100) if v is not None else None
            row[f'best_high_by_day{n}_pct']=p((float(follow.h.iloc[:n].max())/entry-1)*100) if v is not None else None
        rows.append(row)
    perf=pd.DataFrame(rows)
    perf.sort_values(['symbol','signal_date']).to_csv(out/'YDKG_all_signal_dates_through_latest.csv',index=False,encoding='utf-8-sig')
    # Each target has its own lockout: original signals inside active trade are blocked.
    targets=[float(t.strip()) for t in a.targets.split(',') if t.strip()]
    events=[]
    for target in targets:
        for sym, ss in perf.groupby('symbol'):
            g=groups[sym]; dates=g.date.to_numpy()
            lock_until='0000-00-00'
            for r in ss.sort_values('signal_date').itertuples(index=False):
                i=int(np.flatnonzero(dates==r.signal_date)[0]);entry=float(g.c.iloc[i]);win=g.iloc[i+1:i+1+a.horizon]
                if r.signal_date<=lock_until:
                    events.append({'target_pct':target,'symbol':sym,'signal_date':r.signal_date,'status':'BLOCKED_ACTIVE_TRADE','entry_close':p(entry),'exit_date':None,'exit_price':None,'return_pct':None,'sessions_held':None,'db_latest_date':r.db_latest_date})
                    continue
                if len(win)==0: continue
                touches=np.flatnonzero(win.h.to_numpy()>=entry*(1+target/100))
                if len(touches):
                    held=int(touches[0]+1);exit_date=str(win.date.iloc[held-1]);exit_price=entry*(1+target/100);status='TARGET_HIT'
                elif len(win)==a.horizon:
                    held=a.horizon;exit_date=str(win.date.iloc[-1]);exit_price=float(win.c.iloc[-1]);status='TIME_EXIT_30'
                else:
                    held=len(win);exit_date=None;exit_price=None;status='STILL_OPEN_END_DB'
                lock_until=exit_date if exit_date else '9999-12-31'
                events.append({'target_pct':target,'symbol':sym,'signal_date':r.signal_date,'status':status,'entry_close':p(entry),'exit_date':exit_date,'exit_price':p(exit_price) if exit_price is not None else None,'return_pct':p((exit_price/entry-1)*100) if exit_price is not None else None,'sessions_held':held,'db_latest_date':r.db_latest_date})
    ev=pd.DataFrame(events).sort_values(['target_pct','symbol','signal_date'])
    ev.to_csv(out/'YDKG_target_trades_and_blocked.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for t,e in ev.groupby('target_pct'):
        trades=e[e.status!='BLOCKED_ACTIVE_TRADE']; closed=trades[trades.status!='STILL_OPEN_END_DB']
        summary.append({'target_pct':t,'signals':len(e),'blocked':int((e.status=='BLOCKED_ACTIVE_TRADE').sum()),'trades':len(trades),'target_hits':int((e.status=='TARGET_HIT').sum()),'time_exits':int((e.status=='TIME_EXIT_30').sum()),'still_open_db_end':int((e.status=='STILL_OPEN_END_DB').sum()),'mean_closed_return_pct':p(closed.return_pct.mean()) if len(closed) else None,'median_closed_return_pct':p(closed.return_pct.median()) if len(closed) else None})
    pd.DataFrame(summary).to_csv(out/'YDKG_target_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(missing,columns=['symbol','signal_date','reason']).to_csv(out/'YDKG_missing_signals.csv',index=False)
    report={'input_signals':len(sig),'signal_rows_recovered':len(perf),'distinct_symbols':int(perf.symbol.nunique()),'latest_candle_date_max':str(perf.db_latest_date.max()),'latest_candle_date_min':str(perf.db_latest_date.min()),'missing':len(missing),'target_scenarios':summary,'source_file':a.signals,'notes':['Today means latest candle actually stored for each stock in PostgreSQL, NOT an external live price.','Signals are fixed to the original 2024-06/2026 list; no post-2026-06-30 signals discovered.','Return to latest uses the original signal-close anchor even after simulated trade exits; it is price tracking, not realized PnL.','Intraday High target touch approximates limit fill; splits and adjustment errors may severely distort performance.','Trading-session counts, not calendar days.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({'status':'complete','files':list(sorted(x.name for x in out.iterdir())),'input_signals':len(sig),'rows':len(perf),'latest_db_date':str(perf.db_latest_date.max()),'missing':len(missing)},indent=2),flush=True)

if __name__=='__main__':main()
