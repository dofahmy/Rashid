#!/usr/bin/env python3
"""Find literal ATR14% matches to YDKG pre-signal and signal day; evaluate 30 future sessions.
Uses the same ATR definition as breakout100_discovery.py: simple mean(TR[-14:])/close*100.
No percentile, range, tolerance, rounding or candidate future leakage.
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def valid_symbol(s):
    s=str(s).upper().strip()
    return bool(re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?',s)) and not s.endswith(('.CA','.SR'))

def clean(group):
    d=group.copy()
    d['date']=pd.to_datetime(d['date'],errors='coerce').dt.strftime('%Y-%m-%d')
    for col in 'ohlcv': d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['date','o','h','l','c','v'])
    d=d[(d.o>0)&(d.c>0)&(d.l>0)&(d.v>=0)&(d.h>=d[['o','c','l']].max(axis=1))&(d.l<=d[['o','c','h']].min(axis=1))]
    return d.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)

def atr_series(d):
    c=d.c.to_numpy(dtype=float);h=d.h.to_numpy(dtype=float);l=d.l.to_numpy(dtype=float)
    prev=np.r_[c[0],c[:-1]]
    tr=np.maximum.reduce((h-l,np.abs(h-prev),np.abs(l-prev)))
    atr=np.full(len(d),np.nan)
    # 14 exact daily true ranges; same definition as previous discovery engine
    if len(d)>=14:
        atr[13:]=(np.cumsum(np.r_[0.,tr])[14:]-np.cumsum(np.r_[0.,tr])[:-14])/14 / c[13:]*100
        # Recalculate each ATR with numpy mean to preserve exact evaluation order
        # of breakout100_discovery.feature_at, so literal comparison is meaningful.
        atr[13:]=np.array([np.mean(tr[i-13:i+1])/c[i]*100 for i in range(13,len(d))])
    return atr

def iter_db(url,chunksize=100000):
    from sqlalchemy import create_engine,text
    engine=create_engine(url.replace('postgres://','postgresql://',1))
    try:
        with engine.connect().execution_options(stream_results=True) as conn:
            q=text("SELECT symbol, session_date AS date, o,h,l,c,v FROM market_candles_1d "
                   "WHERE symbol !~ '\\.(CA|SR)$' ORDER BY symbol, session_date")
            carry=pd.DataFrame()
            for chunk in pd.read_sql(q,conn,chunksize=chunksize):
                d=pd.concat([carry,chunk],ignore_index=True)
                last=d.symbol.iloc[-1]
                full=d[d.symbol!=last];carry=d[d.symbol==last].copy()
                for _,g in full.groupby('symbol',sort=False):yield g
            if not carry.empty:yield carry
    finally:engine.dispose()

def iter_csv(path):
    d=pd.read_csv(path)
    d=d.rename(columns={'session_date':'date','open':'o','high':'h','low':'l','close':'c','volume':'v'})
    for _,g in d.groupby('symbol',sort=False):yield g

def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--database-url');source.add_argument('--csv')
    p.add_argument('--out',default='/data/ydkg_atr_exact')
    p.add_argument('--reference-date',default='2025-08-22')
    p.add_argument('--start',default='2024-01-01')
    p.add_argument('--end',default='2026-06-30')
    p.add_argument('--horizon',type=int,default=30)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.horizon<1:raise SystemExit('horizon must be >= 1')
    # Extract reference first, without assuming immediately prior calendar day was traded.
    reference=None
    groups=iter_db(a.database_url) if a.database_url else iter_csv(a.csv)
    for group in groups:
        symbol=str(group.symbol.iloc[0]).upper().strip()
        if symbol!='YDKG':continue
        d=clean(group)
        indices=np.flatnonzero(d.date.to_numpy()==a.reference_date)
        if not len(indices):raise SystemExit('YDKG reference date not available in valid daily candles')
        i=int(indices[0]);atr=atr_series(d)
        if i<14:raise SystemExit('YDKG has insufficient pre-signal ATR history')
        reference={'symbol':'YDKG','reference_date':a.reference_date,'previous_date':str(d.date.iloc[i-1]),
                   'atr14_pct_previous':float(atr[i-1]),'atr14_pct_signal':float(atr[i]),
                   'calculation':'mean(last 14 true ranges) / daily close * 100; numpy float64; literal ==',
                   'horizon':a.horizon}
        break
    if reference is None:raise SystemExit('YDKG not present in database')
    print('REFERENCE',json.dumps(reference,ensure_ascii=False),flush=True)
    (out/'reference.json').write_text(json.dumps(reference,indent=2,ensure_ascii=False))
    results=[];studied=0;valid=0;incomplete=0
    groups=iter_db(a.database_url) if a.database_url else iter_csv(a.csv)
    for group in groups:
        symbol=str(group.symbol.iloc[0]).upper().strip()
        if not valid_symbol(symbol):continue
        studied+=1;d=clean(group)
        if len(d)<14+a.horizon:continue
        atr=atr_series(d);dates=d.date.to_numpy();opens=d.o.to_numpy(float);closes=d.c.to_numpy(float)
        highs=d.h.to_numpy(float);lows=d.l.to_numpy(float)
        for i in range(14,len(d)-a.horizon):
            dt=str(dates[i])
            if dt<a.start or dt>a.end:continue
            valid+=1
            prev_match=bool(atr[i-1]==reference['atr14_pct_previous'])
            day_match=bool(atr[i]==reference['atr14_pct_signal'])
            if not (prev_match or day_match):continue
            b=closes[i]
            future_highs=highs[i+1:i+a.horizon+1];future_lows=lows[i+1:i+a.horizon+1]
            if len(future_highs)!=a.horizon:incomplete+=1;continue
            max_hi=float(np.max(future_highs)); min_lo=float(np.min(future_lows))
            max_pct=(max_hi/b-1)*100
            hits100=np.flatnonzero(future_highs>=2*b)
            results.append({'symbol':symbol,'signal_date':dt,'prev_date':str(dates[i-1]),
              'previous_atr14_pct':repr(float(atr[i-1])),'signal_atr14_pct':repr(float(atr[i])),
              'matches_prev_exact':prev_match,'matches_day_exact':day_match,'matches_both_exact':prev_match and day_match,
              'entry_close':b,'entry_open':opens[i],
              'high30':max_hi,'max_high_return_30_pct':round(max_pct,6),
              'close30':float(closes[i+a.horizon]),'close_return_30_pct':round((closes[i+a.horizon]/b-1)*100,6),
              'worst_low_return_30_pct':round((min_lo/b-1)*100,6),
              'hit100_within30':bool(len(hits100)),
              'first_hit100_session':int(hits100[0]+1) if len(hits100) else None,
              'first_max_high_session':int(np.argmax(future_highs)+1)})
        if studied%500==0:print('PROGRESS',studied,'symbols, matching rows',len(results),flush=True)
    columns=['symbol','signal_date','prev_date','previous_atr14_pct','signal_atr14_pct',
              'matches_prev_exact','matches_day_exact','matches_both_exact',
              'entry_close','entry_open','high30','max_high_return_30_pct','close30','close_return_30_pct',
              'worst_low_return_30_pct','hit100_within30','first_hit100_session','first_max_high_session']
    frame=pd.DataFrame(results,columns=columns)
    frame.to_csv(out/'YDKG_ATR_exact_matches_30sessions.csv',index=False)
    summary=[]
    for tag,key in [('previous','matches_prev_exact'),('signal_day','matches_day_exact'),('both','matches_both_exact')]:
        part=frame[frame[key]]
        part.to_csv(out/f'YDKG_ATR_{tag}_exact_30sessions.csv',index=False)
        summary.append({'test':tag,'matched_stock_dates':len(part),
                        'unique_symbols':int(part.symbol.nunique()),
                        'hit100_30':int(part.hit100_within30.sum()),
                        'mean_max_high_30_pct':float(part.max_high_return_30_pct.mean()) if len(part) else None})
    report={'reference':reference,'symbols_processed':studied,'stock_dates_checked':valid,
            'results':summary,'note':'Only full 30-session follow-ups. Exact floating-point equality; no rounding or range. ' 
             'Each matching stock/date is a case, not a unique stock. 30-session High is an ex-post excursion, not execution.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
