#!/usr/bin/env python3
"""Audit and optionally persist Signal -> Base -> Breakout confirmations (Egypt).

Reads the existing EGX30 and EGXALL indicator tables and market OHLCV.
Writes ONLY public.eg_signal_base_confirmation when --write is supplied.
Uses chronological closes; no future information is used to date confirmations.
"""
import argparse
import os
from collections import defaultdict
from datetime import date
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DDL = '''CREATE TABLE IF NOT EXISTS public.eg_signal_base_confirmation (
 benchmark TEXT NOT NULL, symbol TEXT NOT NULL, signal_date DATE NOT NULL,
 signal_close DOUBLE PRECISION NOT NULL,
 status TEXT NOT NULL, confirmation_date DATE,
 confirmation_close DOUBLE PRECISION,
 wait_sessions INTEGER, base_low DOUBLE PRECISION,
 base_high DOUBLE PRECISION, base_close_in_band_pct DOUBLE PRECISION,
 confirmed_gain_from_signal_pct DOUBLE PRECISION,
 method TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(benchmark,symbol,signal_date))'''
METHOD='base15_band20_close70_breakout1_prior10_v1'

def run():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--start',default='2024-01-01')
 p.add_argument('--end',default='2026-10-20')
 p.add_argument('--min-base',type=int,default=15)
 p.add_argument('--max-wait',type=int,default=120)
 p.add_argument('--write',action='store_true')
 a=p.parse_args()
 if a.min_base<10 or a.max_wait<=a.min_base: p.error('min-base >=10 and max-wait > min-base required')
 if date.fromisoformat(a.start)>date.fromisoformat(a.end):p.error('start must be <= end')
 db=os.getenv('DATABASE_URL')
 if not db: p.error('DATABASE_URL missing')
 eng=create_engine(db,pool_pre_ping=True)
 filters='avg_rp5 >= 2 AND plus_di14 <= 30 AND relative_volume20 >= 5'
 with eng.connect() as cn:
  if cn.execute(text("SELECT to_regclass('public.rs_egxall_features')")).scalar() is None:
   p.error('public.rs_egxall_features not found')
  sources={}
  sources['EGX30']=pd.read_sql(text(f'''SELECT symbol,session_date,close FROM public.rs_market_features
   WHERE market='EG' AND session_date BETWEEN :start AND :end AND {filters}'''),cn,params={'start':a.start,'end':a.end})
  sources['EGXALL']=pd.read_sql(text(f'''SELECT a.symbol,a.session_date,a.close
   FROM public.rs_egxall_features a
   JOIN public.rs_market_features f ON f.market='EG' AND f.symbol=a.symbol AND f.session_date=a.session_date
   WHERE a.session_date BETWEEN :start AND :end
   AND a.avg_rp5 >= 2 AND f.plus_di14 <= 30 AND f.relative_volume20 >= 5'''),cn,params={'start':a.start,'end':a.end})
  for k,frame in sources.items():print('SIGNALS',k,len(frame),flush=True)
  all_symbols=sorted(set().union(*(set(df.symbol) for df in sources.values())))
  prices={}
  for n,sym in enumerate(all_symbols,1):
   frame=pd.read_sql(text('''SELECT session_date,c FROM public.market_candles_1d
    WHERE symbol=:symbol AND session_date>=:start AND session_date<=:end ORDER BY session_date'''),cn,
    params={'symbol':sym,'start':a.start,'end':a.end})
   frame['session_date']=pd.to_datetime(frame.session_date)
   frame['c']=pd.to_numeric(frame.c,errors='coerce')
   frame=frame.dropna(subset=['session_date','c']).drop_duplicates('session_date').sort_values('session_date')
   frame=frame[frame.c>0]
   prices[sym]=(frame.session_date.dt.date.to_numpy(),frame.c.to_numpy(float))
   if n%50==0:print('PRICES_LOADED',n,flush=True)
 records=[];statuses=defaultdict(int)
 for bench,signals in sources.items():
  for record in signals.itertuples(index=False):
   dates,close=prices[record.symbol]
   sdate=pd.Timestamp(record.session_date).date()
   at=np.searchsorted(dates,sdate)
   if at>=len(dates) or dates[at]!=sdate:
    statuses['MISSING_SIGNAL_CLOSE']+=1;continue
   anchor=float(close[at]);last=min(len(close)-1,at+a.max_wait)
   result={'benchmark':bench,'symbol':record.symbol,'signal_date':sdate,'signal_close':anchor,
           'status':'PENDING','confirmation_date':None,'confirmation_close':None,
           'wait_sessions':None,'base_low':None,'base_high':None,'base_close_in_band_pct':None,
           'confirmed_gain_from_signal_pct':None,'method':METHOD}
   for j in range(at+a.min_base+1,last+1):
    # The consolidation base ends BEFORE the candidate breakout session.
    base=close[at+1:j]
    if len(base)<a.min_base:continue
    base_low=float(np.min(base));base_high=float(np.max(base))
    inside=float(np.mean((base>=anchor*.80)&(base<=anchor*1.20))*100)
    if inside<70:continue
    prior=close[max(at+1,j-10):j]
    if len(prior)<10:continue
    # First qualifying close: > signal close by 1% AND > past 10 closes by 1%.
    if close[j] > anchor*1.01 and close[j] > np.max(prior)*1.01:
     result.update(status='CONFIRMED',confirmation_date=dates[j],confirmation_close=float(close[j]),
                   wait_sessions=int(j-at),base_low=base_low,base_high=base_high,
                   base_close_in_band_pct=inside,confirmed_gain_from_signal_pct=float((close[j]/anchor-1)*100))
     break
   if result['status']=='PENDING' and last==len(close)-1 and len(close)-1-at<a.max_wait:
    result['status']='OPEN_NOT_YET_CONFIRMED'
   elif result['status']=='PENDING':result['status']='NOT_CONFIRMED_IN_WINDOW'
   records.append(result);statuses[(bench,result['status'])]+=1
 print('OUTPUT_ROWS',len(records),'STATUSES',dict(statuses),flush=True)
 example=[r for r in records if r['symbol']=='UNIT.CA' and r['signal_date']==date(2026,2,10)]
 print('UNIT_FEB_10',example,flush=True)
 if a.write:
  with eng.begin() as cn:
   cn.execute(text(DDL))
   statement=text('''INSERT INTO public.eg_signal_base_confirmation
    (benchmark,symbol,signal_date,signal_close,status,confirmation_date,confirmation_close,
     wait_sessions,base_low,base_high,base_close_in_band_pct,confirmed_gain_from_signal_pct,method)
    VALUES(:benchmark,:symbol,:signal_date,:signal_close,:status,:confirmation_date,:confirmation_close,
     :wait_sessions,:base_low,:base_high,:base_close_in_band_pct,:confirmed_gain_from_signal_pct,:method)
    ON CONFLICT(benchmark,symbol,signal_date) DO UPDATE SET
    signal_close=EXCLUDED.signal_close,status=EXCLUDED.status,
    confirmation_date=EXCLUDED.confirmation_date,confirmation_close=EXCLUDED.confirmation_close,
    wait_sessions=EXCLUDED.wait_sessions,base_low=EXCLUDED.base_low,
    base_high=EXCLUDED.base_high,base_close_in_band_pct=EXCLUDED.base_close_in_band_pct,
    confirmed_gain_from_signal_pct=EXCLUDED.confirmed_gain_from_signal_pct,
    method=EXCLUDED.method,updated_at=now()''')
   for i in range(0,len(records),250):cn.execute(statement,records[i:i+250])
  print('ROWS_UPSERTED',len(records))
 else:print('AUDIT_ONLY_NO_DATABASE_CHANGES')

if __name__=='__main__':run()
