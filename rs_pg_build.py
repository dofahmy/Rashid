#!/usr/bin/env python3
"""Resumable market feature build in separate public.rs_* tables. Never modifies source market tables."""
import argparse, os, time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from six_mirror_batch import iter_db, clean, valid_symbol
from seven_rp_mirrors_90 import features
from rp_stage2_adx import indicators

WINDOWS=(1,5,10,20,60,120,252)
COLS=['symbol','session_date','close','benchmark_close','stock_day_pct','sp500_day_pct'] + [f'rp{x}' for x in WINDOWS]+[f'avg_rp{x}' for x in (5,10,20,60,120)]+['adx14','plus_di14','minus_di14','relative_volume20','relative_volatility14_60']
DDL='''
CREATE TABLE IF NOT EXISTS public.rs_sp500_benchmark(
 session_date DATE PRIMARY KEY,close DOUBLE PRECISION NOT NULL,source TEXT);
CREATE TABLE IF NOT EXISTS public.rs_screen_features(
 symbol TEXT NOT NULL,session_date DATE NOT NULL,close DOUBLE PRECISION,benchmark_close DOUBLE PRECISION,
 stock_day_pct DOUBLE PRECISION,sp500_day_pct DOUBLE PRECISION,
 rp1 DOUBLE PRECISION,rp5 DOUBLE PRECISION,rp10 DOUBLE PRECISION,rp20 DOUBLE PRECISION,
 rp60 DOUBLE PRECISION,rp120 DOUBLE PRECISION,rp252 DOUBLE PRECISION,
 avg_rp5 DOUBLE PRECISION,avg_rp10 DOUBLE PRECISION,avg_rp20 DOUBLE PRECISION,
 avg_rp60 DOUBLE PRECISION,avg_rp120 DOUBLE PRECISION,
 adx14 DOUBLE PRECISION,plus_di14 DOUBLE PRECISION,minus_di14 DOUBLE PRECISION,
 relative_volume20 DOUBLE PRECISION,relative_volatility14_60 DOUBLE PRECISION,
 PRIMARY KEY(symbol,session_date));
CREATE INDEX IF NOT EXISTS rs_features_date_idx ON public.rs_screen_features(session_date);
CREATE TABLE IF NOT EXISTS public.rs_screen_build_progress(
 symbol TEXT PRIMARY KEY, build_start DATE NOT NULL,build_end DATE NOT NULL,
 rows_written INTEGER NOT NULL,finished_at TIMESTAMPTZ DEFAULT NOW());
'''
def num(x):
 try:
  x=float(x)
  return x if np.isfinite(x) else None
 except (ValueError,TypeError):return None

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
 p.add_argument('--start',default='2024-01-01');p.add_argument('--end',default='2026-10-20')
 p.add_argument('--reset-progress',action='store_true',help='Rebuild all symbols, without dropping feature table')
 a=p.parse_args()
 if not a.database_url:raise SystemExit('DATABASE_URL missing')
 eng=create_engine(a.database_url,pool_pre_ping=True)
 with eng.begin() as cn:
  for statement in DDL.split(';'):
   if statement.strip():cn.execute(text(statement))
  # The index source already exists in PostgreSQL; do not re-download Yahoo data.
  src=cn.execute(text('SELECT count(*),min(session_date),max(session_date) FROM public.sp500_seven_index WHERE close>0')).one()
  print('SOURCE_INDEX',{'rows':src[0],'first':str(src[1]),'last':str(src[2])},flush=True)
  if src[0]<260:raise SystemExit('S&P benchmark has insufficient history; check sp500_seven_index')
  cn.execute(text('''INSERT INTO public.rs_sp500_benchmark(session_date,close,source)
   SELECT session_date,close,'sp500_seven_index' FROM public.sp500_seven_index
   WHERE close>0 ON CONFLICT(session_date) DO UPDATE SET close=EXCLUDED.close,source=EXCLUDED.source'''))
  if a.reset_progress:cn.execute(text('DELETE FROM public.rs_screen_build_progress'))
 with eng.connect() as cn:
  br=cn.execute(text('SELECT session_date,close FROM public.rs_sp500_benchmark ORDER BY session_date')).fetchall()
  done=set(cn.execute(text('SELECT symbol FROM public.rs_screen_build_progress WHERE build_start=:s AND build_end=:e'),{'s':a.start,'e':a.end}).scalars())
 bench=pd.DataFrame(br,columns=['date','c']);bench['date']=pd.to_datetime(bench['date'])
 bdates=bench.date.to_numpy(dtype='datetime64[ns]');bc=bench.c.to_numpy(float)
 print('BENCHMARK',len(bench),'RESUME_SYMBOLS',len(done),flush=True)
 nstocks=written=skipped=0
 for raw in iter_db(a.database_url):
  sym=str(raw.symbol.iloc[0]).strip().upper()
  if not valid_symbol(sym) or sym in done:skipped+=1;continue
  d=clean(raw)
  if len(d)<25:continue
  dates=pd.to_datetime(d.date).to_numpy(dtype='datetime64[ns]')
  c=d.c.to_numpy(float)
  pos=np.searchsorted(bdates,dates,side='right')-1
  b=np.full(len(d),np.nan);ok=pos>=0;b[ok]=bc[pos[ok]]
  rp=features(d,bench)
  stock1=np.full(len(d),np.nan);idx1=np.full(len(d),np.nan)
  stock1[1:]=100*(c[1:]/c[:-1]-1)
  idx1[1:]=100*(b[1:]/b[:-1]-1)
  daily=stock1-idx1
  avgs={w:pd.Series(daily).rolling(w,min_periods=w).mean().to_numpy() for w in (5,10,20,60,120)}
  ind=indicators(d)
  select=np.flatnonzero((dates>=np.datetime64(a.start))&(dates<=np.datetime64(a.end)))
  batch=[]
  for i in select:
   if not np.isfinite(b[i]):continue
   row=[sym,str(dates[i])[:10],num(c[i]),num(b[i]),num(stock1[i]),num(idx1[i])]
   row += [num(daily[i]) if w==1 else num(rp[w][i]) for w in WINDOWS]
   row += [num(avgs[w][i]) for w in (5,10,20,60,120)]
   row += [num(ind[k][i]) for k in ('ADX14','PLUS_DI14','MINUS_DI14','RELATIVE_VOLUME20','RELATIVE_VOLATILITY14_60')]
   batch.append(tuple(row))
  # SQLAlchemy-managed batch insert, compatible with the configured DBAPI driver.
  value_tokens=','.join(':'+col for col in COLS)
  sql=text(f"INSERT INTO public.rs_screen_features ({','.join(COLS)}) VALUES ({value_tokens}) "
      "ON CONFLICT(symbol,session_date) DO UPDATE SET " +
      ','.join(f'{col}=EXCLUDED.{col}' for col in COLS[2:]))
  # Features and progress are committed atomically per symbol.
  with eng.begin() as cn:
   for start_at in range(0,len(batch),500):
    chunk=batch[start_at:start_at+500]
    cn.execute(sql,[dict(zip(COLS,row)) for row in chunk])
   cn.execute(text("""INSERT INTO public.rs_screen_build_progress(symbol,build_start,build_end,rows_written)
      VALUES(:symbol,:start,:end,:rows) ON CONFLICT(symbol) DO UPDATE SET build_start=EXCLUDED.build_start,
      build_end=EXCLUDED.build_end,rows_written=EXCLUDED.rows_written,finished_at=NOW()"""),
      {'symbol':sym,'start':a.start,'end':a.end,'rows':len(batch)})
  nstocks+=1;written+=len(batch)
  if nstocks%100==0:print('STOCKS',nstocks,'ROWS_THIS_RUN',written,'SKIPPED',skipped,flush=True)
 with eng.connect() as cn:
  total=cn.execute(text('SELECT count(*) FROM public.rs_screen_features')).scalar()
  first,last=cn.execute(text('SELECT min(session_date),max(session_date) FROM public.rs_screen_features')).one()
 print('DONE',{'new_symbols':nstocks,'new_rows':written,'total_rows':total,'first':str(first),'last':str(last),'postgres_table':'public.rs_screen_features'},flush=True)
if __name__=='__main__':main()
