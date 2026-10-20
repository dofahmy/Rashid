#!/usr/bin/env python3
"""Build full-market, date-indexed, point-in-time RS/technical feature store. No future outcomes."""
import argparse, os, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd
from six_mirror_batch import iter_db, clean, valid_symbol
from seven_rp_mirrors_90 import load_benchmark, features
from rp_stage2_adx import indicators

WINDOWS=(1,5,10,20,60,120,252)
SCHEMA="""CREATE TABLE screen (symbol TEXT NOT NULL, session_date TEXT NOT NULL, close REAL,
benchmark_close REAL, stock_day_pct REAL, sp500_day_pct REAL,
rp1 REAL,rp5 REAL,rp10 REAL,rp20 REAL,rp60 REAL,rp120 REAL,rp252 REAL,
avg_rp5 REAL,avg_rp10 REAL,avg_rp20 REAL,avg_rp60 REAL,avg_rp120 REAL,
adx14 REAL,plus_di14 REAL,minus_di14 REAL,relative_volume20 REAL,relative_volatility14_60 REAL,
PRIMARY KEY (symbol,session_date))"""
COLS=['symbol','session_date','close','benchmark_close','stock_day_pct','sp500_day_pct'] + [f'rp{x}' for x in WINDOWS] + [f'avg_rp{x}' for x in WINDOWS if x!=1 and x!=252] + ['adx14','plus_di14','minus_di14','relative_volume20','relative_volatility14_60']

def num(x):
 try:
  x=float(x)
  return x if np.isfinite(x) else None
 except (ValueError,TypeError):return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
 p.add_argument('--benchmark-csv',default='/data/seven_relative_sp500/sp500_daily.csv')
 p.add_argument('--out',default='/data/rs_dynamic/market_features.sqlite')
 p.add_argument('--start',default='2024-01-01');p.add_argument('--end',default='2026-10-20')
 a=p.parse_args()
 if not a.database_url:raise SystemExit('DATABASE_URL required')
 bench=load_benchmark(a.benchmark_csv)
 dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix('.building.sqlite');tmp.unlink(missing_ok=True)
 conn=sqlite3.connect(tmp);conn.execute('PRAGMA journal_mode=DELETE');conn.execute('PRAGMA synchronous=OFF');conn.execute(SCHEMA)
 ins=f"INSERT INTO screen ({','.join(COLS)}) VALUES ({','.join(['?']*len(COLS))})"
 bdates=bench.date.to_numpy(dtype='datetime64[ns]');bc=bench.c.to_numpy(float)
 nstocks=nrows=0
 try:
  for raw in iter_db(a.database_url):
   sym=str(raw.symbol.iloc[0]).strip().upper()
   if not valid_symbol(sym):continue
   d=clean(raw);nstocks+=1
   if len(d)<25:continue
   dates=pd.to_datetime(d.date).to_numpy(dtype='datetime64[ns]')
   c=d.c.to_numpy(float);pos=np.searchsorted(bdates,dates,side='right')-1
   b=np.full(len(d),np.nan);good=pos>=0;b[good]=bc[pos[good]]
   rp=features(d,bench)
   stock1=np.full(len(d),np.nan);idx1=np.full(len(d),np.nan)
   stock1[1:]=100*(c[1:]/c[:-1]-1)
   idx1[1:]=100*(b[1:]/b[:-1]-1)
   daily=stock1-idx1
   avgs={w:pd.Series(daily).rolling(w,min_periods=w).mean().to_numpy() for w in (5,10,20,60,120)}
   ind=indicators(d)
   selection=np.flatnonzero((dates>=np.datetime64(a.start))&(dates<=np.datetime64(a.end)))
   batch=[]
   for i in selection:
    if not np.isfinite(b[i]):continue
    row=[sym,str(dates[i])[:10],num(c[i]),num(b[i]),num(stock1[i]),num(idx1[i])]
    row += [num(daily[i]) if w==1 else num(rp[w][i]) for w in WINDOWS]
    row += [num(avgs[w][i]) for w in (5,10,20,60,120)]
    row += [num(ind[x][i]) for x in ('ADX14','PLUS_DI14','MINUS_DI14','RELATIVE_VOLUME20','RELATIVE_VOLATILITY14_60')]
    batch.append(row)
   if batch:conn.executemany(ins,batch);nrows+=len(batch)
   if nstocks%200==0:conn.commit();print('STOCKS',nstocks,'ROWS',nrows,flush=True)
  conn.execute('CREATE INDEX idx_date ON screen(session_date)');conn.commit()
 finally:conn.close()
 os.replace(tmp,dest)
 print('DONE',{'stocks':nstocks,'rows':nrows,'path':str(dest),'from':a.start,'to':a.end},flush=True)
if __name__=='__main__':main()
