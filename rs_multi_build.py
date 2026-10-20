#!/usr/bin/env python3
"""Build EG/SA stock screening using saved PostgreSQL benchmark closes; does not query Yahoo."""
import argparse,os,time,datetime,urllib.request,urllib.parse,json
import pandas as pd,numpy as np
from sqlalchemy import create_engine,text
from seven_rp_mirrors_90 import features
from rp_stage2_adx import indicators

INDEX={'EG':('^CASE30','EGX30'),'SA':('^TASI.SR','TASI')}
W=(1,5,10,20,60,120,252)
COLS=['market','symbol','session_date','close','benchmark_close','stock_day_pct','sp500_day_pct']+[f'rp{x}' for x in W]+[f'avg_rp{x}' for x in (5,10,20,60,120)]+['adx14','plus_di14','minus_di14','relative_volume20','relative_volatility14_60']
DDL='''
CREATE TABLE IF NOT EXISTS public.rs_market_benchmark (market TEXT NOT NULL,session_date DATE NOT NULL,close DOUBLE PRECISION NOT NULL,symbol TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(market,session_date));
CREATE TABLE IF NOT EXISTS public.rs_market_features (market TEXT NOT NULL,symbol TEXT NOT NULL,session_date DATE NOT NULL,close DOUBLE PRECISION,benchmark_close DOUBLE PRECISION,stock_day_pct DOUBLE PRECISION,sp500_day_pct DOUBLE PRECISION,rp1 DOUBLE PRECISION,rp5 DOUBLE PRECISION,rp10 DOUBLE PRECISION,rp20 DOUBLE PRECISION,rp60 DOUBLE PRECISION,rp120 DOUBLE PRECISION,rp252 DOUBLE PRECISION,avg_rp5 DOUBLE PRECISION,avg_rp10 DOUBLE PRECISION,avg_rp20 DOUBLE PRECISION,avg_rp60 DOUBLE PRECISION,avg_rp120 DOUBLE PRECISION,adx14 DOUBLE PRECISION,plus_di14 DOUBLE PRECISION,minus_di14 DOUBLE PRECISION,relative_volume20 DOUBLE PRECISION,relative_volatility14_60 DOUBLE PRECISION,PRIMARY KEY(market,symbol,session_date));
CREATE INDEX IF NOT EXISTS rs_mf_market_day_idx ON public.rs_market_features(market,session_date);
CREATE TABLE IF NOT EXISTS public.rs_market_build_progress (market TEXT NOT NULL,symbol TEXT NOT NULL,build_start DATE NOT NULL,build_end DATE NOT NULL,rows_written INTEGER NOT NULL,updated_at TIMESTAMPTZ DEFAULT NOW(),PRIMARY KEY(market,symbol));
'''
def yahoo(symbol,start,end):
 p1=int(datetime.datetime.fromisoformat(start).replace(tzinfo=datetime.timezone.utc).timestamp());p2=int((datetime.datetime.fromisoformat(end)+datetime.timedelta(days=2)).replace(tzinfo=datetime.timezone.utc).timestamp())
 url=f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol,safe="")}?period1={p1}&period2={p2}&interval=1d'
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; Screening/1.0)'})
 with urllib.request.urlopen(req,timeout=35) as resp:obj=json.load(resp)
 if obj.get('chart',{}).get('error'):raise RuntimeError(str(obj['chart']['error']))
 results=obj.get('chart',{}).get('result') or []
 if not results:raise RuntimeError('No Yahoo index chart for '+symbol)
 item=results[0];ts=item.get('timestamp',[]);quote=(item.get('indicators',{}).get('quote') or [{}])[0];cl=quote.get('close') or []
 data={}
 for t,c in zip(ts,cl):
  if c is not None and float(c)>0:data[datetime.datetime.fromtimestamp(t,datetime.timezone.utc).date()]=float(c)
 return sorted(data.items())
def num(x):
 try:
  x=float(x);return x if np.isfinite(x) else None
 except (ValueError,TypeError):return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--database-url',default=os.getenv('DATABASE_URL'));p.add_argument('--market',choices=['EG','SA','ALL'],default='ALL');p.add_argument('--start',default='2015-01-01');p.add_argument('--end',default='2026-10-20');p.add_argument('--refresh',action='store_true',help='Recompute markets even if already in progress')
 a=p.parse_args();eng=create_engine(a.database_url,pool_pre_ping=True);markets=['EG','SA'] if a.market=='ALL' else [a.market]
 with eng.begin() as cn:
  for ddl in DDL.split(';'):
   if ddl.strip():cn.execute(text(ddl))
 for market in markets:
  symbol,index_name=INDEX[market]
  with eng.connect() as cn:
   exists=cn.execute(text("SELECT to_regclass('public.rs_market_benchmark') IS NOT NULL")).scalar()
   if not exists:
    print('INDEX_MISSING',market,'Run import_uploaded_benchmarks.py first',flush=True);continue
   stats=cn.execute(text('SELECT COUNT(*),MIN(session_date),MAX(session_date) FROM public.rs_market_benchmark WHERE market=:m'),{'m':market}).one()
  print('INDEX_FROM_POSTGRES',market,'ROWS',stats[0],'FIRST',str(stats[1]),'LAST',str(stats[2]),flush=True)
  if stats[0]<260:
   print('INDEX_REJECTED',market,'too few benchmark sessions',flush=True);continue
  with eng.connect() as cn:
   b=cn.execute(text('SELECT session_date,close FROM public.rs_market_benchmark WHERE market=:m ORDER BY session_date'),{'m':market}).fetchall()
   done=set(cn.execute(text('SELECT symbol FROM public.rs_market_build_progress WHERE market=:m AND build_start=:s AND build_end=:e'),{'m':market,'s':a.start,'e':a.end}).scalars()) if not a.refresh else set()
   syms=cn.execute(text("SELECT DISTINCT symbol FROM public.market_candles_1d WHERE ( :m='EG' AND upper(symbol) LIKE '%.CA') OR (:m='SA' AND (upper(symbol) LIKE '%.SA' OR upper(symbol) LIKE '%.SR')) ORDER BY symbol"),{'m':market}).scalars().all()
  bench=pd.DataFrame(b,columns=['date','c']);bench['date']=pd.to_datetime(bench.date)
  bdates=bench.date.to_numpy('datetime64[ns]');bc=bench.c.to_numpy(float)
  # Mark contiguous benchmark history segments. Never compare returns over >10 calendar-day gaps.
  bsegment=np.cumsum(np.r_[False,(np.diff(bdates)>np.timedelta64(10,'D'))])
  gaps=[(str(a)[:10],str(b)[:10]) for a,b in zip(bdates[:-1],bdates[1:]) if b-a>np.timedelta64(10,'D')]
  print('BENCHMARK_GAPS',market,len(gaps),gaps[:12],flush=True)

  print('STOCK_SYMBOLS',market,len(syms),'DONE',len(done),flush=True)
  count=nr=0
  for ticker in syms:
   if ticker in done:continue
   with eng.connect() as cn:
    raw=pd.read_sql(text('SELECT session_date AS date,o,h,l,c,v FROM public.market_candles_1d WHERE symbol=:s ORDER BY session_date'),cn,params={'s':ticker})
   if raw.empty:continue
   raw['date']=pd.to_datetime(raw.date,errors='coerce');raw=raw.dropna(subset=['date','c']).drop_duplicates('date',keep='last').sort_values('date').reset_index(drop=True)
   for k in ('o','h','l','c','v'):raw[k]=pd.to_numeric(raw[k],errors='coerce')
   raw=raw.dropna(subset=['o','h','l','c']).reset_index(drop=True)
   if len(raw)<25:continue
   dates=raw.date.to_numpy('datetime64[ns]');c=raw.c.to_numpy(float);pos=np.searchsorted(bdates,dates,side='right')-1
   bval=np.full(len(raw),np.nan);valid=pos>=0;bval[valid]=bc[pos[valid]]
   # If no recent benchmark bar within 7 calendar days, do not use stale index prices.
   valid&=(dates - bdates[np.maximum(pos,0)] <= np.timedelta64(7,'D'))
   bval[~valid]=np.nan
   f=features(raw,bench)
   # Invalidate any RP lookback whose index endpoint is unavailable or stale.
   for w in (5,10,20,60,120,252):
    if len(raw)>w:
     good=valid[w:] & valid[:-w] & (bsegment[np.maximum(pos[w:],0)] == bsegment[np.maximum(pos[:-w],0)])
     f[w][w:][~good]=np.nan
   stock1=np.full(len(raw),np.nan);idx1=stock1.copy()
   stock1[1:]=np.where(c[:-1]>0,100*(c[1:]/c[:-1]-1),np.nan)
   with np.errstate(divide='ignore',invalid='ignore'):
    idx1[1:]=100*(bval[1:]/bval[:-1]-1)
   idx1[~valid]=np.nan
   idx1[1:][bsegment[np.maximum(pos[1:],0)]!=bsegment[np.maximum(pos[:-1],0)]]=np.nan
   idx1[1:][~valid[:-1]]=np.nan
   daily=stock1-idx1
   avgs={w:pd.Series(daily).rolling(w,min_periods=w).mean().to_numpy() for w in (5,10,20,60,120)}
   ind=indicators(raw)
   batches=[]
   for i in np.flatnonzero((dates>=np.datetime64(a.start))&(dates<=np.datetime64(a.end))):
    if not np.isfinite(bval[i]):continue
    row=[market,ticker,str(dates[i])[:10],num(c[i]),num(bval[i]),num(stock1[i]),num(idx1[i])]
    row += [num(daily[i]) if w==1 else num(f[w][i]) for w in W]
    row += [num(avgs[w][i]) for w in (5,10,20,60,120)]
    row += [num(ind[k][i]) for k in ('ADX14','PLUS_DI14','MINUS_DI14','RELATIVE_VOLUME20','RELATIVE_VOLATILITY14_60')]
    batches.append(dict(zip(COLS,row)))
   q=text('INSERT INTO public.rs_market_features('+','.join(COLS)+') VALUES('+','.join(':'+x for x in COLS)+') ON CONFLICT(market,symbol,session_date) DO UPDATE SET '+','.join(x+'=EXCLUDED.'+x for x in COLS[3:]))
   with eng.begin() as cn:
    for i in range(0,len(batches),400):cn.execute(q,batches[i:i+400])
    cn.execute(text('INSERT INTO public.rs_market_build_progress(market,symbol,build_start,build_end,rows_written) VALUES(:market,:symbol,:start,:end,:n) ON CONFLICT(market,symbol) DO UPDATE SET build_start=EXCLUDED.build_start,build_end=EXCLUDED.build_end,rows_written=EXCLUDED.rows_written,updated_at=now()'),{'market':market,'symbol':ticker,'start':a.start,'end':a.end,'n':len(batches)})
   count+=1;nr+=len(batches)
   if count%25==0:print('PROGRESS',market,count,'ROWS',nr,flush=True)
  print('MARKET_DONE',market,'NEW_SYMBOLS',count,'NEW_ROWS',nr,flush=True)
 eng.dispose()
if __name__=='__main__':main()
