#!/usr/bin/env python3
"""Seven independent RP mirrors: point-in-time stock excess percentage-point return vs S&P500."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from six_mirror_batch import iter_db, clean, valid_symbol

REFS={'YDKG':('2025-08-22',137.66,5),'BGMS':('2026-06-04',165.40,1),
      'ZIM':('2024-07-29',172.71,15),'FROG':('2026-04-15',146.96,32),
      'MU':('2026-03-02',203.81,59),'IMSR':('2025-06-24',173.32,78),
      'IMNM':('2025-07-09',237.,82)}
WINDOWS=[5,10,20,60,120,252]

def load_benchmark(path):
 d=pd.read_csv(path)
 if not {'date','c'}.issubset(d.columns):raise ValueError('Benchmark CSV needs date,c columns')
 d['date']=pd.to_datetime(d.date,errors='coerce').dt.normalize()
 d['c']=pd.to_numeric(d.c,errors='coerce')
 d=d.dropna(subset=['date','c']);d=d[d.c>0].drop_duplicates('date',keep='last').sort_values('date')
 if len(d)<270:raise ValueError('Benchmark has fewer than 270 sessions')
 return d[['date','c']].reset_index(drop=True)

def features(d,benchmark):
 dates=pd.to_datetime(d.date).to_numpy(dtype='datetime64[ns]')
 c=d.c.to_numpy(dtype=float)
 # Same stock session endpoints; prior S&P close is used if index session is unavailable.
 bdates=benchmark.date.to_numpy(dtype='datetime64[ns]');bc=benchmark.c.to_numpy(float)
 pos=np.searchsorted(bdates,dates,side='right')-1
 b=np.full(len(d),np.nan);valid=pos>=0;b[valid]=bc[pos[valid]]
 out={}
 for w in WINDOWS:
  arr=np.full(len(d),np.nan)
  if len(d)>w:
   stock_ret=(c[w:]/c[:-w]-1)*100
   bench_ret=(b[w:]/b[:-w]-1)*100
   arr[w:]=stock_ret-bench_ret
  out[w]=arr
 return out

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--database-url',required=True)
 p.add_argument('--benchmark-csv',default='/data/seven_relative_sp500/sp500_daily.csv')
 p.add_argument('--out',default='/data/seven_rp_mirrors')
 p.add_argument('--start',default='2024-01-01');p.add_argument('--end',default='2026-06-30')
 p.add_argument('--horizon',type=int,default=30);p.add_argument('--min-stocks',type=int,default=20)
 p.add_argument('--min-decimals',type=int,default=0);p.add_argument('--max-decimals',type=int,default=6)
 a=p.parse_args()
 if not (a.horizon>0 and a.min_stocks>0 and 0<=a.min_decimals<=a.max_decimals<=8):raise ValueError('Invalid arguments')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 bench=load_benchmark(a.benchmark_csv)
 # The reference features use the last COMPLETED session before the reference date.
 ref_features={};ref_missing={}
 engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
 from sqlalchemy import text
 with engine.connect() as conn:
  for sym,(date,gain,days) in REFS.items():
   d=pd.read_sql(text('SELECT symbol,session_date AS date,o,h,l,c,v FROM market_candles_1d WHERE symbol=:s ORDER BY session_date'),conn,params={'s':sym})
   if d.empty:ref_missing[sym]='missing symbol';continue
   d=clean(d); matches=np.flatnonzero(d.date.to_numpy()==date)
   if not len(matches):ref_missing[sym]='reference date missing';continue
   idx=int(matches[0])-1
   if idx<0:ref_missing[sym]='no previous session';continue
   f=features(d,bench)
   ref_features[sym]={w:float(f[w][idx]) for w in WINDOWS if np.isfinite(f[w][idx])}
   if not ref_features[sym]:ref_missing[sym]='no RP history'
 engine.dispose()
 for sym in ref_missing:ref_features.pop(sym,None)
 print('REFERENCE RP',json.dumps(ref_features,ensure_ascii=False),flush=True)
 if not ref_features:raise RuntimeError('No usable reference RP features')
 precisions=list(range(a.max_decimals,a.min_decimals-1,-1))
 groups={(s,w):{dp:set() for dp in precisions} for s in ref_features for w in ref_features[s]}
 count=0;eligible=0
 # Pass1 measures DISTINCT symbols at each precision, not just stock/date observations.
 for raw in iter_db(a.database_url):
  sym=str(raw.symbol.iloc[0]).strip().upper()
  if not valid_symbol(sym):continue
  count+=1;d=clean(raw)
  if len(d)<a.horizon+6:continue
  dates=d.date.to_numpy();ix=np.flatnonzero((dates>=a.start)&(dates<=a.end));ix=ix[(ix>=1)&(ix+a.horizon<len(d))]
  if not len(ix):continue
  f=features(d,bench);eligible+=len(ix)
  for (ref,w),buckets in groups.items():
   if ref==sym:continue
   vals=f[w][ix-1];valid=np.isfinite(vals)
   if not valid.any():continue
   vals=vals[valid];v=ref_features[ref][w]
   for dp in precisions:
    if np.any(np.round(vals,dp)==np.round(v,dp)):buckets[dp].add(sym)
  if count%500==0:print('PASS1',count,flush=True)
 selected={(s,w):next((dp for dp in precisions if len(b[dp])>=a.min_stocks),None) for (s,w),b in groups.items()}
 print('SELECTED PRECISIONS',json.dumps({f'{s}_RP{w}':dp for (s,w),dp in selected.items()}),flush=True)
 rows={(s,w):[] for s,w in groups};count2=0
 for raw in iter_db(a.database_url):
  sym=str(raw.symbol.iloc[0]).strip().upper()
  if not valid_symbol(sym):continue
  count2+=1;d=clean(raw)
  if len(d)<a.horizon+6:continue
  dates=d.date.to_numpy();ix=np.flatnonzero((dates>=a.start)&(dates<=a.end));ix=ix[(ix>=1)&(ix+a.horizon<len(d))]
  if not len(ix):continue
  f=features(d,bench);c=d.c.to_numpy(float);h=d.h.to_numpy(float);l=d.l.to_numpy(float)
  for (ref,w),dp in selected.items():
   if sym==ref or dp is None:continue
   vals=f[w][ix-1];where=np.flatnonzero(np.isfinite(vals)&(np.round(vals,dp)==np.round(ref_features[ref][w],dp)))
   ref_gain=REFS[ref][1];ref_days=REFS[ref][2]
   for k in where:
    i=int(ix[k]);entry=float(c[i]);future_h=h[i+1:i+a.horizon+1];future_l=l[i+1:i+a.horizon+1]
    hit=np.flatnonzero(future_h>=entry*2);has_horizon=i+ref_days<len(d)
    ref_high=h[i+1:i+ref_days+1] if has_horizon else np.array([])
    rows[(ref,w)].append({'symbol':sym,'signal_date':str(dates[i]),'previous_session':str(dates[i-1]),
     'rp_window':w,'precision':dp,'reference_rp_pp':ref_features[ref][w], 'candidate_rp_pp':float(vals[k]),
     'entry_close':entry,'max_high_return_30_pct':float((future_h.max()/entry-1)*100),
     'hit2_5_30':bool(np.any(future_h>=entry*1.025)), 'hit30_30':bool(np.any(future_h>=entry*1.30)),
     'hit100_30':bool(len(hit)), 'first_hit100_session':int(hit[0]+1) if len(hit) else None,
     'worst_low_30_pct':float((future_l.min()/entry-1)*100), 'close_30_pct':float((c[i+a.horizon]/entry-1)*100),
     'reference_target_gain_pct':ref_gain,'reference_target_sessions':ref_days,
     'reference_horizon_complete':bool(has_horizon),
     'hit_reference_target':bool(np.any(ref_high>=entry*(1+ref_gain/100))) if has_horizon else None})
  if count2%500==0:print('PASS2',count2,flush=True)
 summary=[]
 for (sym,w),bk in groups.items():
  folder=out/sym;folder.mkdir(exist_ok=True)
  dp=selected[(sym,w)];data=pd.DataFrame(rows[(sym,w)])
  data.to_csv(folder/f'{sym}_RP{w}_mirrors.csv',index=False)
  n=len(data)
  summary.append({'reference':sym,'rp_window':w,'reference_rp_pp':ref_features[sym][w],
   'selected_decimals':dp,'matched_distinct_stocks':int(data.symbol.nunique()) if n else 0,
   'stock_date_cases':n,'hit2_5_30':int(data.hit2_5_30.sum()) if n else 0,
   'hit30_30':int(data.hit30_30.sum()) if n else 0,'hit100_30':int(data.hit100_30.sum()) if n else 0,
   'hit100_rate_pct':round(float(data.hit100_30.mean())*100,4) if n else None,
   'reference_target_hits':int(data.hit_reference_target.fillna(False).sum()) if n else 0,
   'reference_horizon_evaluable':int(data.reference_horizon_complete.sum()) if n else 0})
  pd.DataFrame([{'decimals':dp2,'distinct_matching_stocks':len(bk[dp2])} for dp2 in precisions]).to_csv(folder/f'{sym}_RP{w}_precision_counts.csv',index=False)
 pd.DataFrame(summary).to_csv(out/'seven_RP_mirror_comparison.csv',index=False)
 rep={'status':'complete','reference_count':len(ref_features),'missing_references':ref_missing,'reference_values':ref_features,
      'candidate_stock_dates':eligible,'stock_groups_processed':count,'benchmark_source':str(a.benchmark_csv),
      'benchmark_first_date':str(bench.date.iloc[0].date()),'benchmark_last_date':str(bench.date.iloc[-1].date()),
      'reference_features':'RP on previous completed stock session = stock lookback percent return minus S&P 500 lookback percent return, using corresponding stock-date-aligned index closes',
      'precision_policy':'highest decimal count achieving at least min_stocks distinct other tickers; each reference/window independent',
      'outcome':'future daily High in next 30 sessions, exclusive of signal day, complete 30-session follow-up required',
      'warnings':['These are in-sample matched historical observations, not validated predictions.',
                  'Data outside trading dates of the index is carried from its last session; check benchmark alignment.',
                  'Missing RP windows are reported and not treated as matches.',
                  'Potential reverse splits/corporate actions can distort results.',
                  'Original reference horizon may exceed 30 trading sessions.']}
 (out/'report.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False))
 print('DONE',json.dumps({'status':'complete','comparisons':len(summary),'output':str(out)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
