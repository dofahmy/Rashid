#!/usr/bin/env python3
"""Six independent non-YDKG ATR mirror studies; shared DB passes, no future data in features."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

REFERENCES = {
 'BGMS':('2026-06-04',165.40,1), 'ZIM':('2024-07-29',172.71,15),
 'FROG':('2026-04-15',146.96,32), 'MU':('2026-03-02',203.81,59),
 'IMSR':('2025-06-24',173.32,78), 'IMNM':('2025-07-09',237.,82),
}

def valid_symbol(s):
 s=str(s).upper().strip()
 return bool(re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?',s)) and not s.endswith(('.CA','.SR'))

def clean(group):
 d=group.copy(); d['date']=pd.to_datetime(d['date'],errors='coerce').dt.strftime('%Y-%m-%d')
 for col in 'ohlcv':d[col]=pd.to_numeric(d[col],errors='coerce')
 d=d.dropna(subset=['date','o','h','l','c','v'])
 d=d[(d.o>0)&(d.c>0)&(d.l>0)&(d.v>=0)&(d.h>=d[['o','c','l']].max(axis=1))&(d.l<=d[['o','c','h']].min(axis=1))]
 return d.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)

def atr_series(d):
 c=d.c.to_numpy(float);h=d.h.to_numpy(float);l=d.l.to_numpy(float)
 if not len(c):return np.empty(0)
 prev=np.r_[c[0],c[:-1]]
 tr=np.maximum.reduce((h-l,np.abs(h-prev),np.abs(l-prev)))
 atr=np.full(len(c),np.nan)
 if len(c)>=14:
  sums=np.cumsum(np.r_[0.,tr]); atr[13:]=(sums[14:]-sums[:-14])/14/c[13:]*100
  # Match YDKG's numpy mean arithmetic at the reference and for boundary candidates
  atr[13:]=np.array([np.mean(tr[i-13:i+1])/c[i]*100 for i in range(13,len(c))])
 return atr

def iter_db(url,chunksize=100000):
 engine=create_engine(url.replace('postgres://','postgresql://',1))
 try:
  with engine.connect().execution_options(stream_results=True) as conn:
   q=text("SELECT symbol,session_date AS date,o,h,l,c,v FROM market_candles_1d WHERE symbol !~ '\\.(CA|SR)$' ORDER BY symbol,session_date")
   carry=pd.DataFrame()
   for chunk in pd.read_sql(q,conn,chunksize=chunksize):
    whole=pd.concat([carry,chunk],ignore_index=True); last=whole.symbol.iloc[-1]
    full=whole[whole.symbol!=last];carry=whole[whole.symbol==last].copy()
    for _,g in full.groupby('symbol',sort=False):yield g
   if not carry.empty:yield carry
 finally:engine.dispose()

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--database-url',required=True)
 p.add_argument('--out',default='/data/six_mirror_batch')
 p.add_argument('--start',default='2024-01-01');p.add_argument('--end',default='2026-06-30')
 p.add_argument('--horizon',type=int,default=30);p.add_argument('--min-stocks',type=int,default=20)
 p.add_argument('--min-decimals',type=int,default=0);p.add_argument('--max-decimals',type=int,default=8)
 a=p.parse_args()
 if a.horizon<1 or a.min_stocks<1 or a.max_decimals<a.min_decimals or a.min_decimals<0:raise SystemExit('Invalid numerical argument')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 precision=list(range(a.max_decimals,a.min_decimals-1,-1))
 refs={};unavailable={}
 # Targeted reference fetch is fast, avoids six full market passes
 engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
 with engine.connect() as conn:
  for ticker,(date,gain,days) in REFERENCES.items():
   raw=pd.read_sql(text('SELECT symbol, session_date AS date,o,h,l,c,v FROM market_candles_1d WHERE symbol=:s ORDER BY session_date'),conn,params={'s':ticker})
   if raw.empty: unavailable[ticker]='Ticker not present in market_candles_1d';continue
   d=clean(raw); pos=np.flatnonzero(d.date.to_numpy()==date)
   if not len(pos):unavailable[ticker]=f'Reference session {date} absent or invalid';continue
   idx=int(pos[0]);atr=atr_series(d)
   if idx<14 or not np.isfinite(atr[idx-1]) or not np.isfinite(atr[idx]):
    unavailable[ticker]='Insufficient ATR lookback';continue
   refs[ticker]={'reference_date':date,'reference_gain_pct':gain,'reference_peak_days':days,
    'previous_session':str(d.date.iloc[idx-1]),'previous_atr14_pct':float(atr[idx-1]),
    'signal_atr14_pct':float(atr[idx])}
 engine.dispose()
 print('AVAILABLE REFS',json.dumps(refs,ensure_ascii=False),flush=True)
 print('MISSING REFS',unavailable,flush=True)
 if not refs:
  (out/'report.json').write_text(json.dumps({'unavailable_references':unavailable,'status':'no_references'},indent=2));return
 # One shared pass to estimate distinct ticker counts per reference and precision
 buckets={s:{kind:{dp:set() for dp in precision} for kind in ('previous','signal_day','both')} for s in refs}
 count=0;tested=0
 for group in iter_db(a.database_url):
  ticker=str(group.symbol.iloc[0]).upper().strip()
  if not valid_symbol(ticker):continue
  count+=1;d=clean(group)
  if len(d)<15+a.horizon:continue
  atr=atr_series(d);dates=d.date.to_numpy()
  ix=np.flatnonzero((dates>=a.start)&(dates<=a.end));ix=ix[(ix>=14)&(ix+a.horizon<len(d))]
  tested+=len(ix)
  if not len(ix):continue
  previous=atr[ix-1];today=atr[ix]
  for s,r in refs.items():
   if s==ticker:continue
   for dp in precision:
    pflag=np.round(previous,dp)==round(r['previous_atr14_pct'],dp)
    dflag=np.round(today,dp)==round(r['signal_atr14_pct'],dp)
    if pflag.any():buckets[s]['previous'][dp].add(ticker)
    if dflag.any():buckets[s]['signal_day'][dp].add(ticker)
    if (pflag&dflag).any():buckets[s]['both'][dp].add(ticker)
  if count%500==0:print('PASS1',count,'symbols',flush=True)
 selected={s:{kind:next((dp for dp in precision if len(grid[dp])>=a.min_stocks),None) for kind,grid in bk.items()} for s,bk in buckets.items()}
 print('SELECTED PRECISIONS',selected,flush=True)
 # Second shared pass; rows kept per target, even if targets have different peak horizons
 rows={s:[] for s in refs};count2=0
 for group in iter_db(a.database_url):
  ticker=str(group.symbol.iloc[0]).upper().strip()
  if not valid_symbol(ticker):continue
  count2+=1;d=clean(group)
  if len(d)<15+a.horizon:continue
  atr=atr_series(d);dates=d.date.to_numpy();c=d.c.to_numpy(float);h=d.h.to_numpy(float);l=d.l.to_numpy(float)
  ix=np.flatnonzero((dates>=a.start)&(dates<=a.end));ix=ix[(ix>=14)&(ix+a.horizon<len(d))]
  if not len(ix):continue
  for s,r in refs.items():
   if ticker==s:continue
   ps=selected[s]; p=ps['previous'];dd=ps['signal_day'];both=ps['both']
   flags={}
   flags['previous']=np.isfinite(atr[ix-1])&(np.round(atr[ix-1],p)==round(r['previous_atr14_pct'],p)) if p is not None else np.zeros(len(ix),bool)
   flags['signal_day']=np.isfinite(atr[ix])&(np.round(atr[ix],dd)==round(r['signal_atr14_pct'],dd)) if dd is not None else np.zeros(len(ix),bool)
   flags['both']=(np.round(atr[ix-1],both)==round(r['previous_atr14_pct'],both))&(np.round(atr[ix],both)==round(r['signal_atr14_pct'],both)) if both is not None else np.zeros(len(ix),bool)
   matches=np.flatnonzero(flags['previous']|flags['signal_day']|flags['both'])
   for at in matches:
    i=int(ix[at]); entry=float(c[i]);hi=h[i+1:i+1+a.horizon];lo=l[i+1:i+1+a.horizon]
    hits=np.flatnonzero(hi>=2*entry)
    targetlen=int(r['reference_peak_days']);has_long=i+targetlen<len(d)
    longhi=h[i+1:i+1+targetlen] if has_long else np.empty(0)
    rows[s].append({'symbol':ticker,'signal_date':str(dates[i]),'prev_date':str(dates[i-1]),'entry_close':entry,
     'previous_atr14_pct':float(atr[i-1]),'signal_atr14_pct':float(atr[i]),
     'match_previous':bool(flags['previous'][at]),'match_signal_day':bool(flags['signal_day'][at]),'match_both':bool(flags['both'][at]),
     'max_high_return_30_pct':float((hi.max()/entry-1)*100),'hit100_within30':bool(len(hits)),
     'first_hit100_session':int(hits[0]+1) if len(hits) else None,
     'close_return_30_pct':float((c[i+a.horizon]/entry-1)*100),'worst_low_return_30_pct':float((lo.min()/entry-1)*100),
     'reference_horizon':targetlen,'reference_gain_pct':r['reference_gain_pct'],
     'reference_horizon_complete':bool(has_long),
     'max_high_return_reference_horizon_pct':float((longhi.max()/entry-1)*100) if has_long else None,
     'hit_reference_gain_in_reference_horizon':bool(longhi.max()>=entry*(1+r['reference_gain_pct']/100)) if has_long else None})
  if count2%500==0:print('PASS2',count2,'symbols',flush=True)
 summaries=[]
 for s,r in refs.items():
  folder=out/s;folder.mkdir(exist_ok=True)
  df=pd.DataFrame(rows[s]);df.to_csv(folder/f'{s}_all_ATR_mirrors.csv',index=False)
  counts=[{'decimal_places':dp,**{kind+'_distinct_tickers':len(buckets[s][kind][dp]) for kind in ('previous','signal_day','both')}} for dp in precision]
  pd.DataFrame(counts).to_csv(folder/f'{s}_precision_counts.csv',index=False)
  for kind in ('previous','signal_day','both'):
   sub=df[df['match_'+kind]] if len(df) else df
   sub.to_csv(folder/f'{s}_{kind}_mirrors.csv',index=False)
   summaries.append({'reference':s,'reference_date':r['reference_date'],'reference_gain_pct':r['reference_gain_pct'],
    'reference_peak_days':r['reference_peak_days'],'match_type':kind,'decimals':selected[s][kind],
    'distinct_stocks':int(sub.symbol.nunique()) if len(sub) else 0,'stock_date_cases':len(sub),
    'hit100_30':int(sub.hit100_within30.sum()) if len(sub) else 0,
    'rate100_30_pct':round(100*sub.hit100_within30.mean(),4) if len(sub) else None,
    'hit_original_gain_within_original_days':int(sub.hit_reference_gain_in_reference_horizon.fillna(False).sum()) if len(sub) else 0,
    'evaluable_original_horizon':int(sub.reference_horizon_complete.sum()) if len(sub) else 0})
  (folder/'reference.json').write_text(json.dumps(r,indent=2))
 pd.DataFrame(summaries).to_csv(out/'all_six_comparison.csv',index=False)
 report={'status':'complete','available_references':refs,'unavailable_references':unavailable,'stock_groups_processed':count,
  'stock_date_candidates_examined':tested,'min_distinct_other_tickers':a.min_stocks,'evaluation_30_sessions_requires_complete_window':True,
  'reference_horizon_additional_metric_requires_complete_window':True,'selected_precisions':selected,
  'cautions':['ATR rounding is in-sample signal matching, not predictive validation.',
  'Historical raw/split-adjusted prices can contain reverse splits; audit exceptional gains.',
  'Each target is independent; YDKG parameters are not imposed on other targets.',
  'Do not assume reference peak return was first achieved exactly on stated day.',
  'Signals after end date or lacking 30 sessions are not included.']}
 (out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps({'available':list(refs),'unavailable':unavailable,'results':summaries},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
