#!/usr/bin/env python3
"""FROG: nested fair comparison of previous-day vs both-day ATR at identical precision."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from six_mirror_batch import iter_db, clean, atr_series, valid_symbol

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--database-url',required=True)
 p.add_argument('--out',default='/data/frog_fair_atr')
 p.add_argument('--start',default='2024-01-01')
 p.add_argument('--end',default='2026-06-30')
 p.add_argument('--horizon',type=int,default=30)
 p.add_argument('--decimals',default='3,2,1')
 a=p.parse_args()
 dps=list(dict.fromkeys(int(v) for v in a.decimals.split(',')))
 if not dps or any(x<0 or x>8 for x in dps) or a.horizon<=0:raise SystemExit('Invalid decimal places or horizon')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
 with engine.connect() as conn:
  raw=pd.read_sql(text("SELECT symbol,session_date AS date,o,h,l,c,v FROM market_candles_1d WHERE symbol='FROG' ORDER BY session_date"),conn)
 engine.dispose()
 ref=clean(raw)
 pos=np.flatnonzero(ref.date.to_numpy()=='2026-04-15')
 if not len(pos) or pos[0]<14:raise SystemExit('Missing FROG reference or 14-bar prior history')
 i=int(pos[0]);atr=atr_series(ref);prev=float(atr[i-1]);today=float(atr[i]);
 if not np.isfinite(prev) or not np.isfinite(today):raise SystemExit('Reference ATR unavailable')
 print('FROG REFERENCE',str(ref.date.iloc[i-1]),prev,str(ref.date.iloc[i]),today,flush=True)
 data={dp:[] for dp in dps};groups=0;eligible=0
 for raw in iter_db(a.database_url):
  ticker=str(raw.symbol.iloc[0]).upper().strip()
  if ticker=='FROG' or not valid_symbol(ticker):continue
  groups+=1;d=clean(raw)
  if len(d)<a.horizon+15:continue
  arr=atr_series(d);dates=d.date.to_numpy()
  indices=np.flatnonzero((dates>=a.start)&(dates<=a.end));indices=indices[(indices>=14)&(indices+a.horizon<len(d))]
  if not len(indices):continue
  pv=arr[indices-1];dv=arr[indices];eligible+=len(indices)
  cls=d.c.to_numpy(float);highs=d.h.to_numpy(float);lows=d.l.to_numpy(float)
  for dp in dps:
   before=np.isfinite(pv)&(np.round(pv,dp)==np.round(prev,dp))
   both=before & np.isfinite(dv)&(np.round(dv,dp)==np.round(today,dp))
   # The nested-invariant MUST always hold for each precision.
   assert not np.any(both&~before), 'Nestedness invariant failed'
   matches=np.flatnonzero(before)
   for j in matches:
    at=int(indices[j]);entry=float(cls[at]);h=highs[at+1:at+1+a.horizon];l=lows[at+1:at+1+a.horizon]
    data[dp].append({'symbol':ticker,'signal_date':str(dates[at]),'previous_date':str(dates[at-1]),'entry_close':entry,'previous_atr_pct':float(pv[j]),'signal_day_atr_pct':float(dv[j]),'previous_match':True,'both_match':bool(both[j]),'previous_only':bool(not both[j]),'hit_2_5':bool(np.any(h>=entry*1.025)),'hit_30':bool(np.any(h>=entry*1.3)),'hit_100':bool(np.any(h>=entry*2)),'max_high_gain_pct':float((np.max(h)/entry-1)*100),'worst_low_pct':float((np.min(l)/entry-1)*100),'close_day30_pct':float((cls[at+a.horizon]/entry-1)*100)})
  if groups%500==0:print('PROGRESS',groups,'symbols, candidates',eligible,flush=True)
 rows=[]
 for dp in dps:
  df=pd.DataFrame(data[dp]);df.to_csv(out/f'FROG_precision_{dp}_all_previous_matches.csv',index=False)
  for name,mask in [('PREVIOUS',np.ones(len(df),bool)),('BOTH',df.both_match.to_numpy(bool) if len(df) else np.zeros(0,bool)),('PREVIOUS_ONLY',~df.both_match.to_numpy(bool) if len(df) else np.zeros(0,bool))]:
   s=df.loc[mask]
   rows.append({'decimals':dp,'group':name,'stock_date_cases':len(s),'distinct_stocks':int(s.symbol.nunique()) if len(s) else 0,'hit_2_5':int(s.hit_2_5.sum()) if len(s) else 0,'hit_2_5_rate_pct':round(100*s.hit_2_5.mean(),3) if len(s) else None,'hit_30':int(s.hit_30.sum()) if len(s) else 0,'hit_30_rate_pct':round(100*s.hit_30.mean(),3) if len(s) else None,'hit_100':int(s.hit_100.sum()) if len(s) else 0,'hit_100_rate_pct':round(100*s.hit_100.mean(),3) if len(s) else None,'median_max_high_gain_pct':round(float(s.max_high_gain_pct.median()),3) if len(s) else None,'median_close_day30_pct':round(float(s.close_day30_pct.median()),3) if len(s) else None})
  print('PRECISION',dp,'previous',len(df),'both',int(df.both_match.sum()) if len(df) else 0,'previous_only',int((~df.both_match).sum()) if len(df) else 0,flush=True)
 summary=pd.DataFrame(rows);summary.to_csv(out/'FROG_fair_precision_comparison.csv',index=False)
 report={'status':'complete','reference':'FROG','reference_date':'2026-04-15','reference_prev_ATR_pct':prev,'reference_day_ATR_pct':today,'precisions':dps,'eligible_stock_dates':eligible,'ticker_groups_processed':groups,'date_start':a.start,'date_end':a.end,'forward_sessions':a.horizon,'comparison_logic':'at every precision BOTH = PREVIOUS match AND signal-day match, so BOTH is always a subset of PREVIOUS','reference_ticker_excluded':True,'forward_window':'complete 30 sessions required','scope':'independent ATR comparisons; no OR/AND discovery filters applied','warning':['In-sample historical associations are not out-of-sample proof.','Same symbol can have multiple signal dates; count cases separately from unique stocks.','High target touches are not guaranteed executable fills.','Corporate actions and reverse splits may distort large returns.']}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(summary.to_string(index=False),flush=True)
 print('DONE',str(out),flush=True)
if __name__=='__main__':main()
