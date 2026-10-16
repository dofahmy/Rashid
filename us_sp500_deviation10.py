#!/usr/bin/env python3
"""US stocks matching S&P 500's EXACT FOUR realized price metrics, each within a 10% RELATIVE deviation, on 18 signal starts. Read only; no Time-Price."""
import argparse, csv
from pathlib import Path
from sqlalchemy import text
from us_tp_strategy_60pct import get_bars
from sp500_price_only_bottom import build_rows

DATES=('2018-02-08','2018-03-23','2018-10-11','2018-11-20','2018-12-07','2019-06-03','2020-02-25','2022-01-21','2022-03-07','2022-04-26','2022-06-13','2022-09-20','2023-03-10','2023-09-26','2023-10-26','2024-08-05','2025-03-10','2026-03-20')
FEATURES=(('prior5_pct','prior5'),('prior20_pct','prior20'),('fromhigh60_pct','fromhigh60'),('pos60_pct','pos60'))
SQL="""
WITH ranked AS (
 SELECT b.symbol,b.day,b.c AS close,b.h,b.l,
  LAG(b.c,5) OVER (PARTITION BY b.symbol ORDER BY b.day) AS c5,
  LAG(b.c,20) OVER (PARTITION BY b.symbol ORDER BY b.day) AS c20,
  MAX(b.h) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS high60,
  MIN(b.l) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS low60,
  COUNT(*) OVER (PARTITION BY b.symbol ORDER BY b.day ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS n60
 FROM us_tp_daily_bars b
 JOIN market_stocks s ON s.symbol=b.symbol AND s.market='US'
 WHERE b.day<=:lastday
), evaluated AS (
 SELECT r.symbol,r.day,s.company,r.close,r.n60,r.c5,r.c20,r.high60,r.low60,
  100.0*(r.close/NULLIF(r.c5,0)-1) AS prior5_pct,
  100.0*(r.close/NULLIF(r.c20,0)-1) AS prior20_pct,
  100.0*(r.close/NULLIF(r.high60,0)-1) AS fromhigh60_pct,
  CASE WHEN r.high60>r.low60 THEN 100.0*(r.close-r.low60)/(r.high60-r.low60) ELSE NULL END AS pos60_pct
 FROM ranked r JOIN market_stocks s ON s.symbol=r.symbol AND s.market='US'
 WHERE r.day IN ("""+','.join(':d'+str(i) for i in range(len(DATES)))+"""))
SELECT * FROM evaluated ORDER BY day,symbol
"""

def save(path,rows,cols):
 with open(path,'w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',default='/tmp/us_sp500_deviation10')
 p.add_argument('--deviation',type=float,default=10.0,help='Maximum relative deviation percent PER metric (default 10)')
 args=p.parse_args()
 if not 0<=args.deviation<=100:p.error('--deviation must be between 0 and 100')
 tol=args.deviation/100.0
 bars=get_bars('^GSPC')
 index_by_day={r['date']:r for r in build_rows(bars)}
 missing=[d for d in DATES if d not in index_by_day]
 if missing:raise RuntimeError('S&P 500 missing dates: '+', '.join(missing))
 from core import database
 engine=database()
 summaries={d:dict(date=d,matching=0,assessable=0,insufficient=0,has_bar=0) for d in DATES}
 matches=[]
 with engine() as session:
  total=session.execute(text("SELECT COUNT(*) FROM market_stocks WHERE market='US'")).scalar()
  rows=session.execute(text(SQL),{'lastday':max(DATES),**{'d'+str(i):d for i,d in enumerate(DATES)}}).mappings()
  for raw in rows:
   d=str(raw['day']); record=summaries[d];record['has_bar']+=1
   if raw['n60']<60 or any(raw[k] is None for k,_ in FEATURES) or any(raw[k] is None or float(raw[k])<=0 for k in ('c5','c20','high60')):
    record['insufficient']+=1;continue
   record['assessable']+=1
   idx=index_by_day[d]; deviations={};same=True
   for col,ikey in FEATURES:
    ref=float(idx[ikey]); val=float(raw[col]); diff=abs(val-ref)
    # Relative to the INDEX reading; zero baseline requires exact equality.
    max_diff=abs(ref)*tol
    deviations[col]=100.0*diff/abs(ref) if ref!=0 else (0.0 if diff==0 else float('inf'))
    if diff>max_diff+1e-9:same=False
   if same:
    record['matching']+=1
    matches.append({'date':d,'symbol':raw['symbol'],'company':raw['company'],'close':round(float(raw['close']),4),
      **{'index_'+col:round(float(idx[ikey]),5) for col,ikey in FEATURES},
      **{col:round(float(raw[col]),5) for col,_ in FEATURES},
      **{'deviation_'+col:round(deviations[col],3) for col,_ in FEATURES},
      'max_deviation_pct':round(max(deviations.values()),3)})
 output=[]
 print('RULE: Each stock must match ALL FOUR actual S&P 500 metric readings to within +/-',args.deviation,'% RELATIVE deviation; not threshold matching',flush=True)
 print('DATES:',len(DATES),'US UNIVERSE:',total,'NO Time-Price, READ ONLY',flush=True)
 for d in DATES:
  rec=summaries[d]; rec['no_bar']=total-rec['has_bar'];rec['pct_matching_of_assessable']=round(100*rec['matching']/rec['assessable'],3) if rec['assessable'] else ''
  rec['status']='NO_CACHED_BARS' if not rec['has_bar'] else 'PARTIAL_COVERAGE' if rec['assessable']<total else 'COMPLETE'
  output.append(rec)
  print('DATE',d,'MATCHING:',rec['matching'],'ASSESSABLE:',rec['assessable'],'NO_BAR:',rec['no_bar'],'INSUFFICIENT:',rec['insufficient'],'STATUS:',rec['status'],flush=True)
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 save(str(out)+'_counts.csv',output,list(output[0]))
 cols=['date','symbol','company','close']+[x for col,_ in FEATURES for x in ('index_'+col,col,'deviation_'+col)]+['max_deviation_pct']
 save(str(out)+'_stocks.csv',matches,cols)
 print('FILES:',str(out)+'_counts.csv',str(out)+'_stocks.csv',flush=True)
 print('NOTE: No bars in 2018-2020 = no data, NOT zero matching stocks. Current US universe has survivorship bias.',flush=True)

if __name__=='__main__':main()
