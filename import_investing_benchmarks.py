#!/usr/bin/env python3
"""Validated, non-destructive Investing.com CSV/HTML index import into Railway PostgreSQL.
Usage: python import_investing_benchmarks.py --database-url "$DATABASE_URL" --market ALL --eg-csv egx.csv --sa-csv tasi.csv
Online HTML mode fetches PUBLICLY visible rows only; it cannot fetch login-protected bulk historical ranges.
"""
import argparse, datetime as dt, io, json, os, re, math, sys
from pathlib import Path
import pandas as pd
import requests
from sqlalchemy import create_engine, text

PAGES={'EG':('EGX30','https://www.investing.com/indices/egx30-historical-data','^CASE30'),
       'SA':('TASI','https://www.investing.com/indices/tasi-historical-data','^TASI.SR')}
HEAD={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36', 'Accept-Language':'en-US,en;q=0.9'}
DATE_KEYS=['date','dates','day','trading date','التاريخ','تاريخ','日期']
CLOSE_KEYS=['price','close','closing price','last','last price','last traded price','اخر سعر','آخر سعر','السعر','收盘']

def clean_name(x):return re.sub(r'\s+',' ',str(x).lower().strip().replace('\ufeff',''))
def parse_date(x):
 s=str(x).strip().replace('\u200f','').replace('\u200e','').replace('年','-').replace('月','-').replace('日','').replace('Sept ','Sep ')
 if not s or s.lower() in ['nan','nat','none']:return None
 for fmt in ['%b %d, %Y','%B %d, %Y','%d %b %Y','%d %B %Y','%Y-%m-%d','%m/%d/%Y','%d/%m/%Y','%d-%m-%Y','%Y/%m/%d','%d.%m.%Y','%Y-%m-%d %H:%M:%S']:
  try:return dt.datetime.strptime(s,fmt).date()
  except ValueError:pass
 return None

def parse_price(x):
 s=str(x).strip().replace(',','').replace(' ','').replace('\u200e','').replace('\u200f','')
 if not s:return None
 try:
  v=float(s)
  return v if math.isfinite(v) and v>0 else None
 except ValueError:return None

def parse_dataframe(df):
 df=df.copy();df.columns=[str(t[-1]) if isinstance(t,tuple) else str(t) for t in df.columns]
 keys={clean_name(c):c for c in df.columns}
 datecol=next((keys[k] for k in DATE_KEYS if k in keys),None)
 pricecol=next((keys[k] for k in CLOSE_KEYS if k in keys),None)
 if datecol is None or pricecol is None:raise ValueError('Missing Date/Price columns: '+str(list(df.columns)))
 rows={};invalid=0;conflict=[]
 for _,r in df.iterrows():
  d=parse_date(r[datecol]);v=parse_price(r[pricecol])
  if d is None or v is None:invalid+=1;continue
  if d in rows and abs(rows[d]/v-1)>0.003:conflict.append(str(d))
  rows[d]=v
 if conflict:raise ValueError('Same-date conflicting closes: '+', '.join(conflict[:4]))
 if not rows:raise ValueError('No valid dates and closes; sample='+str(df.head(2).to_dict('records'))[:450])
 return rows,{'source_rows':len(df),'invalid_rows':invalid,'date_col':datecol,'close_col':pricecol}

def read_csv_file(path):
 content=Path(path).read_bytes();errors=[]
 for enc in ['utf-8-sig','utf-16','cp1256','latin1']:
  for sep in [None,',',';','\t']:
   try:
    d=pd.read_csv(io.BytesIO(content),encoding=enc,sep=sep,engine='python',dtype=str)
    return parse_dataframe(d)
   except Exception as exc:errors.append(str(exc)[:80])
 raise ValueError('Unable to parse CSV. Examples: '+str(errors[-3:]))

def fetch_public_html(url):
 r=requests.get(url,headers=HEAD,timeout=35)
 r.raise_for_status()
 if any(z in r.text.lower() for z in ['captcha','access denied','enable javascript']):
  raise RuntimeError('Website challenge/login/JavaScript; please use CSV export')
 tables=pd.read_html(io.StringIO(r.text))
 for tb in tables:
  try:return parse_dataframe(tb)
  except (ValueError,KeyError):pass
 raise RuntimeError('No readable historical table in public HTML; export CSV manually instead')

def insert_rows(engine,market,label,symbol,url,rows,allow_replace=False):
 with engine.begin() as c:
  c.execute(text('''CREATE TABLE IF NOT EXISTS public.rs_market_benchmark
   (market TEXT NOT NULL, session_date DATE NOT NULL, close DOUBLE PRECISION NOT NULL,
   symbol TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY (market,session_date))'''))
  columns={v[0] for v in c.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='rs_market_benchmark'"))}
  symcol='symbol' if 'symbol' in columns else ('benchmark' if 'benchmark' in columns else None)
  if not symcol:raise RuntimeError('Benchmark table has no symbol/benchmark column')
  existent={r[0]:(float(r[1]),str(r[2])) for r in c.execute(text('SELECT session_date,close,source FROM public.rs_market_benchmark WHERE market=:m'),{'m':market})}
  conflicts=[{'day':str(d),'stored':round(existent[d][0],3),'new':round(v,3),'source':existent[d][1]}
             for d,v in rows.items() if d in existent and abs(existent[d][0]/v-1)>.003]
  if conflicts and not allow_replace:
   raise RuntimeError('PRICE_CONFLICT: import aborted for this market; sample='+json.dumps(conflicts[:5]))
  inserts=[{'m':market,'d':d,'v':v,'s':'investing_csv' if 'csv' in label else 'investing_public_html','symbol':symbol} for d,v in sorted(rows.items())]
  # Existing closes are preserved by default. With explicit --replace-conflicts, the user may replace them.
  stmt=(f'INSERT INTO public.rs_market_benchmark (market,session_date,close,{symcol},source) '
         f'VALUES (:m,:d,:v,:symbol,:s) ON CONFLICT (market,session_date) '+
         ('DO UPDATE SET close=EXCLUDED.close,source=EXCLUDED.source' if allow_replace else 'DO NOTHING'))
  for i in range(0,len(inserts),250):c.execute(text(stmt),inserts[i:i+250])
  result=c.execute(text('SELECT count(*),min(session_date),max(session_date) FROM public.rs_market_benchmark WHERE market=:m'),{'m':market}).one()
 return {'count':int(result[0]),'first':str(result[1]),'last':str(result[2]),'overlap_conflicts':len(conflicts)}

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
 p.add_argument('--market',choices=['EG','SA','ALL'],default='ALL')
 p.add_argument('--eg-csv');p.add_argument('--sa-csv')
 p.add_argument('--online',action='store_true',help='Attempt recent public HTML rows; may be blocked or limited to one month')
 p.add_argument('--end',default=str(dt.date.today()))
 p.add_argument('--replace-conflicts',action='store_true',help='Only with manual source comparison and approval')
 p.add_argument('--report',help='Optional local JSON report, NOT a permanent record')
 a=p.parse_args();eng=create_engine(a.database_url,pool_pre_ping=True)
 markets=['EG','SA'] if a.market=='ALL' else [a.market];out={}
 for market in markets:
  name,url,symbol=PAGES[market];csvfile=a.eg_csv if market=='EG' else a.sa_csv
  try:
   if csvfile:data,parsed=read_csv_file(csvfile);source='csv '+csvfile
   elif a.online:data,parsed=fetch_public_html(url);source='public html '+url
   else:raise RuntimeError('Provide --'+market.lower()+'-csv (for Egypt --eg-csv; for Saudi --sa-csv) or --online')
   last=max(data);end=dt.date.fromisoformat(a.end);staleness=(end-last).days
   gaps=[{'from':str(x),'to':str(y),'days':(y-x).days} for x,y in zip(sorted(data),sorted(data)[1:]) if (y-x).days>10]
   saved=insert_rows(eng,market,source,symbol,url,data,a.replace_conflicts)
   out[market]={'status':'IMPORTED' if staleness<=10 else 'STALE_PARTIAL','source':source,'parsed':parsed,
                'source_rows':len(data),'first':str(min(data)),'last':str(last),'ready_for_full_screening':False,
                'days_stale':staleness,'gaps_over_10_days':len(gaps),'examples':gaps[:5],'postgres':saved}
  except Exception as exc:out[market]={'status':'ERROR_NO_WRITE','error':str(exc)[:900]}
  print('MARKET_RESULT',market,json.dumps(out[market],ensure_ascii=False),flush=True)
 if a.report:
  dest=Path(a.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print('SUMMARY',json.dumps({m:r['status'] for m,r in out.items()}))
if __name__=='__main__':main()
