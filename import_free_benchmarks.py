#!/usr/bin/env python3
"""Import free third-party EGX30/TASI historical CSV data into separate PostgreSQL benchmark table.
The source is not authoritative; validation reports explicitly warn on gaps. Existing
market_candles_1d and US RS tables are never modified.
"""
import argparse, datetime as dt, io, json, os, re, sys, urllib.parse
from pathlib import Path
import requests
import pandas as pd
from sqlalchemy import create_engine, text

SOURCES = {
 'EG': ('EGX30','https://raw.githubusercontent.com/markraouf10/EGX30-Analysis/main/data/EGX_30_Historical_Data.csv'),
 'SA': ('TASI','https://raw.githubusercontent.com/VickykciV/Stock-Market-Network/main/TASI%20Historical%20Data.csv'),
}
HEADERS={'User-Agent':'Mozilla/5.0 (compatible; MarketBenchmarkImporter/1.0)'}
DATE_KEYS={'date','日期','التاريخ','tarih'}
CLOSE_KEYS={'price','close','last','收盘','اخر سعر','آخر سعر','السعر','السعر الأخير','closing price'}

def normalize(k): return re.sub(r'\s+',' ',str(k).strip().lower().lstrip('\ufeff'))

def get_col(df, choices):
    d={normalize(c):c for c in df.columns}
    for key in choices:
        if key in d:return d[key]
    raise ValueError('Missing required column among '+str(list(df.columns)))

def parse_date(val):
    s=str(val).strip()
    if not s:return None
    for fmt in ('%m/%d/%Y','%Y-%m-%d','%d/%m/%Y','%b %d, %Y','%d %b %Y','%d.%m.%Y','%Y/%m/%d'):
        try:return dt.datetime.strptime(s,fmt).date()
        except ValueError:pass
    return None

def parse_price(val):
    s=str(val).strip().replace(',','').replace(' ','').replace('\u200f','')
    if not s or s in ('-','nan','None'):return None
    try:
        v=float(s)
        return v if 0 < v < 1e9 else None
    except ValueError:return None

def parse_frame(content, label):
    df=pd.read_csv(io.BytesIO(content), encoding='utf-8-sig',dtype=str)
    dc=get_col(df,DATE_KEYS)
    pc=get_col(df,CLOSE_KEYS)
    parsed={}; invalid=0; conflicting=[]
    for idx,row in df.iterrows():
        d=parse_date(row[dc]); v=parse_price(row[pc])
        if d is None or v is None: invalid+=1;continue
        if d in parsed and abs(parsed[d]-v)>0.005:conflicting.append(str(d))
        parsed[d]=v
    if conflicting:raise ValueError(f'Conflicting same-date closes in {label}: {conflicting[:5]}')
    if not parsed:raise ValueError(f'No valid dates and closes in {label}')
    return parsed,{'source_rows':len(df),'invalid_rows':invalid,'date_column':dc,'close_column':pc}

def fetch(url):
    response=requests.get(url,headers=HEADERS,timeout=60)
    response.raise_for_status()
    if len(response.content)<100 or len(response.content)>15_000_000:raise RuntimeError('Unexpected size')
    return response.content

def validate(data, last_required, minimum, first_expected):
    dates=sorted(data)
    all_gaps=[(a,b,(b-a).days) for a,b in zip(dates,dates[1:]) if (b-a).days>10]
    long_gaps=[dict(after=str(a),before=str(b),calendar_days=n) for a,b,n in all_gaps]
    usable=sum(1 for x in dates if x>=first_expected)
    age=(last_required-dates[-1]).days
    return {'rows':len(dates),'rows_from_requested_start':usable,'first':str(dates[0]),'last':str(dates[-1]),
            'days_stale_vs_requested_end':age, 'gap_over_10_calendar_days_count':len(long_gaps),'example_gaps':long_gaps[:15],
            'historical_coverage_ok':usable>=minimum and dates[0]<=first_expected and not long_gaps,
            'recent_coverage_ok':-3<=age<=10,
            'ready_for_screening':usable>=minimum and dates[0]<=first_expected and not long_gaps and -3<=age<=10}

def safe_table_setup(conn):
    conn.execute(text('''CREATE TABLE IF NOT EXISTS public.rs_market_benchmark (
       market VARCHAR(2) NOT NULL, benchmark VARCHAR(24) NOT NULL, session_date DATE NOT NULL,
       close DOUBLE PRECISION NOT NULL CHECK (close > 0), source TEXT NOT NULL,
       retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       PRIMARY KEY(market, session_date)
    )'''))

def import_market(engine, market, data, label):
    benchmark=SOURCES[market][0]
    with engine.begin() as conn:
        safe_table_setup(conn)
        # Do not overwrite an existing benchmark's conflicting prices without an explicit review.
        existing=conn.execute(text('''SELECT session_date, close, source FROM public.rs_market_benchmark
                 WHERE market=:m'''),{'m':market}).fetchall()
        conflicts=[{'date':str(d),'stored':float(v),'incoming':data[d],'stored_source':source}
                  for d,v,source in existing if d in data and abs(v-data[d])/max(v,1)>0.003]
        if conflicts:raise ValueError(f'Existing data conflicts >0.3%, no writes made: {conflicts[:8]}')
        rows=[{'m':market,'b':benchmark,'d':d,'c':close,'s':label} for d,close in sorted(data.items())]
        stmt=text('''INSERT INTO public.rs_market_benchmark (market,benchmark,session_date,close,source)
            VALUES (:m,:b,:d,:c,:s) ON CONFLICT (market,session_date) DO NOTHING''')
        for i in range(0,len(rows),500):conn.execute(stmt,rows[i:i+500])
        saved=conn.execute(text('''SELECT COUNT(*),MIN(session_date),MAX(session_date)
                      FROM public.rs_market_benchmark WHERE market=:m'''),{'m':market}).one()
    return {'stored_rows':saved[0],'stored_first':str(saved[1]),'stored_last':str(saved[2])}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    a.add_argument('--market',choices=['EG','SA','ALL'],default='ALL')
    a.add_argument('--end',default='2026-10-20')
    a.add_argument('--start-eg',default='2019-01-01')
    a.add_argument('--start-sa',default='2016-10-16')
    a.add_argument('--eg-csv',help='Optional newer historical EGX30 CSV exported from Investing.com')
    a.add_argument('--sa-csv',help='Optional newer historical TASI CSV exported from Investing.com')
    a.add_argument('--report',default='benchmark_import_report.json')
    a.add_argument('--dry-run',action='store_true')
    args=a.parse_args()
    if not args.database_url and not args.dry_run:raise SystemExit('DATABASE_URL missing')
    last=dt.date.fromisoformat(args.end)
    markets=['EG','SA'] if args.market=='ALL' else [args.market]
    engine=create_engine(args.database_url) if not args.dry_run else None
    results={}
    for m in markets:
        label,url=SOURCES[m]
        print('FETCH',m,label,url,flush=True)
        try:
            combined,diag=parse_frame(fetch(url),label+' community CSV')
            extra_file=args.eg_csv if m=='EG' else args.sa_csv
            extras={}; extra_diag={}
            if extra_file:
                extras,extra_diag=parse_frame(Path(extra_file).read_bytes(),label+' investing export')
                overlap=[(d,combined[d],v) for d,v in extras.items() if d in combined and abs(combined[d]-v)/combined[d]>0.003]
                if overlap:raise ValueError(f'CSV sources disagree >0.3%: {overlap[:5]}')
                combined.update(extras)
            first=dt.date.fromisoformat(args.start_eg if m=='EG' else args.start_sa)
            quality=validate(combined,last,500,first)
            info={'benchmark':label,'source':url,'parsed':diag,'extra_csv':str(extra_file) if extra_file else None,
                  'extra':extra_diag,'quality':quality}
            if engine:info['postgres']=import_market(engine,m,combined,('community_git_csv + manual_investing_csv' if extras else 'community_git_csv'))
            info['status']='READY' if quality['ready_for_screening'] else 'PARTIAL_DO_NOT_BUILD_RS'
            results[m]=info
            print('MARKET_RESULT',m,json.dumps(info,ensure_ascii=False,default=str),flush=True)
        except Exception as ex:
            results[m]={'status':'ERROR_NO_IMPORT','error':str(ex)}
            print('MARKET_ERROR',m,str(ex),flush=True)
    Path(args.report).write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('REPORT',args.report)
    if any(x['status']=='ERROR_NO_IMPORT' for x in results.values()):sys.exit(1)
    if any(x['status']!='READY' for x in results.values()):sys.exit(2)

if __name__=='__main__':main()
