#!/usr/bin/env python3
"""Import free third-party EGX30/TASI historical CSV data into separate PostgreSQL benchmark table.
The source is not authoritative; validation reports explicitly warn on gaps. Existing
market_candles_1d and US RS tables are never modified.
"""
import argparse, datetime as dt, io, json, os, re, sys, urllib.parse, math
from pathlib import Path
import requests
import pandas as pd
from sqlalchemy import create_engine, text

SOURCES = {
 'EG': ('EGX30','https://raw.githubusercontent.com/markraouf10/EGX30-Analysis/main/data/EGX_30_Historical_Data.csv'),
 'SA': ('TASI','https://raw.githubusercontent.com/VickykciV/Stock-Market-Network/main/TASI%20Historical%20Data.csv'),
}
HEADERS={'User-Agent':'Mozilla/5.0 (compatible; MarketBenchmarkImporter/1.0)'}
DATE_KEYS={'date','dates','day','trading date','trading_date','datetime','日期','التاريخ','tarih'}
CLOSE_KEYS={'price','close','last','closing price','closing_price','close price','last price','adj close','adj_close','收盘','اخر سعر','آخر سعر','السعر','السعر الأخير'}

def normalize(k): return re.sub(r'\s+',' ',str(k).strip().lower().lstrip('\ufeff'))

def get_col(df, choices):
    d={normalize(c):c for c in df.columns}
    for key in choices:
        if key in d:return d[key]
    raise ValueError('Missing required column among '+str(list(df.columns)))

def parse_date(val):
    s=str(val).strip().replace('\u200e','').replace('\u200f','')
    if not s or s.lower() in ('nan','nat','none'):return None
    for fmt in ('%m/%d/%Y','%Y-%m-%d','%d/%m/%Y','%b %d, %Y','%B %d, %Y',
                '%d %b %Y','%d %B %Y','%d-%b-%Y','%d-%B-%Y',
                '%d.%m.%Y','%Y/%m/%d','%d-%m-%Y','%m-%d-%Y',
                '%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S'):
        try:return dt.datetime.strptime(s,fmt).date()
        except ValueError:pass
    try:
        ts=pd.to_datetime(s,errors='coerce',dayfirst=False)
        return None if pd.isna(ts) else ts.date()
    except (ValueError,TypeError,OverflowError):return None

def parse_price(val):
    s=str(val).strip().replace(',','').replace(' ','').replace('\u200f','').replace('\u200e','')
    if not s or s.lower() in ('-','nan','none','null'):return None
    # Volumes may have K/M/B suffixes; closing prices must not.
    try:
        v=float(s)
        return v if math.isfinite(v) and 0 < v < 1e9 else None
    except ValueError:return None

def parse_frame(content, label):
    try:
        df=pd.read_csv(io.BytesIO(content),encoding='utf-8-sig',dtype=str,sep=None,engine='python')
    except Exception as exc:
        raise ValueError(f'CSV read failed ({label}): {exc}') from exc
    if df.empty:raise ValueError(f'Empty CSV: {label}')
    try:
        dc=get_col(df,DATE_KEYS)
        pc=get_col(df,CLOSE_KEYS)
    except ValueError as exc:
        raise ValueError(f'{exc}; sample_rows={df.head(3).to_dict(orient="records")}') from exc
    parsed={}; invalid=0; conflicting=[]
    for idx,row in df.iterrows():
        d=parse_date(row[dc]); v=parse_price(row[pc])
        if d is None or v is None: invalid+=1;continue
        if d in parsed and abs(parsed[d]-v)>0.005:conflicting.append(str(d))
        parsed[d]=v
    if conflicting:raise ValueError(f'Conflicting same-date closes in {label}: {conflicting[:5]}')
    if not parsed:raise ValueError(f'No valid dates and closes in {label}; columns={list(df.columns)}; sample_rows={df.head(5).to_dict(orient="records")}')
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
    # Existing rs_multi_build.py created this table with symbol, not benchmark.
    # Do not ALTER or DROP any existing table; adapt inserts to its actual schema.
    conn.execute(text('''CREATE TABLE IF NOT EXISTS public.rs_market_benchmark (
       market TEXT NOT NULL, session_date DATE NOT NULL,
       close DOUBLE PRECISION NOT NULL, symbol TEXT NOT NULL, source TEXT NOT NULL,
       PRIMARY KEY(market,session_date))'''))
    info=conn.execute(text('''SELECT column_name, is_nullable, column_default
       FROM information_schema.columns
       WHERE table_schema='public' AND table_name='rs_market_benchmark' ''')).fetchall()
    columns={r[0] for r in info}
    if not {'market','session_date','close','source'}.issubset(columns):
        raise RuntimeError('Existing benchmark table has incompatible schema: '+str(sorted(columns)))
    if 'symbol' in columns:return 'symbol'
    if 'benchmark' in columns:return 'benchmark'
    raise RuntimeError('Existing table has neither symbol nor benchmark column')

def import_market(engine,market,data,label):
    sym={'EG':'^CASE30','SA':'^TASI.SR'}[market]
    with engine.begin() as conn:
        name_col=safe_table_setup(conn)
        existing=conn.execute(text('''SELECT session_date,close,source FROM public.rs_market_benchmark
                         WHERE market=:m'''),{'m':market}).fetchall()
        # Validate before any data writes, and never silently replace conflicting prices.
        conflicts=[{'date':str(d),'stored':float(v),'incoming':data[d],'stored_source':source}
                  for d,v,source in existing if d in data and abs(float(v)-data[d])/max(abs(float(v)),1)>0.003]
        if conflicts:raise ValueError(f'Existing data conflicts >0.3%; no writes: {conflicts[:8]}')
        rows=[{'m':market,'b':sym,'d':d,'c':close,'s':label} for d,close in sorted(data.items())]
        stmt=text(f'''INSERT INTO public.rs_market_benchmark(market,{name_col},session_date,close,source)
                     VALUES (:m,:b,:d,:c,:s)
                     ON CONFLICT(market,session_date) DO NOTHING''')
        for i in range(0,len(rows),300):conn.execute(stmt,rows[i:i+300])
        saved=conn.execute(text('''SELECT COUNT(*),MIN(session_date),MAX(session_date)
                  FROM public.rs_market_benchmark WHERE market=:m'''),{'m':market}).one()
    return {'table_column':name_col,'stored_rows':saved[0],
            'stored_first':str(saved[1]),'stored_last':str(saved[2])}

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
    if args.report:
        report_path=Path(args.report)
        try:
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            print('REPORT',args.report)
        except OSError as exc:
            print('REPORT_FILE_WARNING',str(exc),'full report already printed above')
    print('SUMMARY',json.dumps({m:{'status':v.get('status'),'postgres':v.get('postgres'),'quality':v.get('quality')} for m,v in results.items()},ensure_ascii=False,default=str))
    if any(x['status']=='ERROR_NO_IMPORT' for x in results.values()):sys.exit(1)
    if any(x['status']!='READY' for x in results.values()):sys.exit(2)

if __name__=='__main__':main()
