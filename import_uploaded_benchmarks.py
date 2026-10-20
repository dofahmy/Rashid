#!/usr/bin/env python3
"""Safe append-only Investing.com CSV importer to existing PostgreSQL benchmark table."""
import argparse, csv, datetime as dt, json, math, os
from pathlib import Path
from sqlalchemy import create_engine, text

PAIRS={'EG':('EGX 30 Historical Data.csv','^CASE30'),'SA':('Tadawul All Share Historical Data.csv','^TASI.SR')}

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        required={'Date','Price'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: columns {reader.fieldnames} missing Date/Price')
        values={}; invalid=[]
        for row in reader:
            try:
                day=dt.datetime.strptime(row['Date'].strip(),'%m/%d/%Y').date()
                close=float(row['Price'].replace(',','').strip())
                if not (math.isfinite(close) and close>0): raise ValueError('nonpositive close')
                if day in values and abs(values[day]-close)>.0001:
                    raise ValueError('duplicate date conflicting price')
                values[day]=close
            except Exception as e:invalid.append([row.get('Date'),str(e)])
    if invalid:raise ValueError('Invalid rows: '+repr(invalid[:8]))
    if not values:raise ValueError('no valid rows')
    return values

def gaps(data,threshold=10):
    dates=sorted(data)
    return [{'after':str(a),'before':str(b),'calendar_days':(b-a).days} for a,b in zip(dates,dates[1:]) if (b-a).days>threshold]

def run(engine,market,path,dry_run=False):
    v=read_csv(path);file_dates=sorted(v)
    res={'market':market,'file_rows':len(v),'file_first':str(file_dates[0]),'file_last':str(file_dates[-1]),'file_gaps_gt10_days':gaps(v)[:20]}
    with engine.begin() as conn:
        conn.execute(text('''CREATE TABLE IF NOT EXISTS public.rs_market_benchmark (
          market TEXT NOT NULL, session_date DATE NOT NULL, close DOUBLE PRECISION NOT NULL,
          symbol TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY(market,session_date))'''))
        cols={r[0] for r in conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='rs_market_benchmark'"))}
        symcol='symbol' if 'symbol' in cols else 'benchmark' if 'benchmark' in cols else None
        if symcol is None or not {'market','session_date','close','source'}.issubset(cols):
            raise ValueError('Existing table incompatible: '+repr(sorted(cols)))
        prior={r[0]:float(r[1]) for r in conn.execute(text('SELECT session_date,close FROM public.rs_market_benchmark WHERE market=:m'),{'m':market})}
        overlap=set(v)&set(prior)
        conflicts=[{'date':str(d),'db':prior[d],'csv':v[d],'difference_pct':round((v[d]/prior[d]-1)*100,4)} for d in sorted(overlap) if abs(v[d]/prior[d]-1)>.003]
        res.update(existing_count=len(prior),overlap_count=len(overlap),conflicts_count=len(conflicts),conflicts_examples=conflicts[:8])
        if conflicts:
            res['status']='CONFLICT_NO_WRITE';return res
        new_rows=[{'m':market,'d':d,'c':v[d],'sym':PAIRS[market][1],'src':'investing_manual_csv'} for d in file_dates if d not in prior]
        res['inserted_count']=0 if dry_run else len(new_rows)
        if not dry_run:
            stmt=text(f'''INSERT INTO public.rs_market_benchmark(market,session_date,close,{symcol},source)
                          VALUES (:m,:d,:c,:sym,:src) ON CONFLICT(market,session_date) DO NOTHING''')
            for i in range(0,len(new_rows),400):conn.execute(stmt,new_rows[i:i+400])
        combined={**prior,**({} if dry_run else {d:v[d] for d in v if d not in prior})}
        dd=sorted(combined)
        res.update(db_total_after=len(combined),db_first=str(dd[0]),db_last=str(dd[-1]),
                   gaps_gt10_days=gaps(combined)[:30],gap_count=len(gaps(combined)),
                   status='DRY_RUN' if dry_run else 'IMPORTED_WITH_GAPS' if gaps(combined) else 'IMPORTED_NO_LONG_GAPS')
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    ap.add_argument('--data-dir',default=str(Path(__file__).parent))
    ap.add_argument('--market',choices=['EG','SA','ALL'],default='ALL')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--report',default='')
    a=ap.parse_args()
    if not a.database_url:ap.error('DATABASE_URL is required')
    eng=create_engine(a.database_url)
    out={}
    for m in (['EG','SA'] if a.market=='ALL' else [a.market]):
        try:out[m]=run(eng,m,Path(a.data_dir)/PAIRS[m][0],a.dry_run)
        except Exception as e:out[m]={'status':'ERROR','error':str(e)}
        print('MARKET_RESULT',m,json.dumps(out[m],ensure_ascii=False,default=str),flush=True)
    if a.report:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    if any(x['status'] in ('ERROR','CONFLICT_NO_WRITE') for x in out.values()):raise SystemExit(1)
if __name__=='__main__':main()
