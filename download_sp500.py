#!/usr/bin/env python3
"""Download S&P 500 (^GSPC) historical daily CLOSE to local CSV, no DB mutation."""
import argparse, datetime as dt, json, time
from pathlib import Path
import pandas as pd
import requests

def yahoo_chart(symbol, start, end):
    start_ts=int(pd.Timestamp(start,tz='UTC').timestamp())
    end_ts=int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)).timestamp())
    headers={'User-Agent':'Mozilla/5.0 (compatible; Research/1.0)','Accept':'application/json'}
    errors=[]
    for host in ('query1.finance.yahoo.com','query2.finance.yahoo.com'):
        url=f'https://{host}/v8/finance/chart/{requests.utils.quote(symbol,safe="")}'
        try:
            r=requests.get(url,params={'period1':start_ts,'period2':end_ts,'interval':'1d'},headers=headers,timeout=45)
            r.raise_for_status()
            obj=r.json()['chart']; err=obj.get('error')
            if err:raise RuntimeError(str(err))
            res=obj['result'][0]
            stamps=res['timestamp']; quotes=res['indicators']['quote'][0]['close']
            d=pd.DataFrame({'date':pd.to_datetime(stamps,unit='s',utc=True).date.astype(str),'c':quotes})
            d['c']=pd.to_numeric(d.c,errors='coerce')
            d=d.dropna().loc[lambda x:x.c>0].drop_duplicates('date').sort_values('date').reset_index(drop=True)
            if len(d)<260:raise RuntimeError(f'Only {len(d)} rows')
            return d,host
        except Exception as exc:
            errors.append(f'{host}: {exc}')
            time.sleep(1)
    raise RuntimeError('Cannot retrieve historical ^GSPC from Yahoo: '+' | '.join(errors))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/data/seven_relative_sp500/sp500_daily.csv')
    p.add_argument('--start',default='2023-01-01')
    p.add_argument('--end',default=dt.date.today().isoformat())
    a=p.parse_args()
    d,host=yahoo_chart('^GSPC',a.start,a.end)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(target,index=False)
    print(json.dumps({'status':'complete','symbol':'^GSPC','provider_host':host,'rows':len(d),'first':d.date.iloc[0],'last':d.date.iloc[-1],'out':str(target)},indent=2))
if __name__=='__main__':main()
