#!/usr/bin/env python3
"""Point-in-time relative performance of seven breakout reference stocks vs S&P 500 benchmark."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

REFS={'YDKG':'2025-08-22','BGMS':'2026-06-04','ZIM':'2024-07-29','FROG':'2026-04-15','MU':'2026-03-02','IMSR':'2025-06-24','IMNM':'2025-07-09'}
WINDOWS=[5,10,20,60,120,252]
BENCHMARK_CHOICES=['^GSPC','GSPC','SPX','SP500','S&P500','SPY','VOO','IVV']

def read_one(conn,ticker):
    q=text('SELECT session_date AS date,c FROM market_candles_1d WHERE symbol = :symbol ORDER BY session_date')
    d=pd.read_sql(q,conn,params={'symbol':ticker})
    if d.empty:return d
    d['date']=pd.to_datetime(d.date,errors='coerce').dt.normalize()
    d['c']=pd.to_numeric(d.c,errors='coerce')
    return d.dropna(subset=['date','c']).loc[lambda x:x.c>0].drop_duplicates('date',keep='last').sort_values('date').reset_index(drop=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--database-url',required=True)
    p.add_argument('--out',default='/data/seven_relative_sp500')
    p.add_argument('--benchmark',default='auto',help='Index ticker available in DB, or auto. SPY proxy is clearly labeled.')
    p.add_argument('--benchmark-csv',default=None,help='Local CSV with date,c for the exact S&P 500 daily closes; read-only, no DB write.')
    p.add_argument('--include-signal-day',action='store_true',help='Additional diagnostic only; never include in pre-signal features.')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
    with engine.connect() as conn:
        choices=BENCHMARK_CHOICES if a.benchmark=='auto' else [a.benchmark]
        attempted=[];bench=None;bench_name=None
        if a.benchmark_csv:
            file=Path(a.benchmark_csv)
            if not file.is_file():raise FileNotFoundError(f'Benchmark file missing: {file}')
            d=pd.read_csv(file)
            if not {'date','c'}.issubset(d.columns):raise ValueError('Benchmark CSV must contain date,c columns')
            d['date']=pd.to_datetime(d.date,errors='coerce').dt.normalize()
            d['c']=pd.to_numeric(d.c,errors='coerce')
            d=d.dropna(subset=['date','c']).loc[lambda x:x.c>0].drop_duplicates('date',keep='last').sort_values('date').reset_index(drop=True)
            attempted.append({'symbol':'^GSPC_CSV','rows':int(len(d))})
            if len(d)>=260:bench=d;bench_name='^GSPC_CSV'
        else:
            for candidate in choices:
                d=read_one(conn,candidate);attempted.append({'symbol':candidate,'rows':int(len(d))})
                if len(d)>=260:bench=d;bench_name=candidate;break
        if bench is None:
            report={'status':'missing_benchmark','attempted':attempted,'message':'No usable S&P500 index/ETF series in market_candles_1d. Supply/import S&P500 benchmark historical daily closes and rerun; cannot calculate valid stock-vs-index relative performance.'}
            (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(json.dumps(report,indent=2,ensure_ascii=False));return
        print('BENCHMARK',bench_name,'ROWS',len(bench),flush=True)
        benchmark=bench.set_index('date')['c']
        items=[];missing=[]
        for ticker,date in REFS.items():
            stock=read_one(conn,ticker);sig=pd.Timestamp(date)
            row=stock.index[stock.date.eq(sig)]
            if not len(row):missing.append({'symbol':ticker,'date':date,'reason':'No matching reference candle'});continue
            i=int(row[0]);j=i-1
            if j<1:missing.append({'symbol':ticker,'date':date,'reason':'Insufficient history'});continue
            prev=stock.iloc[j]
            # Align benchmark by stock-session dates; previous asof allowed for closed market holidays.
            joined=pd.merge_asof(stock[['date','c']].rename(columns={'c':'stock_close'}),bench[['date','c']].rename(columns={'c':'benchmark_close'}),on='date',direction='backward')
            d={'symbol':ticker,'signal_date':date,'previous_session':str(prev.date.date()),'stock_prev_close':float(prev.c),'benchmark_symbol':bench_name,'benchmark_kind':('ETF_PROXY' if bench_name in ('SPY','VOO','IVV') else 'INDEX'),'benchmark_prev_close':float(joined.iloc[j].benchmark_close) if pd.notna(joined.iloc[j].benchmark_close) else np.nan}
            for n in WINDOWS:
                k=j-n
                if k<0 or pd.isna(joined.iloc[k].benchmark_close) or pd.isna(joined.iloc[j].benchmark_close):
                    for name in ('stock','sp500','relative_pp','relative_ratio_pct','stock_excess_ratio'):d[f'{name}_{n}d']=np.nan
                    continue
                s=float(joined.iloc[j].stock_close/joined.iloc[k].stock_close-1)*100
                b=float(joined.iloc[j].benchmark_close/joined.iloc[k].benchmark_close-1)*100
                d[f'stock_{n}d']=s;d[f'sp500_{n}d']=b
                d[f'relative_pp_{n}d']=s-b
                d[f'relative_ratio_pct_{n}d']=((1+s/100)/(1+b/100)-1)*100
                d[f'stock_excess_ratio_{n}d']=float(joined.iloc[j].stock_close/joined.iloc[j].benchmark_close)/(float(joined.iloc[k].stock_close/joined.iloc[k].benchmark_close))
            for n in (20,60,120):
                if j>=n:
                    prices=stock.c.iloc[j-n+1:j+1]
                    d[f'drawdown_from_prior_{n}d_high_close_proxy_pct']=(float(prev.c)/float(prices.max())-1)*100
                else:d[f'drawdown_from_prior_{n}d_high_close_proxy_pct']=np.nan
            if a.include_signal_day:
                d['signal_close_return_pct']=(float(stock.iloc[i].c/stock.iloc[j].c)-1)*100
            items.append(d)
            print('REF',ticker,'PREVIOUS',str(prev.date.date()),'RP20',d.get('relative_pp_20d'),'RP60',d.get('relative_pp_60d'),flush=True)
    engine.dispose()
    result=pd.DataFrame(items)
    result.to_csv(out/'seven_reference_relative_performance.csv',index=False,encoding='utf-8-sig')
    long=[]
    for r in items:
        for n in WINDOWS:
            long.append({'symbol':r['symbol'],'signal_date':r['signal_date'],'previous_session':r['previous_session'],'lookback_sessions':n,'stock_return_pct':r[f'stock_{n}d'],'sp500_return_pct':r[f'sp500_{n}d'],'relative_performance_pp':r[f'relative_pp_{n}d'],'relative_ratio_pct':r[f'relative_ratio_pct_{n}d'],'benchmark_symbol':bench_name})
    pd.DataFrame(long).to_csv(out/'seven_reference_relative_by_window.csv',index=False,encoding='utf-8-sig')
    report={'status':'complete','reference_count':len(items),'missing_references':missing,'benchmark':bench_name,'benchmark_type':('ETF proxy tracking S&P500, not exact S&P500 index' if bench_name in ('SPY','VOO','IVV') else 'Index'),'benchmark_candidates_examined':attempted,'indicator_timing':'Previous completed trading session only','lookback_trading_sessions':WINDOWS,'relative_performance_pp':'Stock percent return minus index percent return, same stock-session endpoints','relative_ratio_pct':'(1+stock_return)/(1+index_return)-1, in percent','cautions':['Reference sample is only seven stocks; cannot infer predictive performance from it alone.','Compare with other stock-date controls, and validate on data not used to fit rules.','Corporate-action-adjusted price consistency is critical.','SPY/VOO/IVV is a tracking ETF proxy, NOT the exact S&P500 index.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'status':'complete','benchmark':bench_name,'references':len(items),'out':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
