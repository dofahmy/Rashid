#!/usr/bin/env python3
"""Progressive decimal rounding of ATR14% to reach >= N distinct other US tickers, then evaluate 30 sessions."""
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def valid_symbol(s):
    s=str(s).upper().strip()
    return bool(re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?',s)) and not s.endswith(('.CA','.SR'))

def clean(group):
    d=group.copy()
    d['date']=pd.to_datetime(d['date'],errors='coerce').dt.strftime('%Y-%m-%d')
    for col in 'ohlcv': d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['date','o','h','l','c','v'])
    d=d[(d.o>0)&(d.c>0)&(d.l>0)&(d.v>=0)&(d.h>=d[['o','c','l']].max(axis=1))&(d.l<=d[['o','c','h']].min(axis=1))]
    return d.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)

def atr_series(d):
    c=d.c.to_numpy(dtype=float);h=d.h.to_numpy(dtype=float);l=d.l.to_numpy(dtype=float)
    prev=np.r_[c[0],c[:-1]]
    tr=np.maximum.reduce((h-l,np.abs(h-prev),np.abs(l-prev)))
    atr=np.full(len(d),np.nan)
    # 14 exact daily true ranges; same definition as previous discovery engine
    if len(d)>=14:
        atr[13:]=(np.cumsum(np.r_[0.,tr])[14:]-np.cumsum(np.r_[0.,tr])[:-14])/14 / c[13:]*100
        # Recalculate each ATR with numpy mean to preserve exact evaluation order
        # of breakout100_discovery.feature_at, so literal comparison is meaningful.
        atr[13:]=np.array([np.mean(tr[i-13:i+1])/c[i]*100 for i in range(13,len(d))])
    return atr

def iter_db(url,chunksize=100000):
    from sqlalchemy import create_engine,text
    engine=create_engine(url.replace('postgres://','postgresql://',1))
    try:
        with engine.connect().execution_options(stream_results=True) as conn:
            q=text("SELECT symbol, session_date AS date, o,h,l,c,v FROM market_candles_1d "
                   "WHERE symbol !~ '\\.(CA|SR)$' ORDER BY symbol, session_date")
            carry=pd.DataFrame()
            for chunk in pd.read_sql(q,conn,chunksize=chunksize):
                d=pd.concat([carry,chunk],ignore_index=True)
                last=d.symbol.iloc[-1]
                full=d[d.symbol!=last];carry=d[d.symbol==last].copy()
                for _,g in full.groupby('symbol',sort=False):yield g
            if not carry.empty:yield carry
    finally:engine.dispose()

def iter_csv(path):
    d=pd.read_csv(path)
    d=d.rename(columns={'session_date':'date','open':'o','high':'h','low':'l','close':'c','volume':'v'})
    for _,g in d.groupby('symbol',sort=False):yield g

def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--database-url');source.add_argument('--csv')
    p.add_argument('--out',default='/data/ydkg_atr_exact')
    p.add_argument('--reference-date',default='2025-08-22')
    p.add_argument('--start',default='2024-01-01')
    p.add_argument('--end',default='2026-06-30')
    p.add_argument('--horizon',type=int,default=30)
    p.add_argument('--min-stocks',type=int,default=20)
    p.add_argument('--max-decimals',type=int,default=8)
    p.add_argument('--min-decimals',type=int,default=0)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.horizon<1 or a.min_stocks<1 or a.min_decimals<0 or a.max_decimals<a.min_decimals:
        raise SystemExit('Invalid horizon, min-stocks or decimal precision')
    # Extract reference first, without assuming immediately prior calendar day was traded.
    reference=None
    groups=iter_db(a.database_url) if a.database_url else iter_csv(a.csv)
    for group in groups:
        symbol=str(group.symbol.iloc[0]).upper().strip()
        if symbol!='YDKG':continue
        d=clean(group)
        indices=np.flatnonzero(d.date.to_numpy()==a.reference_date)
        if not len(indices):raise SystemExit('YDKG reference date not available in valid daily candles')
        i=int(indices[0]);atr=atr_series(d)
        if i<14:raise SystemExit('YDKG has insufficient pre-signal ATR history')
        reference={'symbol':'YDKG','reference_date':a.reference_date,'previous_date':str(d.date.iloc[i-1]),
                   'atr14_pct_previous':float(atr[i-1]),'atr14_pct_signal':float(atr[i]),
                   'calculation':'mean(last 14 true ranges) / daily close * 100; numpy float64; literal ==',
                   'horizon':a.horizon}
        break
    if reference is None:raise SystemExit('YDKG not present in database')
    print('REFERENCE',json.dumps(reference,ensure_ascii=False),flush=True)
    (out/'reference.json').write_text(json.dumps(reference,indent=2,ensure_ascii=False))
    # First pass: count distinct non-reference tickers at each precision, independently per test.
    precisions=list(range(a.max_decimals,a.min_decimals-1,-1))
    buckets={kind:{dp:set() for dp in precisions} for kind in ('previous','signal_day','both')}
    valid=0;studied=0
    groups=iter_db(a.database_url) if a.database_url else iter_csv(a.csv)
    for group in groups:
        symbol=str(group.symbol.iloc[0]).upper().strip()
        if not valid_symbol(symbol) or symbol=='YDKG':continue
        studied+=1;d=clean(group)
        if len(d)<14+a.horizon:continue
        atr=atr_series(d); dates=d.date.to_numpy()
        ix=np.flatnonzero((dates>=a.start)&(dates<=a.end))
        ix=ix[(ix>=14)&(ix+a.horizon<len(d))]
        valid+=len(ix)
        if not len(ix):continue
        previous=atr[ix-1];today=atr[ix]
        for dp in precisions:
            pm=np.round(previous,dp)==round(reference['atr14_pct_previous'],dp)
            dm=np.round(today,dp)==round(reference['atr14_pct_signal'],dp)
            if pm.any():buckets['previous'][dp].add(symbol)
            if dm.any():buckets['signal_day'][dp].add(symbol)
            if (pm&dm).any():buckets['both'][dp].add(symbol)
        if studied%500==0:print('COUNT PROGRESS',studied,'symbols',flush=True)
    chosen={}
    for kind in buckets:
        chosen[kind]=next((dp for dp in precisions if len(buckets[kind][dp])>=a.min_stocks),None)
    print('CHOSEN DECIMAL PLACES',chosen,flush=True)
    count_report=[]
    for dp in precisions:
        count_report.append({'decimal_places':dp,**{kind+'_unique_other_stocks':len(buckets[kind][dp]) for kind in buckets}})
    pd.DataFrame(count_report).to_csv(out/'YDKG_ATR_rounding_counts.csv',index=False)
    # Second pass: evaluate only at selected precision. No repeated discovery computations.
    results=[];studied2=0
    groups=iter_db(a.database_url) if a.database_url else iter_csv(a.csv)
    for group in groups:
        symbol=str(group.symbol.iloc[0]).upper().strip()
        if not valid_symbol(symbol) or symbol=='YDKG':continue
        studied2+=1;d=clean(group)
        if len(d)<14+a.horizon:continue
        atr=atr_series(d);dates=d.date.to_numpy();opens=d.o.to_numpy(float)
        closes=d.c.to_numpy(float);highs=d.h.to_numpy(float);lows=d.l.to_numpy(float)
        ix=np.flatnonzero((dates>=a.start)&(dates<=a.end))
        ix=ix[(ix>=14)&(ix+a.horizon<len(d))]
        for i in ix:
            flags={}
            for kind,v,r in [('previous',atr[i-1],reference['atr14_pct_previous']),('signal_day',atr[i],reference['atr14_pct_signal'])]:
                dp=chosen[kind]
                flags[kind]=dp is not None and bool(round(float(v),dp)==round(float(r),dp))
            bothdp=chosen['both']
            flags['both']=bothdp is not None and bool(round(float(atr[i-1]),bothdp)==round(reference['atr14_pct_previous'],bothdp) and round(float(atr[i]),bothdp)==round(reference['atr14_pct_signal'],bothdp))
            if not any(flags.values()):continue
            b=closes[i];future_highs=highs[i+1:i+a.horizon+1];future_lows=lows[i+1:i+a.horizon+1]
            hits100=np.flatnonzero(future_highs>=2*b)
            results.append({'symbol':symbol,'signal_date':str(dates[i]),'prev_date':str(dates[i-1]),
              'previous_atr14_pct':repr(float(atr[i-1])),'signal_atr14_pct':repr(float(atr[i])),
              'match_previous':flags['previous'],'match_signal_day':flags['signal_day'],'match_both':flags['both'],
              'entry_close':b,'entry_open':opens[i], 'high30':float(np.max(future_highs)),
              'max_high_return_30_pct':round((np.max(future_highs)/b-1)*100,6),
              'close30':float(closes[i+a.horizon]),'close_return_30_pct':round((closes[i+a.horizon]/b-1)*100,6),
              'worst_low_return_30_pct':round((np.min(future_lows)/b-1)*100,6),
              'hit100_within30':bool(len(hits100)),
              'first_hit100_session':int(hits100[0]+1) if len(hits100) else None,
              'first_max_high_session':int(np.argmax(future_highs)+1)})
        if studied2%500==0:print('EVALUATION PROGRESS',studied2,'symbols; matches',len(results),flush=True)
    columns=['symbol','signal_date','prev_date','previous_atr14_pct','signal_atr14_pct',
             'match_previous','match_signal_day','match_both','entry_close','entry_open','high30',
             'max_high_return_30_pct','close30','close_return_30_pct','worst_low_return_30_pct',
             'hit100_within30','first_hit100_session','first_max_high_session']
    frame=pd.DataFrame(results,columns=columns)
    frame.to_csv(out/'YDKG_ATR_rounded_matches_30sessions.csv',index=False)
    summary=[]
    for kind,key in [('previous','match_previous'),('signal_day','match_signal_day'),('both','match_both')]:
        part=frame[frame[key]]
        part.to_csv(out/f'YDKG_ATR_{kind}_rounded_30sessions.csv',index=False)
        summary.append({'test':kind,'selected_decimal_places':chosen[kind],
                        'unique_other_stocks':int(part.symbol.nunique()),'matched_stock_dates':len(part),
                        'hit100_30':int(part.hit100_within30.sum()),
                        'unique_hit_stocks':int(part[part.hit100_within30].symbol.nunique()),
                        'mean_max_high_30_pct':float(part.max_high_return_30_pct.mean()) if len(part) else None})
    report={'reference':reference,'symbols_processed_excluding_reference':studied,
            'stock_dates_checked':valid,'min_distinct_other_stocks':a.min_stocks,
            'precision_search':count_report,'results':summary,
            'note':'Round both reference and candidate to the same decimal places, without percentile range. Exclude YDKG; minimum 20 DISTINCT tickers, not 20 stock-dates. Separate previous/day/both precision. Only cases with 30 follow-up sessions.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
