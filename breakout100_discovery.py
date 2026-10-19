#!/usr/bin/env python3
"""US 100% breakout discovery. Research only; no orders or future-leaking predictors."""
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

SHOWCASE = {'BGMS':'2026-06-04','YDKG':'2025-08-22','ZIM':'2024-07-29',
            'FROG':'2026-04-15','MU':'2026-03-02','IMSR':'2025-06-24','IMNM':'2025-07-09'}
FEATURES=['ret5','ret10','ret20','ret60','ret120','volatility20','atr14_pct',
          'rvol20','dollar_volume20','rsi14','dist_ma20','dist_ma50','dist_ma120',
          'distance_high60_pct','range20_pct','up_days20_pct','daily_candle_pct']

def us_symbol(x):
    x=str(x).upper().strip()
    return bool(re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?',x)) and not x.endswith(('.CA','.SR'))

def safe_div(a,b):return a/b if b and np.isfinite(b) else np.nan

def feature_at(df,i):
    # Feature window ends on index i: signal day close, no bars after i.
    a=df.iloc[:i+1]; c=a.c.to_numpy(dtype=float); h=a.h.to_numpy(dtype=float)
    l=a.l.to_numpy(dtype=float); v=a.v.to_numpy(dtype=float); o=a.o.to_numpy(dtype=float)
    if len(c)<121 or c[-1]<=0: return None
    d=np.diff(c); gains=np.maximum(d[-14:],0).mean(); losses=np.maximum(-d[-14:],0).mean()
    rsi=100 if losses==0 and gains>0 else 50 if losses==0 else 100-100/(1+gains/losses)
    prev=np.r_[c[0],c[:-1]]
    tr=np.maximum.reduce([h-l,abs(h-prev),abs(l-prev)])
    ret=np.diff(np.log(c[-21:]))
    out={'signal_close':c[-1],'signal_open':o[-1],
         'ret5':(c[-1]/c[-6]-1)*100,'ret10':(c[-1]/c[-11]-1)*100,
         'ret20':(c[-1]/c[-21]-1)*100,'ret60':(c[-1]/c[-61]-1)*100,
         'ret120':(c[-1]/c[-121]-1)*100,
         'volatility20':np.std(ret,ddof=1)*np.sqrt(252)*100,
         'atr14_pct':tr[-14:].mean()/c[-1]*100,
         'rvol20':safe_div(v[-1],np.nanmean(v[-21:-1])),
         'dollar_volume20':np.nanmedian(c[-20:]*v[-20:]),
         'rsi14':rsi,
         'dist_ma20':(c[-1]/c[-20:].mean()-1)*100,
         'dist_ma50':(c[-1]/c[-50:].mean()-1)*100,
         'dist_ma120':(c[-1]/c[-120:].mean()-1)*100,
         'distance_high60_pct':(c[-1]/h[-60:].max()-1)*100,
         'range20_pct':(h[-20:].max()/l[-20:].min()-1)*100 if l[-20:].min()>0 else np.nan,
         'up_days20_pct':np.mean(np.diff(c[-21:])>0)*100,
         'daily_candle_pct':(c[-1]/o[-1]-1)*100 if o[-1]>0 else np.nan}
    return out

def analyze_symbol(df, start_date, end_date, min_dollar_volume):
    symbol=df.symbol.iloc[0]; dates=df.date.to_numpy(); n=len(df)
    month_seen=set(); out=[]
    for i in range(120,n):
        dt=str(dates[i])[:10]
        if dt<start_date or dt>end_date:continue
        month=dt[:7]
        showcase=(symbol in SHOWCASE and SHOWCASE[symbol]==dt)
        monthly=month not in month_seen
        if not monthly and not showcase:continue
        if monthly:month_seen.add(month)
        if i+90>=n:continue # 90-session full follow-up for unbiased labels
        features=feature_at(df,i)
        if features is None or features['signal_close']<=0:continue
        if features['dollar_volume20'] < min_dollar_volume and not showcase:continue
        base=features['signal_close']
        row={'symbol':symbol,'date':dt,'month':month,'anchor':'showcase' if showcase else 'monthly',**features}
        for horizon in (30,60,90):
            highs=df.h.iloc[i+1:i+horizon+1].to_numpy(dtype=float)
            hits=np.flatnonzero(highs>=2*base)
            row[f'hit100_{horizon}']=int(len(hits)>0)
            row[f'max_high_return_{horizon}']=round((np.nanmax(highs)/base-1)*100,3)
            row[f'first_hit100_{horizon}']=int(hits[0]+1) if len(hits) else None
        out.append(row)
    return out

def load_csv(path):
    d=pd.read_csv(path)
    aliases={'session_date':'date','o':'o','h':'h','l':'l','c':'c','v':'v','open':'o','high':'h','low':'l','close':'c','volume':'v','ticker':'symbol'}
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d.columns and v not in d.columns})
    return d

def load_db(url, chunksize=100000):
    from sqlalchemy import create_engine, text
    engine=create_engine(url.replace('postgres://','postgresql://',1))
    # Stream ordered symbols in chunks to ensure one full ticker is processed at a time.
    with engine.connect().execution_options(stream_results=True) as conn:
        q=text("SELECT symbol, session_date AS date, o,h,l,c,v FROM market_candles_1d "
               "WHERE symbol !~ '\\.(CA|SR)$' ORDER BY symbol, session_date")
        carry=pd.DataFrame();
        for chunk in pd.read_sql(q,conn,chunksize=chunksize):
            d=pd.concat([carry,chunk],ignore_index=True)
            last=d.symbol.iloc[-1]; full=d[d.symbol!=last]; carry=d[d.symbol==last].copy()
            for _,g in full.groupby('symbol',sort=False):yield g
        if not carry.empty:yield carry

def prepare(d):
    d=d.copy(); d['symbol']=d.symbol.astype(str).str.upper().str.strip()
    d=d[d.symbol.map(us_symbol)]
    for x in ('o','h','l','c','v'):d[x]=pd.to_numeric(d[x],errors='coerce')
    d['date']=pd.to_datetime(d.date,errors='coerce').dt.strftime('%Y-%m-%d')
    d=d.dropna(subset=['symbol','date','o','h','l','c','v'])
    d=d[(d.o>0)&(d.c>0)&(d.h>=d[['o','c','l']].max(axis=1))&(d.l>0)&(d.l<=d[['o','c','h']].min(axis=1))&(d.v>=0)]
    return d.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)

def score(df, out):
    # Fixed, deliberately simple diagnostic; no optimized strategy fitted to 7 examples.
    train=df[df.date<'2025-01-01']; val=df[(df.date>='2025-01-01')&(df.date<'2026-01-01')]; test=df[df.date>='2026-01-01']
    summary=[]
    for name,d in (('discovery_2024',train),('validation_2025',val),('holdout_2026',test)):
        for h in (30,60,90):
            z=d[d.anchor=='monthly']; y=z[f'hit100_{h}']
            summary.append({'period':name,'horizon':h,'stock_months':len(z),'hits':int(y.sum()),
                            'base_rate':float(y.mean()) if len(y) else None})
    pd.DataFrame(summary).to_csv(out/'market_baselines.csv',index=False)
    show=df[df.anchor=='showcase'].copy()
    show.to_csv(out/'seven_showcase_features.csv',index=False)
    profile=[]
    for f in FEATURES:
        v=show[f].dropna(); b=df[df.anchor=='monthly'][f].dropna()
        profile.append({'feature':f,'showcase_n':len(v),'showcase_min':v.min() if len(v) else None,
                        'showcase_median':v.median() if len(v) else None,'showcase_max':v.max() if len(v) else None,
                        'market_median':b.median() if len(b) else None})
    pd.DataFrame(profile).to_csv(out/'seven_vs_market_feature_profile.csv',index=False)
    return {'showcase_found':sorted(show.symbol.unique().tolist()),
            'showcase_missing':[s for s in SHOWCASE if s not in set(show.symbol)],
            'unique_stocks':int(df.symbol.nunique()),'total_anchors':len(df),
            'disclaimer':'Showcase handpicked after success. Do not interpret its feature ranges as validated predictive rules. 2026 holdout is not untouched if its winners informed feature selection. Adjust splits/corporate actions and survivorship before execution.'}

def main():
    p=argparse.ArgumentParser(); source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--database-url',default=None);source.add_argument('--csv',default=None)
    p.add_argument('--out',default='/data/breakout100_discovery')
    p.add_argument('--start',default='2024-01-01');p.add_argument('--end',default='2026-06-30')
    p.add_argument('--min-dollar-volume',type=float,default=5_000_000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    groups=load_db(a.database_url) if a.database_url else (g for _,g in load_csv(a.csv).groupby('symbol',sort=False))
    allrows=[];count=0
    for g in groups:
        if not us_symbol(g.symbol.iloc[0]):continue
        count+=1
        allrows.extend(analyze_symbol(prepare(g),a.start,a.end,a.min_dollar_volume))
        if count%500==0:print('PROGRESS',count,'symbols',len(allrows),'anchors',flush=True)
    if not allrows:raise SystemExit('No mature anchors. Check source history, dates and 90-session follow-up.')
    d=pd.DataFrame(allrows)
    d.to_csv(out/'all_market_anchors.csv',index=False)
    report=score(d,out);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
