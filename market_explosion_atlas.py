#!/usr/bin/env python3
"""US Market Explosion Atlas: point-in-time monthly winners and pre-month DNA.

Inputs: daily OHLCV in CSV/Parquet files, one or many files, or SQL database.
No market data is downloaded or fabricated. All dated features use bars before
month start. Winners are retrospective labels, never historical buy signals.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sqlite3, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='1.0.0'
ALIASES = {'symbol':['symbol','ticker','stock','code'], 'date':['date','datetime','timestamp','time'],
           'open':['open','o'], 'high':['high','h'], 'low':['low','l'],
           'close':['close','c','adj_close','adjusted_close'], 'volume':['volume','vol','v']}

def args_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input', help='CSV/Parquet file or directory containing daily bars')
    g.add_argument('--database-url', help='SQLAlchemy URL (SQLite/Postgres), e.g. from DATABASE_URL')
    p.add_argument('--table', default='daily_prices',help='SQL table with daily OHLCV (SQL input)')
    p.add_argument('--out',default='/tmp/market_explosion_atlas')
    p.add_argument('--start',default='2000-01-01')
    p.add_argument('--end',default=None,help='Inclusive final completed calendar month; partial month excluded by default')
    p.add_argument('--include-partial',action='store_true',help='Include unfinished final calendar month (not recommended for comparisons)')
    p.add_argument('--top',type=int,default=20)
    p.add_argument('--explosion-threshold',type=float,default=100., help='monthly return percent')
    p.add_argument('--min-bars-month',type=int,default=10)
    p.add_argument('--min-bars-history',type=int,default=120)
    p.add_argument('--min-price',type=float,default=0.50, help='previous-month final close (USD)')
    p.add_argument('--min-median-dollar-volume',type=float,default=100000., help='last 20 sessions before month')
    p.add_argument('--max-controls-per-month',type=int,default=100)
    p.add_argument('--train-end',default='2018-12-31')
    p.add_argument('--validation-end',default='2022-12-31')
    p.add_argument('--allow-unadjusted',action='store_true', help='Acknowledge that unchecked corporate actions may create false explosions')
    p.add_argument('--adjusted',action='store_true', help='Declare source OHLC data consistently split-adjusted (volume must be adjusted consistently)')
    p.add_argument('--rebuild',action='store_true',help='Ignore already completed symbol cache')
    return p

def safe_col(df, name):
    lc={str(c).lower().strip():c for c in df.columns}
    return next((lc[x] for x in ALIASES[name] if x in lc),None)

def normalize(raw, fallback=None):
    ren={}
    for k in ALIASES:
        c=safe_col(raw,k)
        if c is not None: ren[c]=k
    df=raw.rename(columns=ren).copy()
    if 'symbol' not in df:
        if fallback is None: raise ValueError('No symbol column; use one file per ticker, e.g. AAPL.csv')
        df['symbol']=fallback
    required=['symbol','date','open','high','low','close','volume']
    missing=[x for x in required if x not in df]
    if missing:raise ValueError('Missing '+','.join(missing))
    df=df[required].copy()
    df['symbol']=df['symbol'].astype(str).str.upper().str.strip()
    df['date']=pd.to_datetime(df['date'],errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['symbol','date','open','high','low','close'])
    df=df[(df['symbol']!='')&(df['close']>0)&(df['open']>0)&(df['high']>0)&(df['low']>0)&
          (df['high']>=df[['open','close','low']].max(axis=1))&
          (df['low']<=df[['open','close','high']].min(axis=1))]
    return df.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')

def read_sources(a):
    if a.input:
        path=Path(a.input)
        paths=sorted([p for p in path.rglob('*') if p.suffix.lower() in ['.csv','.parquet','.pq']]) if path.is_dir() else [path]
        if not paths:raise ValueError('No CSV/Parquet files found')
        for p in paths:
            try:
                raw=pd.read_csv(p) if p.suffix.lower()=='.csv' else pd.read_parquet(p)
                yield normalize(raw,p.stem.upper())
            except Exception as e: print(f'WARN INPUT {p}: {e}',flush=True)
    else:
        from sqlalchemy import create_engine,inspect,text
        url=a.database_url
        if url.startswith('postgres://'):url='postgresql://'+url[len('postgres://'):]
        engine=create_engine(url)
        if not a.table.replace('_','').isalnum():raise ValueError('Invalid SQL table name')
        inspector=inspect(engine)
        available=inspector.get_table_names()
        if a.table not in available:raise ValueError(f'Table {a.table!r} not found; examples: {available[:30]}')
        fields=[c['name'] for c in inspector.get_columns(a.table)]
        pick={}
        for k in ALIASES:
            match=next((c for c in fields if c.lower() in ALIASES[k]),None)
            if match:pick[k]=match
        if 'symbol' not in pick or 'date' not in pick:raise ValueError('SQL table needs symbol and date columns')
        query=f'SELECT * FROM "{a.table}"'
        with engine.connect() as con:
            # Streaming iterator avoids huge database allocation at once.
            for chunk in pd.read_sql_query(text(query),con,chunksize=150000):
                yield normalize(chunk)

def feature_frame(history):
    """Only history strictly before the first date of the ranked month."""
    if history.empty:return {}
    h=history.sort_values('date')
    c=h.close.to_numpy(dtype=float);v=h.volume.to_numpy(dtype=float)
    hi=h.high.to_numpy(dtype=float);lo=h.low.to_numpy(dtype=float)
    def ret(n):return 100*(c[-1]/c[-n-1]-1) if len(c)>n and c[-n-1]>0 else np.nan
    def rng(n):return 100*(np.nanmax(hi[-n:])/np.nanmin(lo[-n:])-1) if len(c)>=n and min(lo[-n:])>0 else np.nan
    def volatility(n):
        if len(c)<n+1:return np.nan
        z=np.diff(np.log(c[-n-1:]))
        return np.std(z,ddof=1)*np.sqrt(252)*100 if len(z)>1 else np.nan
    def avg(z):
        z=np.asarray(z,dtype=float);z=z[np.isfinite(z)]
        return float(z.mean()) if len(z) else np.nan
    def ratio(x,y):return x/y if np.isfinite(x) and np.isfinite(y) and y>0 else np.nan
    dv=c*v; dv[~np.isfinite(dv)]=np.nan
    daily_change=np.diff(c[-21:]) if len(c)>=21 else np.array([])
    vw=v[-20:] if len(v)>=20 else np.array([])
    ups=vw[daily_change>0] if len(vw)==20 and len(daily_change)==20 else np.array([])
    downs=vw[daily_change<0] if len(vw)==20 and len(daily_change)==20 else np.array([])
    prior_high=np.max(hi[-120:]) if len(c)>=120 else np.nan
    ma20=avg(c[-20:]) if len(c)>=20 else np.nan
    ma60=avg(c[-60:]) if len(c)>=60 else np.nan
    ma120=avg(c[-120:]) if len(c)>=120 else np.nan
    m20=avg(v[-20:]) if len(c)>=20 else np.nan
    m60=avg(v[-60:]) if len(c)>=60 else np.nan
    return {
      'pre_month_close':float(c[-1]), 'prior_history_bars':len(h),
      'median_dollar_volume20':float(np.nanmedian(dv[-20:])) if len(dv)>=20 and np.any(np.isfinite(dv[-20:])) else np.nan,
      'pre_ret5_pct':ret(5),'pre_ret10_pct':ret(10),'pre_ret20_pct':ret(20),'pre_ret60_pct':ret(60),'pre_ret120_pct':ret(120),'pre_ret252_pct':ret(252),
      'pre_range10_pct':rng(10),'pre_range20_pct':rng(20),'pre_range60_pct':rng(60),
      'pre_volatility20_pct':volatility(20),'pre_volatility60_pct':volatility(60),
      'pre_vol_ratio20to60':ratio(volatility(20),volatility(60)),
      'pre_volume20_over60':ratio(m20,m60),
      'pre_up_down_vol_ratio20':ratio(avg(ups),avg(downs)),
      'pre_up_days_share20_pct':float(100*np.mean(daily_change>0)) if len(daily_change)==20 else np.nan,
      'pre_drawdown_from120high_pct':float(100*(c[-1]/prior_high-1)) if np.isfinite(prior_high) and prior_high>0 else np.nan,
      'pre_distance_ma20_pct':float(100*(c[-1]/ma20-1)) if np.isfinite(ma20) and ma20>0 else np.nan,
      'pre_distance_ma60_pct':float(100*(c[-1]/ma60-1)) if np.isfinite(ma60) and ma60>0 else np.nan,
      'pre_distance_ma120_pct':float(100*(c[-1]/ma120-1)) if np.isfinite(ma120) and ma120>0 else np.nan,
      'pre_last20_compression_ratio':ratio(rng(20),rng(60))}

def compute_symbol(s,g,a):
    g=g.sort_values('date').drop_duplicates('date')
    g=g[g.date<=pd.Timestamp(a.end)] if a.end else g
    if len(g)<a.min_bars_history+a.min_bars_month:return []
    months=g.date.dt.to_period('M')
    output=[]
    for m, sub in g.groupby(months,sort=True):
        month_first=m.to_timestamp()
        if month_first<pd.Timestamp(a.start).to_period('M').to_timestamp():continue
        if not a.include_partial and m == pd.Timestamp.today().to_period('M'):continue
        history=g[g.date<month_first]
        if len(history)<a.min_bars_history or len(sub)<a.min_bars_month:continue
        pre=feature_frame(history)
        if pre['pre_month_close']<a.min_price or not np.isfinite(pre['median_dollar_volume20']) or pre['median_dollar_volume20']<a.min_median_dollar_volume:continue
        prior=pre['pre_month_close']
        # Consistent start: previous month's last close. Low-to-high is a separate
        # look-ahead DESCRIPTIVE statistic, not a tradable return.
        month_close=float(sub.close.iloc[-1]);max_high=float(sub.high.max())
        min_low=float(sub.low.min())
        ret_close=100*(month_close/prior-1)
        max_from_start=100*(max_high/prior-1)
        max_low_to_high=100*(max_high/min_low-1) if min_low>0 else np.nan
        output.append({'symbol':s,'month':str(m),'month_start':month_first.date().isoformat(),
          'month_last_bar':sub.date.max().date().isoformat(),'month_bars':len(sub),
          'month_open':float(sub.open.iloc[0]),'month_close':month_close,
          'month_return_pct':ret_close,'month_max_from_prior_close_pct':max_from_start,
          'month_low_to_high_range_pct':max_low_to_high,
          'month_min_low_date':sub.loc[sub.low.idxmin(),'date'].date().isoformat(),
          'month_max_high_date':sub.loc[sub.high.idxmax(),'date'].date().isoformat(),
          'month_max_drawdown_from_prior_close_pct':100*(min_low/prior-1),
          'month_volume_total':float(sub.volume.sum(min_count=1)) if sub.volume.notna().any() else np.nan,
          **pre})
    return output

def atomic_csv(df,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        df.to_csv(tmp,index=False)
    os.replace(tmp,path)

def run(a):
    if not a.adjusted and not a.allow_unadjusted:
        raise SystemExit('SAFETY CHECK: pass --adjusted for split-adjusted OHLCV or --allow-unadjusted to explicitly accept possible false return spikes.')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cache=out/'symbol_cache';cache.mkdir(exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ('database_url','rebuild')}
    config['input_source']=a.input or 'SQL_DATABASE'
    confighash=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()[:12]
    meta=out/'config.json';old=json.loads(meta.read_text()) if meta.exists() else {}
    if old.get('hash')!=confighash and any(cache.glob('*.csv')):
        raise SystemExit('Settings changed; use a new --out path (or delete old symbol_cache).')
    meta.write_text(json.dumps({'hash':confighash,'version':VERSION,'settings':config},indent=2,default=str))
    print('ATLAS START',json.dumps({'out':str(out),'config_hash':confighash}),flush=True)
    # SQL chunks may split symbols; consolidate per-symbol across chunks in local staging.
    staging=out/'staging';staging.mkdir(exist_ok=True)
    n_input=0
    for frame in read_sources(a):
        if frame.empty:continue
        for symbol,sg in frame.groupby('symbol',sort=False):
            # Stable hash to keep filenames safe for tickers like BRK.B
            sid=hashlib.sha1(symbol.encode()).hexdigest()[:16]
            f=staging/f'{sid}.csv'
            # Merge multiple source chunks for one ticker; last wins on duplicate dates.
            if f.exists():sg=pd.concat([pd.read_csv(f),sg],ignore_index=True)
            sg=normalize(sg,symbol)
            atomic_csv(sg,f)
            n_input+=len(sg)
    source_files=sorted(staging.glob('*.csv'))
    print('SOURCE STAGED SYMBOLS',len(source_files),'STAGED ROWS approx',n_input,flush=True)
    processed=0; skipped=0
    for file in source_files:
        output_file=cache/file.name
        if output_file.exists() and not a.rebuild:skipped+=1;continue
        try:
            df=normalize(pd.read_csv(file))
            symbol=str(df.symbol.iloc[0])
            rows=compute_symbol(symbol,df,a)
            atomic_csv(pd.DataFrame(rows) if rows else pd.DataFrame(columns=['symbol','month']),output_file)
        except Exception as ex:
            print('WARN SYMBOL',file.name,str(ex),flush=True)
            continue
        processed+=1
        if (processed+skipped)%100==0:
            print(f'PROGRESS done={processed+skipped}/{len(source_files)} calculated={processed} cached={skipped}',flush=True)
    frames=[]
    for file in cache.glob('*.csv'):
        try:
            f=pd.read_csv(file)
            if len(f) and 'month_return_pct' in f:frames.append(f)
        except Exception as ex:print('WARN CACHE',file,ex,flush=True)
    if not frames:raise SystemExit('No eligible month rows. Check source schema/history and filters.')
    df=pd.concat(frames,ignore_index=True).drop_duplicates(['symbol','month'])
    df=df.sort_values(['month','month_return_pct','symbol'],ascending=[True,False,True])
    df['rank_in_month']=df.groupby('month').cumcount()+1
    df['is_top_n']=df.rank_in_month<=a.top
    df['is_100pct_explosion']=df.month_return_pct>=a.explosion_threshold
    df['is_atlas_winner']=df.is_top_n|df.is_100pct_explosion
    df['year']=df.month.str[:4].astype(int)
    train=pd.Timestamp(a.train_end);val=pd.Timestamp(a.validation_end)
    df['split']=np.select([pd.to_datetime(df.month_start)<=train,pd.to_datetime(df.month_start)<=val],['TRAIN','VALIDATION'],default='HOLDOUT')
    atomic_csv(df,out/'all_eligible_stock_months.csv')
    winners=df[df.is_atlas_winner].copy()
    atomic_csv(winners,out/'monthly_explosion_winners.csv')
    # Controls drawn deterministically using fixed seed and price/liquidity era-matched
    # same-month pool; excludes all winners (including top N) and retrospective 100% labels.
    controls=[]
    for month, group in df[~df.is_atlas_winner].groupby('month'):
        n=min(a.max_controls_per_month,len(group))
        if n:
            controls.append(group.sample(n=n,random_state=20261016))
    ctrl=pd.concat(controls,ignore_index=True) if controls else df.iloc[0:0]
    atomic_csv(ctrl,out/'monthly_nonwinner_controls.csv')
    study=pd.concat([winners.assign(target=1),ctrl.assign(target=0)],ignore_index=True)
    atomic_csv(study,out/'dna_study_winners_controls.csv')
    # Note: top-N relative winner is label of rank, NOT necessarily an explosive return.
    summary=df.groupby('month').agg(eligible_stocks=('symbol','nunique'),top_n=('is_top_n','sum'),
        above_threshold=('is_100pct_explosion','sum'),max_monthly_return_pct=('month_return_pct','max'),
        median_monthly_return_pct=('month_return_pct','median')).reset_index()
    summary['split']=np.select([pd.to_datetime(summary.month+'-01')<=train,pd.to_datetime(summary.month+'-01')<=val],['TRAIN','VALIDATION'],default='HOLDOUT')
    atomic_csv(summary,out/'monthly_market_summary.csv')
    byyear=summary.assign(year=summary.month.str[:4]).groupby('year').agg(
        months=('month','size'),median_eligible_stocks=('eligible_stocks','median'),
        threshold_breakouts=('above_threshold','sum'),best_monthly_return_pct=('max_monthly_return_pct','max')).reset_index()
    atomic_csv(byyear,out/'annual_coverage_quality.csv')
    coverage={'source_symbols':len(source_files),'symbols_with_eligible_months':int(df.symbol.nunique()),
      'eligible_stock_months':len(df),'months':int(df.month.nunique()),'atlas_winner_rows':len(winners),
      'distinct_winner_symbols':int(winners.symbol.nunique()),'explosion_100pct_rows':int(df.is_100pct_explosion.sum()),
      'control_rows':len(ctrl),'min_date':str(df.month.min()),'max_date':str(df.month.max()),
      'adjusted_declared':a.adjusted,'warning':'Retrospective labels only, not trading signals. Survivorship, delistings, corporate actions and data gaps require vendor checks.'}
    (out/'quality_report.json').write_text(json.dumps(coverage,indent=2))
    print('ATLAS COMPLETED',json.dumps(coverage,indent=2),flush=True)
    print('FILES:',', '.join(p.name for p in out.glob('*.csv')),flush=True)

if __name__=='__main__':
    run(args_parser().parse_args())
