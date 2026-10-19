#!/usr/bin/env python3
"""Read-only, point-in-time drawdown audit of fixed YDKG signal dates.
Measures PRE-signal lookback High-to-previous Close (no signal/future leakage),
then compares previously saved forward 30-session outcomes. Does NOT optimize rules.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam

HORIZONS=(20,60,120,252)
BINS=[-0.00001,38,50,61,80,101]
LABELS=['<38%','38–50%','50–61%','61–80%','>=80%']

def f(x):
    try:
        a=float(x)
        return round(a,5) if np.isfinite(a) else None
    except (ValueError,TypeError):return None

def clean_candles(df):
    if df.empty:return df
    df=df.copy()
    df['date']=pd.to_datetime(df['date'],errors='coerce').dt.strftime('%Y-%m-%d')
    for k in ('h','c'):df[k]=pd.to_numeric(df[k],errors='coerce')
    df=df.dropna(subset=['date','h','c']);df=df[(df.h>0)&(df.c>0)&(df.h>=df.c)]
    return df.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')

def drawdown(g,i,n):
    # i is SIGNAL session index; use n historical sessions ending at i-1.
    if i<n:return None,None
    window=g.iloc[i-n:i]
    peak=float(window.h.max())
    prev=float(g.c.iloc[i-1])
    return (1-prev/peak)*100, peak

def summaries(df,col,groupcols):
    results=[]
    for grp,ss in df.groupby(groupcols,dropna=False,observed=True):
        if not isinstance(grp,tuple):grp=(grp,)
        row={k:v for k,v in zip(groupcols,grp)}
        for label,group in [('all',ss),('winner100',ss[ss.pct_max_high_first30>=100])]:
            if label=='all':
                row.update(cases=len(group),symbols=int(group.symbol.nunique()),
                           hit_2_5=int((group.pct_max_high_first30>=2.5).sum()),
                           hit_30=int((group.pct_max_high_first30>=30).sum()),
                           hit_100=int((group.pct_max_high_first30>=100).sum()),
                           median_max_high_30_pct=f(group.pct_max_high_first30.median()),
                           median_close_30_pct=f(group.pct_close_session30.median()),
                           median_latest_close_pct=f(group.pct_latest_close_from_signal.median()) if 'pct_latest_close_from_signal' in group else None,
                           median_worst_low_30_pct=f(group.pct_worst_low_first30.median()) if 'pct_worst_low_first30' in group else None)
        row['rate_2_5_pct']=f(row['hit_2_5']/row['cases']*100) if row['cases'] else None
        row['rate_30_pct']=f(row['hit_30']/row['cases']*100) if row['cases'] else None
        row['rate_100_pct']=f(row['hit_100']/row['cases']*100) if row['cases'] else None
        results.append(row)
    return pd.DataFrame(results)

def build_lock(df, groups, hold=30):
    # Conservative reference: no second signal in same stock for 30 sessions,
    # irrespective of profit targets (target-specific lock handled separately elsewhere).
    rows=[]
    for sym,g in df.sort_values(['symbol','signal_date']).groupby('symbol'):
        candles=groups.get(sym); nlocked=-1
        date_to_i={d:i for i,d in enumerate(candles.date)} if candles is not None else {}
        for r in g.itertuples():
            ix=date_to_i.get(r.signal_date)
            blocked=ix is None or ix <= nlocked
            if not blocked:nlocked=ix+hold
            rows.append((r.Index,'BLOCKED_30_SESSION_LOCK' if blocked else 'ELIGIBLE_30_SESSION_LOCK'))
    return pd.Series(dict(rows))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--database-url',default=os.environ.get('DATABASE_URL'))
    a.add_argument('--signals',default='/data/ydkg_full_performance/YDKG_all_signal_dates_through_latest.csv')
    a.add_argument('--out',default='/data/ydkg_drawdown_audit')
    a.add_argument('--market-symbols',default='SPY,QQQ,IWM,^GSPC,SPX')
    a.add_argument('--sector-map',default=None,help='Optional CSV with symbol,sector,sector_etf fields; ETFs must exist in DB')
    a.add_argument('--source-exclusion',type=float,default=300.0)
    x=a.parse_args()
    if not x.database_url:a.error('DATABASE_URL missing')
    out=Path(x.out);out.mkdir(parents=True,exist_ok=True)
    sig=pd.read_csv(x.signals,dtype={'symbol':str,'signal_date':str})
    for k in ('symbol','signal_date','pct_max_high_first30'):
        if k not in sig.columns:a.error('Missing input column '+k)
    sig['symbol']=sig.symbol.str.upper().str.strip();sig['signal_date']=sig.signal_date.str[:10]
    for k in ('pct_max_high_first30','pct_close_session30','pct_latest_close_from_signal','pct_worst_low_first30'):
        if k not in sig:sig[k]=np.nan
        sig[k]=pd.to_numeric(sig[k],errors='coerce')
    sig=sig.sort_values(['symbol','signal_date']).reset_index(drop=True)
    sector_map=None
    if x.sector_map:
        sector_map=pd.read_csv(x.sector_map,dtype=str).fillna('')
        if not {'symbol','sector','sector_etf'}.issubset(sector_map.columns):a.error('sector map requires symbol,sector,sector_etf columns')
        sector_map['symbol']=sector_map.symbol.str.upper().str.strip()
        sector_map=sector_map.drop_duplicates('symbol').set_index('symbol')
    market_options=[s.strip().upper() for s in x.market_symbols.split(',') if s.strip()]
    more=sorted(set(sector_map.sector_etf.str.upper())-set([''])) if sector_map is not None else []
    symbols=sorted(set(sig.symbol)|set(market_options)|set(more))
    engine=create_engine(x.database_url.replace('postgres://','postgresql://',1))
    stmt=text('SELECT symbol, session_date AS date, h, c FROM market_candles_1d WHERE symbol IN :syms ORDER BY symbol,session_date').bindparams(bindparam('syms',expanding=True))
    with engine.connect() as conn:raw=pd.read_sql(stmt,conn,params={'syms':symbols})
    engine.dispose()
    raw=clean_candles(raw)
    groups={s:g.reset_index(drop=True) for s,g in raw.groupby('symbol')}
    market=next((s for s in market_options if s in groups and len(groups[s])>260),None)
    print('MARKET_PROXY', market or 'UNAVAILABLE','CANDLES',len(raw),'SIGNALS',len(sig),flush=True)
    def market_mov(sym,day,n):
        g=groups.get(sym)
        if g is None:return None
        ix=g.index[g.date.eq(day)]
        if not len(ix):return None
        i=int(ix[0]);
        if i<n or i<1:return None
        prev=float(g.c.iloc[i-1]);base=float(g.c.iloc[i-n]);
        return (prev/base-1)*100
    rows=[]
    for r in sig.itertuples(index=False):
        g=groups.get(r.symbol)
        if g is None: continue
        hits=g.index[g.date.eq(r.signal_date)]
        if not len(hits):continue
        i=int(hits[0]);q=r._asdict();q['previous_session']=str(g.date.iloc[i-1]) if i>=1 else None
        for n in HORIZONS:
            dd,peak=drawdown(g,i,n)
            q[f'pre_drawdown_{n}_pct']=f(dd)
            q[f'pre_peak_{n}_high']=f(peak)
        q['prior_60_return_pct']=f((g.c.iloc[i-1]/g.c.iloc[i-61]-1)*100) if i>=61 else None
        q['prior_120_return_pct']=f((g.c.iloc[i-1]/g.c.iloc[i-121]-1)*100) if i>=121 else None
        q['market_proxy']=market
        q['market_prior60_return_pct']=f(market_mov(market,r.signal_date,60)) if market else None
        q['market_prior120_return_pct']=f(market_mov(market,r.signal_date,120)) if market else None
        q['prior60_minus_market_pct']=f(q['prior_60_return_pct']-q['market_prior60_return_pct']) if q['prior_60_return_pct'] is not None and q['market_prior60_return_pct'] is not None else None
        q['prior120_minus_market_pct']=f(q['prior_120_return_pct']-q['market_prior120_return_pct']) if q['prior_120_return_pct'] is not None and q['market_prior120_return_pct'] is not None else None
        mp=sector_map.loc[r.symbol] if sector_map is not None and r.symbol in sector_map.index else None
        q['sector']=str(mp['sector']) if mp is not None else None
        q['sector_etf']=str(mp['sector_etf']).upper() if mp is not None else None
        q['sector_prior60_return_pct']=f(market_mov(q['sector_etf'],r.signal_date,60)) if q['sector_etf'] else None
        q['sector_prior120_return_pct']=f(market_mov(q['sector_etf'],r.signal_date,120)) if q['sector_etf'] else None
        q['prior60_minus_sector_pct']=f(q['prior_60_return_pct']-q['sector_prior60_return_pct']) if q['sector_prior60_return_pct'] is not None and q['prior_60_return_pct'] is not None else None
        q['sector_confirmed_declining60']= bool(q['sector_prior60_return_pct']<0) if q['sector_prior60_return_pct'] is not None else None
        q['market_confirmed_declining60']= bool(q['market_prior60_return_pct']<0) if q['market_prior60_return_pct'] is not None else None
        rows.append(q)
    df=pd.DataFrame(rows)
    if df.empty:raise RuntimeError('None of the signal dates found in PostgreSQL')
    # Exclude whole SYMBOL (every date) if any signal's first30 High > cutoff.
    excluded=set(df.loc[df.pct_max_high_first30>x.source_exclusion,'symbol'])
    df['cohort']='EXCLUDED_SYMBOL_GT_300' if False else 'KEPT'
    df.loc[df.symbol.isin(excluded),'cohort']='EXCLUDED_SYMBOL_GT_300'
    df['lock_status']=build_lock(df,groups)
    # Primary 252-session category: a missing 252 lookback is UNKNOWN, never presumed safe.
    for n in HORIZONS:
        vals=df[f'pre_drawdown_{n}_pct'].clip(lower=0)
        df[f'drawdown_bin_{n}']=pd.cut(vals,bins=BINS,labels=LABELS,right=False).astype(object)
        df[f'drawdown_bin_{n}']=df[f'drawdown_bin_{n}'].fillna('UNKNOWN')
        df[f'pre_80_exclusion_{n}']=np.where(df[f'pre_drawdown_{n}_pct'].isna(),'UNKNOWN',np.where(df[f'pre_drawdown_{n}_pct']>=80,'EXCLUDE_GE_80','ALLOW_LT_80'))
    df['preferred_38_50_252']=df['drawdown_bin_252'].eq('38–50%')
    df['sector_check']='NO_SECTOR_DATA'
    df.loc[df['sector_prior60_return_pct'].notna(),'sector_check']='AVAILABLE'
    df['market_check']=np.where(df['market_prior60_return_pct'].notna(),'AVAILABLE','NO_MARKET_PROXY_DATA')
    df.to_csv(out/'all_signals_drawdown_audit.csv',index=False,encoding='utf-8-sig')
    sub=df[df.cohort=='KEPT'].copy()
    sub.to_csv(out/'kept_61_signals_drawdown_audit.csv',index=False,encoding='utf-8-sig')
    separate=[]
    for name,g in [('ALL_71',df),('EXCLUDING_GT300',sub),('EXCLUDING_GT300_LOCK30',sub[sub.lock_status=='ELIGIBLE_30_SESSION_LOCK'])]:
        for n in HORIZONS:
            q=summaries(g,f'drawdown_bin_{n}',['drawdown_bin_'+str(n)])
            q['cohort']=name;q['lookback_sessions']=n
            separate.append(q)
    result=pd.concat(separate,ignore_index=True)
    result=result[['cohort','lookback_sessions']+[v for v in result.columns if v not in ('cohort','lookback_sessions')]]
    result.to_csv(out/'drawdown_buckets_20_60_120_252.csv',index=False,encoding='utf-8-sig')
    # Show 252 scenarios in full including absolute winners and failures, not just winning precision.
    checks=[]
    for name,g in [('ALL_71',df),('EXCLUDING_GT300',sub),('EXCLUDING_GT300_LOCK30',sub[sub.lock_status=='ELIGIBLE_30_SESSION_LOCK'])]:
        for strat,mask in [('NO_EXTRA_FILTER',pd.Series(True,index=g.index)),('ONLY_38_TO_50',g.preferred_38_50_252),('ALLOW_DROP_BELOW_61',g.pre_drawdown_252_pct.lt(61)),('ALLOW_DROP_BELOW_80',g.pre_drawdown_252_pct.lt(80)),('ONLY_38_TO_61',g.pre_drawdown_252_pct.ge(38)&g.pre_drawdown_252_pct.lt(61)),('EXCLUDE_GE80_AND_REQUIRE_SECTOR_DOWN',g.pre_drawdown_252_pct.lt(80)&g.sector_confirmed_declining60.eq(True))]:
            v=g[mask]
            checks.append({'cohort':name,'filter':strat,'signals':len(v),'stocks':int(v.symbol.nunique()),'historical_100_hits':int((v.pct_max_high_first30>=100).sum()),'historical_30_hits':int((v.pct_max_high_first30>=30).sum()),'rate_100_pct':f((v.pct_max_high_first30>=100).mean()*100) if len(v) else None,'rate_30_pct':f((v.pct_max_high_first30>=30).mean()*100) if len(v) else None,'median_high_30_pct':f(v.pct_max_high_first30.median()),'median_close_30_pct':f(v.pct_close_session30.median()),'unknown_252':int(v.pre_drawdown_252_pct.isna().sum()),'sector_data_available':int(v.sector_prior60_return_pct.notna().sum())})
    pd.DataFrame(checks).to_csv(out/'hypothesis_38_50_vs_61_80_and_sector.csv',index=False,encoding='utf-8-sig')
    rep={'status':'complete','signals_loaded':int(len(sig)),'signals_found_in_db':int(len(df)),
         'symbols_in_full_sample':int(df.symbol.nunique()),'whole_symbols_excluded_gt_300pct':sorted(excluded),
         'remaining_signals':int(len(sub)),'remaining_symbols':int(sub.symbol.nunique()),
         'market_proxy':market,'market_proxy_data_points':int(df.market_prior60_return_pct.notna().sum()),
         'sector_returns_available':int(df.sector_prior60_return_pct.notna().sum()),
         'prior252_missing':int(df.pre_drawdown_252_pct.isna().sum()),
         'prior252_missing_kept':int(sub.pre_drawdown_252_pct.isna().sum()),
         'rule':'252-session PRE-signal High to previous day Close, excluding the signal day; no forward data used for classification',
         'benchmark':'same-date 60/120 session CLOSE returns ending on PREVIOUS session, not drawdown since sector index peak',
         'warning':['Historical same-sample investigation: associations, not validated predictive rules.',
         'Missing market/sector source is UNKNOWN, never a confirmed synchronized decline.',
         'Raw High/Close may be split-adjusted or inconsistent. Extreme reverse-split stocks require manual validation.',
         'The optional 30-session no-overlap lock is conservative, not dynamic TP-based closing.',
         'The >300% exclusion is itself forward-looking and may be used ONLY for retrospective analysis, NEVER live stock selection.']}
    (out/'report.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(rep,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
