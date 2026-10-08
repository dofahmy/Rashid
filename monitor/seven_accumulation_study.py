"""Historical research: Does Double-7 Repeat add information beyond price consolidation?

Runs against the project's existing EGX/SP500 loaders without altering any tables.
Uses only signals known as of their historical signal date. No lookahead in features.
Run: python -m monitor.seven_accumulation_study --market egypt --output-dir /tmp/seven-study
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS=(5,10,20,40,60,100)
TARGETS=(10,20,50)


def _prices(df):
    x=df.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True).copy()
    for k in ('close','high','low','volume'):
        if k not in x: x[k]=np.nan
        x[k]=pd.to_numeric(x[k],errors='coerce')
    x['date']=pd.to_datetime(x['date']).dt.normalize()
    return x


def _has_ohlc(x, end):
    sub=x.iloc[max(0,end-59):end+1]
    return all(c in sub and sub[c].notna().sum()>=min(15,len(sub)) for c in ('high','low'))


def features(x, i):
    """Backward-only consolidation descriptors; None means not measurable."""
    result={}
    for n in (20,40,60):
        z=x.iloc[max(0,i-n+1):i+1]
        prefix=f'pre{n}_'
        if len(z)<n or z.close.isna().any() or (z.close<=0).any():
            for key in ('range_pct','return_pct','range_contraction','volume_trend','higher_lows'):
                result[prefix+key]=None
            continue
        hi=z.high if z.high.notna().all() else z.close
        lo=z.low if z.low.notna().all() else z.close
        result[prefix+'range_pct']=round((hi.max()/lo.min()-1)*100,4) if lo.min()>0 else None
        result[prefix+'return_pct']=round((z.close.iloc[-1]/z.close.iloc[0]-1)*100,4)
        halves=np.array_split(np.arange(n),2)
        def pct_range(positions):
            h=hi.iloc[positions].max(); l=lo.iloc[positions].min()
            return (h/l-1)*100 if l>0 else np.nan
        first,second=map(pct_range,halves)
        result[prefix+'range_contraction']=round(second/first,4) if first>0 and np.isfinite(second) else None
        vs=z.volume
        a=vs.iloc[:n//2].mean();b=vs.iloc[n//2:].mean()
        result[prefix+'volume_trend']=round(b/a,4) if np.isfinite(a) and a>0 and np.isfinite(b) else None
        result[prefix+'higher_lows']=bool(lo.iloc[n//2:].min()>lo.iloc[:n//2].min())
    # fixed, pre-declared proxy; no future data
    c=result.get('pre20_range_contraction');v=result.get('pre20_volume_trend');r=result.get('pre20_return_pct')
    result['consolidation_proxy']=(bool(c<=0.8 and v<=1.2 and r is not None and abs(r)<=15)
                                   if c is not None and v is not None and r is not None else None)
    return result


def outcomes(x,i):
    """Forward outcomes from next session OPEN (executable proxy), not hindsight repeat price."""
    future=x.iloc[i+1:i+101]
    opening=x.open.iloc[i+1] if 'open' in x and i+1<len(x) else np.nan
    entry=float(opening) if pd.notna(opening) and opening>0 else np.nan
    result={'entry_next_open':entry if np.isfinite(entry) else None,
            'available_forward_sessions':len(future)}
    for h in HORIZONS:
        z=future.iloc[:h];complete=len(z)==h
        result[f'complete_{h}d']=complete
        high=z.high if len(z) and z.high.notna().all() else pd.Series(dtype=float)
        low=z.low if len(z) and z.low.notna().all() else pd.Series(dtype=float)
        good=np.isfinite(entry) and len(high)==len(z) and len(low)==len(z) and len(z)>0
        result[f'max_rise_{h}d']=round((high.max()/entry-1)*100,3) if good else None
        result[f'max_drawdown_{h}d']=round((low.min()/entry-1)*100,3) if good else None
        for t in TARGETS:
            result[f'hit_{t}pct_{h}d']=(bool((high>=entry*(1+t/100)).any()) if good and complete else None)
    for t in TARGETS:
        result[f'first_{t}pct_session']=next((j for j,r in enumerate(future.itertuples(),1)
            if np.isfinite(entry) and pd.notna(r.high) and r.high>=entry*(1+t/100)),None)
    # Profit-target/stop hit ordering, same daily bar => unknown.
    for stop in (3,5):
        status='PENDING' if len(future)<20 else 'NEITHER'; at=None
        if not np.isfinite(entry): status='NO_OPEN'
        else:
            for j,r in enumerate(future.iloc[:20].itertuples(),1):
                if pd.isna(r.high) or pd.isna(r.low):status='NO_OHLC';break
                up=r.high>=entry*1.05;dn=r.low<=entry*(1-stop/100)
                if up or dn:
                    status='AMBIGUOUS' if up and dn else ('TARGET_FIRST' if up else 'STOP_FIRST')
                    at=j;break
        result[f'plus5_vs_minus{stop}']=status
        result[f'first_decision_minus{stop}']=at
    return result


def historical_repeats(x, tolerance=1.0, max_gap=20):
    # Reuses the existing algorithm exactly, then looks at ALL formed clusters, not just latest.
    from monitor.seven_system import _single,_repeat_double7_levels
    single,_=_single(x,True)
    return _repeat_double7_levels(single,tolerance,max_gap)


def _stats(rows):
    out=[]
    if rows.empty:return out
    for group,sub in rows.groupby('cohort'):
        for h in HORIZONS:
            eligible=sub[sub[f'complete_{h}d'] & sub[f'max_rise_{h}d'].notna()]
            data={'cohort':group,'horizon':h,'n':len(eligible),
                  'avg_max_rise':round(eligible[f'max_rise_{h}d'].mean(),3) if len(eligible) else None,
                  'median_max_drawdown':round(eligible[f'max_drawdown_{h}d'].median(),3) if len(eligible) else None}
            for t in TARGETS:
                data[f'hit_{t}pct_rate']=(round(100*eligible[f'hit_{t}pct_{h}d'].mean(),2) if len(eligible) else None)
            out.append(data)
    return out


def analyze(frames,seed=1729,tolerance=1.0,max_gap=20,min_history=60):
    rng=np.random.default_rng(seed)
    all_rows=[]
    for sym,raw in frames.items():
        x=_prices(raw)
        if len(x)<min_history+101:continue
        events=historical_repeats(x,tolerance,max_gap)
        if not events:continue
        indices={pd.Timestamp(d):i for i,d in enumerate(x.date)}
        repeat_ix=[]
        for ev in events:
            i=indices.get(pd.Timestamp(ev['end_date']))
            if i is None or i<min_history:continue
            repeat_ix.append((i,ev))
            d={'symbol':sym,'date':str(x.date.iloc[i].date()),'cohort':'repeat',
               'gap':ev['max_session_gap'],'touches':ev['touches'],'spread_pct':ev['price_spread_pct']}
            all_rows.append({**d,**features(x,i),**outcomes(x,i)})
        if not repeat_ix:continue
        used={i for i,_ in repeat_ix}
        # Match by symbol and calendar quarter, avoiding Repeat dates.
        for i,ev in repeat_ix:
            q=x.date.iloc[i].to_period('Q')
            candidates=[j for j in range(min_history,len(x)-1) if j not in used and x.date.iloc[j].to_period('Q')==q]
            if not candidates:continue
            j=int(rng.choice(candidates))
            d={'symbol':sym,'date':str(x.date.iloc[j].date()),'cohort':'random_matched',
               'gap':None,'touches':None,'spread_pct':None}
            all_rows.append({**d,**features(x,j),**outcomes(x,j)})
    df=pd.DataFrame(all_rows)
    if df.empty:return df,[]
    # Select a consolidation-only cohort from random matched dates, using backward-only definition.
    mask=(df.cohort=='random_matched')&(df.consolidation_proxy==True)
    matched=df.loc[mask].copy();matched['cohort']='consolidation_no_repeat'
    df=pd.concat([df,matched],ignore_index=True)
    return df,_stats(df)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--market',choices=['egypt','sp500'],default='egypt')
    p.add_argument('--output-dir',default='/tmp/seven_accumulation_study')
    p.add_argument('--max-symbols',type=int,default=0)
    p.add_argument('--price-tolerance-pct',type=float,default=1.0)
    p.add_argument('--max-gap-sessions',type=int,default=20)
    args=p.parse_args()
    if args.market=='egypt':
        from monitor.seven_system import _load_stocks
    else:
        from monitor.sp500_seven_system import _load_stocks
    frames=_load_stocks()
    if args.max_symbols:frames=dict(list(sorted(frames.items()))[:args.max_symbols])
    df,summary=analyze(frames,tolerance=args.price_tolerance_pct,max_gap=args.max_gap_sessions)
    target=Path(args.output_dir);target.mkdir(parents=True,exist_ok=True)
    df.to_csv(target/f'{args.market}_historical_events.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(target/f'{args.market}_cohort_summary.csv',index=False,encoding='utf-8-sig')
    (target/f'{args.market}_study_info.json').write_text(json.dumps({'market':args.market,'symbols_loaded':len(frames),'signals':int((df.cohort=='repeat').sum()) if not df.empty else 0,'method':'next-session open, high/low follow-through; compare per-symbol same-quarter random dates, consolidation proxy is exploratory','warning':'Exploratory comparison; random cohort is not matched for liquidity/volatility, and current universe may exhibit survivorship bias. Cross-validation and embargoed out-of-sample testing required before trading.'},indent=2),encoding='utf-8')
    print(f'Wrote {target}; {len(df)} total cohort rows; repeat events: {int((df.cohort=="repeat").sum()) if not df.empty else 0}')

if __name__=='__main__':main()
