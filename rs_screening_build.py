#!/usr/bin/env python3
"""Read-only, point-in-time seven RP mirror scanner; write atomic dashboard snapshot.
Do not treat selected historical winner envelopes as validated prediction signals.
"""
import argparse, json, os, tempfile
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from six_mirror_batch import iter_db, clean, valid_symbol
from seven_rp_mirrors_90 import load_benchmark, features
from rp_stage2_adx import indicators, FEATURES

# Each of the seven reference stocks has its OWN RP anchor, window, precision.
RULES = {
 'BGMS': (252,-122.94436924386827,1),
 'YDKG': (10,-22.4616818784678,2),
 'FROG': (5,-14.914486555060458,2),
 'MU': (252,306.54079462899887,0),
 'IMNM': (5,-8.396022640906263,3),
 'ZIM': (20,-21.15946378106055,3),
 'IMSR': (60,-0.29378970293643114,3),
}

def read_ranges(path,mode):
    df=pd.read_csv(path)
    result={}
    for ref,(w,_,_) in RULES.items():
        sub=df[(df.reference==ref)&(df.rp_window==w)&(df.approach==mode)]
        result[ref]={}
        for _,row in sub.iterrows():
            if pd.notna(row.lower) and pd.notna(row.upper):
                result[ref][str(row.feature)]=(float(row.lower),float(row.upper))
        if set(result[ref])!=set(FEATURES):
            raise RuntimeError('Missing stage-2 indicator ranges for '+ref+' mode '+mode)
    return result

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.rs-',suffix='.json')
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    p.add_argument('--benchmark-csv',default='/data/seven_relative_sp500/sp500_daily.csv')
    p.add_argument('--ranges',default='/data/rp_stage2_adx/winner_indicator_ranges.csv')
    p.add_argument('--out',default='/data/rs_screening/snapshot.json')
    p.add_argument('--recent-days',type=int,default=30)
    args=p.parse_args()
    if not args.database_url:raise SystemExit('DATABASE_URL missing')
    if args.recent_days<1 or args.recent_days>365:raise SystemExit('recent-days must be 1..365')
    bench=load_benchmark(args.benchmark_csv)
    ceiling=bench.date.iloc[-1].date()
    cutoff=(pd.Timestamp(ceiling)-pd.Timedelta(days=args.recent_days)).strftime('%Y-%m-%d')
    rng={m:read_ranges(args.ranges,m) for m in ('minmax','trimmed')}
    out=[];count=0
    for raw in iter_db(args.database_url):
        sym=str(raw.symbol.iloc[0]).strip().upper()
        if not valid_symbol(sym):continue
        d=clean(raw);count+=1
        if len(d)<110:continue
        dates=d.date.to_numpy()
        indices=np.flatnonzero((dates>=cutoff)&(dates<=str(ceiling)))
        if not len(indices):continue
        rs=features(d,bench)
        # Reuse exactly the same Wilder/volume/volatility definitions as stage-2 research.
        ind=indicators(d)
        for i in indices:
            if i<1:continue
            for ref,(window,reference,dp) in RULES.items():
                if ref==sym:continue
                rv=float(rs[window][i-1])
                if not np.isfinite(rv) or np.round(rv,dp)!=np.round(reference,dp):continue
                vals={k:(float(arr[i-1]) if np.isfinite(arr[i-1]) else None) for k,arr in ind.items()}
                statuses={}
                for mode in ('minmax','trimmed'):
                    statuses[mode]=all(vals[k] is not None and lo<=vals[k]<=hi for k,(lo,hi) in rng[mode][ref].items())
                out.append({'symbol':sym,'signal_date':str(dates[i]),'reference':ref,'rp_window':window,
                    'rp_value_pp':round(rv,6),'rp_reference_pp':reference,'selected_decimals':dp,
                    'signal_close':round(float(d.c.iloc[i]),6),
                    'first_stage':True,'minmax_pass':statuses['minmax'],'trimmed_pass':statuses['trimmed'],
                    **{k.lower():round(v,5) if v is not None else None for k,v in vals.items()}})
        if count%500==0:print('SCANNED',count,'MATCHES',len(out),flush=True)
    out.sort(key=lambda r:(r['signal_date'],r['reference'],r['symbol']),reverse=True)
    snap={'status':'complete','benchmark_last_date':str(ceiling),'recent_days':args.recent_days,
      'stock_groups':count,'total_rs_matches':len(out),'minmax_pass':sum(r['minmax_pass'] for r in out),
      'trimmed_pass':sum(r['trimmed_pass'] for r in out),
      'strategy_rules':{k:{'rp_window':v[0],'rp_reference_pp':v[1],'precision':v[2]} for k,v in RULES.items()},
      'rows':out,'disclaimer':'Historical winner envelopes are in-sample. No future outcome is inferred. Signal-day close is shown retrospectively and may not be executable at that price.'}
    atomic_json(args.out,snap)
    print('DONE',json.dumps({k:v for k,v in snap.items() if k not in ('rows','strategy_rules')}),flush=True)
if __name__=='__main__':main()
