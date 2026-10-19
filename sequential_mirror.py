#!/usr/bin/env python3
"""Sequential exact-indicator mirror discovery (research only).  No Squaring modifications."""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam

REFERENCES = {
 'YDKG':('2025-08-22',137.66,5), 'ZIM':('2024-07-29',172.71,15),
 'FROG':('2026-04-15',146.96,32), 'MU':('2026-03-02',203.81,59),
 'IMSR':('2025-06-24',173.32,78), 'IMNM':('2025-07-09',237.00,82),
}
FEATURES=['ret5','ret10','ret20','ret60','ret120','volatility20','atr14_pct','rvol20','dollar_volume20','rsi14','dist_ma20','dist_ma50','dist_ma120','distance_high60_pct','range20_pct','up_days20_pct','daily_candle_pct']

def query_bars(db, symbols, max_days):
    engine=create_engine(db.replace('postgres://','postgresql://',1))
    query=text('SELECT symbol,session_date AS date,h,c FROM market_candles_1d WHERE symbol IN :syms ORDER BY symbol,session_date').bindparams(bindparam('syms',expanding=True))
    result={}
    with engine.connect() as con:
        for start in range(0,len(symbols),100):
            frame=pd.read_sql(query,con,params={'syms':symbols[start:start+100]})
            if frame.empty:continue
            frame['date']=frame.date.astype(str).str[:10]
            for k in ('h','c'):frame[k]=pd.to_numeric(frame[k],errors='coerce')
            frame=frame.drop_duplicates(['symbol','date'],keep='last')
            for sym,g in frame.groupby('symbol',sort=False):
                g=g.sort_values('date');result[sym]=(g.date.to_numpy(),g.h.to_numpy(dtype=float))
            if start and start%500==0:print('BARS LOADED',start,'/',len(symbols),flush=True)
    engine.dispose()
    return result

def outcomes(frame,bars,target,days):
    """Compute future High outcome only when complete window is available; no day 0."""
    hit=np.zeros(len(frame),dtype=bool); valid=np.zeros(len(frame),dtype=bool)
    first=np.full(len(frame),np.nan); max_gain=np.full(len(frame),np.nan)
    for i,(sym,dt,price) in enumerate(zip(frame.symbol,frame.date,frame.signal_close)):
        pair=bars.get(sym)
        if pair is None or not np.isfinite(price) or price<=0:continue
        dates,highs=pair
        ix=np.searchsorted(dates,dt)
        if ix>=len(dates) or dates[ix]!=dt:continue
        future=highs[ix+1:ix+1+days]
        if len(future)!=days or not np.isfinite(future).all():continue
        valid[i]=True
        returns=(future/price-1)*100
        max_gain[i]=float(np.max(returns))
        h=np.flatnonzero(returns>=target-1e-9)
        if len(h):hit[i]=True;first[i]=int(h[0]+1)
    return valid,hit,first,max_gain

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--anchors',default='/data/breakout100_discovery/all_market_anchors.csv')
    ap.add_argument('--showcase',default='/data/breakout100_discovery/seven_showcase_features.csv')
    ap.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    ap.add_argument('--reference',choices=[*REFERENCES,'all'],default='YDKG')
    ap.add_argument('--out',default='/data/sequential_mirror')
    ap.add_argument('--percentile-tolerance',type=float,default=5.,help='Absolute percentile points around reference; individual indicator pass/fail')
    ap.add_argument('--min-matches',type=int,default=20,help='Minimum complete-outcome rows for a filter step')
    ap.add_argument('--min-hit-retention',type=int,default=1,help='Minimum true outcomes retained while selecting a step')
    ap.add_argument('--ranking',choices=['coverage','precision'],default='coverage',help='coverage: largest matching universe, then hits; precision: most validated hits, then hit rate')
    args=ap.parse_args()
    if not args.database_url:ap.error('DATABASE_URL required to evaluate timing and price, not just similarity')
    if not 0 < args.percentile_tolerance <= 50:ap.error('percentile tolerance must be between 0 and 50')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    anchors=pd.read_csv(args.anchors,low_memory=False)
    showcase=pd.read_csv(args.showcase)
    if 'anchor' in anchors:anchors=anchors[anchors.anchor.eq('monthly')].copy()
    anchors['date']=anchors.date.astype(str).str[:10]
    anchors=anchors.drop_duplicates(['symbol','date']).reset_index(drop=True) # one opportunity per stock/date
    for c in FEATURES+['signal_close']:
        anchors[c]=pd.to_numeric(anchors[c],errors='coerce')
    anchors['year']=anchors.date.str[:4]
    refs=REFERENCES if args.reference=='all' else {args.reference:REFERENCES[args.reference]}
    market_2024=anchors[anchors.date<'2025-01-01']
    bars=query_bars(args.database_url,anchors.symbol.dropna().unique().tolist(),max(v[2] for v in refs.values()))
    outputs=[];totals=[]
    for ref,(dt,target,days) in refs.items():
        exemplar=showcase[(showcase.symbol==ref)&(showcase.date.astype(str).str[:10]==dt)]
        if exemplar.empty:
            print('SKIPPED',ref,'missing showcase reference',flush=True);continue
        exemplar=exemplar.iloc[0]
        valid,hit,first,max_gain=outcomes(anchors,bars,target,days)
        eligible=valid & ~((anchors.symbol==ref)&(anchors.date==dt)).to_numpy()
        # Fit calibration on 2024 universe, without using outcome labels.
        # Each numeric indicator is binarized: percentile difference within tolerance.
        matches={};ref_values={}
        for feature in FEATURES:
            cal=market_2024[feature].dropna().to_numpy(dtype=float)
            if len(cal)<50 or not np.isfinite(float(exemplar[feature])):continue
            cal.sort()
            candidate=anchors[feature].to_numpy(dtype=float)
            ref_value=float(exemplar[feature])
            ref_rank=np.searchsorted(cal,ref_value,side='right')/len(cal)
            rank=np.searchsorted(cal,candidate,side='right')/len(cal)
            pred=np.isfinite(candidate)&(np.abs(rank-ref_rank)<=args.percentile_tolerance/100+1e-12)
            matches[feature]=pred
            ref_values[feature]={'reference_value':ref_value,'reference_percentile_2024':round(100*ref_rank,3),'lower_bound_2024':float(np.quantile(cal,max(0,ref_rank-args.percentile_tolerance/100))),'upper_bound_2024':float(np.quantile(cal,min(1,ref_rank+args.percentile_tolerance/100)))}
        active=eligible.copy();remaining=list(matches);stages=[]
        for stage in range(1,len(remaining)+1):
            candidates=[]
            for f in remaining:
                mask=active & matches[f]
                n=int(mask.sum());h=int((mask & hit).sum())
                if n<args.min_matches or h<args.min_hit_retention:continue
                candidates.append((f,n,h,h/n))
            if not candidates:break
            if args.ranking=='coverage':candidates.sort(key=lambda x:(-x[1],-x[2],x[0]))
            else:candidates.sort(key=lambda x:(-x[2],-x[3],-x[1],x[0]))
            best=candidates[0];f=best[0]
            active &= matches[f];remaining.remove(f)
            row={'reference':ref,'stage':stage,'feature_added':f,'reference_value':ref_values[f]['reference_value'],
                 'reference_percentile':ref_values[f]['reference_percentile_2024'],
                 'approx_lower_bound':ref_values[f]['lower_bound_2024'],'approx_upper_bound':ref_values[f]['upper_bound_2024'],
                 'matched_valid_anchors':int(active.sum()),'mirror_hits':int((active&hit).sum()),
                 'mirror_rate_pct':round(100*(active&hit).sum()/active.sum(),5),
                 'baseline_rate_pct':round(100*(eligible&hit).sum()/eligible.sum(),5),
                 'lift_vs_baseline':round(((active&hit).sum()/active.sum())/((eligible&hit).sum()/eligible.sum()),4) if (eligible&hit).sum() else np.nan,
                 'matched_symbols':int(anchors.loc[active,'symbol'].nunique()),
                 'matched_2024':int((active&(anchors.year=='2024').to_numpy()).sum()),
                 'matched_2025':int((active&(anchors.year=='2025').to_numpy()).sum()),
                 'matched_2026':int((active&(anchors.year=='2026').to_numpy()).sum())}
            stages.append(row);outputs.append(row)
            print(ref,'STAGE',stage,f,'MATCHES',row['matched_valid_anchors'],'HITS',row['mirror_hits'],'RATE',row['mirror_rate_pct'],flush=True)
            selection=anchors.loc[active,['symbol','date','signal_close','year']].copy()
            selection['hit_reference_price_in_days']=hit[active]
            selection['first_hit_day']=first[active]
            selection['max_high_return_pct']=max_gain[active]
            selection.to_csv(out/f'{ref}_stage_{stage:02d}_candidates.csv',index=False)
        pd.DataFrame(stages).to_csv(out/f'{ref}_filter_stages.csv',index=False)
        (out/f'{ref}_indicator_bounds.json').write_text(json.dumps(ref_values,indent=2))
        totals.append({'reference':ref,'target_return_pct':target,'target_days':days,'valid_anchors':int(eligible.sum()),
                       'baseline_hits':int((eligible&hit).sum()),'stages':len(stages),'missing_outcomes':int((~valid).sum())})
    pd.DataFrame(outputs).to_csv(out/'all_filter_stages.csv',index=False)
    (out/'report.json').write_text(json.dumps({'summary':totals,'indicator_match_rule':'Each indicator individually passes if its 2024-market percentile is within the chosen percentile tolerance of reference. This is a Boolean match, NOT identical numeric value.','cautions':'Exploratory/in-sample greedy stages selected using the same historical cases. Coverage ranking can prefer permissive uninformative indicators; review hit counts/lift and test on unselected dates. No claims of calibrated predictive success. Trading execution, splits, and same-date adjustment require independent audits.'},indent=2))
    print((out/'report.json').read_text())
if __name__=='__main__':main()
