#!/usr/bin/env python3
"""Seven independent reference mirrors; research-only, observational metrics."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Original showcase peak returns and time-to-peak as supplied by user.
# These are targets, NOT claims of first passage to +100%.
REFERENCES = {
    'BGMS': ('2026-06-04', 165.40, 1),
    'YDKG': ('2025-08-22', 137.66, 5),
    'ZIM': ('2024-07-29', 172.71, 15),
    'FROG': ('2026-04-15', 146.96, 32),
    'MU': ('2026-03-02', 203.81, 59),
    'IMSR': ('2025-06-24', 173.32, 78),
    'IMNM': ('2025-07-09', 237.00, 82),
}
FEATURES = [
    'ret5','ret10','ret20','ret60','ret120', 'volatility20','atr14_pct',
    'rvol20','dollar_volume20','rsi14','dist_ma20','dist_ma50','dist_ma120',
    'distance_high60_pct','range20_pct','up_days20_pct','daily_candle_pct',
]
# Each DNA is a single exemplar; zero feature tuning against outcomes.
# Weights only encode declared feature families, not optimized against winners.
FAMILIES = {
    'momentum':['ret5','ret10','ret20','ret60','ret120'],
    'volatility':['volatility20','atr14_pct','range20_pct'],
    'volume':['rvol20','dollar_volume20'],
    'trend':['dist_ma20','dist_ma50','dist_ma120','distance_high60_pct','rsi14','up_days20_pct'],
    'candle':['daily_candle_pct'],
}

def perc_mapping(values, references):
    vals = np.asarray(values, dtype=float)
    sorted_vals = np.sort(np.asarray(references, dtype=float))
    sorted_vals = sorted_vals[np.isfinite(sorted_vals)]
    if len(sorted_vals) < 20:
        raise ValueError('Not enough reference market observations to calibrate ranks')
    ranks = np.full(vals.shape, np.nan, dtype=float)
    ok = np.isfinite(vals)
    ranks[ok] = np.searchsorted(sorted_vals, vals[ok], side='right')/len(sorted_vals)
    return ranks

def add_similarity(anchors, exemplar, calibration):
    z = anchors.copy()
    # Nonlinear features are rank transformed, which avoids enormous outliers
    # dominating distance; each family gets equal total weight.
    distances = []
    family_sim = {}
    for family, cols in FAMILIES.items():
        cols = [c for c in cols if c in z and c in exemplar and pd.notna(exemplar[c])]
        if not cols: continue
        pieces=[]
        for f in cols:
            x = np.log1p(z[f].clip(lower=0)) if f=='dollar_volume20' else z[f]
            cal = np.log1p(calibration[f].clip(lower=0)) if f=='dollar_volume20' else calibration[f]
            ref = np.log1p(max(float(exemplar[f]),0)) if f=='dollar_volume20' else float(exemplar[f])
            candidate_rank = perc_mapping(pd.to_numeric(x,errors='coerce'), pd.to_numeric(cal,errors='coerce'))
            ref_rank = float(perc_mapping([ref],pd.to_numeric(cal,errors='coerce'))[0])
            # rank distance 0..1 -> similarity 100..0
            pieces.append(np.abs(candidate_rank-ref_rank))
        family_distance = np.nanmean(np.vstack(pieces), axis=0)
        family_sim[family] = 100*(1-family_distance)
        distances.append(family_distance)
    if not distances: raise ValueError('No feature families available')
    z['dna_similarity_pct'] = np.round(100*(1-np.nanmean(np.vstack(distances),axis=0)),2)
    for k,v in family_sim.items(): z[f'dna_{k}_pct']=np.round(v,2)
    return z

def get_daily_bars(url, symbols, first_date, last_date, batch_size=100):
    """Fetch candidate/ref bars only, not a full market re-scan."""
    from sqlalchemy import create_engine, text, bindparam
    engine=create_engine(url.replace('postgres://','postgresql://',1))
    frames=[]; names=sorted(set(symbols))
    query=text('SELECT symbol,session_date AS date,o,h,l,c,v FROM market_candles_1d '
               'WHERE symbol IN :symbols AND session_date >= :start AND session_date <= :end '
               'ORDER BY symbol,session_date').bindparams(bindparam('symbols',expanding=True))
    with engine.connect() as con:
        for i in range(0,len(names),batch_size):
            chunk=pd.read_sql(query,con,params={'symbols':names[i:i+batch_size],
                                                   'start':first_date,'end':last_date})
            if not chunk.empty: frames.append(chunk)
    if not frames: return {}
    all_bars=pd.concat(frames,ignore_index=True)
    all_bars['date']=all_bars['date'].astype(str).str.slice(0,10)
    for f in ('o','h','l','c','v'):all_bars[f]=pd.to_numeric(all_bars[f],errors='coerce')
    all_bars=all_bars.dropna(subset=['h','c'])
    return {s:d.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
            for s,d in all_bars.groupby('symbol')}

def exact_performance(row, bars, target_gain, target_days):
    """Only future session High values; day-0 is excluded. Requires entire target horizon."""
    if bars is None:return {'evaluated':False,'reason':'missing_bars'}
    dates=bars.date.to_numpy()
    # Exact anchor must exist. Never substitute another day without disclosure.
    at=np.flatnonzero(dates==str(row['date'])[:10])
    if not len(at):return {'evaluated':False,'reason':'missing_anchor_date'}
    ix=int(at[0]); days=int(target_days)
    future=bars.h.iloc[ix+1:ix+1+days].to_numpy(dtype=float)
    price=float(row['signal_close'])
    if len(future)<days or price<=0 or not np.all(np.isfinite(future)):
        return {'evaluated':False,'reason':'incomplete_forward_window'}
    gains=(future/price-1)*100
    threshold=float(target_gain)
    hits=np.flatnonzero(gains>=threshold-1e-9)
    first=int(hits[0]+1) if len(hits) else None
    max_gain=float(np.max(gains)); peakday=int(np.argmax(gains)+1)
    # Reference-timing target: reaches reference peak gain in <= reference days.
    # Exact curve-shape similarity is not inferred from a single peak statistic.
    price_coverage=max_gain/threshold if threshold>0 else np.nan
    price_closeness=100*min(max_gain,threshold)/max(max_gain,threshold) if max_gain>0 and threshold>0 else 0
    time_closeness=100*(1-abs(first-days)/max(first,days)) if first is not None else 0
    return {'evaluated':True,'reason':'',
            'max_high_return_at_reference_days_pct':round(max_gain,3),
            'peak_day_within_reference_window':peakday,
            'reached_reference_gain':bool(first is not None),
            'first_day_reference_gain':first,
            'full_mirror':bool(first is not None and first<=days),
            'price_coverage_pct':round(price_coverage*100,2),
            'price_closeness_pct':round(price_closeness,2),
            'time_closeness_pct':round(time_closeness,2)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--anchors',default='/data/breakout100_discovery/all_market_anchors.csv')
    ap.add_argument('--showcase',default='/data/breakout100_discovery/seven_showcase_features.csv')
    ap.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    ap.add_argument('--out',default='/data/seven_mirror_strategies')
    ap.add_argument('--top',type=int,default=100,help='Candidate matches per strategy and year')
    ap.add_argument('--min-similarity',type=float,default=0)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    anchors=pd.read_csv(args.anchors)
    showcase=pd.read_csv(args.showcase)
    anchors=anchors[anchors.anchor.eq('monthly')].copy()
    for f in FEATURES:
        anchors[f]=pd.to_numeric(anchors[f],errors='coerce')
    anchors['date']=anchors.date.astype(str).str.slice(0,10)
    anchors['year']=anchors.date.str[:4]
    # Periods selected due to observed winners: report results as retrospective,
    # not prospective validation even if shown per year.
    calibrate=anchors[anchors.date<'2025-01-01'].copy()
    if len(calibrate)<100: raise SystemExit('2024 calibration anchors insufficient')
    reports=[]; candidate_parts=[]; references_used=[]
    for ticker,(dt,target_gain,days) in REFERENCES.items():
        matching=showcase[(showcase.symbol==ticker)&(showcase.date.astype(str).str[:10]==dt)]
        if matching.empty:
            reports.append({'reference':ticker,'status':'REFERENCE_MISSING',
                            'detail':'Missing reference feature row; strategy not scored'})
            continue
        exemplar=matching.iloc[0]
        references_used.append({'reference':ticker,'date':dt,'target_gain_pct':target_gain,
                                'target_days':days,'reference_close_in_db':exemplar['signal_close']})
        scored=add_similarity(anchors,exemplar,calibrate)
        scored['reference']=ticker
        scored['target_gain_pct']=target_gain
        scored['target_days']=days
        # For a fair historical diagnostic, exclude same ticker and same month as exemplar.
        scored=scored[~((scored.symbol==ticker)&(scored.date==dt))]
        scored=scored[scored.dna_similarity_pct>=args.min_similarity]
        scored=scored.sort_values(['dna_similarity_pct','symbol'],ascending=[False,True])
        # Top per year makes 2024/2025/2026 interpretable with balanced retrieval.
        for year,group in scored.groupby('year'):
            candidate_parts.append(group.head(args.top).copy())
        reports.append({'reference':ticker,'status':'FEATURES_FOUND',
                        'reference_date':dt,'target_gain_pct':target_gain,
                        'target_days':days,'market_anchors_scored':int(len(scored))})
    (out/'reference_status.csv').write_text(pd.DataFrame(reports).to_csv(index=False))
    pd.DataFrame(references_used).to_csv(out/'reference_prices.csv',index=False)
    if not candidate_parts:
        raise SystemExit('No references usable; check showcase CSV')
    picks=pd.concat(candidate_parts,ignore_index=True)
    summary=[{'reference':t,'status':next(x['status'] for x in reports if x['reference']==t)} for t in REFERENCES]
    if args.database_url:
        # Allow full 90-session outcomes after latest anchor; fetch bar data through
        # 2026-12-31 to cover these dates when available. Missing remains unevaluated.
        symbols=picks.symbol.unique().tolist()
        first_date=min(picks.date.min(),min(x[0] for x in REFERENCES.values()))
        bars=get_daily_bars(args.database_url,symbols,first_date,'2027-06-30')
        performance=[]
        for _,r in picks.iterrows():
            perf=exact_performance(r,bars.get(r.symbol),r.target_gain_pct,r.target_days)
            performance.append(perf)
        picks=pd.concat([picks.reset_index(drop=True),pd.DataFrame(performance)],axis=1)
        # Rates are conditioned on selected top similar candidates; no estimates for
        # unseen non-shortlisted market or forward prediction.
        valid=picks[picks.evaluated==True]
        for (ref,year),g in valid.groupby(['reference','year']):
            summary.append({'reference':ref,'year':year,'status':'EVALUATED',
                            'matches_evaluated':len(g),'mean_similarity_pct':round(g.dna_similarity_pct.mean(),2),
                            'mirror_hits':int(g.full_mirror.sum()),'mirror_rate_pct':round(g.full_mirror.mean()*100,3),
                            'median_max_return_pct':round(g.max_high_return_at_reference_days_pct.median(),3),
                            'median_price_coverage_pct':round(g.price_coverage_pct.median(),2)})
    else:
        print('WARNING: DATABASE_URL missing; similarity only. Performance NOT evaluated.',flush=True)
    picks.to_csv(out/'top_similar_candidates.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'mirror_results_by_year.csv',index=False)
    report={'status':'complete','anchors_considered':len(anchors),'references':reports,
            'shortlisted_rows':len(picks),'outcome_evaluated':int(picks.evaluated.sum()) if 'evaluated' in picks else 0,
            'disclosure':('Retrospective exploratory comparison. Reference stocks chosen after extraordinary gains. '
                          'No untouched holdout or predictive success probabilities. Similarity is percentile-'
                          'distance score, not success likelihood. Price data need split/corporate-action audits. '
                          'Base uses database daily CLOSE; this can differ from Squaring repeat entry prices. '
                          'Reference TARGETS come from supplied historical peak table, not validated in DB.')}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
