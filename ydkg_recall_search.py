#!/usr/bin/env python3
"""Recall-first beam search for separate YDKG mirror research; no Squaring edits."""
import argparse, json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
from sequential_mirror import FEATURES, REFERENCES, query_bars, outcomes


def choose_beams(matches, eligible, hit, train, min_hits, min_matches, beam_width=30, max_filters=10):
    """Keep up to beam_width combinations, preferring high hit recall and then specificity."""
    names=list(matches)
    base=eligible & train
    allhits=int(np.count_nonzero(base & hit))
    beams=[((),base)]
    stages=[]
    for depth in range(1,min(max_filters,len(names))+1):
        candidates=[]
        seen=set()
        for chosen,mask in beams:
            for name in names:
                if name in chosen:continue
                comb=tuple(sorted((*chosen,name)))
                if comb in seen:continue
                seen.add(comb)
                new=mask & matches[name]
                n=int(new.sum())
                if n<min_matches:continue
                h=int(np.count_nonzero(new&hit))
                if h<min_hits:continue
                candidates.append((comb,new,h,n))
        if not candidates:break
        # Rank: preserve the most genuine hits first; within same count, remove more false positives.
        candidates.sort(key=lambda x:(-x[2],x[3],x[0]))
        beams=[(c[0],c[1]) for c in candidates[:beam_width]]
        best=candidates[0]
        stages.append({'stage':depth,'features':best[0],'train_hits':best[2],
                       'train_matches':best[3],'train_hit_rate_pct':round(100*best[2]/best[3],5),
                       'train_recall_pct':round(100*best[2]/allhits,2) if allhits else None,
                       'alternatives_at_depth':len(candidates)})
    return stages,allhits


def main():
    ap=argparse.ArgumentParser(description='YDKG recall-first multi-filter search')
    ap.add_argument('--anchors',default='/data/breakout100_discovery/all_market_anchors.csv')
    ap.add_argument('--showcase',default='/data/breakout100_discovery/seven_showcase_features.csv')
    ap.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    ap.add_argument('--out',default='/data/ydkg_recall_search')
    ap.add_argument('--reference',default='YDKG',choices=list(REFERENCES))
    ap.add_argument('--percentile-tolerance',type=float,default=5)
    ap.add_argument('--min-hits',type=int,default=6,help='minimum discovery-period successful cases kept')
    ap.add_argument('--min-matches',type=int,default=20)
    ap.add_argument('--beam-width',type=int,default=30)
    ap.add_argument('--max-filters',type=int,default=10)
    ap.add_argument('--train-before',default='2026-01-01',help='selection only on dates earlier than this')
    args=ap.parse_args()
    if not args.database_url:ap.error('DATABASE_URL is required')
    if not 0<args.percentile_tolerance<=50:ap.error('percentile tolerance must be 0-50')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    anchors=pd.read_csv(args.anchors,low_memory=False)
    showcase=pd.read_csv(args.showcase)
    if 'anchor' in anchors:anchors=anchors[anchors.anchor.eq('monthly')].copy()
    anchors['date']=anchors.date.astype(str).str[:10]
    anchors=anchors.drop_duplicates(['symbol','date']).reset_index(drop=True)
    anchors['symbol']=anchors.symbol.astype(str)
    for f in FEATURES+['signal_close']:anchors[f]=pd.to_numeric(anchors[f],errors='coerce')
    refdate,target,days=REFERENCES[args.reference]
    exemplar=showcase[(showcase.symbol==args.reference)&(showcase.date.astype(str).str[:10]==refdate)]
    if exemplar.empty:raise ValueError('Missing reference features: '+args.reference)
    exemplar=exemplar.iloc[0]
    bars=query_bars(args.database_url,anchors.symbol.dropna().unique().tolist(),days)
    valid,hit,first,max_gain=outcomes(anchors,bars,target,days)
    eligible=valid & ~((anchors.symbol==args.reference)&(anchors.date==refdate)).to_numpy()
    train=anchors.date.lt(args.train_before).to_numpy()
    test=~train
    calibration=anchors[anchors.date.lt('2025-01-01')]
    matches={}; bounds={}
    for f in FEATURES:
        cal=np.sort(calibration[f].dropna().to_numpy(dtype=float))
        if len(cal)<50 or not np.isfinite(float(exemplar[f])):continue
        rv=float(exemplar[f]);refpct=np.searchsorted(cal,rv,side='right')/len(cal)
        arr=anchors[f].to_numpy(dtype=float)
        pct=np.searchsorted(cal,arr,side='right')/len(cal)
        matches[f]=np.isfinite(arr)&(np.abs(pct-refpct)<=args.percentile_tolerance/100+1e-12)
        bounds[f]={'reference_value':rv,'percentile':round(refpct*100,2),
                   'lower':float(np.quantile(cal,max(0,refpct-args.percentile_tolerance/100))),
                   'upper':float(np.quantile(cal,min(1,refpct+args.percentile_tolerance/100)))}
    train_hits=int(np.count_nonzero(eligible&train&hit))
    if args.min_hits>train_hits:raise ValueError(f'--min-hits={args.min_hits} but only {train_hits} positive training cases. Lower --min-hits.')
    stages,baseline=choose_beams(matches,eligible,hit,train,args.min_hits,args.min_matches,args.beam_width,args.max_filters)
    rows=[]
    for s in stages:
        mask=eligible.copy()
        for f in s['features']:mask &=matches[f]
        tr=mask&train;te=mask&test
        nh=int(np.count_nonzero(tr&hit));nt=int(tr.sum());vh=int(np.count_nonzero(te&hit));vt=int(te.sum())
        base_test_hit=int(np.count_nonzero(eligible&test&hit))
        row={'stage':s['stage'],'filters_added':s['stage'],'features':' & '.join(s['features']),
             'last_added':next((f for f in s['features'] if f not in (rows[-1]['features'].split(' & ') if rows else [])),''),
             'train_matches':nt,'train_hits':nh,'train_recall_pct':round(100*nh/baseline,2) if baseline else None,
             'train_precision_pct':round(100*nh/nt,5) if nt else None,
             'test_matches_2026':vt,'test_hits_2026':vh,'test_precision_pct':round(100*vh/vt,5) if vt else None,
             'test_recall_pct':round(100*vh/base_test_hit,2) if base_test_hit else None,
             'alternatives_considered':s['alternatives_at_depth']}
        rows.append(row)
        print('STAGE',row['stage'],row['features'],'TRAIN',nh,'/',nt,'TEST',vh,'/',vt,flush=True)
    pd.DataFrame(rows).to_csv(out/f'{args.reference}_recall_first_stages.csv',index=False)
    # Indicator-only diagnostics, no feature ranking concealed.
    individual=[]
    for f,m in matches.items():
        tr=eligible&train&m;te=eligible&test&m
        individual.append({'feature':f,'train_matches':int(tr.sum()),'train_hits':int((tr&hit).sum()),
                           'test_matches_2026':int(te.sum()),'test_hits_2026':int((te&hit).sum())})
    pd.DataFrame(individual).sort_values(['train_hits','train_matches'],ascending=[False,True]).to_csv(out/f'{args.reference}_single_indicators.csv',index=False)
    (out/f'{args.reference}_indicator_bounds.json').write_text(json.dumps(bounds,indent=2))
    report={'reference':args.reference,'target_gain_pct':target,'target_days':days,
            'train_valid':int((eligible&train).sum()),'train_hits':train_hits,
            'test_valid_2026':int((eligible&test).sum()),'test_hits_2026':int((eligible&test&hit).sum()),
            'min_hits_required':args.min_hits,'stages':len(rows),
            'cautions':'Exploratory. Reference success known in advance, 2026 NOT untouched because previously inspected. Outcomes for 2026 within extraction end may be censored. Percentile tolerance is a Boolean feature match, not literal equality. Baseline compares anchors only, not daily universe. Split/corporate-action adjustments should be audited.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
