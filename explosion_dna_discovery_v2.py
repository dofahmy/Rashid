#!/usr/bin/env python3
"""Exploratory monthly breakout DNA study, US candidates only; no database writes."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['pre_ret5_pct','pre_ret10_pct','pre_ret20_pct','pre_ret60_pct','pre_ret120_pct',
 'pre_range10_pct','pre_range20_pct','pre_range60_pct','pre_volatility20_pct',
 'pre_volatility60_pct','pre_vol_ratio20to60','pre_volume20_over60',
 'pre_up_down_vol_ratio20','pre_up_days_share20_pct',
 'pre_drawdown_from120high_pct','pre_distance_ma20_pct','pre_distance_ma60_pct',
 'pre_distance_ma120_pct','pre_last20_compression_ratio', 'median_dollar_volume20']

def metrics(df, k=20):
    if len(df)==0:return dict(n=0, positives=0, base_rate=None, picks=0, hits=0, precision=None, lift=None, recall=None, months=0)
    pick = df.sort_values(['month','score','symbol'], ascending=[True,False,True]).groupby('month',group_keys=False).head(k)
    base = float(df.y.mean()); precision=float(pick.y.mean()) if len(pick) else 0
    return dict(n=len(df),positives=int(df.y.sum()), base_rate=round(base,6),picks=len(pick),hits=int(pick.y.sum()),precision=round(precision,6),lift=round(precision/base,3) if base else None,recall=round(float(pick.y.sum()/df.y.sum()),4) if df.y.sum() else None,months=int(df.month.nunique()),unique_pick_symbols=int(pick.symbol.nunique()))

def fit_bins(train, n_bins, smoothing):
    overall=float(train.y.mean())
    rules=[]
    # Fit feature response on TRAIN only; equal-frequency bins, smoothed toward TRAIN prevalence.
    for feature in FEATURES:
        v=pd.to_numeric(train[feature],errors='coerce').replace([np.inf,-np.inf],np.nan)
        valid=v.notna()
        if valid.sum()<200:continue
        edges=np.unique(np.nanquantile(v[valid],np.linspace(0,1,n_bins+1)[1:-1]))
        if not len(edges):continue
        idx=np.searchsorted(edges,v[valid].values,side='right')
        levels=[]
        for b in range(len(edges)+1):
            m=idx==b; n=int(m.sum()); hits=int(train.loc[valid,'y'].to_numpy()[m].sum())
            rate=(hits+smoothing*overall)/(n+smoothing)
            # center relative to train average; clip influence per feature
            levels.append(float(np.clip(np.log((rate+0.0001)/(overall+0.0001)), -0.7,0.7)))
        rules.append(dict(feature=feature,edges=edges.tolist(),levels=levels,n_valid=int(valid.sum())))
    return rules

def score_frame(df,rules):
    vals=np.zeros(len(df),dtype=float)
    for rule in rules:
        v=pd.to_numeric(df[rule['feature']],errors='coerce').to_numpy(dtype=float)
        idx=np.searchsorted(rule['edges'],np.where(np.isfinite(v),v,0),side='right')
        scores=np.asarray(rule['levels'])[idx]
        vals+=np.where(np.isfinite(v),scores,0)
    return vals

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='/data/market_explosion_atlas_v2/US_CANDIDATE/all_eligible_stock_months.csv')
    p.add_argument('--out',default='/data/explosion_dna_v2')
    p.add_argument('--min-dollar-volume',type=float,default=100000)
    p.add_argument('--pick-per-month',type=int,default=20)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input,low_memory=False)
    required={'symbol','month','month_return_pct','prior_history_bars','median_dollar_volume20'}|set(FEATURES)
    missing=sorted(required-set(d.columns))
    if missing:raise SystemExit('Missing columns: '+str(missing))
    d['symbol']=d.symbol.astype(str).str.upper().str.strip()
    # A provisional US-candidate category; verify listing exchange before production use.
    d=d[~d.symbol.str.endswith('.CA')].copy()
    d=d[(d.month>='2024-04')&(d.month<='2026-09')].copy()
    d=d.drop_duplicates(['symbol','month'],keep='first')
    d['y']=(pd.to_numeric(d.month_return_pct,errors='coerce')>=100).astype(int)
    d['median_dollar_volume20']=pd.to_numeric(d.median_dollar_volume20,errors='coerce')
    # Criteria based only on prior-month knowledge; no current-month filters.
    eligible=d[(pd.to_numeric(d.prior_history_bars,errors='coerce')>=120)&(d.median_dollar_volume20>=a.min_dollar_volume)].copy()
    eligible['period']=np.select([eligible.month<'2025-01',eligible.month<'2026-01'],['TRAIN','VALIDATION'],default='HOLDOUT')
    print('COUNTS',eligible.groupby('period').agg(rows=('y','size'),hits=('y','sum'),months=('month','nunique')).to_string(),flush=True)
    tr=eligible[eligible.period=='TRAIN'].copy();va=eligible[eligible.period=='VALIDATION'].copy();ho=eligible[eligible.period=='HOLDOUT'].copy()
    if tr.y.sum()<10:raise SystemExit('Too few positive training cases; cannot fit useful model.')
    candidates=[]
    for bins in (3,4,5):
        for smoothing in (100,300,1000):
            rules=fit_bins(tr,bins,smoothing)
            vv=va.copy();vv['score']=score_frame(vv,rules)
            m=metrics(vv,a.pick_per_month)
            candidates.append(dict(bins=bins,smoothing=smoothing,validation=m,rules=rules))
    # Select solely using validation. No holdout-driven tuning.
    candidates.sort(key=lambda z:(z['validation']['hits'],z['validation']['precision'] or 0,-z['bins'],-z['smoothing']),reverse=True)
    best=candidates[0];rules=best['rules']
    report={'definition':'month_return_pct >= 100: prior-month close to current month final close',
       'universe':'non-.CA symbols, provisional US candidate only',
       'price_adjustment':'UNVERIFIED: corporate action and delisting audits required',
       'training':'2024-04..2024-12','validation':'2025-01..2025-12','holdout':'2026-01..2026-09',
       'min_dollar_volume':a.min_dollar_volume,'pick_per_month':a.pick_per_month,
       'selected_hyperparams':{'bins':best['bins'],'smoothing':best['smoothing']},'periods':{}}
    for name,frame in [('TRAIN',tr),('VALIDATION',va),('HOLDOUT',ho)]:
        frame=frame.copy();frame['score']=score_frame(frame,rules)
        report['periods'][name]=metrics(frame,a.pick_per_month)
        frame[['symbol','month','period','score','y','month_return_pct','median_dollar_volume20']].sort_values(['month','score'],ascending=[True,False]).groupby('month',group_keys=False).head(a.pick_per_month).to_csv(out/(name.lower()+'_top20.csv'),index=False)
    pd.DataFrame([{'feature':r['feature'],'edges':json.dumps(r['edges']),'levels':json.dumps(r['levels'])} for r in rules]).to_csv(out/'frozen_feature_bins.csv',index=False)
    with open(out/'frozen_rules.json','w') as f:json.dump(rules,f,indent=2)
    with open(out/'metrics.json','w') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    print('RESULTS',json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    print('FILES',out,flush=True)
if __name__=='__main__':main()
