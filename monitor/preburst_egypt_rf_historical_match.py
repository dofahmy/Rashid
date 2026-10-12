"""Exploratory IN-SAMPLE classification for Egyptian preburst cycles, 2023-26.
Run: python -m monitor.preburst_egypt_rf_historical_match --output-dir /tmp/egypt-rf-historical
This is historical memorization capacity, NOT predictive performance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    'rsi14','momentum5_pct','momentum20_pct','close_vs_sma20_pct',
    'close_vs_sma50_pct','bbz20','atr14_pct','return_20','position_20_pct',
    'volume5_vs_50','repeat_count_1pct_60'
]
DEFAULT_INPUT='/tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-csv',default=DEFAULT_INPUT)
    ap.add_argument('--output-dir',default='/tmp/egypt-rf-historical')
    args=ap.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    if not Path(args.input_csv).is_file():
        raise SystemExit(f'Missing {args.input_csv}. Run us_preburst_250day_outcomes, then preburst_egypt_feature_study in same Railway deployment.')
    raw=pd.read_csv(args.input_csv)
    required=['outcome']+FEATURES
    missing=[x for x in required if x not in raw.columns]
    if missing: raise SystemExit(f'Missing columns: {missing}')
    data=raw.loc[raw.outcome.isin(['winner','failure'])].copy().reset_index(drop=True)
    X=data[FEATURES].apply(pd.to_numeric, errors='coerce')
    if X.isna().any().any():
        raise SystemExit(f'Incomplete features: {X.isna().sum()[X.isna().sum()>0].to_dict()}')
    y=(data['outcome']=='winner').to_numpy().astype(int)
    n_win=int(y.sum()); n_fail=int((1-y).sum())
    targets=[100,95,90]
    results=[]; saved=[]; importance=[]
    configs=[(d,l) for d in [None,12,7,4] for l in [1,2,5,10]]
    for depth,leaf in configs:
        clf=RandomForestClassifier(n_estimators=300, max_depth=depth,
          min_samples_leaf=leaf, random_state=42, n_jobs=-1, class_weight=None)
        clf.fit(X,y)
        scores=clf.predict_proba(X)[:,1]
        label=f'depth_{depth if depth is not None else "none"}_leaf_{leaf}'
        for target in targets:
            k=int(np.ceil(n_win*target/100))
            # largest possible threshold retaining >= k winners; ties handled inclusively
            threshold=float(np.sort(scores[y==1])[::-1][k-1])
            select=scores >= threshold - 1e-12
            nw=int((select & (y==1)).sum())
            nf=int((select & (y==0)).sum())
            results.append(dict(model=label,max_depth=depth,min_samples_leaf=leaf,
              desired_winner_recall_pct=target, threshold=round(threshold,10),
              selected=int(select.sum()), winners_retained=nw, winners_missed=n_win-nw,
              failures_selected=nf, failures_excluded=n_fail-nf,
              precision_pct=round(100*nw/max(1,nw+nf),2),
              winner_recall_pct=round(100*nw/n_win,2)))
        importance.extend([dict(model=label,indicator=c,importance=float(v))
                            for c,v in zip(FEATURES,clf.feature_importances_)])
        saved.append((label,scores))
    table=pd.DataFrame(results)
    table.to_csv(output/'all_models_and_thresholds.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(importance).to_csv(output/'feature_importances.csv',index=False,encoding='utf-8-sig')
    # save best model scores for each recall objective; ties: fewer failures, more winners, simpler model
    champion=[]
    for target in targets:
        sub=table[table.desired_winner_recall_pct==target].copy()
        sub=sub.sort_values(['failures_excluded','winners_retained','min_samples_leaf'],ascending=[False,False,False])
        best=sub.iloc[0].to_dict(); champion.append(best)
        model_scores=dict(saved)[best['model']]
        out=data.copy()
        out['historical_rf_probability']=np.round(model_scores,7)
        out['selected']=out['historical_rf_probability']>=float(best['threshold'])-0.00000011
        out.to_csv(output/f'best_historical_recall_{target}pct.csv',index=False,encoding='utf-8-sig')
    summary={'version':'EG-RF-HISTORICAL-v1', 'period':'2023-2026',
      'scope':'original filter resolved cycles only', 'input_cycles':len(data),
      'winners':n_win,'failures':n_fail,'models_tested':len(configs),
      'tree_counts_per_model':300,'historical_same_data_training_and_scoring':True,
      'best_by_recall_target':champion,
      'files':['all_models_and_thresholds.csv','feature_importances.csv',
        'best_historical_recall_100pct.csv','best_historical_recall_95pct.csv',
        'best_historical_recall_90pct.csv'],
      'warnings':['All scores use the same examples used to fit models; high fit can mean memorization, not forecast skill.',
       'All 191 winners include early-confirmed wins not yet mature; pending unresolved cases excluded.',
       'Original filter cohort only, not whole-market rescan; outputs do not tell if new stock opportunities exist.',
       'Adjusted EGX prices and corporate actions have not been verified.']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
