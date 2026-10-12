"""Exploratory decision-tree thresholds for Egyptian preburst 250-session winners.
Does not mutate database or Flask. Training (2023-24) and review (2025+ completed).
Optional pre-2023 replay with --historical; it uses ALL daily indicator rows and
250-session suppression, not the original filtered cycles.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.impute import SimpleImputer

FEATURES=['rsi14','momentum5_pct','momentum20_pct','close_vs_sma20_pct','close_vs_sma50_pct','bbz20','atr14_pct','return_20','position_20_pct','volume5_vs_50','repeat_count_1pct_60']
TARGETS=[1.0,.95,.90]

def choose_cut(proba,y,recall):
    # allow ties; conservative threshold chosen among positive sample scores
    positives=np.asarray(proba)[np.asarray(y)==1]
    if not len(positives):return float('inf')
    scores=np.unique(positives)
    feasible=[t for t in scores if (positives>=t-1e-12).mean()+1e-12>=recall]
    return float(max(feasible)) if feasible else float(min(scores))

def metrics(scores, y, threshold):
    scores=np.asarray(scores); y=np.asarray(y,dtype=int)
    keep=scores>=threshold-1e-12
    nw=int(((y==1)&keep).sum()); nwin=int((y==1).sum()); nf=int((y==0).sum()); fp=int(((y==0)&keep).sum())
    return dict(total=len(y),winners=nwin,failures=nf,selected=int(keep.sum()),winners_retained=nw,failures_selected=fp,failures_excluded=nf-fp,recall_pct=round(100*nw/nwin,2) if nwin else None,precision_pct=round(100*nw/(nw+fp),2) if nw+fp else None)

def prep(df):
    missing=set(FEATURES+['outcome','signal_start_utc'])-set(df.columns)
    if missing:raise ValueError('Missing columns: '+str(sorted(missing)))
    df=df[df.outcome.isin(['winner','failure'])].copy()
    df['year']=pd.to_datetime(df.signal_start_utc).dt.year
    df['y']=df.outcome.eq('winner').astype(int)
    for f in FEATURES:df[f]=pd.to_numeric(df[f],errors='coerce')
    return df

def fit_tree(train,depth,leaf):
    if train.empty or train.y.nunique()<2:raise ValueError('Not enough positive/negative training cases')
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    x=imputer.fit_transform(train[FEATURES])
    tree=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=42)
    tree.fit(x,train.y)
    return tree,imputer

def scores(tree,imp,frame):
    x=imp.transform(frame[FEATURES])
    ps=tree.predict_proba(x)
    classes=list(tree.classes_)
    return ps[:,classes.index(1)]

def score_group(name,frame,model,imp,thresholds):
    arr=scores(model,imp,frame)
    return [dict(segment=name,requested_training_recall_pct=int(target*100),threshold=round(th,9),**metrics(arr,frame.y.to_numpy(),th)) for target,th in thresholds.items()]

def run(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    path=Path(args.input_csv)
    if not path.exists():raise FileNotFoundError(f'{path} missing. First run python -m monitor.us_preburst_250day_outcomes --market egypt --output-dir /tmp/preburst-250d-egypt && python -m monitor.preburst_egypt_feature_study --output-dir /tmp/preburst-egypt-features')
    df=prep(pd.read_csv(path))
    training=df[df.year.isin([2023,2024])]
    review=df[df.year>=2025]
    if training.empty:raise ValueError('No resolved 2023-2024 training records')
    comparison=[];rules=[]
    for depth in [2,3,4,5]:
      for leaf in [8,15,25]:
        model,imp=fit_tree(training,depth,leaf)
        training_scores=scores(model,imp,training)
        ths={t:choose_cut(training_scores,training.y.to_numpy(),t) for t in TARGETS}
        for segment,group in [('train_2023_2024',training),('review_2025_2026_mature_only',review),('all_2023_2026_in_sample_selection',df)]:
            for r in score_group(segment,group,model,imp,ths):comparison.append(dict(depth=depth,min_samples_leaf=leaf,**r))
        rules.append(f'===== depth={depth}, leaf={leaf} =====\n'+export_text(model,feature_names=FEATURES,decimals=4))
    # Separate descriptive upper-bound study: fit on ALL resolved cycles to
    # preserve 191 winners at chosen 100%-recall threshold, never call this validated.
    fullfit=[]
    for depth in [2,3,4,5]:
      for leaf in [8,15,25]:
        full_model,full_imp=fit_tree(df,depth,leaf)
        full_scores=scores(full_model,full_imp,df)
        for target in TARGETS:
            cutoff=choose_cut(full_scores,df.y.to_numpy(),target)
            fullfit.append(dict(depth=depth,min_samples_leaf=leaf,target_recall_pct=int(target*100),threshold=round(cutoff,9),**metrics(full_scores,df.y.to_numpy(),cutoff)))
    pd.DataFrame(fullfit).to_csv(out/'full_period_in_sample_fit_NOT_VALIDATED.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(comparison).to_csv(out/'tree_threshold_comparison.csv',index=False,encoding='utf-8-sig')
    (out/'tree_readable_rules.txt').write_text('\n\n'.join(rules),encoding='utf-8')
    comp=pd.DataFrame(comparison)
    review100=comp[(comp.segment=='review_2025_2026_mature_only')&(comp.requested_training_recall_pct==100)].sort_values(['failures_excluded','winners_retained'],ascending=False)
    summary={'version':'EG-TREE-EXPLORATORY-v1','input_rows_resolved':len(df),'winners':int(df.y.sum()),'failures':int((1-df.y).sum()),'train_2023_2024':{'rows':len(training),'winners':int(training.y.sum())},'review_2025_2026_mature_only':{'rows':len(review),'winners':int(review.y.sum())},'model_configs_tested':12,'best_full_period_191_recall_in_sample':pd.DataFrame(fullfit).query('target_recall_pct == 100').sort_values('failures_excluded',ascending=False).head(5).to_dict('records'),'targets_training_recall_pct':[100,95,90], 'top_5_review_results_at_train_recall_100pct':review100.head(5).to_dict('records'),'files':['tree_threshold_comparison.csv','full_period_in_sample_fit_NOT_VALIDATED.csv','tree_readable_rules.txt','summary.json'], 'warnings':['191/191 cannot be guaranteed by a model trained only on 2023-24. Model outputs on 2025-26 are a time review, not a fully untouched holdout because prior feature rule exploration inspected 2025-26.','A model that fits 191/191 and 0/226 in-sample would be memorization risk, not evidence of future edge.','These rows are historical baseline-filter signals; model has not yet been scanned over all Egypt daily indicator rows.','Pending signals excluded from denominator; price corporate actions not verified.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input-csv',default='/tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv');p.add_argument('--output-dir',default='/tmp/preburst-egypt-tree-study');run(p.parse_args())
