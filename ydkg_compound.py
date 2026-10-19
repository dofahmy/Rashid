#!/usr/bin/env python3
"""Exploratory YDKG: OR of two-indicator AND branches; all historical wins retained.
Values are previous-trading-session features and threshold fitting is in-sample.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine,text,bindparam
from ydkg_zero_loss_thresholds import FEATURES,feature_at


def load_features(base, database_url):
    symbols=sorted(set(base.symbol))
    engine=create_engine(database_url.replace('postgres://','postgresql://',1))
    q=text('SELECT symbol,session_date AS date,o,h,l,c,v FROM market_candles_1d WHERE symbol IN :symbols ORDER BY symbol,session_date').bindparams(bindparam('symbols',expanding=True))
    lookup={}
    with engine.connect() as conn:
        for start in range(0,len(symbols),100):
            b=pd.read_sql(q,conn,params={'symbols':symbols[start:start+100]})
            if b.empty: continue
            b['date']=b.date.astype(str).str[:10]
            for x in 'ohlcv': b[x]=pd.to_numeric(b[x],errors='coerce')
            b=b.dropna(subset=['date','o','h','l','c','v'])
            b=b[(b.o>0)&(b.c>0)&(b.l>0)&(b.v>=0)&(b.h>=b[['o','c','l']].max(axis=1))&(b.l<=b[['o','c','h']].min(axis=1))]
            for sym,g in b.groupby('symbol',sort=False):
                lookup[sym]=g.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
    engine.dispose()
    values={f:np.full(len(base),np.nan) for f in FEATURES}
    for i,r in base.iterrows():
        bars=lookup.get(r.symbol)
        if bars is None:continue
        idx=np.flatnonzero(bars.date.to_numpy()==r.prev_date)
        if len(idx)==0:continue
        row=feature_at(bars,int(idx[0]))
        for f in FEATURES:values[f][i]=row[f]
    return values


def conditions(values,winners,min_branch_winners):
    all_conds=[]; n=len(winners)
    for f,vals in values.items():
        finite=np.isfinite(vals)
        valid_w=vals[finite&winners]
        if len(valid_w)<min_branch_winners:continue
        # Candidate thresholds are actual winner measurements, no arbitrary +/- range.
        # Percentiles only pick candidate threshold locations; displayed values are literal thresholds.
        vals_unique=np.unique(valid_w)
        if len(vals_unique)>20:
            ids=np.unique(np.round(np.linspace(0,len(vals_unique)-1,13)).astype(int))
            vals_unique=vals_unique[ids]
        for op in ('>=','<='):
            for t in vals_unique:
                # UNKNOWN never proves a condition. A separate unverified ledger is saved.
                mask=finite&((vals>=t) if op=='>=' else (vals<=t))
                nw=int((mask&winners).sum()); nf=int((mask&~winners).sum())
                if nw>=min_branch_winners and nw<n:
                    all_conds.append((f,op,float(t),mask,nw,nf))
    return all_conds


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matches',default='/data/ydkg_atr_rounded/YDKG_ATR_previous_rounded_30sessions.csv')
    p.add_argument('--database-url',default=os.environ.get('DATABASE_URL'))
    p.add_argument('--out',default='/data/ydkg_compound')
    p.add_argument('--max-branches',type=int,default=4)
    p.add_argument('--min-branch-winners',type=int,default=3)
    p.add_argument('--max-pair-candidates',type=int,default=20000)
    a=p.parse_args()
    if not a.database_url:p.error('DATABASE_URL missing')
    base=pd.read_csv(a.matches).copy()
    base['symbol']=base.symbol.astype(str).str.upper().str.strip()
    base['prev_date']=base.prev_date.astype(str).str[:10]
    base=base[(base.symbol!='YDKG')&(base.match_previous.astype(str).str.lower()=='true')].reset_index(drop=True)
    winners=base.hit100_within30.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
    if not winners.any():p.error('No winning cases in input')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    print('LOADING FEATURES',len(base),'CASES',flush=True)
    vals=load_features(base,a.database_url)
    missing=pd.DataFrame([{'indicator':f,'winner_missing':int((~np.isfinite(v)&winners).sum()),'failure_missing':int((~np.isfinite(v)&~winners).sum())} for f,v in vals.items()])
    missing.to_csv(out/'feature_missing_audit.csv',index=False)
    conds=conditions(vals,winners,a.min_branch_winners)
    print('SINGLE CONDITIONS:',len(conds),flush=True)
    pairs=[]
    for i,(f,op,t,mask,_,_) in enumerate(conds):
        for g,opp,u,mask2,_,_ in conds[i+1:]:
            if f==g:continue
            both=mask&mask2
            w=int(np.sum(both&winners)); bad=int(np.sum(both&~winners))
            if w<a.min_branch_winners:continue
            pairs.append((f,op,t,g,opp,u,both,w,bad))
    # Drop equivalent masks and keep the simplest lexicographic representation.
    bymask={}
    for e in pairs:
        key=np.packbits(e[6]).tobytes()
        if key not in bymask:bymask[key]=e
    pairs=list(bymask.values())
    pairs.sort(key=lambda e:(-e[7]/(e[8]+1),-e[7],e[8]))
    if len(pairs)>a.max_pair_candidates:pairs=pairs[:a.max_pair_candidates]
    print('AND PAIRS:',len(pairs),flush=True)
    # Each OR branch must add >=1 previously uncovered winner, and final result must
    # cover EVERY winner. We do not silently present partial recall as success.
    covered=np.zeros(len(base),dtype=bool);chosen=[];stage=[]
    for number in range(1,a.max_branches+1):
        remaining=winners&~covered
        if not np.any(remaining):break
        remaining_w=int(remaining.sum())
        best=None
        for pair in pairs:
            new=int(np.sum(pair[6]&remaining))
            if new==0:continue
            candidate=covered|pair[6]
            failed=int(np.sum(candidate&~winners))
            # Prioritize recall, but avoid selecting a broad branch indiscriminately.
            # The final stage must cover the remaining wins if possible.
            objective=(new/remaining_w)/(1+failed/max(1,int((~winners).sum())))**2
            if number==a.max_branches and new!=remaining_w:continue
            rank=(objective,new,-failed)
            if best is None or rank>best[0]:best=(rank,pair)
        if best is None:break
        e=best[1];new_winners=int(np.sum(e[6]&winners&~covered));covered|=e[6];chosen.append(e)
        stage.append({'branch':number,'condition_1':f'{e[0]} {e[1]} {e[2]:.15g}',
                      'condition_2':f'{e[3]} {e[4]} {e[5]:.15g}',
                      'new_winners':new_winners,
                      'retained_winners':int(np.sum(covered&winners)),
                      'failed_cases_remaining':int(np.sum(covered&~winners)),
                      'cases_remaining':int(covered.sum())})
        print('BRANCH',number,stage[-1],flush=True)
    # Also evaluate sequential conjunction from the prior threshold file if user has it.
    pd.DataFrame(stage,columns=['branch','condition_1','condition_2','new_winners','retained_winners','failed_cases_remaining','cases_remaining']).to_csv(out/'YDKG_OR_AND_stages.csv',index=False)
    candidates=base.copy();candidates['is_winner']=winners;candidates['passes_compound']=covered
    for i,e in enumerate(chosen,1):candidates[f'branch_{i}_matched']=e[6]
    candidates.to_csv(out/'YDKG_cases_with_compound_flags.csv',index=False)
    all_w=bool(np.all(covered[winners]))
    result={'reference':'YDKG','input_cases':len(base),'input_winners':int(winners.sum()),'input_failures':int((~winners).sum()),
            'single_conditions_tested':len(conds),'distinct_AND_pairs':len(pairs),'OR_branches':len(chosen),
            'all_winners_retained':all_w,'retained_winners':int(np.sum(covered&winners)),
            'final_failed_cases':int(np.sum(covered&~winners)),
            'removed_failed_cases':int(np.sum(~covered&~winners)),
            'result_valid_under_zero_loss_requirement':all_w,
            'note':'Pure threshold AND pairs combined by OR. Missing measurements fail each numeric condition; thus small-history winners may be excluded. In-sample fitted thresholds are NOT predictive validation.'}
    (out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
