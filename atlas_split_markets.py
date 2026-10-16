#!/usr/bin/env python3
"""Rebuild monthly rankings and chronological splits from the existing full Atlas CSV.
This is retrospective research only; .CA classification is provisional and MUST be
verified against a trusted exchange/security master before live use.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='all_eligible_stock_months.csv from Atlas V1.1')
    p.add_argument('--out',required=True)
    p.add_argument('--top',type=int,default=20)
    p.add_argument('--threshold',type=float,default=100)
    p.add_argument('--controls',type=int,default=100)
    p.add_argument('--train-end',default='2022-12')
    p.add_argument('--validation-end',default='2024-12')
    p.add_argument('--verify-symbols',help='Optional CSV with symbol,market columns; market must be US or EGX')
    a=p.parse_args()
    src=Path(a.input); root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(src,low_memory=False)
    required={'symbol','month','month_return_pct','pre_month_close','month_start'}
    missing=required-set(df.columns)
    if missing:raise SystemExit(f'Missing columns: {sorted(missing)}')
    df['symbol']=df.symbol.astype(str).str.upper().str.strip()
    df['month']=df.month.astype(str)
    df['month_return_pct']=pd.to_numeric(df.month_return_pct,errors='coerce')
    df=df.dropna(subset=['month_return_pct'])
    df=df.drop_duplicates(['symbol','month'])
    if a.verify_symbols:
        mapping=pd.read_csv(a.verify_symbols)
        mapping.columns=mapping.columns.str.lower().str.strip()
        if not {'symbol','market'}.issubset(mapping):raise SystemExit('Symbol mapping needs symbol,market columns')
        mapping['symbol']=mapping.symbol.astype(str).str.upper().str.strip()
        mapping['market']=mapping.market.astype(str).str.upper().str.strip()
        mapping=mapping.drop_duplicates('symbol')
        df=df.merge(mapping[['symbol','market']],how='left',on='symbol')
        df['market']=df.market.fillna('UNVERIFIED')
    else:
        df['market']=np.where(df.symbol.str.endswith('.CA'),'EGX','US_CANDIDATE')
    df['split']=np.select([df.month<=a.train_end,df.month<=a.validation_end],['TRAIN','VALIDATION'],default='HOLDOUT')
    manifest={'source_rows':len(df),'classification':'verified mapping' if a.verify_symbols else 'provisional suffix-only (.CA versus other)', 'train_end':a.train_end,'validation_end':a.validation_end,'markets':{}}
    for market, sub in df.groupby('market'):
        out=root/market;out.mkdir(exist_ok=True)
        sub=sub.sort_values(['month','month_return_pct','symbol'],ascending=[True,False,True]).copy()
        sub['rank_in_month']=sub.groupby('month').cumcount()+1
        sub['is_top_n']=sub.rank_in_month<=a.top
        sub['is_100pct_explosion']=sub.month_return_pct>=a.threshold
        sub['is_atlas_winner']=sub.is_top_n|sub.is_100pct_explosion
        sub['target_explosion_100pct']=sub.is_100pct_explosion.astype(int)
        sub['target_atlas_rank']=sub.is_atlas_winner.astype(int)
        sub.to_csv(out/'all_eligible_stock_months.csv',index=False)
        winners=sub.loc[sub.is_atlas_winner].copy()
        winners.to_csv(out/'monthly_explosion_winners.csv',index=False)
        controls=[]
        for month,g in sub.loc[~sub.is_atlas_winner].groupby('month'):
            controls.append(g.sample(n=min(a.controls,len(g)),random_state=20261016))
        ctrl=pd.concat(controls,ignore_index=True) if controls else sub.iloc[0:0].copy()
        ctrl.to_csv(out/'monthly_nonwinner_controls.csv',index=False)
        pd.concat([winners,ctrl],ignore_index=True).to_csv(out/'dna_study_winners_controls.csv',index=False)
        summ=sub.groupby('month').agg(eligible_stocks=('symbol','nunique'),top_n=('is_top_n','sum'),explosions_100pct=('is_100pct_explosion','sum'),max_monthly_return_pct=('month_return_pct','max')).reset_index()
        summ['split']=np.select([summ.month<=a.train_end,summ.month<=a.validation_end],['TRAIN','VALIDATION'],default='HOLDOUT')
        summ.to_csv(out/'monthly_market_summary.csv',index=False)
        stats={'rows':int(len(sub)),'symbols':int(sub.symbol.nunique()),'months':int(sub.month.nunique()),'winners':int(len(winners)),'unique_winner_symbols':int(winners.symbol.nunique()),'explosions_100pct':int(sub.is_100pct_explosion.sum()),'controls':int(len(ctrl)),'splits':sub.groupby('split').size().to_dict()}
        manifest['markets'][market]=stats
        print(market,json.dumps(stats,ensure_ascii=False),flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('DONE',root,flush=True)

if __name__=='__main__':main()
