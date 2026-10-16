#!/usr/bin/env python3
"""Exploratory 14-stock fingerprint and full-universe diagnostic; no forward signal claims."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SYMBOLS=['RGTI','QBTS','IONQ','SOUN','BBAI','RKLB','ASTS','LUNR','ACHR','OKLO','SMR','IREN','CIFR','HIMS']
FEATURES=[
 'pre_ret5_pct','pre_ret10_pct','pre_ret20_pct','pre_ret60_pct','pre_ret120_pct',
 'pre_range20_pct','pre_range60_pct','pre_volatility20_pct','pre_volatility60_pct',
 'pre_vol_ratio20to60','pre_volume20_over60','pre_up_down_vol_ratio20',
 'pre_up_days_share20_pct','pre_drawdown_from120high_pct',
 'pre_distance_ma20_pct','pre_distance_ma60_pct','pre_distance_ma120_pct',
 'pre_last20_compression_ratio', 'median_dollar_volume20'
]
# Predeclared core categories: fixed in advance, rather than optimizing on all future labels.
CORE=['pre_ret20_pct','pre_ret60_pct','pre_drawdown_from120high_pct',
      'pre_volatility60_pct','pre_volume20_over60','pre_distance_ma60_pct']

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input',default='/data/market_explosion_atlas_v2/US_CANDIDATE/all_eligible_stock_months.csv')
 ap.add_argument('--out',default='/data/liquid14_dna_audit')
 ap.add_argument('--min-dollar-volume',type=float,default=4_500_000)
 ap.add_argument('--target-rise-pct',type=float,default=50)
 ap.add_argument('--required-core',type=int,default=4)
 a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.input,low_memory=False)
 req=['symbol','month','month_max_from_prior_close_pct','median_dollar_volume20', 'prior_history_bars'] + FEATURES
 missing=sorted(set(req)-set(d.columns))
 if missing:raise SystemExit('Missing columns: '+str(missing))
 d['symbol']=d.symbol.astype(str).str.upper().str.strip()
 d['month']=d.month.astype(str)
 d=d[(~d.symbol.str.endswith('.CA'))&(d.month>='2024-04')&(d.month<='2026-09')].drop_duplicates(['symbol','month']).copy()
 for c in set(FEATURES+['month_max_from_prior_close_pct','prior_history_bars']):
  d[c]=pd.to_numeric(d[c],errors='coerce')
 eligible=d[(d['median_dollar_volume20']>=a.min_dollar_volume)&(d['prior_history_bars']>=120)].copy()
 eligible['exploded']=(eligible.month_max_from_prior_close_pct>=a.target_rise_pct).astype(int)
 # One retrospective 'best month' per showcase stock; NOT an ex-ante event-selection strategy.
 known=eligible[eligible.symbol.isin(SYMBOLS)].copy()
 known=known.sort_values(['symbol','month_max_from_prior_close_pct','month'],ascending=[True,False,True])
 best=known.drop_duplicates('symbol').sort_values('symbol').copy()
 missing_symbols=sorted(set(SYMBOLS)-set(best.symbol))
 if missing_symbols:print('WARNING: showcase stocks missing from liquidity-eligible data:',missing_symbols,flush=True)
 if len(best)<8:raise SystemExit('Insufficient showcase examples for diagnostic.')
 best.to_csv(root/'showcase14_best_months.csv',index=False)
 # Report pre-month distribution for every candidate attribute.
 profiles=[]
 for f in FEATURES:
  v=best[f].dropna(); base=eligible[f].dropna()
  if not len(v):continue
  profiles.append(dict(feature=f,examples=len(v),median=float(v.median()),p10=float(v.quantile(.10)),p25=float(v.quantile(.25)),p75=float(v.quantile(.75)),p90=float(v.quantile(.90)),market_median=float(base.median()) if len(base) else np.nan))
 pd.DataFrame(profiles).to_csv(root/'shared_feature_profile.csv',index=False)
 # Transparent *illustrative* shared fingerprint: 10th--90th percentile bounds on SIX predeclared features,
 # score=number of conditions met. Bounds use best-month showcase examples, thus in-sample by construction.
 condcols=[];rules=[]
 for f in CORE:
  v=best[f].dropna()
  if len(v)<8:continue
  low=float(v.quantile(.10)); high=float(v.quantile(.90))
  name='match_'+f
  eligible[name]=eligible[f].between(low,high,inclusive='both') & eligible[f].notna()
  best[name]=best[f].between(low,high,inclusive='both') & best[f].notna()
  condcols.append(name)
  rules.append(dict(feature=f,lower_p10=low,upper_p90=high,showcase_matches=int(best[name].sum()),showcase_total=int(len(best))))
 if not condcols:raise SystemExit('No usable fingerprint characteristics.')
 need=min(a.required_core,len(condcols))
 eligible['matching_conditions']=eligible[condcols].sum(axis=1)
 best['matching_conditions']=best[condcols].sum(axis=1)
 eligible['flagged']=eligible.matching_conditions>=need
 best['flagged']=best.matching_conditions>=need
 # Coverage of all historical liquid explosions and all negative stock-months.
 tp=int((eligible.flagged & (eligible.exploded==1)).sum())
 fp=int((eligible.flagged & (eligible.exploded==0)).sum())
 fn=int((~eligible.flagged & (eligible.exploded==1)).sum())
 tn=int((~eligible.flagged & (eligible.exploded==0)).sum())
 selected=eligible[eligible.flagged].copy()
 exploded=eligible[eligible.exploded==1].copy()
 selected.sort_values(['month','matching_conditions','symbol'],ascending=[True,False,True]).to_csv(root/'all_flagged_stock_months.csv',index=False)
 selected[selected.exploded==1].sort_values(['month','month_max_from_prior_close_pct'],ascending=[True,False]).to_csv(root/'correct_explosions.csv',index=False)
 selected[selected.exploded==0].sort_values(['month','matching_conditions'],ascending=[True,False]).to_csv(root/'false_positive_stock_months.csv',index=False)
 exploded[~exploded.flagged].sort_values(['month','month_max_from_prior_close_pct'],ascending=[True,False]).to_csv(root/'missed_explosions.csv',index=False)
 best.to_csv(root/'showcase14_rule_checks.csv',index=False)
 pd.DataFrame(rules).to_csv(root/'illustrative_common_rules.csv',index=False)
 permonth=eligible.groupby('month').apply(lambda x:pd.Series({'eligible':len(x),'explosions':int(x.exploded.sum()),'flags':int(x.flagged.sum()),'correct':int((x.flagged & (x.exploded==1)).sum()),'false_positives':int((x.flagged & (x.exploded==0)).sum()),'missed':int(((~x.flagged)&(x.exploded==1)).sum())}),include_groups=False).reset_index()
 permonth.to_csv(root/'monthly_scorecard.csv',index=False)
 base_rate=float(eligible.exploded.mean());precision=tp/(tp+fp) if tp+fp else None
 report={'universe':'non-.CA provisional US symbols','eligibility':{'min_pre_month_median_daily_dollar_volume':a.min_dollar_volume,'min_prior_bars':120},'target':f'month_max_from_prior_close_pct >= {a.target_rise_pct}',
 'showcase_stocks_requested':len(SYMBOLS),'showcase_stocks_found':len(best),'showcase_missing':missing_symbols,
 'showcase_fingerprint_matches':int(best.flagged.sum()),'fingerprint':{'method':'Predeclared six features, p10-p90 intervals computed from retrospectively best month of each showcase stock; score is number of matching intervals','conditions_required':need,'conditions_available':len(condcols)},
 'full_liquid_universe':{'eligible_stock_months':len(eligible),'unique_symbols':int(eligible.symbol.nunique()),'explosion_stock_months':int(eligible.exploded.sum()),'unique_explosion_stocks':int(exploded.symbol.nunique()),'flagged':tp+fp,'correct':tp,'false_positives':fp,'missed_explosions':fn,'true_negatives':tn,'precision':precision,'recall':tp/(tp+fn) if tp+fn else None,'base_rate':base_rate,'lift':precision/base_rate if base_rate and precision is not None else None},
 'caveat':'IN-SAMPLE: all 14 examples intentionally chosen after observing explosions, 2024-2026 labels already inspected; this is feature mining, not independent validation. Price adjustment/splits, universe security type and delistings unverified. One ticker can create multiple stock-months.'}
 (root/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print('SHOWCASE 14',best[['symbol','month','month_max_from_prior_close_pct','median_dollar_volume20','matching_conditions','flagged']].to_string(index=False),flush=True)
 print('FINGERPRINT RULES',pd.DataFrame(rules).to_string(index=False),flush=True)
 print('RESULT',json.dumps(report,indent=2,ensure_ascii=False),flush=True)
 print('FILES',root,flush=True)
if __name__=='__main__':main()
