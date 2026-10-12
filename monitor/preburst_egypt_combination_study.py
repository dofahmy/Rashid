"""Egypt-only exploratory comparison of predefined combinations of Pre-Burst rules.

Run: python -m monitor.preburst_egypt_combination_study \
 --input /tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv \
 --output-dir /tmp/preburst-egypt-combinations

Read only: no database access, no changes to existing screening or signal records.
Note: 2025 data was already inspected in development, so it is not a pristine holdout.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED=['symbol','signal_start_utc','classification','full_250_days',
          'repeat_count_1pct_60','close_vs_sma50_pct','rsi14','volume5_vs_50']
BASELINE_LABEL='original_filter'

def calc(input_file, output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(input_file,dtype={'symbol':str},low_memory=False)
    absent=[c for c in REQUIRED if c not in df.columns]
    if absent:raise ValueError('Missing columns: '+', '.join(absent))
    df['year']=pd.to_datetime(df['signal_start_utc'],errors='coerce',utc=True).dt.year
    df=df[df.year.between(2023,2026)].copy()
    for c in REQUIRED[4:]:df[c]=pd.to_numeric(df[c],errors='coerce')
    if 'outcome' not in df:
        df['outcome']=np.select([df.classification.isin(['متفوق','ناجح']),df.classification.eq('لم يحقق خلال 250 يوم')],['winner','failure'],default='pending')
    full=df.full_250_days.astype(str).str.lower().isin(['true','1','1.0'])
    df['mature']=full & df.outcome.isin(['winner','failure'])
    df['confirmed_winner']=df.outcome.eq('winner')
    a=df.repeat_count_1pct_60.gt(3)&df.repeat_count_1pct_60.le(6)
    b=df.close_vs_sma50_pct.gt(-1.7608)
    c=df.rsi14.gt(41.163)
    d=df.volume5_vs_50.le(.43982)
    df['rule_repeat_4_to_6']=a
    df['rule_sma50_above_minus1_7608']=b
    df['rule_rsi_above_41_163']=c
    df['rule_volume_under_0_43982']=d
    df['rule_count']=a.astype(int)+b.astype(int)+c.astype(int)+d.astype(int)
    strategies={
      'baseline':pd.Series(True,index=df.index),
      'selective_3_of_3_repeat_sma_rsi':a&b&c,
      'balanced_at_least_2_of_4':df.rule_count>=2,
      'broad_at_least_1_of_4':df.rule_count>=1,
      'repeat_only':a,
      'sma50_only':b,
      'rsi_only':c,
      'volume_only':d,
    }
    for name, mask in strategies.items():df['selected_'+name]=mask
    slices={'train_2023_2024':df.year.isin([2023,2024]),
            'review_2025_2026':df.year.isin([2025,2026]),
            'all_2023_2026':pd.Series(True,index=df.index)}
    records=[]
    for segment, segmask in slices.items():
        seg=df[segmask]; mature=seg[seg.mature]; total_winners=int(mature.confirmed_winner.sum())
        for name in strategies:
            allselected=seg[seg['selected_'+name]]
            picked=allselected[allselected.mature]
            successes=int(picked.confirmed_winner.sum()); n=len(picked)
            records.append({'segment':segment,'strategy':name,'total_cycles':len(seg),
              'mature_cycles_baseline':len(mature),'mature_winners_baseline':total_winners,
              'selected_mature':n,'selected_mature_winners':successes,
              'selected_mature_failures':n-successes,
              'win_rate_pct':round(100*successes/n,2) if n else None,
              'baseline_win_rate_pct':round(100*total_winners/len(mature),2) if len(mature) else None,
              'lift_percentage_points':round(100*successes/n-100*total_winners/len(mature),2) if n and len(mature) else None,
              'winner_recall_pct':round(100*successes/total_winners,2) if total_winners else None,
              'selected_stocks':int(picked.symbol.nunique()),
              'selected_all_including_pending':len(allselected),
              'selected_confirmed_winners_all':int(allselected.confirmed_winner.sum()),
              'selected_pending':int((allselected.outcome=='pending').sum()),
              'selected_early_winners_immature':int((allselected.confirmed_winner & ~allselected.mature).sum())})
    summary=pd.DataFrame(records)
    summary.to_csv(out/'strategy_comparison.csv',index=False,encoding='utf-8-sig')
    per_year=[]
    for year, group in df.groupby('year'):
        mature=group[group.mature]; wins=int(mature.confirmed_winner.sum())
        for name in strategies:
            selected=mature[mature['selected_'+name]]
            n=len(selected); w=int(selected.confirmed_winner.sum())
            per_year.append({'year':int(year),'strategy':name,'mature_baseline':len(mature),
            'baseline_winners':wins,'selected_mature':n,'winners':w,'failures':n-w,
            'win_rate_pct':round(100*w/n,2) if n else None})
    pd.DataFrame(per_year).to_csv(out/'strategies_by_year.csv',index=False,encoding='utf-8-sig')
    df.to_csv(out/'every_cycle_with_strategy_flags.csv',index=False,encoding='utf-8-sig')
    info={'scope':'Egypt 2023-2026, +100% within 250 daily trading sessions',
      'total_cycles':len(df),'mature_cycles':int(df.mature.sum()),
      'strategies':list(strategies),'rules':{
      'repeat':'3 < repeat_count_1pct_60 <= 6','sma50':'close_vs_sma50_pct > -1.7608',
      'rsi':'rsi14 > 41.163','volume':'volume5_vs_50 <= 0.43982'},
      'results':summary.to_dict(orient='records'),
      'cautions':['Exploratory: the 2025-26 rule rankings were previously examined; this is NOT an untouched out-of-sample test.',
       'Win rate uses mature 250-session cycles only; early winners from immature cycles shown separately.',
       'No new backtest trades are simulated: combinations are applied to entry indicators for the PREVIOUS baseline cycles.',
       'Multiple nonoverlapping cycles of the same stock are dependent; verify adjusted price series and corporate actions.',
       'Missing numeric indicator never satisfies its rule.']}
    (out/'comparison_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'output_dir':str(out),'cycles':len(df),'mature':int(df.mature.sum()),
            'comparison':summary[summary.segment=='review_2025_2026'].to_dict(orient='records'),
            'files':['strategy_comparison.csv','strategies_by_year.csv','every_cycle_with_strategy_flags.csv','comparison_summary.json']}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='/tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv')
    p.add_argument('--output-dir',default='/tmp/preburst-egypt-combinations')
    a=p.parse_args()
    print(json.dumps(calc(a.input,a.output_dir),ensure_ascii=False,indent=2),flush=True)
