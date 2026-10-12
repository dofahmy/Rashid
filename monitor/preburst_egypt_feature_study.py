"""Egypt-only PreBurst study. Read-only indicators + offline outcomes CSV.
Run: python -m monitor.preburst_egypt_feature_study --cycles /tmp/preburst-250d-egypt/nonoverlapping_cycles_250_trading_days.csv --output-dir /tmp/preburst-egypt-features
"""
from __future__ import annotations
import argparse, json, csv, math
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text

FIELDS=['rsi14','momentum5_pct','momentum20_pct','close_vs_sma20_pct','close_vs_sma50_pct','bbz20','atr14_pct','return_20','position_20_pct','volume5_vs_50','repeat_count_1pct_60']
TABLE='us_preburst_full_history_v5'

def load_features(base):
    """Batch-join Egypt indicators at the exact signal date, not a future candle."""
    from core import database
    from sqlalchemy import text
    wanted=base[['symbol','signal_date']].drop_duplicates().reset_index(drop=True)
    rows=[]
    with database()() as sess:
        for off in range(0,len(wanted),150):
            batch=wanted.iloc[off:off+150]; params={}; vals=[]
            for i,r in enumerate(batch.itertuples(index=False)):
                params[f's{i}']=str(r.symbol);params[f'd{i}']=str(r.signal_date)
                vals.append(f'(:s{i}, CAST(:d{i} AS DATE))')
            query=f"""WITH wanted(symbol, signal_date) AS (VALUES {','.join(vals)})
                SELECT w.symbol,w.signal_date, h.reference_utc, {','.join('h."'+k+'" AS "'+k+'"' for k in FIELDS)}
                FROM wanted w LEFT JOIN {TABLE} h
                  ON h.market='egypt' AND h.symbol=w.symbol
                 AND CAST(h.reference_utc AS DATE)=w.signal_date"""
            rows.extend(dict(r) for r in sess.execute(text(query),params).mappings())
    z=pd.DataFrame(rows)
    if not z.empty:
        z['signal_date']=pd.to_datetime(z.signal_date).dt.strftime('%Y-%m-%d')
        # Never silently pick a different bar if multiple share the same date.
        dup=z.duplicated(['symbol','signal_date'],keep=False)
        if dup.any():
            raise RuntimeError(f'Duplicate indicator dates for {int(dup.sum())} records; investigate rather than selecting arbitrarily')
    return z

def stats(df, cols):
    output=[]
    for col in cols:
        for grp,name in [(df[df.outcome=='winner'],'winner_100pct'),(df[df.outcome=='failure'],'failure_250_complete')]:
            v=pd.to_numeric(grp[col],errors='coerce').dropna()
            output.append({'indicator':col,'group':name,'n':int(len(v)),'median':round(float(v.median()),5) if len(v) else None,'mean':round(float(v.mean()),5) if len(v) else None,'q25':round(float(v.quantile(.25)),5) if len(v) else None,'q75':round(float(v.quantile(.75)),5) if len(v) else None})
    return pd.DataFrame(output)

def build_candidates(train,test):
    """Single-indicator rules, discovered in training only; 2025+ is untouched test."""
    candidates=[]; train=train.copy();test=test.copy()
    for col in FIELDS:
        t=pd.to_numeric(train[col],errors='coerce')
        valid=t.dropna()
        if len(valid)<30: continue
        boundaries=sorted(set(float(x) for x in valid.quantile([.2,.4,.6,.8]).tolist()))
        rules=[]
        for q in boundaries: rules += [('<=',q),('>',q)]
        edges=[-np.inf]+boundaries+[np.inf]
        rules += [('between',(edges[i],edges[i+1])) for i in range(len(edges)-1)]
        for op,cut in rules:
            def matches(frame):
                v=pd.to_numeric(frame[col],errors='coerce')
                if op=='<=':return v<=cut
                if op=='>':return v>cut
                return (v>cut[0])&(v<=cut[1])
            mask=matches(train)
            tr=train[mask]; ts=test[matches(test)]
            if len(tr)<20 or len(tr)<.08*len(train):continue
            base=train.outcome.eq('winner').mean(); rate=tr.outcome.eq('winner').mean()
            if rate <= base:continue
            def result(a,prefix):
                return {prefix+'_count':int(len(a)),prefix+'_winners':int(a.outcome.eq('winner').sum()),prefix+'_win_rate_pct':round(float(100*a.outcome.eq('winner').mean()),2) if len(a) else None}
            cut_txt=f'{cut:.5g}' if op!='between' else f'({cut[0]:.5g}, {cut[1]:.5g}]'
            candidates.append({'indicator':col,'rule':f'{op} {cut_txt}','train_lift_x':round(float(rate/base),3) if base else None,**result(tr,'train'),**result(ts,'test')})
    return pd.DataFrame(candidates).sort_values(['train_lift_x','train_count'],ascending=False) if candidates else pd.DataFrame(columns=['indicator','rule'])

def run(cycles, out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(cycles,dtype={'symbol':str}); mandatory={'symbol','signal_start_utc','classification','full_250_days','max_rise_250d_pct'}
    if not mandatory.issubset(d.columns):raise ValueError('Missing CSV columns: '+str(mandatory-set(d.columns)))
    d['signal_date']=pd.to_datetime(d.signal_start_utc,errors='coerce').dt.strftime('%Y-%m-%d')
    d['year']=pd.to_datetime(d.signal_date).dt.year
    d=d[d.year.between(2023,2026)].copy()
    d['outcome']=np.select([d.classification.isin(['متفوق','ناجح']),d.classification.eq('لم يحقق خلال 250 يوم')],['winner','failure'],default='pending')
    # Full, non-success is a failure; early +100% is a confirmed win, even if immature.
    def summarise(data):return {'rows':int(len(data)),'winner':int((data.outcome=='winner').sum()),'failure':int((data.outcome=='failure').sum()),'pending':int((data.outcome=='pending').sum())}
    f=load_features(d); merged=d.merge(f,on=['symbol','signal_date'],how='left',validate='one_to_one')
    merged['indicator_matched']=merged.reference_utc.notna()
    merged.to_csv(out/'egypt_cycles_with_entry_indicators_2023_2026.csv',index=False,encoding='utf-8-sig')
    matched=merged[merged.indicator_matched].copy()
    mature=matched[matched.outcome.isin(['winner','failure'])].copy()
    eligible=mature[mature.full_250_days.astype(str).str.lower().isin(['true','1'])].copy()
    train=eligible[eligible.year.isin([2023,2024])]
    # 2025-26 only includes confirmed winners + mature non-winners, with count reported.
    hold=eligible[eligible.year.isin([2025,2026])]
    stat=stats(mature,FIELDS)
    stat.to_csv(out/'winner_vs_failure_indicators.csv',index=False,encoding='utf-8-sig')
    candidates=build_candidates(train,hold)
    candidates.to_csv(out/'train_rules_with_holdout_check.csv',index=False,encoding='utf-8-sig')
    year_table=matched.groupby('year',dropna=False).outcome.value_counts().unstack(fill_value=0)
    year_table.to_csv(out/'year_outcomes_matched.csv',encoding='utf-8-sig')
    info={'market':'egypt','period':'2023-2026','target_pct':100,'followup':'250 distinct daily sessions',
          'all_2023_2026':summarise(d),'matched_indicators':summarise(matched),'missing_indicator_rows':int((~merged.indicator_matched).sum()),
          'fully_mature_with_indicators':summarise(eligible),'train_2023_2024':summarise(train),'holdout_2025_2026':summarise(hold),
          'rules_created':int(len(candidates)),
          'cautions':['Candidates are exploratory and selected using 2023-2024 only; 2025-2026 testing is limited by right censoring.',
                      'Descriptive comparison includes early confirmed wins. Candidate rule discovery and holdout use full 250-session cohorts only; pending cohorts excluded.',
                      'Source prices must be checked for corporate actions, splits and stale/erroneous quotes before financial conclusions.',
                      'Same symbol may have several non-overlapping cycles, so records are not independent observations.']}
    (out/'study_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return info

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycles',default='/tmp/preburst-250d-egypt/nonoverlapping_cycles_250_trading_days.csv');p.add_argument('--output-dir',default='/tmp/preburst-egypt-features');a=p.parse_args()
    print(json.dumps(run(a.cycles,a.output_dir),ensure_ascii=False,indent=2),flush=True)
