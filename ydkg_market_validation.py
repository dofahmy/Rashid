#!/usr/bin/env python3
"""Freeze YDKG ATR + three OR-of-AND rules, evaluate every complete US stock/date.
Read-only PostgreSQL. No refitting. Uses same ATR and indicator definitions as discovery.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from ydkg_atr_rounded import iter_db,clean,atr_series,valid_symbol
from ydkg_zero_loss_thresholds import feature_at

RULES = [
    (('ret5','<=',2.9064047335132), ('ret120','<=',-67.3110719187771)),
    (('ret10','>=',-12.6760557074163), ('distance_high60_pct','>=',-58.1554212616052)),
    (('rsi14','>=',34.5520082996184), ('dist_ma20','<=',-30.8753095661602)),
]
REF_PREV_ATR = 32.20779220779221

def test_rule(vals, cond):
    f,op,threshold=cond
    v=vals.get(f,np.nan)
    if v is None or not np.isfinite(v):return False
    return bool(v<=threshold if op=='<=' else v>=threshold)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--database-url',default=os.environ.get('DATABASE_URL'))
    ap.add_argument('--previous-matches',default='/data/ydkg_atr_rounded/YDKG_ATR_previous_rounded_30sessions.csv')
    ap.add_argument('--out',default='/data/ydkg_market_validation')
    ap.add_argument('--start',default='2024-01-01')
    ap.add_argument('--end',default='2026-06-30')
    ap.add_argument('--horizon',type=int,default=30)
    ap.add_argument('--atr-decimals',type=int,default=1)
    a=ap.parse_args()
    if not a.database_url:ap.error('DATABASE_URL missing')
    if a.horizon<1:ap.error('horizon must be >= 1')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    known=set()
    if a.previous_matches and Path(a.previous_matches).exists():
        train=pd.read_csv(a.previous_matches,usecols=['symbol','signal_date'])
        known={(str(r.symbol).upper(),str(r.signal_date)[:10]) for r in train.itertuples()}
    rows=[]; totals={y:{'eligible':0,'atr':0,'compound':0,'hits_all':0,'hits_atr':0,'hits_compound':0} for y in ('2024','2025','2026')}
    processed=0;ignored_no_full=0
    for group in iter_db(a.database_url):
        sym=str(group.symbol.iloc[0]).upper().strip()
        if not valid_symbol(sym) or sym=='YDKG':continue
        processed+=1
        d=clean(group)
        if len(d)<14+a.horizon+1:continue
        dates=d.date.to_numpy();close=d.c.to_numpy(float);high=d.h.to_numpy(float);low=d.l.to_numpy(float)
        candidates=np.flatnonzero((dates>=a.start)&(dates<=a.end))
        candidates=candidates[candidates>=14]
        full=candidates[candidates+a.horizon<len(d)]
        ignored_no_full+=len(candidates)-len(full)
        if not len(full):continue
        atr=atr_series(d)
        match=np.isfinite(atr[full-1])&(np.round(atr[full-1],a.atr_decimals)==round(REF_PREV_ATR,a.atr_decimals))
        mask_idx=full[match]
        # Daily-market baseline uses exactly same fully-observed stock-dates.
        for i in full:
            year=str(dates[i])[:4]
            if year not in totals:continue
            t=totals[year];t['eligible']+=1
            hit=bool(np.any(high[i+1:i+a.horizon+1]>=2*close[i]))
            t['hits_all']+=int(hit)
        for i in mask_idx:
            year=str(dates[i])[:4];t=totals.get(year)
            if t is None:continue
            t['atr']+=1
            prior_feats=feature_at(d,int(i-1))
            branches=[all(test_rule(prior_feats,c) for c in branch) for branch in RULES]
            passes=any(branches)
            future_high=high[i+1:i+a.horizon+1];future_low=low[i+1:i+a.horizon+1]
            hit_arr=np.flatnonzero(future_high>=2*close[i]);hit=bool(len(hit_arr))
            t['hits_atr']+=int(hit)
            t['compound']+=int(passes)
            t['hits_compound']+=int(hit and passes)
            rows.append({'symbol':sym,'signal_date':str(dates[i]),'prev_date':str(dates[i-1]),'year':year,
                'in_original_135_cases':(sym,str(dates[i])) in known,
                'atr14_pct_previous':float(atr[i-1]),'atr_rounded':round(float(atr[i-1]),a.atr_decimals),
                'branch1':branches[0],'branch2':branches[1],'branch3':branches[2],
                'passes_compound':passes,'entry_close':float(close[i]),
                'max_high_return_30_pct':round((float(future_high.max())/close[i]-1)*100,6),
                'worst_low_return_30_pct':round((float(future_low.min())/close[i]-1)*100,6),
                'close_return_30_pct':round((float(close[i+a.horizon])/close[i]-1)*100,6),
                'hit100_within30':hit,
                'first_hit100_session':int(hit_arr[0]+1) if hit else None,
                'first_max_high_session':int(future_high.argmax()+1)})
        if processed%500==0:print(f'PROGRESS {processed} symbols, ATR matches {len(rows)}',flush=True)
    df=pd.DataFrame(rows,columns=['symbol','signal_date','prev_date','year','in_original_135_cases',
        'atr14_pct_previous','atr_rounded','branch1','branch2','branch3','passes_compound','entry_close',
        'max_high_return_30_pct','worst_low_return_30_pct','close_return_30_pct','hit100_within30',
        'first_hit100_session','first_max_high_session'])
    df.to_csv(out/'all_ATR_matching_candidates.csv',index=False)
    df[df.passes_compound].sort_values(['signal_date','symbol']).to_csv(out/'compound_passed_all.csv',index=False)
    df[df.passes_compound & df.hit100_within30].sort_values(['first_hit100_session','max_high_return_30_pct'],ascending=[True,False]).to_csv(out/'compound_winners_100pct.csv',index=False)
    df[(~df.in_original_135_cases)&df.passes_compound].to_csv(out/'compound_passed_outside_original_135.csv',index=False)
    report=[]
    for label, sub, global_t in [('2024',df[df.year=='2024'],totals['2024']),('2025',df[df.year=='2025'],totals['2025']),('2026',df[df.year=='2026'],totals['2026']),('ALL',df, {k:sum(v[k] for v in totals.values()) for k in totals['2024']})]:
        for scope,ss in [('all_atr_matches',sub),('outside_original_135',sub[~sub.in_original_135_cases])]:
            apass=ss[ss.passes_compound]; w=apass[apass.hit100_within30]
            report.append({'year':label,'scope':scope,'market_eligible_stock_dates':global_t['eligible'],
              'market_hits_100_30':global_t['hits_all'],'market_hit_rate_pct':round(100*global_t['hits_all']/global_t['eligible'],6) if global_t['eligible'] else None,
              'atr_cases':len(ss),'atr_hits':int(ss.hit100_within30.sum()),'atr_rate_pct':round(100*ss.hit100_within30.mean(),6) if len(ss) else None,
              'compound_cases':len(apass),'compound_hits':len(w),
              'compound_rate_pct':round(100*len(w)/len(apass),6) if len(apass) else None,
              'compound_unique_stocks':int(apass.symbol.nunique()),'winner_unique_stocks':int(w.symbol.nunique()),
              'median_first_hit100_sessions':float(w.first_hit100_session.median()) if len(w) else None,
              'median_max_high_return_pct':float(w.max_high_return_30_pct.median()) if len(w) else None})
    pd.DataFrame(report).to_csv(out/'results_by_year.csv',index=False)
    summary={'status':'complete','symbol_groups_processed':processed,'excluded_reference':'YDKG',
        'atr_reference_previous_pct':REF_PREV_ATR,'atr_decimals':a.atr_decimals,'outcome':'High >= 2 * signal-day Close in next 30 trading sessions',
        'complete_forward_sessions_required':a.horizon,'incomplete_date_candidates_excluded':ignored_no_full,
        'original_known_cases_count':len(known),'rules':[[{'feature':f,'operator':op,'threshold':t} for f,op,t in branch] for branch in RULES],
        'rule_logic':'previous-day ATR rounded to 1 decimal AND ((ret5 <= threshold AND ret120 <= threshold) OR (ret10 >= threshold AND distance_high60_pct >= threshold) OR (rsi14 >= threshold AND dist_ma20 <= threshold))',
        'cautions':['Rules fitted to 135 selected historical cases; excluding the 135 only is NOT an untouched holdout.',
                    'Universe and corporate actions may have survivorship or split issues; high prices do not guarantee fill.',
                    'Repeated stock-date observations correlated; summaries report cases and distinct symbols separately.',
                    'Missing indicators fail numeric AND conditions.']}
    (out/'report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({'status':'complete','atr_cases':len(df),'compound_cases':int(df.passes_compound.sum()),
        'compound_winners':int((df.passes_compound&df.hit100_within30).sum()),'output':str(out)},indent=2),flush=True)

if __name__=='__main__':main()
