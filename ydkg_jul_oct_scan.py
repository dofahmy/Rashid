#!/usr/bin/env python3
"""Frozen YDKG search including recent, right-censored signals; no future-data eligibility filter."""
import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from ydkg_atr_rounded import iter_db,clean,atr_series,valid_symbol
from ydkg_zero_loss_thresholds import feature_at
from ydkg_market_validation import RULES,REF_PREV_ATR,test_rule

COLS=['symbol','signal_date','year_month','entry_close','atr_previous','branch1','branch2','branch3','sessions_available','mature_30','latest_symbol_date','latest_close','latest_return_pct','max_high_observed_pct','max_high_first30_pct','min_low_first30_pct','close_30_return_pct','hit_2_5_first30','hit_30_first30','hit_100_first30','hit_2_5_to_latest','hit_100_to_latest','first_2_5_session','first_100_session','runner_5pct_status','lock30_status','locked_by_date']

def first_hit(high, price):
    w=np.flatnonzero(high>=price)
    return int(w[0])+1 if len(w) else None

def report_rows(df, label):
    result=[]
    for ym,g in [('ALL',df)]+list(df.groupby('year_month',sort=True)):
        matured=g[g.mature_30]
        row={'scope':label,'month':ym,'signals':len(g),'unique_stocks':int(g.symbol.nunique()) if len(g) else 0,
             'mature_30':len(matured),'unmatured':len(g)-len(matured)}
        for k in ['hit_2_5_first30','hit_30_first30','hit_100_first30']:
            n=int(matured[k].sum()) if len(matured) else 0
            row[k]=n;row[k+'_pct']=round(100*n/len(matured),3) if len(matured) else None
        for k in ['hit_2_5_to_latest','hit_100_to_latest']:
            n=int(g[k].sum()) if len(g) else 0
            row[k]=n;row[k+'_pct']=round(100*n/len(g),3) if len(g) else None
        row['median_latest_return_pct']=round(float(g.latest_return_pct.median()),3) if len(g) else None
        result.append(row)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    p.add_argument('--out',default='/data/ydkg_jul_oct')
    p.add_argument('--start',default='2026-01-01',help='Start early to respect previously open same-symbol positions')
    p.add_argument('--show-from',default='2026-07-01')
    p.add_argument('--end',default='2026-10-16')
    p.add_argument('--horizon',type=int,default=30)
    p.add_argument('--atr-decimals',type=int,default=1)
    a=p.parse_args()
    if not a.database_url:p.error('DATABASE_URL required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    allrows=[]; lockrows=[]; total_processed=0;latest=''
    for raw in iter_db(a.database_url):
        sym=str(raw.symbol.iloc[0]).strip().upper()
        if sym=='YDKG' or not valid_symbol(sym):continue
        total_processed+=1;d=clean(raw)
        if len(d)<16:continue
        dates=d.date.to_numpy(); cls=d.c.to_numpy(float);hi=d.h.to_numpy(float);lo=d.l.to_numpy(float)
        latest=max(latest,str(dates[-1]))
        ix=np.flatnonzero((dates>=a.start)&(dates<=a.end)&(np.arange(len(d))>=14))
        if not len(ix):continue
        atr=atr_series(d)
        lock30=-1; runner_lock=-1;runner_entry_date=None
        for i in ix:
            if not np.isfinite(atr[i-1]) or round(float(atr[i-1]),a.atr_decimals)!=round(REF_PREV_ATR,a.atr_decimals):continue
            f=feature_at(d,int(i-1))
            branches=[all(test_rule(f,c) for c in pair) for pair in RULES]
            if not any(branches):continue
            fwd=hi[i+1:min(i+a.horizon+1,len(d))];all_fwd=hi[i+1:]; lows=lo[i+1:min(i+a.horizon+1,len(d))]
            n_observed=len(fwd);mature=n_observed>=a.horizon
            t25=first_hit(fwd,cls[i]*1.025); t100=first_hit(fwd,cls[i]*2.)
            full_t25=first_hit(all_fwd,cls[i]*1.025); full_t100=first_hit(all_fwd,cls[i]*2.)
            # When there is no open 5% runner, allow a new trade; otherwise suppress it.
            runner_block=bool(i<=runner_lock)
            lock30_block=bool(i<=lock30)
            row={'symbol':sym,'signal_date':str(dates[i]),'year_month':str(dates[i])[:7],
                 'entry_close':round(float(cls[i]),7),'atr_previous':round(float(atr[i-1]),6),
                 'branch1':branches[0],'branch2':branches[1],'branch3':branches[2],
                 'sessions_available':n_observed,'mature_30':mature,'latest_symbol_date':str(dates[-1]),
                 'latest_close':round(float(cls[-1]),7),'latest_return_pct':round((float(cls[-1])/cls[i]-1)*100,4),
                 'max_high_observed_pct':round((float(all_fwd.max())/cls[i]-1)*100,4) if len(all_fwd) else None,
                 'max_high_first30_pct':round((float(fwd.max())/cls[i]-1)*100,4) if n_observed else None,
                 'min_low_first30_pct':round((float(lows.min())/cls[i]-1)*100,4) if n_observed else None,
                 'close_30_return_pct':round((float(cls[i+a.horizon])/cls[i]-1)*100,4) if mature else None,
                 'hit_2_5_first30':bool(t25),'hit_30_first30':bool(first_hit(fwd,cls[i]*1.3)),
                 'hit_100_first30':bool(t100),'hit_2_5_to_latest':bool(full_t25),
                 'hit_100_to_latest':bool(full_t100),'first_2_5_session':t25,'first_100_session':t100,
                 'runner_5pct_status':'BLOCKED_SAME_SYMBOL_OPEN' if runner_block else ('CLOSED_100' if full_t100 else ('SOLD_95_RUNNER_OPEN' if full_t25 else 'FULL_POSITION_OPEN')),
                 'lock30_status':'BLOCKED_SAME_SYMBOL' if lock30_block else 'ACCEPT',
                 'locked_by_date':runner_entry_date if runner_block else None}
            allrows.append(row)
            if not lock30_block:lock30=i+(t100 if t100 else a.horizon)
            if not runner_block:
                runner_entry_date=str(dates[i]);runner_lock=i+full_t100 if full_t100 else len(d)-1
            if runner_block:lockrows.append({'symbol':sym,'signal_date':str(dates[i]),'blocked_by_signal':runner_entry_date})
        if total_processed%500==0:print('PROGRESS',total_processed,'symbol groups','qualifying',len(allrows),flush=True)
    df=pd.DataFrame(allrows,columns=COLS).sort_values(['signal_date','symbol'])
    recent=df[(df.signal_date>=a.show_from)&(df.signal_date<=a.end)].copy()
    recent.to_csv(out/'recent_all_signals.csv',index=False)
    recent[recent.runner_5pct_status!='BLOCKED_SAME_SYMBOL_OPEN'].to_csv(out/'recent_actionable_no_runner_overlap.csv',index=False)
    recent[recent.runner_5pct_status=='BLOCKED_SAME_SYMBOL_OPEN'].to_csv(out/'recent_blocked_open_runner.csv',index=False)
    df.to_csv(out/'all_2026_signals_including_prior_locks.csv',index=False)
    recent[~recent.mature_30].to_csv(out/'recent_pending_30_sessions.csv',index=False)
    summary=[]
    summary+=report_rows(recent,'ALL_RAW')
    summary+=report_rows(recent[recent.runner_5pct_status!='BLOCKED_SAME_SYMBOL_OPEN'],'ACTIONABLE_RUNNER_LOCK')
    summary+=report_rows(recent[recent.lock30_status=='ACCEPT'],'LOCK30_COMPARISON')
    pd.DataFrame(summary).to_csv(out/'summary_by_month.csv',index=False)
    rep={'status':'complete','last_database_date_any_symbol':latest,'signals_start':a.start,'recent_start':a.show_from,
         'end':a.end,'symbols_processed':total_processed,'raw_all_2026_signals':len(df),
         'raw_recent_signals':len(recent),'recent_without_open_runner_overlap':int((recent.runner_5pct_status!='BLOCKED_SAME_SYMBOL_OPEN').sum()),
         'recent_blocked_by_open_runner':int((recent.runner_5pct_status=='BLOCKED_SAME_SYMBOL_OPEN').sum()),
         'recent_without_30session_lock':int((recent.lock30_status=='ACCEPT').sum()),
         'recent_fully_evaluated':int(recent.mature_30.sum()),'recent_still_under_30_sessions':int((~recent.mature_30).sum()),
         'strict_100_target_requires_first_following_high_ge_2x_close':True,
         'method':'Frozen YDKG ATR + 3 OR/AND paths on previous-day indicators. Recent signals retained even without 30 forward bars.',
         'runner_lock':'For the 95% at +2.5% and 5% at +100% concept, block subsequent signals on same ticker until the remaining 5% hits +100% (no fixed time exit).',
         'warnings':['Not real-time; prices reflect last DB candle for each symbol; incomplete forward windows are right censored, not failures.',
           'The first 95% exit alone does not free a same-symbol slot while its 5% runner remains open.',
           'Signal-close entry and limit-high touch are optimistic execution assumptions; split adjusted prices may invalidate extreme returns.',
           'Do not apply retrospective greater-than-300% winners exclusion to live recommendations.',
           'Rules were fitted on 2024-Jun 2026 data; July-Oct is prospective-on-history validation only if its data were never used to tune filters.']}
    (out/'report.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    print(json.dumps({'status':'complete','recent_signals':len(recent),'actionable':rep['recent_without_open_runner_overlap'],'open_runner_blocked':rep['recent_blocked_by_open_runner'],'pending_30':rep['recent_still_under_30_sessions'],'out':str(out)},indent=2))
if __name__=='__main__':main()
