#!/usr/bin/env python3
"""YDKG fixed ATR + OR/AND backtest. Exit at first high touch of TP within 30 sessions;
otherwise close at session 30. Do not re-enter the same ticker during an open trade.
Research assumptions: entry at signal-day close, target fill at limit price, no slippage/fees.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from ydkg_atr_rounded import iter_db, clean, atr_series, valid_symbol
from ydkg_zero_loss_thresholds import feature_at
from ydkg_market_validation import RULES, REF_PREV_ATR, test_rule

TARGETS=(10,20,30,50,75,100,137.66)

def backtest_candidates(candidates, close, high, low, dates, horizon, target_pct):
    """All potential signals (sorted day indices); lock ticker until exit day inclusive.
    Returns trades, ignored candidate count. Events touch target at High; no intra-day order assumption.
    """
    trades=[]; ignored=0; locked_through=-1
    for i, branches in candidates:
        if i<=locked_through:
            ignored+=1; continue
        entry=float(close[i]); window_high=high[i+1:i+horizon+1];window_low=low[i+1:i+horizon+1]
        if len(window_high)!=horizon: continue
        target=entry*(1.+target_pct/100.)
        touched=np.flatnonzero(window_high>=target)
        if len(touched):
            duration=int(touched[0])+1
            exit_price=target
            status='target_hit'
        else:
            duration=horizon; exit_price=float(close[i+duration]);status='time_exit'
        locked_through=i+duration
        realized=(exit_price/entry-1.)*100.
        # Observed maximum/minimum only until actual exit; can't claim time order inside last candle.
        hmax=float(window_high[:duration].max()); lmin=float(window_low[:duration].min())
        trades.append({'symbol':None,'signal_date':str(dates[i]),'exit_date':str(dates[locked_through]),
                       'entry_close':entry,'target_pct':target_pct,'exit_type':status,
                       'exit_price_assumed':round(exit_price,8),'days_open':duration,
                       'realized_return_pct_assumed':round(realized,6),
                       'max_high_before_exit_pct':round((hmax/entry-1)*100,6),
                       'worst_low_before_exit_pct':round((lmin/entry-1)*100,6),
                       'branch_1':bool(branches[0]),'branch_2':bool(branches[1]),'branch_3':bool(branches[2])})
    return trades, ignored

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--database-url',default=os.environ.get('DATABASE_URL'))
    ap.add_argument('--out',default='/data/ydkg_nonoverlap')
    ap.add_argument('--start',default='2024-01-01');ap.add_argument('--end',default='2026-06-30')
    ap.add_argument('--horizon',type=int,default=30)
    ap.add_argument('--atr-decimals',type=int,default=1)
    ap.add_argument('--targets',default=','.join(map(str,TARGETS)))
    a=ap.parse_args()
    if not a.database_url:ap.error('DATABASE_URL required')
    if a.horizon<1:ap.error('--horizon must be positive')
    targets=sorted(set(float(t.strip()) for t in a.targets.split(',')))
    if not targets or any(t<=0 for t in targets):ap.error('Targets must be positive')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    all_trades=[]; eligible=0; processed=0; missing_followup=0; ignored_by_target={t:0 for t in targets}
    for raw in iter_db(a.database_url):
        sym=str(raw.symbol.iloc[0]).upper().strip()
        if sym=='YDKG' or not valid_symbol(sym):continue
        processed+=1
        d=clean(raw)
        if len(d)<14+a.horizon+1:continue
        dates=d.date.to_numpy();close=d.c.to_numpy(float);high=d.h.to_numpy(float);low=d.l.to_numpy(float)
        candidates_idx=np.flatnonzero((dates>=a.start)&(dates<=a.end)&(np.arange(len(d))>=14))
        missing_followup+=int((candidates_idx+a.horizon>=len(d)).sum())
        candidates_idx=candidates_idx[candidates_idx+a.horizon<len(d)]
        if not len(candidates_idx):continue
        atr=atr_series(d)
        candidates=[]
        for i in candidates_idx:
            if not np.isfinite(atr[i-1]) or round(float(atr[i-1]),a.atr_decimals)!=round(REF_PREV_ATR,a.atr_decimals):continue
            f=feature_at(d,int(i-1))
            branches=tuple(all(test_rule(f,c) for c in br) for br in RULES)
            if any(branches):candidates.append((int(i),branches))
        eligible+=len(candidates)
        if not candidates:continue
        for t in targets:
            trades,ignored=backtest_candidates(candidates,close,high,low,dates,a.horizon,t)
            ignored_by_target[t]+=ignored
            for r in trades:r['symbol']=sym
            all_trades.extend(trades)
        if processed%500==0:print(f'PROGRESS {processed} stocks; raw qualifying signals {eligible}',flush=True)
    df=pd.DataFrame(all_trades,columns=['symbol','signal_date','exit_date','entry_close','target_pct','exit_type',
        'exit_price_assumed','days_open','realized_return_pct_assumed','max_high_before_exit_pct',
        'worst_low_before_exit_pct','branch_1','branch_2','branch_3'])
    df.to_csv(out/'all_nonoverlapping_trades.csv',index=False)
    summaries=[]
    for t in targets:
        sub=df[df.target_pct==t]
        for label,ss in [('ALL',sub)]+[(year,sub[sub.signal_date.str[:4]==year]) for year in ('2024','2025','2026')]:
            n=len(ss); hits=int((ss.exit_type=='target_hit').sum())
            summaries.append({'target_pct':t,'entry_year':label,'raw_signals_before_lock':eligible if label=='ALL' else None,
                'blocked_same_stock_signals':ignored_by_target[t] if label=='ALL' else None,
                'trades':n,'unique_stocks':int(ss.symbol.nunique()),'target_hits':hits,
                'hit_rate_pct':round(100*hits/n,5) if n else None,
                'median_days_to_target':round(float(ss.loc[ss.exit_type=='target_hit','days_open'].median()),3) if hits else None,
                'mean_realized_return_pct_assumed':round(float(ss.realized_return_pct_assumed.mean()),5) if n else None,
                'median_realized_return_pct_assumed':round(float(ss.realized_return_pct_assumed.median()),5) if n else None,
                'worst_realized_return_pct_assumed':round(float(ss.realized_return_pct_assumed.min()),5) if n else None,
                'median_worst_low_before_exit_pct':round(float(ss.worst_low_before_exit_pct.median()),5) if n else None})
    pd.DataFrame(summaries).to_csv(out/'summary_by_target_and_year.csv',index=False)
    report={'raw_compound_matching_signal_dates':eligible,'symbols_processed':processed,'missing_30_followup_candidates':missing_followup,
         'target_levels_pct':targets,'horizon_trading_sessions':a.horizon,'entry':'signal-day daily CLOSE (assumed)',
         'exit':'first following-session HIGH touches target -> assumed target-limit fill; otherwise CLOSE at 30th session',
         'no_overlap':'per symbol and per target scenario; ignore any qualifying signals on or before an existing trade exit session',
         'signals_after_exit':'eligible on next trading session if they independently pass filters',
         'warning':['Full-history results are in-sample for fixed rules fitted to original 135 signals.',
             'Target touch at HIGH does not guarantee executable fill; gaps, splits, spreads, fees excluded.',
             'Assumes entry at signal CLOSE; cannot trade earlier using that closing information.',
             'Requires full 30 future sessions, including trades that hit target sooner; newer dates excluded.',
             'Dollar returns are not additive and simultaneous different-ticker exposure is unrestricted.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'status':'complete','raw_compound_signals':eligible,'target_summary':[{k:v for k,v in r.items() if k in ('target_pct','trades','target_hits','hit_rate_pct','blocked_same_stock_signals')} for r in summaries if r['entry_year']=='ALL'],'output':str(out)},indent=2),flush=True)

if __name__=='__main__':main()
