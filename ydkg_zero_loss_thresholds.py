#!/usr/bin/env python3
"""Greedy zero-winning-case-loss indicator filters, YDKG previous-day reference.
Input only prior output (135 cases), not the entire market. Retains ALL original winning cases.
Research / in-sample, not a validated predictive rule.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam

FEATURES=['ret5','ret10','ret20','ret60','ret120','volatility20','rvol20',
 'dollar_volume20','rsi14','dist_ma20','dist_ma50','dist_ma120',
 'distance_high60_pct','range20_pct','up_days20_pct','daily_candle_pct']
# ATR14% is already the fixed first filter, so not re-tested.

def feature_at(d,i):
    """Evaluate each feature only when ITS required lookback is available."""
    a=d.iloc[:i+1]
    c=a.c.to_numpy(float); h=a.h.to_numpy(float); l=a.l.to_numpy(float)
    v=a.v.to_numpy(float); o=a.o.to_numpy(float)
    def div(x,y): return x/y if np.isfinite(y) and y != 0 else np.nan
    def valid(n):
        return len(c)>=n and np.all(np.isfinite(c[-n:])) and np.all(c[-n:]>0)
    def output(value):
        try:
            value=float(value)
            return value if np.isfinite(value) else np.nan
        except (ValueError,TypeError,ZeroDivisionError):return np.nan
    f={x:np.nan for x in FEATURES}
    for n in (5,10,20,60,120):
        if valid(n+1): f['ret'+str(n)]=output((c[-1]/c[-n-1]-1)*100)
    if valid(21):
        logret=np.diff(np.log(c[-21:]))
        f['volatility20']=output(np.std(logret,ddof=1)*np.sqrt(252)*100)
        f['up_days20_pct']=output(np.mean(np.diff(c[-21:])>0)*100)
    if len(v)>=21 and np.all(np.isfinite(v[-21:])):
        f['rvol20']=output(div(v[-1],np.nanmean(v[-21:-1])))
    if valid(20) and np.all(np.isfinite(v[-20:])):
        f['dollar_volume20']=output(np.nanmedian(c[-20:]*v[-20:]))
    if valid(15):
        dif=np.diff(c[-15:]);gain=np.maximum(dif,0).mean();loss=np.maximum(-dif,0).mean()
        f['rsi14']=output(100. if loss==0 and gain>0 else 50. if loss==0 else 100-100/(1+gain/loss))
    for n in (20,50,120):
        if valid(n): f['dist_ma'+str(n)]=output((c[-1]/c[-n:].mean()-1)*100)
    if valid(60) and np.all(np.isfinite(h[-60:])) and np.max(h[-60:])>0:
        f['distance_high60_pct']=output((c[-1]/h[-60:].max()-1)*100)
    if valid(20) and np.all(np.isfinite(h[-20:])) and np.all(np.isfinite(l[-20:])) and np.min(l[-20:])>0:
        f['range20_pct']=output((h[-20:].max()/l[-20:].min()-1)*100)
    if len(c) and np.isfinite(c[-1]) and np.isfinite(o[-1]) and o[-1]>0:
        f['daily_candle_pct']=output((c[-1]/o[-1]-1)*100)
    return f

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matches',default='/data/ydkg_atr_rounded/YDKG_ATR_previous_rounded_30sessions.csv')
    p.add_argument('--database-url',default=os.environ.get('DATABASE_URL'))
    p.add_argument('--out',default='/data/ydkg_zero_loss')
    p.add_argument('--min-decimals',type=int,default=0)
    p.add_argument('--max-decimals',type=int,default=8)
    p.add_argument('--reference',default='YDKG')
    p.add_argument('--reference-date',default='2025-08-22')
    a=p.parse_args()
    if not a.database_url: p.error('DATABASE_URL missing')
    if a.min_decimals<0 or a.max_decimals<a.min_decimals:p.error('Invalid decimal range')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(a.matches,low_memory=False)
    needed={'symbol','signal_date','prev_date','hit100_within30'}
    if not needed.issubset(base):p.error('Missing columns: '+str(needed-set(base.columns)))
    base=base.copy().reset_index(drop=True)
    base['symbol']=base.symbol.astype(str).str.upper().str.strip()
    base['signal_date']=base.signal_date.astype(str).str[:10]
    base['prev_date']=base.prev_date.astype(str).str[:10]
    base=base[(base.symbol!=a.reference)&(base.match_previous.astype(str).str.lower()=='true')].reset_index(drop=True)
    base['winner']=base.hit100_within30.astype(str).str.lower().isin(['true','1','1.0'])
    if base.empty:p.error('No previous ATR matches')
    symbols=sorted(set(base.symbol)|{a.reference})
    engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
    query=text('SELECT symbol,session_date AS date,o,h,l,c,v FROM market_candles_1d '
               'WHERE symbol IN :symbols ORDER BY symbol,session_date').bindparams(bindparam('symbols',expanding=True))
    lookup={}
    with engine.connect() as conn:
        for ix in range(0,len(symbols),100):
            block=pd.read_sql(query,conn,params={'symbols':symbols[ix:ix+100]})
            if block.empty:continue
            block['date']=block.date.astype(str).str[:10]
            for col in 'ohlcv':block[col]=pd.to_numeric(block[col],errors='coerce')
            block=block.dropna(subset=['date','o','h','l','c','v'])
            block=block[(block.o>0)&(block.c>0)&(block.l>0)&(block.v>=0)&
                        (block.h>=block[['o','c','l']].max(axis=1))&
                        (block.l<=block[['o','c','h']].min(axis=1))]
            for symbol,g in block.groupby('symbol',sort=False):
                g=g.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
                lookup[symbol]=g
    engine.dispose()
    ref=lookup.get(a.reference)
    if ref is None:raise RuntimeError('Reference bars missing')
    ri=np.where(ref.date.to_numpy()==a.reference_date)[0]
    if not len(ri) or ri[0]<121:raise RuntimeError('Reference day or previous 120 bars missing')
    ref_feats=feature_at(ref,int(ri[0]-1))
    if ref_feats is None:raise RuntimeError('Reference feature calculation failed')
    columns={f:np.full(len(base),np.nan) for f in FEATURES}
    missing=[]
    for i,r in base.iterrows():
        bars=lookup.get(r.symbol)
        if bars is None:missing.append((r.symbol,r.signal_date,'symbol missing'));continue
        bi=np.where(bars.date.to_numpy()==r.prev_date)[0]
        if not len(bi):missing.append((r.symbol,r.signal_date,'prev date missing'));continue
        feat=feature_at(bars,int(bi[0]))
        if feat is None:missing.append((r.symbol,r.signal_date,'feature calculation unavailable'));continue
        for f in FEATURES: columns[f][i]=float(feat[f])
    # No successful case may be removed. Missing winner data disallows making claims.
    active=np.ones(len(base),dtype=bool); winners=base.winner.to_numpy(bool)
    if any(not np.isfinite(columns[f][winners]).all() for f in FEATURES):
        missing_features=[f for f in FEATURES if not np.isfinite(columns[f][winners]).all()]
        print('NOTICE: some winners lack features:',missing_features,flush=True)
    feature_audit=[]
    for f in FEATURES:
        vv=columns[f]; rv=float(ref_feats[f]); ok=np.isfinite(vv)
        feature_audit.append({'indicator':f,'reference_value':rv,
             'winner_values_available':int((ok&winners).sum()),'winner_values_missing':int((~ok&winners).sum()),
             'loser_values_available':int((ok&~winners).sum()),
             'winner_min':float(np.min(vv[ok&winners])) if (ok&winners).any() else None,
             'winner_max':float(np.max(vv[ok&winners])) if (ok&winners).any() else None})
    pd.DataFrame(feature_audit).to_csv(out/'YDKG_feature_coverage_audit.csv',index=False)
    # Per-feature directional thresholds derived from ALL current winners, not YDKG value.
    # Missing measurements pass provisionally, never silently excluded.
    candidates=[];stages=[];remaining=set(FEATURES)
    original_w=int(winners.sum());original_stocks=int(base.loc[winners,'symbol'].nunique())
    audit=[]
    for stage in range(1,len(FEATURES)+1):
        possibilities=[]
        for f in sorted(remaining):
            vals=columns[f]; measured=np.isfinite(vals)
            observed_w=active&winners&measured
            missing_w=active&winners&~measured
            if observed_w.sum()==0:
                audit.append({'stage':stage,'indicator':f,'reason':'no winners with measured feature'})
                continue
            lo=float(vals[observed_w].min()); hi=float(vals[observed_w].max())
            # inclusive exact threshold; equal-valued winners always kept.
            for op,threshold in [('>=',lo),('<=',hi)]:
                passed=(vals>=threshold) if op=='>=' else (vals<=threshold)
                # Pass missing values as UNKNOWN, not a confirmed numerical match.
                eligible=passed|~measured
                kept=active&eligible
                win_kept=int((kept&winners).sum())
                removed=int((active&~winners&~eligible).sum())
                missing_losers=int((active&~winners&~measured).sum())
                trial={'stage':stage,'indicator':f,'operator':op,'threshold':threshold,
                       'cases_remaining':int(kept.sum()),'winners_retained':win_kept,
                       'winners_missing_value':int(missing_w.sum()),
                       'failed_cases_removed':removed,'failed_cases_missing_value':missing_losers,
                       'eligible':bool(removed>0 and win_kept==original_w)}
                candidates.append(trial)
                if removed>0 and win_kept==original_w:
                    possibilities.append((removed,int(missing_w.sum()),f,op,threshold,eligible))
        if not possibilities:break
        possibilities.sort(key=lambda t:(-t[0],t[1],t[2],t[3]))
        removed,missing_count,f,op,threshold,mask=possibilities[0]
        active &= mask;remaining.remove(f)
        row={'stage':stage,'indicator':f,'operator':op,'threshold':threshold,
             'cases_remaining':int(active.sum()),'winning_cases':int((active&winners).sum()),
             'winning_distinct_stocks':int(base.loc[active&winners,'symbol'].nunique()),
             'failed_cases_remaining':int((active&~winners).sum()),
             'failed_cases_removed_this_stage':removed,
             'winners_missing_indicator':missing_count,
             'precision_pct':round(100*int((active&winners).sum())/int(active.sum()),5)}
        stages.append(row)
        print('STAGE',stage,f,op,format(threshold,'.15g'),'CASES',row['cases_remaining'],
              'WINNERS',row['winning_cases'],'FAILED REMOVED',removed,flush=True)
        base.loc[active].to_csv(out/f'YDKG_stage_{stage:02d}_remaining.csv',index=False)
    pd.DataFrame(stages,columns=['stage','indicator','operator','threshold','cases_remaining',
         'winning_cases','winning_distinct_stocks','failed_cases_remaining',
         'failed_cases_removed_this_stage','winners_missing_indicator','precision_pct']).to_csv(out/'YDKG_zero_loss_stages.csv',index=False)
    pd.DataFrame(candidates,columns=['stage','indicator','operator','threshold','cases_remaining',
           'winners_retained','winners_missing_value','failed_cases_removed',
           'failed_cases_missing_value','eligible']).to_csv(out/'YDKG_all_indicator_trials.csv',index=False)
    base.loc[active].to_csv(out/'YDKG_final_remaining.csv',index=False)
    pd.DataFrame(missing,columns=['symbol','signal_date','reason']).to_csv(out/'missing_features.csv',index=False)
    report={'feature_coverage_audit':'YDKG_feature_coverage_audit.csv', 'reference':a.reference,
     'time_of_indicators':'previous trading session before each candidate signal',
     'baseline_cases':len(base),'baseline_unique_stocks':int(base.symbol.nunique()),
     'baseline_winning_cases':original_w,'baseline_unique_winners':original_stocks,
     'filters_added':len(stages),'no_zero_loss_filter_found':len(stages)==0,
     'final_cases':int(active.sum()),'final_winning_cases':int((active&winners).sum()),
     'final_unique_winners':int(base.loc[active&winners,'symbol'].nunique()),
     'final_failed_cases':int((active&~winners).sum()),
     'missing_policy':'Keep missing measurements as UNKNOWN; never interpret as passing numeric rule. Report missing winning features.',
     'caution':'Thresholds fitted to same historical winners, NOT an out-of-sample strategy. Review corporate actions, survivorship, and trade execution.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
