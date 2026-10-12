"""EGX RF whole-indicator-history scan. Retrospective research, NOT validated forecasting.
Run: python -m monitor.preburst_egypt_rf_market_scan --output-dir /tmp/egypt-rf-market
Requires original feature-study CSV, historical indicator SQL table, and Egyptian price bars.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from bisect import bisect_right
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump
from monitor.preburst_egypt_winner_range_replay import load_indicators, summarize
from monitor.preburst_egypt_winner_mean_scan import load_prices, calc_outcome
from monitor.preburst_egypt_feature_study import FIELDS

VERSION = 'EG-RF-WHOLE-MARKET-v1'
START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-12-31')

def build_cycles(history, prices, first, last):
    rows = []
    for symbol, group in history.groupby('symbol',sort=False):
        bars = prices.get(symbol, [])
        if not bars: continue
        day_list = [b[0] for b in bars]
        days = group['day'].tolist()
        accept = group['rf_pass'].tolist()
        probs = group['rf_probability'].tolist()
        next_allowed = pd.Timestamp.min
        for i, match in enumerate(accept):
            if not match or (i > 0 and accept[i-1]): continue
            day = days[i]
            if not first <= day <= last or day <= next_allowed: continue
            idx = bisect_right(day_list, day)-1
            if idx < 0 or day_list[idx] != day or bars[idx][2] <= 0: continue
            entry = bars[idx][2]
            result = calc_outcome(bars, day, entry)
            next_allowed = bars[min(idx+250, len(bars)-1)][0]
            rows.append(dict(symbol=str(symbol),signal_date=str(day.date()),entry_close=float(entry),
                rf_probability=float(probs[i]),**result))
    return pd.DataFrame(rows,columns=['symbol','signal_date','entry_close','rf_probability','outcome',
        'first_100pct_bar','bars_observed','complete_250','max_rise_pct','peak_bar','peak_date'])

def run(input_csv, output_dir, depth, leaf, estimators):
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    raw = pd.read_csv(input_csv,dtype={'symbol':str})
    absent = [c for c in ['symbol','signal_start_utc','outcome']+list(FIELDS) if c not in raw.columns]
    if absent: raise ValueError('Missing training columns: '+str(absent))
    resolved = raw[raw.outcome.isin(['winner','failure'])].copy()
    if len(resolved)!=417 or int(resolved.outcome.eq('winner').sum())!=191:
        raise ValueError(f'Unexpected training cohort: {len(resolved)} resolved, {resolved.outcome.eq("winner").sum()} winners; expected 417 and 191')
    imputer = SimpleImputer(strategy='median')
    train_x = imputer.fit_transform(resolved[FIELDS].apply(pd.to_numeric,errors='coerce'))
    y = resolved.outcome.eq('winner').astype(int).to_numpy()
    clf = RandomForestClassifier(n_estimators=estimators,max_depth=depth,min_samples_leaf=leaf,
        random_state=42,n_jobs=-1)
    clf.fit(train_x,y)
    train_prob=clf.predict_proba(train_x)[:,1]
    cutoff = float(np.min(train_prob[y==1]))
    # use full precision, not 0.530263 rounding of prior console output
    train_pass = train_prob >= cutoff-1e-12
    training = dict(total=len(y), winners=int(y.sum()), failures=int((y==0).sum()),
        threshold_full_precision=cutoff,retained_winners=int(np.sum(train_pass & (y==1))),
        selected_failures=int(np.sum(train_pass & (y==0))))
    history=load_indicators()
    # missing indicators are imputed with training-only medians
    whole_x=imputer.transform(history[FIELDS].apply(pd.to_numeric,errors='coerce'))
    history['rf_probability'] = clf.predict_proba(whole_x)[:,1]
    history['rf_pass'] = history.rf_probability >= cutoff-1e-12
    matched=history[['symbol','day','rf_probability','rf_pass']].copy()
    matched['signal_date']=matched.day.dt.strftime('%Y-%m-%d')
    original=resolved[['symbol','signal_start_utc','outcome']].copy()
    original['signal_date']=pd.to_datetime(original.signal_start_utc).dt.strftime('%Y-%m-%d')
    linked=original.merge(matched[['symbol','signal_date','rf_probability','rf_pass']],
        on=['symbol','signal_date'],how='left',validate='many_to_one')
    linked.to_csv(out/'original_417_entry_scores.csv',index=False,encoding='utf-8-sig')
    prices=load_prices()
    period=build_cycles(history,prices,START,END)
    period.to_csv(out/'cycles_2023_2026.csv',index=False,encoding='utf-8-sig')
    whole=build_cycles(history,prices,pd.Timestamp('1900-01-01'),pd.Timestamp('2100-01-01'))
    whole.to_csv(out/'cycles_all_history.csv',index=False,encoding='utf-8-sig')
    before=whole[whole.signal_date<'2023-01-01']
    before.to_csv(out/'cycles_before_2023.csv',index=False,encoding='utf-8-sig')
    # report strict-date matching separately, since entry events shift under suppression
    original_winners=linked[linked.outcome=='winner'].copy()
    score_recall=dict(winners=191,entry_day_scored=int(original_winners.rf_probability.notna().sum()),
        entry_day_passed=int(original_winners.rf_pass.fillna(False).sum()),
        exact_date_selected_in_scanned_2023_26=len(set(zip(period.symbol,period.signal_date)) &
                        set(zip(original_winners.symbol,original_winners.signal_date))))
    daily=history[['symbol','day','rf_probability','rf_pass']].copy()
    daily['day']=daily.day.dt.strftime('%Y-%m-%d')
    daily[daily.rf_pass].to_csv(out/'all_daily_matches.csv',index=False,encoding='utf-8-sig')
    importance=pd.DataFrame({'indicator':FIELDS,'importance':clf.feature_importances_})
    importance.sort_values('importance',ascending=False).to_csv(out/'feature_importances.csv',index=False,encoding='utf-8-sig')
    dump({'model':clf,'imputer':imputer,'features':list(FIELDS),'threshold':cutoff},out/'rf_model_and_imputer.joblib')
    summary=dict(version=VERSION,parameters={'n_estimators':estimators,'max_depth':depth,
        'min_samples_leaf':leaf,'random_state':42},training_fit=training,
        history_indicator_rows=len(history),history_stocks=int(history.symbol.nunique()),
        daily_matches=int(history.rf_pass.sum()),original_191_entry_recall=score_recall,
        same_period_2023_2026=summarize(period),historical_before_2023=summarize(before),
        all_available_history=summarize(whole),
        files=['cycles_2023_2026.csv','cycles_before_2023.csv','cycles_all_history.csv',
            'all_daily_matches.csv','original_417_entry_scores.csv','feature_importances.csv','rf_model_and_imputer.joblib'],
        cautions=['Training fit is in-sample memorization; do not interpret 191/191 as predictive accuracy.',
            '2019-22 is a retrospective prior-period stress test; 2023-26 includes training observations.',
            'Independent cycles use first matched day in a consecutive streak and suppress subsequent 250 trading sessions.',
            'Historical all-period cycles may differ from separately restarted 2023-26 cycles.',
            '250 subsequent daily High bars; price splits and corporate actions not adjusted or verified.',
            'NaN indicators imputed using medians of training cohort only.'])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input-csv',default='/tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv')
    p.add_argument('--output-dir',default='/tmp/egypt-rf-market')
    p.add_argument('--depth',type=int,default=12)
    p.add_argument('--leaf',type=int,default=2)
    p.add_argument('--estimators',type=int,default=400)
    a=p.parse_args()
    print(json.dumps(run(a.input_csv,a.output_dir,a.depth,a.leaf,a.estimators),ensure_ascii=False,indent=2),flush=True)
