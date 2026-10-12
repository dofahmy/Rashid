"""Egypt winner extrema replay: historical independent scan + exact 191 entry recall.
Run: python -m monitor.preburst_egypt_winner_range_replay --output-dir /tmp/egypt-winner-range
Read-only; no app.py or DB changes. Requires existing V5 indicator history and daily prices.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from bisect import bisect_right
from collections import Counter
import pandas as pd
from sqlalchemy import text
from core import database
from monitor.preburst_egypt_feature_study import FIELDS,TABLE
from monitor.preburst_egypt_winner_mean_scan import load_prices,calc_outcome

VERSION='EG-WINNER-EXTREMA-REPLAY-v1'
# Displayed min/max rounded to 3 decimals. Add 0.001 to both sides to
# avoid excluding a winner solely due to rounding of the console printout.
EXTREMA={
'rsi14':(27.024,42.997),'momentum5_pct':(-8.901,0.000),
'momentum20_pct':(-21.046,-4.015),'close_vs_sma20_pct':(-12.582,-3.056),
'close_vs_sma50_pct':(-31.193,6.672),'bbz20':(-4.359,-0.659),
'atr14_pct':(1.065,11.543),'return_20':(-19.939,-4.068),
'position_20_pct':(2.133,23.620),'volume5_vs_50':(0.115,57.420),
'repeat_count_1pct_60':(0.000,24.000)
}

def load_indicators():
    fields=', '.join('"'+c+'"' for c in FIELDS)
    stmt=text(f'''SELECT symbol,reference_utc,{fields} FROM {TABLE}
        WHERE market='egypt' ORDER BY symbol,reference_utc''').execution_options(stream_results=True,yield_per=5000)
    with database()() as sess:
        rows=[dict(r) for r in sess.execute(stmt).mappings()]
    if not rows:raise RuntimeError('No Egypt indicator history in database')
    d=pd.DataFrame(rows)
    d['day']=pd.to_datetime(d.reference_utc,errors='coerce').dt.normalize()
    d=d[d.day.notna()].copy()
    if d.duplicated(['symbol','day']).any():
        raise RuntimeError('Duplicate indicator rows for symbol and day; fix before scanning.')
    for c in FIELDS:d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def make_ranges(winners_path, eps):
    src='provided_191_displayed_extrema_3_decimal_places'
    if winners_path and Path(winners_path).is_file():
        w=pd.read_csv(winners_path)
        if 'outcome' in w.columns:w=w[w.outcome.eq('winner')]
        elif 'classification' in w.columns:w=w[w.classification.isin(['متفوق','ناجح'])]
        if len(w)!=191:raise ValueError(f'Expected exactly 191 winner cycles, found {len(w)}')
        bounds={c:(float(pd.to_numeric(w[c],errors='coerce').min()),float(pd.to_numeric(w[c],errors='coerce').max())) for c in FIELDS}
        if any(pd.isna(lo) or pd.isna(hi) for lo,hi in bounds.values()):raise ValueError('Winner indicator has missing/invalid values')
        src='exact_extrema_computed_from_191_winners_csv'
        return bounds,w,src
    return {k:(v[0]-eps,v[1]+eps) for k,v in EXTREMA.items()},None,src

def scan(d,prices,period_start,period_end):
    cycles=[]
    for sym,g in d.groupby('symbol',sort=False):
        bars=prices.get(sym,[])
        if not bars:continue
        dates=[b[0] for b in bars]
        matched=g['all_11_in_range'].tolist()
        ds=g['day'].tolist()
        next_allowed=pd.Timestamp.min
        for i,yes in enumerate(matched):
            if not yes or (i>0 and matched[i-1]):continue
            day=ds[i]
            if day<period_start or day>period_end:continue
            if day<=next_allowed:continue
            at=bisect_right(dates,day)-1
            if at<0 or dates[at]!=day or bars[at][2]<=0:continue
            entry=bars[at][2]
            result=calc_outcome(bars,day,entry)
            next_allowed=bars[min(at+250,len(bars)-1)][0]
            cycles.append({'symbol':sym,'signal_date':str(day.date()),'entry_close':entry,**result})
    return pd.DataFrame(cycles,columns=['symbol','signal_date','entry_close','outcome','first_100pct_bar','bars_observed','complete_250','max_rise_pct','peak_bar','peak_date'])

def summarize(d):
    n=Counter(d.outcome) if len(d) else Counter()
    winners=n['متفوق']+n['ناجح']; failures=n['لم يحقق'];pending=n['قيد المتابعة']
    return {'cycles':len(d),'superior':n['متفوق'],'successful':n['ناجح'],'confirmed_100pct':winners,'complete_failures':failures,'pending':pending,
            'resolved_win_rate_pct':round(100*winners/(winners+failures),2) if winners+failures else None,
            'distinct_stocks':int(d.symbol.nunique()) if len(d) else 0}

def run(output_dir,winners_csv,original_csv,epsilon):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    bounds,winners,source=make_ranges(winners_csv,epsilon)
    d=load_indicators()
    for c,(lo,hi) in bounds.items():
        d[c+'_pass']=d[c].between(lo,hi,inclusive='both').fillna(False)
    pass_cols=[c+'_pass' for c in FIELDS]
    d['matched_count']=d[pass_cols].sum(axis=1)
    d['all_11_in_range']=d.matched_count.eq(len(FIELDS))
    (out/'winner_extrema_ranges.csv').write_text(pd.DataFrame([{'indicator':k,'lower':v[0],'upper':v[1]} for k,v in bounds.items()]).to_csv(index=False),encoding='utf-8-sig')
    if winners is None and Path(original_csv).is_file():
        old=pd.read_csv(original_csv,dtype={'symbol':str})
        old['signal_date']=pd.to_datetime(old.signal_start_utc).dt.strftime('%Y-%m-%d')
        if 'classification' in old:winners=old[(old.classification.isin(['متفوق','ناجح'])) & (old.signal_date>='2023-01-01') & (old.signal_date<='2026-12-31')].copy()
        elif 'outcome' in old:winners=old[(old.outcome.eq('winner')) & (old.signal_date>='2023-01-01') & (old.signal_date<='2026-12-31')].copy()
    recall=None
    if winners is not None:
        w=winners.copy()
        w['signal_date']=pd.to_datetime(w['signal_start_utc']).dt.strftime('%Y-%m-%d') if 'signal_start_utc' in w else pd.to_datetime(w['signal_date']).dt.strftime('%Y-%m-%d')
        lookup=d[['symbol','day','matched_count','all_11_in_range']].copy();lookup['signal_date']=lookup.day.dt.strftime('%Y-%m-%d')
        w=w.merge(lookup[['symbol','signal_date','matched_count','all_11_in_range']],on=['symbol','signal_date'],how='left',validate='many_to_one')
        w['all_11_in_range']=w.all_11_in_range.fillna(False).astype(bool)
        w[['symbol','signal_date','matched_count','all_11_in_range']].to_csv(out/'original_191_entry_recall.csv',index=False,encoding='utf-8-sig')
        recall={'original_winner_cycles':len(w),'original_winners_matching_all_11_at_entry':int(w.all_11_in_range.sum()),
                'original_winners_not_matching':int((~w.all_11_in_range).sum()),'missing_entry_indicator_rows':int(w.matched_count.isna().sum())}
    prices=load_prices()
    start=pd.Timestamp('2023-01-01');end=pd.Timestamp('2026-12-31')
    part=scan(d,prices,start,end)
    part.to_csv(out/'egypt_2023_2026_strict_11_ranges_cycles.csv',index=False,encoding='utf-8-sig')
    all_=scan(d,prices,pd.Timestamp('1900-01-01'),pd.Timestamp('2100-01-01'))
    all_.to_csv(out/'egypt_all_history_strict_11_ranges_cycles.csv',index=False,encoding='utf-8-sig')
    pre=all_[all_.signal_date<'2023-01-01'].copy()
    pre.to_csv(out/'egypt_before_2023_strict_11_ranges_cycles.csv',index=False,encoding='utf-8-sig')
    # Cycles crossing Jan 2023 boundary differ between independently scoped scans.
    selected_keys=set(zip(part.symbol,part.signal_date))
    cycle_recall=None
    if winners is not None:
        keys=set(zip(w.symbol.astype(str),w.signal_date))
        cycle_recall={'original_191_exact_signal_dates_selected_as_independent_new_cycles':len(keys&selected_keys),
          'note':'First matching day of each matching streak and 250-session suppression can change cycle starts, so raw entry recall and cycle recall differ.'}
    summary={'version':VERSION,'range_source':source,'range_count':len(bounds),
      'history_indicator_rows':len(d),'history_stocks':int(d.symbol.nunique()),'history_first_day':str(d.day.min().date()),'history_last_day':str(d.day.max().date()),
      'original_191_entry_recall':recall,'original_cycle_start_recall_2023_2026':cycle_recall,
      'same_period_2023_2026':summarize(part),'historical_before_2023':summarize(pre),'all_available_history':summarize(all_),
      'cautions':['Min/max fitted to known winners: 191/191 raw coverage is in-sample by construction, not predictive evidence.',
        'Full-history scan restarts 250-bar cycles before 2023; 2023+ results should be compared using same period-only scan.',
        'Future high >= 2x entry close within next 250 daily bars; not executable return; corporate actions unadjusted.',
        'Original 191 counts successful signal CYCLES, not distinct stocks. More signals can be old rallies with different entry dates.',
        'Winner bounds learned from 2023-26; before-2023 is prior-history stress test, not a prospective future holdout.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',default='/tmp/egypt-winner-range')
    p.add_argument('--winners-csv',default='/tmp/preburst-egypt-features/egypt_191_winners_all_indicators.csv')
    p.add_argument('--original-csv',default='/tmp/preburst-250d-egypt/nonoverlapping_cycles_250_trading_days.csv')
    p.add_argument('--epsilon',type=float,default=0.001)
    a=p.parse_args()
    print(json.dumps(run(a.output_dir,a.winners_csv,a.original_csv,a.epsilon),ensure_ascii=False,indent=2),flush=True)
