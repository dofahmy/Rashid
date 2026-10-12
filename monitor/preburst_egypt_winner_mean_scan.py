"""Exploratory independent Egypt 2023-26 mean-envelope scanner.
python -m monitor.preburst_egypt_winner_mean_scan --output-dir /tmp/egypt-mean-scan
Read-only. Scans the complete indicator timeline, not just old winning signals.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from bisect import bisect_right
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from core import database
from monitor.preburst_egypt_feature_study import FIELDS, TABLE

VERSION='EG-MEAN-SCAN-v1'
# Fixed winner cohort from the user-verified 191 Egyptian winning cycles, 2023-2026.
MEAN={'rsi14':37.570,'momentum5_pct':-4.916,'momentum20_pct':-9.350,'close_vs_sma20_pct':-6.860,'close_vs_sma50_pct':-5.493,'bbz20':-1.856,'atr14_pct':4.559,'return_20':-9.541,'position_20_pct':11.474,'volume5_vs_50':0.986,'repeat_count_1pct_60':3.780}
STD={'rsi14':4.153,'momentum5_pct':2.422,'momentum20_pct':3.999,'close_vs_sma20_pct':2.408,'close_vs_sma50_pct':5.893,'bbz20':0.572,'atr14_pct':1.768,'return_20':3.559,'position_20_pct':6.080,'volume5_vs_50':4.166,'repeat_count_1pct_60':4.177}
THRESHOLDS=[11,9,7]

def load_history():
    fields=', '.join('"'+c+'"' for c in FIELDS)
    stmt=text(f'''SELECT symbol, reference_utc, {fields} FROM {TABLE}
      WHERE market='egypt' AND reference_utc >= :start AND reference_utc < :end
      ORDER BY symbol,reference_utc''').execution_options(stream_results=True,yield_per=5000)
    with database()() as sess:
        data=[dict(row) for row in sess.execute(stmt,{'start':'2023-01-01','end':'2027-01-01'}).mappings()]
    if not data:raise RuntimeError('No Egypt indicators in 2023-2026; first build V5 history.')
    d=pd.DataFrame(data)
    d['day']=pd.to_datetime(d.reference_utc).dt.normalize()
    if d.duplicated(['symbol','day']).any():raise RuntimeError('Multiple indicator bars for same EGX symbol/day; cannot safely scan')
    for k in FIELDS:d[k]=pd.to_numeric(d[k],errors='coerce')
    return d

def load_prices():
    from monitor.gann_analysis import _daily_table
    prices=defaultdict(list)
    with database()() as sess:
        t=_daily_table(sess); fmt=sess.get_bind().dialect.identifier_preparer
        names={col.name.lower():col for col in t.c}
        def pick(*x):return next((names[v] for v in x if v in names),None)
        sy,dt,hi,cl=pick('symbol','ticker','sym'),pick('session_date','date','d'),pick('h','high'),pick('c','close')
        if any(x is None for x in [sy,dt,hi,cl]):raise RuntimeError('EGX table lacks symbol/date/high/close')
        s,d,h,c=[fmt.quote(x.name) for x in (sy,dt,hi,cl)]
        q=text(f'''SELECT {s} AS symbol, CAST({d} AS DATE) AS bar_day,
          MAX(CAST({h} AS DOUBLE PRECISION)) AS bar_high,MAX(CAST({c} AS DOUBLE PRECISION)) AS bar_close
          FROM {fmt.format_table(t)} WHERE {s} LIKE '%.CA'
          GROUP BY {s}, CAST({d} AS DATE) ORDER BY {s},bar_day''').execution_options(stream_results=True,yield_per=5000)
        for r in sess.execute(q).mappings():
            if r['bar_high'] is not None and r['bar_close'] is not None:
                prices[str(r['symbol'])].append((pd.Timestamp(r['bar_day']),float(r['bar_high']),float(r['bar_close'])))
    return prices

def calc_outcome(bars, date, entry):
    days=[x[0] for x in bars]
    at=bisect_right(days,date)
    future=bars[at:at+250]; done=len(future)==250
    hit=None; peak=None; peak_date=None; peak_bar=None
    for i,(dt,h,c) in enumerate(future,1):
        if h>0:
            if peak is None or h>peak:peak,peak_date,peak_bar=h,dt,i
            if hit is None and h>=2*entry:hit=i
    label=('متفوق' if hit<100 else 'ناجح') if hit is not None else ('لم يحقق' if done else 'قيد المتابعة')
    return {'outcome':label,'first_100pct_bar':hit,'bars_observed':len(future),'complete_250':done,
            'max_rise_pct':round(100*(peak/entry-1),4) if peak else None,'peak_bar':peak_bar,
            'peak_date':str(peak_date.date()) if peak_date is not None else None}

def run(output_dir, width=1.0, known_csv='/tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv'):
    if width<=0:raise ValueError('width must be positive')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    features=load_history(); prices=load_prices()
    score=pd.Series(0,index=features.index,dtype='int32')
    metadata=[]
    for k in FIELDS:
        lo,hi=MEAN[k]-width*STD[k],MEAN[k]+width*STD[k]
        score += features[k].between(lo,hi,inclusive='both').fillna(False).astype('int32')
        metadata.append({'indicator':k,'winner_mean':MEAN[k],'winner_std':STD[k],'lower':round(lo,5),'upper':round(hi,5)})
    features['mean_match_count']=score
    pd.DataFrame(metadata).to_csv(out/'mean_indicator_ranges.csv',index=False,encoding='utf-8-sig')
    candidates=features[score>=min(THRESHOLDS)].copy()
    all_results=[]
    known=set(); have_known=False
    known_file=Path(known_csv)
    if known_file.exists():
        old=pd.read_csv(known_file,dtype={'symbol':str})
        known={(str(s),str(d)[:10]) for s,d in zip(old.symbol,old.signal_start_utc)}
        have_known=True
    for level in THRESHOLDS:
        count=Counter(); rows=[]
        for sym, grp in features.groupby('symbol',sort=False):
            # Consecutive matching candles form ONE signal; first matching candle only.
            dates=grp['day'].tolist(); mask=(grp.mean_match_count>=level).tolist()
            signals=[(dates[i],int(grp.iloc[i].mean_match_count)) for i,on in enumerate(mask) if on and (i==0 or not mask[i-1])]
            bars=prices.get(sym,[])
            if not bars:continue
            price_days=[b[0] for b in bars]
            next_allowed=pd.Timestamp.min
            for dt,matched in signals:
                if dt<=next_allowed:continue
                ix=bisect_right(price_days,dt)-1
                if ix<0 or price_days[ix]!=dt or bars[ix][2]<=0:continue
                entry=bars[ix][2]
                result=calc_outcome(bars,dt,entry)
                future_end=bars[min(ix+250,len(bars)-1)][0]
                next_allowed=future_end if future_end>dt else pd.Timestamp.max
                key=(str(sym),str(dt.date()))
                row={'market':'egypt','match_required':level,'symbol':sym,'signal_date':str(dt.date()),
                     'entry_close':entry,'matched_indicators':matched,
                     'same_exact_start_as_old_191_winner': key in known and bool((old[(old.symbol==sym)&(old.signal_start_utc.astype(str).str[:10]==key[1])].outcome=='winner').any()) if have_known else None,
                     'matches_old_signal_start':key in known if have_known else None,
                     **result}
                rows.append(row);count[result['outcome']]+=1
        name=f'egypt_mean_{level}_of_11_cycles.csv'
        pd.DataFrame(rows).to_csv(out/name,index=False,encoding='utf-8-sig')
        eligible=sum(count[k] for k in ['متفوق','ناجح','لم يحقق'])
        win=count['متفوق']+count['ناجح']
        all_results.append({'required_matches':level,'cycles':len(rows),'superior':count['متفوق'],'successful':count['ناجح'],
           'complete_failures':count['لم يحقق'],'pending':count['قيد المتابعة'],
           'confirmed_100pct':win,'resolved_win_rate_pct':round(100*win/eligible,2) if eligible else None,
           'overlap_old_start_count':sum(bool(r['matches_old_signal_start']) for r in rows) if have_known else None,
           'new_start_vs_old_signal_ledger':sum(not r['matches_old_signal_start'] for r in rows) if have_known else None,
           'new_start_confirmed_100pct':sum(not r['matches_old_signal_start'] and r['outcome'] in ['متفوق','ناجح'] for r in rows) if have_known else None,
           'file':name})
    summary={'version':VERSION,'market':'egypt','period':'2023-2026','mean_envelope_std_multiplier':width,
      'source_winner_191':True,'source_total_indicator_rows':len(features),'stock_count_scanned':int(features.symbol.nunique()),
      'old_results_csv_found':have_known,'strategies':all_results,
      'cautions':['This is retrospective, winner-derived exploratory screening, NOT out-of-sample validation.',
      'New means a different symbol+entry date from the original ledger, not necessarily a new profitable stock.',
      '250 *subsequent trading sessions*; signal day HIGH excluded. Price series corporate actions not validated.',
      'Original results covered 2023-26 and original signals already met original filter: alternative rule can select others.',
      'Mean envelope is a modeling assumption (mean +/- standard deviation), not a literal equality.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='/tmp/preburst-egypt-mean-scan');p.add_argument('--width',type=float,default=1.0)
    p.add_argument('--known-csv',default='/tmp/preburst-egypt-features/egypt_cycles_with_entry_indicators_2023_2026.csv')
    args=p.parse_args();print(json.dumps(run(args.output_dir,args.width,args.known_csv),ensure_ascii=False,indent=2),flush=True)
