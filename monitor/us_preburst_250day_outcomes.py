"""Classify ALL stored Pre-Burst ledger signals against +100% over 250 trading days.

Usage: python -m monitor.us_preburst_250day_outcomes --market us --output-dir /tmp/preburst-250d-us
No ledger or market data is changed. Incomplete cases remain explicitly visible.
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from sqlalchemy import text
from core import database
from monitor.us_preburst_ledger import LEDGER, fingerprint
from monitor.us_preburst_setup import FEATURES, DEFAULT_ACTIVE, parse_filters

VERSION = "V7.3-first-signal-per-stock"

FIELDS=['market','symbol','signal_start_utc','entry_close','filter_end_utc','filter_status','days_observed','first_hit_day','first_hit_date','classification','max_rise_250d_pct','max_rise_day','max_rise_date','max_rise_high','full_250_days','last_price_date','data_flag']

def day(x):
    return x.date() if isinstance(x,datetime) else (x if isinstance(x,date) else datetime.fromisoformat(str(x)[:10]).date())

def get_signals(market, fid, members=None):
    with database()() as sess:
        stmt=text(f'''SELECT symbol,start_utc,entry_close,end_utc,status FROM {LEDGER}
          WHERE market=:m AND filter_id=:f ORDER BY symbol,start_utc''')
        return [dict(r) for r in sess.execute(stmt, {'m':market,'f':fid}).mappings() if members is None or r['symbol'] in members]

def daily_rows(market):
    """Yield sorted (symbol, session_date, high, close) from original price source."""
    with database()() as sess:
        if market=='us':
            stmt=text('''SELECT symbol,CAST(bar_time AS DATE) AS trade_date, MAX(high) AS day_high, MAX(close) AS day_close
             FROM us_all_seven_hourly_bars WHERE high>0 GROUP BY symbol,CAST(bar_time AS DATE)
             ORDER BY symbol,trade_date''').execution_options(stream_results=True,yield_per=4000)
        else:
            from monitor.gann_analysis import _daily_table
            t=_daily_table(sess); pre=sess.get_bind().dialect.identifier_preparer
            names={c.name.lower():c for c in t.c}
            def pick(*opts): return next((names[k] for k in opts if k in names),None)
            sy=pick('symbol','ticker','sym'); dt=pick('session_date','date','d'); hi=pick('h','high'); cl=pick('c','close')
            if any(v is None for v in (sy,dt,hi,cl)): raise ValueError('Missing Egypt symbol/date/high/close')
            s,d,h,c=(pre.quote(k.name) for k in (sy,dt,hi,cl)); tn=pre.format_table(t)
            stmt=text(f'''SELECT {s} AS symbol, CAST({d} AS DATE) AS trade_date, MAX(CAST({h} AS DOUBLE PRECISION)) AS day_high, MAX(CAST({c} AS DOUBLE PRECISION)) AS day_close
               FROM {tn} WHERE {s} LIKE '%.CA' GROUP BY {s},CAST({d} AS DATE)
               ORDER BY {s},trade_date''').execution_options(stream_results=True,yield_per=4000)
        for r in sess.execute(stmt).mappings(): yield str(r['symbol']),day(r['trade_date']),r['day_high'],r['day_close']

def analyze(symbol, signal, bars, market):
    entry=signal['entry_close']; start=day(signal['start_utc'])
    data=[r for r in bars if r[0]>start] # Conservatively exclude signal date; no same-day lookahead.
    window=data[:250]; hit=None; peak=None; peak_day=None; peak_date=None
    if entry is not None and float(entry)>0:
        goal=float(entry)*2
        for i,(dt,hi,cl) in enumerate(window,1):
            if hi is None: continue
            try: price=float(hi)
            except (ValueError,TypeError): continue
            if price>0:
                if peak is None or price>peak:
                    peak=price; peak_day=i; peak_date=dt
                if hit is None and price>=goal: hit=(i,dt)
    complete=len(window)>=250
    if entry is None or float(entry)<=0: label='سعر دخول غير صالح'
    elif hit and hit[0]<100: label='متفوق'
    elif hit and hit[0]<=250: label='ناجح'
    elif complete: label='لم يحقق خلال 250 يوم'
    else: label='قيد المتابعة — بيانات أقل من 250 يوم'
    return {'market':market,'symbol':symbol,'signal_start_utc':str(signal['start_utc']),
            'entry_close':entry,'filter_end_utc':str(signal['end_utc'] or ''),'filter_status':signal['status'],
            'days_observed':len(window),'first_hit_day':hit[0] if hit else '',
            'first_hit_date':str(hit[1]) if hit else '', 'classification':label,
            'max_rise_250d_pct':round((peak/float(entry)-1)*100,4) if peak is not None and entry and float(entry)>0 else '',
            'max_rise_day':peak_day if peak_day is not None else '',
            'max_rise_date':str(peak_date) if peak_date is not None else '',
            'max_rise_high':round(peak,6) if peak is not None else '',
            'full_250_days':complete,'last_price_date':str(bars[-1][0]) if bars else '',
            'data_flag':'لا توجد أسعار تالية' if not data else ('الفترة مكتملة' if complete else 'الفترة غير مكتملة')}

def run(market, out_dir, query_string=''):
    if query_string:
        from urllib.parse import parse_qs
        a={k:v[-1] for k,v in parse_qs(query_string.split('?',1)[-1],keep_blank_values=True).items()}
        selected,settings=parse_filters(a)
        target=float(a.get('target_pct','100'))
    else:
        selected=list(DEFAULT_ACTIVE);settings={k:{'min':float(v[1]),'max':float(v[2])} for k,v in FEATURES.items()};target=100.
    if target!=100: raise ValueError('This +100% classification requires target_pct=100; do not mix filter fingerprints.')
    storage='egypt' if market=='egypt' else 'us'
    members=None
    if market=='sp500':
        from monitor.sp500_seven_system import list_symbols
        members=set(list_symbols())
    fid=fingerprint(selected,settings,target)
    signals=get_signals(storage,fid,members)
    if not signals: raise RuntimeError(f'No ledger rows for market={storage}, filter_id={fid}; build ledger with same Setup settings first.')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    # Signals are sorted by symbol/start_utc. Keep ONLY the earliest signal per stock.
    by_symbol={}
    for signal in signals:
        by_symbol.setdefault(signal['symbol'],signal)
    counts=Counter(); rows=0; matched=set();current=None;bars=[]
    rises_by_status={'لم يحقق خلال 250 يوم':[], 'قيد المتابعة — بيانات أقل من 250 يوم':[]}
    buckets=Counter()
    with (out/'first_signal_per_stock_250_trading_days.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=FIELDS);writer.writeheader()
        def emit(symbol, barlist):
            nonlocal rows
            if symbol not in by_symbol: return
            matched.add(symbol)
            item=analyze(symbol,by_symbol[symbol],barlist,market)
            writer.writerow(item)
            counts[item['classification']]+=1;rows+=1
            cls=item['classification']
            if cls in rises_by_status and isinstance(item['max_rise_250d_pct'],float):
                val=item['max_rise_250d_pct']
                rises_by_status[cls].append(val)
                if val<0: band='below_0pct'
                elif val<10: band='0_to_under_10pct'
                elif val<25: band='10_to_under_25pct'
                elif val<50: band='25_to_under_50pct'
                elif val<75: band='50_to_under_75pct'
                else: band='75_to_under_100pct'
                buckets[f'{cls}__{band}']+=1
        for symbol,dt,hi,cl in daily_rows(storage):
            if symbol!=current:
                if current is not None: emit(current,bars)
                current=symbol;bars=[]
            bars.append((dt,hi,cl))
        if current is not None:emit(current,bars)
        for symbol in sorted(by_symbol.keys()-matched):emit(symbol,[])
    evaluated=counts['متفوق']+counts['ناجح']+counts['لم يحقق خلال 250 يوم']
    from statistics import mean, median
    rise_stats={}
    for category,vals in rises_by_status.items():
        rise_stats[category]={
            'stocks_with_later_price_data':len(vals),
            'average_peak_rise_pct':round(mean(vals),2) if vals else None,
            'median_peak_rise_pct':round(median(vals),2) if vals else None,
            'best_peak_rise_pct':round(max(vals),2) if vals else None,
            'worst_peak_rise_pct':round(min(vals),2) if vals else None,
        }
    summary={'market':market,'filter_id':fid,'original_signal_rows':len(signals),
             'distinct_stocks_first_signal_only':len(by_symbol),'ignored_repeat_signals':len(signals)-len(by_symbol),
             'all_signals':len(by_symbol),'processed_signals':rows,
             'superior_before_day_100':counts['متفوق'],'successful_days_100_through_250':counts['ناجح'],
             'complete_no_100pct':counts['لم يحقق خلال 250 يوم'],
             'incomplete_unresolved':counts['قيد المتابعة — بيانات أقل من 250 يوم'],
             'invalid_entry':counts['سعر دخول غير صالح'],
             'decided_signals':evaluated,
             'success_rate_among_decided_pct':round(100*(counts['متفوق']+counts['ناجح'])/evaluated,2) if evaluated else None,
             'peak_rise_for_not_reached_100pct':rise_stats,
             'peak_rise_buckets':dict(sorted(buckets.items())),
             'csv_file':'first_signal_per_stock_250_trading_days.csv',
             'note':'ONE earliest signal per symbol, all later occurrences ignored permanently. Peak rise uses subsequent daily HIGH within 250 observed trading sessions (even when filter exits). A negative peak rise means every subsequent daily HIGH was below entry. Incomplete future data is not a failure. Excludes signal-day HIGH to avoid lookahead. Corporate actions, missing sessions and adjusted/unadjusted prices can distort results.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',required=True,choices=['egypt','us','sp500']);p.add_argument('--output-dir',required=True);p.add_argument('--query-string',default='')
    a=p.parse_args(); print(f'[PreBurst 250d] {VERSION} | file={__file__}', flush=True);print(json.dumps(run(a.market,a.output_dir,a.query_string),ensure_ascii=False,indent=2),flush=True)
