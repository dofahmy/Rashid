"""Classify Pre-Burst signals against +100% over 250 hourly bars (US) or trading sessions (Egypt).

Usage: python -m monitor.us_preburst_250day_outcomes --market us --output-dir /tmp/preburst-250d-us
No ledger or market data is changed. Incomplete cases remain explicitly visible.
"""
from __future__ import annotations
import argparse, csv, json
from bisect import bisect_right
from statistics import mean, median
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from sqlalchemy import text
from core import database
from monitor.us_preburst_ledger import LEDGER, fingerprint
from monitor.us_preburst_setup import FEATURES, DEFAULT_ACTIVE, parse_filters

VERSION = "V7.5-us-250-hourly-bars-egypt-250-sessions"

FIELDS=['cycle_number','source_signal_count_during_cycle','first_price_date','last_price_date','available_trading_dates','first_signal_to_first_price_days','market_price_source','market_data_end_date','market_data_start_date','market_history_sessions','market_250_day_coverage','market_note','market','symbol','signal_start_utc','entry_close','filter_end_utc','filter_status','days_observed','first_hit_day','first_hit_date','classification','max_rise_250d_pct','max_rise_day','max_rise_date','max_rise_high','full_250_days','last_price_date','data_flag']

def point(x, market):
    if market=='egypt': return day(x)
    if isinstance(x,datetime): return x.replace(tzinfo=None)
    if isinstance(x,date): return datetime.combine(x,datetime.min.time())
    return datetime.fromisoformat(str(x).replace('Z','+00:00')).replace(tzinfo=None)

def day(x):
    return x.date() if isinstance(x,datetime) else (x if isinstance(x,date) else datetime.fromisoformat(str(x)[:10]).date())

def get_signals(market, fid, members=None):
    with database()() as sess:
        stmt=text(f'''SELECT symbol,start_utc,entry_close,end_utc,status FROM {LEDGER}
          WHERE market=:m AND filter_id=:f ORDER BY symbol,start_utc''')
        return [dict(r) for r in sess.execute(stmt, {'m':market,'f':fid}).mappings() if members is None or r['symbol'] in members]

def daily_rows(market):
    """Yield (symbol, timestamp/day, high, close) sorted by symbol, period."""
    with database()() as sess:
        if market=='us':
            stmt=text('''SELECT symbol, bar_time AS bar_period, high AS bar_high, close AS bar_close
             FROM us_all_seven_hourly_bars WHERE high>0
             ORDER BY symbol,bar_time''').execution_options(stream_results=True,yield_per=4000)
        else:
            from monitor.gann_analysis import _daily_table
            t=_daily_table(sess); pre=sess.get_bind().dialect.identifier_preparer
            names={c.name.lower():c for c in t.c}
            def pick(*opts): return next((names[k] for k in opts if k in names),None)
            sy=pick('symbol','ticker','sym'); dt=pick('session_date','date','d'); hi=pick('h','high'); cl=pick('c','close')
            if any(v is None for v in (sy,dt,hi,cl)): raise ValueError('Missing Egypt symbol/date/high/close')
            s,d,h,c=(pre.quote(k.name) for k in (sy,dt,hi,cl)); tn=pre.format_table(t)
            stmt=text(f'''SELECT {s} AS symbol, CAST({d} AS DATE) AS bar_period, MAX(CAST({h} AS DOUBLE PRECISION)) AS bar_high, MAX(CAST({c} AS DOUBLE PRECISION)) AS bar_close
               FROM {tn} WHERE {s} LIKE '%.CA' GROUP BY {s},CAST({d} AS DATE)
               ORDER BY {s},trade_date''').execution_options(stream_results=True,yield_per=4000)
        for r in sess.execute(stmt).mappings(): yield str(r['symbol']),point(r['bar_period'],market),r['bar_high'],r['bar_close']

def analyze(symbol, signal, bars, market):
    entry=signal['entry_close']; start=point(signal['start_utc'],market)
    data=[r for r in bars if r[0]>start] # Exclude signal BAR for US, signal DAY for Egypt; no lookahead.
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
    # Group signals by symbol; never discard an independent signal after an expired cycle.
    by_symbol={}
    for signal in signals:
        by_symbol.setdefault(signal['symbol'],[]).append(signal)
    counts=Counter(); rows=0; matched=set(); current=None; bars=[]
    rises_by_status={'لم يحقق خلال 250 يوم':[], 'قيد المتابعة — بيانات أقل من 250 يوم':[]}
    buckets=Counter(); cycles_per_symbol=Counter(); market_dates=set()
    source_details={}
    csv_name='nonoverlapping_cycles_250_trading_days.csv'
    with (out/csv_name).open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader()
        def emit(symbol, barlist):
            nonlocal rows
            if symbol not in by_symbol:return
            matched.add(symbol)
            dates=[bar[0] for bar in barlist]
            source_details[symbol]={'first_price_date':str(dates[0]) if dates else None,
                 'last_price_date':str(dates[-1]) if dates else None,
                 'available_trading_dates':len(dates)}
            # A new cycle can only start AFTER 250 observed sessions following its opening signal.
            # If price history has ended, no new cycle can be established.
            allowed_at=date.min if market=='egypt' else datetime.min; cycle_idx=0; suppressed=0
            for sig in by_symbol[symbol]:
                opened=point(sig['start_utc'],market)
                if opened<=allowed_at:
                    suppressed+=1;continue
                if not dates or opened>dates[-1]:
                    # No prices past an entry: still report as incomplete; do not create
                    # further overlapping cycles after this one.
                    after=[]
                else:
                    after_index=bisect_right(dates,opened)
                    after=dates[after_index:after_index+250]
                cycle_idx+=1
                item=analyze(symbol,sig,barlist,market)
                item['cycle_number']=cycle_idx
                item['source_signal_count_during_cycle']=sum(
                    1 for other in by_symbol[symbol]
                    if opened<=point(other['start_utc'],market)<= (after[-1] if after else (date.max if market=='egypt' else datetime.max))
                )
                item['first_price_date']=str(dates[0]) if dates else ''
                item['last_price_date']=str(dates[-1]) if dates else ''
                item['available_trading_dates']=len(dates)
                item['first_signal_to_first_price_days']=(dates[0]-opened).total_seconds()/86400 if dates else ''
                item['market_price_source']='EGX market_candles_1d' if market=='egypt' else 'US native 1H bars'
                item['market_data_end_date']=str(dates[-1]) if dates else ''
                item['market_data_start_date']=str(dates[0]) if dates else ''
                item['market_history_sessions']=len(dates)
                item['market_250_day_coverage']='complete' if item['full_250_days'] else 'incomplete'
                item['market_note']='High derived from stored bars; adjusted prices/corporate actions not checked'
                writer.writerow(item)
                counts[item['classification']]+=1;rows+=1;cycles_per_symbol[symbol]+=1
                cls=item['classification']
                if cls in rises_by_status and isinstance(item['max_rise_250d_pct'],float):
                    val=item['max_rise_250d_pct'];rises_by_status[cls].append(val)
                    if val<0:band='below_0pct'
                    elif val<10:band='0_to_under_10pct'
                    elif val<25:band='10_to_under_25pct'
                    elif val<50:band='25_to_under_50pct'
                    elif val<75:band='50_to_under_75pct'
                    else:band='75_to_under_100pct'
                    buckets[f'{cls}__{band}']+=1
                if len(after)<250:
                    allowed_at=(date.max if market=='egypt' else datetime.max) # unresolved window: suppress all follow-up signals
                else:
                    allowed_at=after[-1]
        for symbol,dt,hi,cl in daily_rows(storage):
            market_dates.add(dt)
            if symbol!=current:
                if current is not None:emit(current,bars)
                current=symbol;bars=[]
            bars.append((dt,hi,cl))
        if current is not None:emit(current,bars)
        for symbol in sorted(by_symbol.keys()-matched):emit(symbol,[])
    evaluated=counts['متفوق']+counts['ناجح']+counts['لم يحقق خلال 250 يوم']
    rise_stats={}
    for category,vals in rises_by_status.items():
        rise_stats[category]={
            'cycles_with_later_price_data':len(vals),
            'average_peak_rise_pct':round(mean(vals),2) if vals else None,
            'median_peak_rise_pct':round(median(vals),2) if vals else None,
            'best_peak_rise_pct':round(max(vals),2) if vals else None,
            'worst_peak_rise_pct':round(min(vals),2) if vals else None}
    # Only fully mature cycles are eligible for a final success rate. Early winners
    # from incomplete 250-day cohorts must not inflate that denominator.
    mature=0; mature_wins=0; early_provisional=0
    with (out/csv_name).open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['full_250_days']=='True':
                mature+=1
                if r['classification'] in ('متفوق','ناجح'):mature_wins+=1
            elif r['classification'] in ('متفوق','ناجح'):
                early_provisional+=1
    hist_days=len(market_dates)
    summary={'version':VERSION,'market':market,'filter_id':fid,
             'original_signal_rows':len(signals),'distinct_stocks':len(by_symbol),
             'independent_250_session_cycles':rows,'follow_up_unit':'hourly_bars' if storage=='us' else 'trading_days',
             'suppressed_overlapping_signals':len(signals)-rows,
             'stocks_with_multiple_independent_cycles':sum(v>1 for v in cycles_per_symbol.values()),
             'all_signals':rows,'processed_signals':rows,
             'superior_before_day_100':counts['متفوق'],
             'successful_days_100_through_250':counts['ناجح'],
             'complete_no_100pct':counts['لم يحقق خلال 250 يوم'],
             'incomplete_unresolved':counts['قيد المتابعة — بيانات أقل من 250 يوم'],
             'invalid_entry':counts['سعر دخول غير صالح'],
             'decided_signals_provisional_and_mature':evaluated,
             'mature_250_session_cycles':mature,
             'mature_100pct_winners':mature_wins,
             'mature_success_rate_pct':round(mature_wins*100/mature,2) if mature else None,
             'early_winners_in_incomplete_cohorts':early_provisional,
             'market_price_history':{'source':'market_candles_1d (Egypt daily)' if storage=='egypt' else 'us_all_seven_hourly_bars (US 1H)',
                 'first_date':str(min(market_dates)) if market_dates else None,
                 'last_date':str(max(market_dates)) if market_dates else None,
                 'distinct_market_periods':hist_days,
                 'unit':'hourly_bars' if storage=='us' else 'trading_days',
                 'coverage_warning':('Fewer than 251 distinct market bars/sessions; no complete 250-period follow-up can be expected for recent signals.' if hist_days<251 else 'Historical coverage varies by symbol and signal date; inspect per-row follow-up.'),
                 'price_source_warning':'US uses native hourly Highs; verify bar continuity and provider adjustments.' if storage=='us' else 'EGX price series may include unadjusted corporate actions.'},
             'peak_rise_for_not_reached_100pct':rise_stats,
             'peak_rise_buckets':dict(sorted(buckets.items())),
             'csv_file':csv_name,
             'note':'US=250 subsequent stored 1H candles, Egypt=250 subsequent daily trading sessions. Nonoverlapping cycles per stock. Incomplete windows are not failures. Mature-only success rate. First hit from subsequent bar High excluding signal bar. Corporate actions and missing bars can distort results.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',required=True,choices=['egypt','us','sp500']);p.add_argument('--output-dir',required=True);p.add_argument('--query-string',default='')
    a=p.parse_args();print(f'[PreBurst 250d] {VERSION} | file={__file__}',flush=True);print(json.dumps(run(a.market,a.output_dir,a.query_string),ensure_ascii=False,indent=2),flush=True)
