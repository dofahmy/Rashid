#!/usr/bin/env python3
"""EGX daily time-price angle level monitor (parameterized Gann-style angles).

This is a quantitative convention, not a unique/canonical 45-degree market price scale.
1x1 means 1 anchor ATR for each --unit-days trading sessions.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

FEATURES = ('symbol','session_date','h','l','c')
RATIOS = [("1x2",0.5),("1x1",1.0),("2x1",2.0)]

def bars_from_db():
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise RuntimeError('DATABASE_URL is missing')
    engine = create_engine(url, pool_pre_ping=True)
    query = text("""
        SELECT symbol, session_date, h, l, c
        FROM market_candles_1d
        WHERE UPPER(symbol) LIKE '%.CA'
          AND session_date IS NOT NULL
          AND h IS NOT NULL AND l IS NOT NULL AND c IS NOT NULL
        ORDER BY symbol, session_date
    """)
    with engine.connect() as con:
        df = pd.read_sql(query, con)
    if df.empty:
        raise RuntimeError('No EGX .CA rows')
    df['session_date'] = pd.to_datetime(df['session_date']).dt.normalize()
    for c in ('h','l','c'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['h','l','c'])
    df = df[(df.l > 0) & (df.h >= df.l) & (df.c > 0)]
    return df.drop_duplicates(['symbol','session_date'], keep='last')

def atr14(g):
    pc = g.c.shift(1)
    tr = pd.concat([(g.h-g.l),(g.h-pc).abs(),(g.l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(14,min_periods=14).mean()

def pivots(g, side, wing, lookback, min_atr):
    # Fully confirmed pivots: there must be wing trading days AFTER the pivot, before the latest day.
    # Never use current day's high/low when selecting pivots.
    n = len(g)
    end = n - wing - 1
    start = max(wing, n-lookback)
    out=[]
    for i in range(start,end):
        a = g.iloc[i]
        ai = float(a.atr)
        if not np.isfinite(ai) or ai<=0: continue
        segment=g.iloc[i-wing:i+wing+1]
        if side == 'LOW':
            val=float(a.l)
            if val != float(segment.l.min()):continue
            # meaningful reversal relative to surrounding prices
            if (float(segment.h.max()) - val) < min_atr*ai: continue
        else:
            val=float(a.h)
            if val != float(segment.h.max()):continue
            if (val - float(segment.l.min())) < min_atr*ai: continue
        out.append((i,val,ai))
    return out

def run(args):
    df=bars_from_db()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];latest=[];skipped=[]
    # No fabricated bars: trading days measured by actual saved sessions, not calendar days.
    for symbol,g in df.groupby('symbol',sort=True):
        g=g.sort_values('session_date').reset_index(drop=True)
        if len(g) < max(40,2*args.pivot_wing+16):
            skipped.append({'symbol':symbol,'reason':'insufficient_bars','bars':len(g)})
            continue
        g['atr']=atr14(g)
        last=g.iloc[-1];last_day=last.session_date
        latest.append({'symbol':symbol,'last_date':str(last_day.date()),'bars':len(g)})
        if not np.isfinite(last.atr) or last.atr<=0:
            skipped.append({'symbol':symbol,'reason':'missing_latest_atr'});continue
        prev_close=float(g.iloc[-2].c)
        lo=float(last.l);hi=float(last.h);close=float(last.c)
        for kind,sign in [('LOW',1),('HIGH',-1)]:
            anchors=pivots(g,kind,args.pivot_wing,args.pivot_lookback,args.pivot_min_atr)
            if not anchors:
                skipped.append({'symbol':symbol,'reason':f'no_confirmed_{kind}_pivot'});continue
            # Latest confirmed swing anchor of each kind, objectively chosen.
            idx,base,unit_atr=anchors[-1]
            age=(len(g)-1)-idx
            for label,ratio in RATIOS:
                level=base + sign*age*(unit_atr/args.unit_days)*ratio
                if level <= 0:continue
                touched=lo<=level<=hi
                near=abs(close-level)<=float(last.atr)*args.near_atr
                if prev_close < level and close>level:status='CROSS_UP_CLOSE'
                elif prev_close > level and close<level:status='CROSS_DOWN_CLOSE'
                elif touched and close>=level:status='TOUCHED_CLOSE_ABOVE'
                elif touched:status='TOUCHED_CLOSE_BELOW'
                elif near:status='NEAR_CLOSE'
                else:status='NO_TOUCH'
                records.append({
                    'symbol':symbol,'session_date':str(last_day.date()),
                    'last_close':round(close,6),'last_low':round(lo,6),'last_high':round(hi,6),
                    'anchor_type':kind,'anchor_date':str(g.iloc[idx].session_date.date()),
                    'anchor_price':round(base,6),'anchor_atr14':round(unit_atr,6),
                    'elapsed_trading_sessions':int(age),'angle':label,'angle_multiplier':ratio,
                    'price_units_per_trading_day':round(unit_atr/args.unit_days,8),
                    'squaring_level':round(level,6), 'distance_close_pct':round(100*(close/level-1),4),
                    'touched_by_last_bar':bool(touched),'near_close':bool(near),'status':status,
                    'last_atr14':round(float(last.atr),6)
                })
    result=pd.DataFrame(records)
    if result.empty: raise RuntimeError('No levels computed. Check pivot settings and data history.')
    result=result.sort_values(['symbol','anchor_type','angle'])
    hit=result[result.touched_by_last_bar].copy()
    interested=result[result.touched_by_last_bar | result.near_close].copy()
    result.to_csv(out/'all_stocks_levels.csv',index=False,encoding='utf-8-sig')
    hit.to_csv(out/'touched_levels.csv',index=False,encoding='utf-8-sig')
    interested.to_csv(out/'touched_or_near_levels.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(latest).to_csv(out/'last_dates_by_stock.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(skipped).to_csv(out/'skipped.csv',index=False,encoding='utf-8-sig')
    most_common=max(set(x['last_date'] for x in latest),key=lambda d:sum(z['last_date']==d for z in latest))
    summary={
      'version':'EGX-TIME-PRICE-DAILY-v1',
      'method':'Fixed numerical angle slopes from last CONFIRMED swing LOW and HIGH; 1x1 means anchor ATR14 divided by unit-days of observed trading sessions; 1x2 half-slope, 2x1 double-slope',
      'calibration':{'atr14_per_n_sessions':args.unit_days,'pivot_wing':args.pivot_wing,'pivot_lookback':args.pivot_lookback,'minimum_pivot_range_in_anchor_atr':args.pivot_min_atr,'near_tolerance_latest_atr':args.near_atr},
      'source_stocks':int(df.symbol.nunique()),'stocks_with_levels':int(result.symbol.nunique()),
      'latest_session_mode':most_common,'stocks_on_latest_session':sum(x['last_date']==most_common for x in latest),
      'touching_stocks_all_own_last_dates':int(hit.symbol.nunique()),'touched_levels_all_own_last_dates':len(hit),
      'touching_stocks_latest_session':int(hit[hit.session_date==most_common].symbol.nunique()),
      'touched_levels_latest_session':int((hit.session_date==most_common).sum()),
      'near_or_touch_stocks_latest_session':int(interested[interested.session_date==most_common].symbol.nunique()),
      'unconfirmed_or_short_history':len(skipped),
      'files':['all_stocks_levels.csv','touched_levels.csv','touched_or_near_levels.csv','last_dates_by_stock.csv','skipped.csv'],
      'limitations':['Angle labels are parameterized, not a unique physical 45 degree line. Zoom does not change calculated prices.',
                     'Only confirmed past pivots are used; this is a historical geometry study, not a predictive guarantee.',
                     'Some symbols may have stale last dates; use latest_session_mode filter.',
                     'Unadjusted corporate actions, illiquid candles, and data errors can distort signals.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\nLATEST SESSION TOUCHES (first 30)')
    columns=['symbol','session_date','anchor_type','anchor_date','angle','squaring_level','last_low','last_high','last_close','status']
    print(hit[hit.session_date==most_common][columns].head(30).to_string(index=False) if len(hit) else 'None')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='/tmp/egx-time-price-daily')
    ap.add_argument('--unit-days',type=float,default=20.,help='1x1 slope: one anchor ATR per N trading sessions')
    ap.add_argument('--pivot-wing',type=int,default=5)
    ap.add_argument('--pivot-lookback',type=int,default=180)
    ap.add_argument('--pivot-min-atr',type=float,default=1.5)
    ap.add_argument('--near-atr',type=float,default=0.15)
    a=ap.parse_args()
    if a.unit_days<=0 or a.pivot_wing<1 or a.pivot_lookback<2*a.pivot_wing+1:ap.error('invalid calibration')
    run(a)
