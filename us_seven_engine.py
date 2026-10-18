"""Public US repeat screener. Uses the original Repeat computation unchanged.
Only US ticker universe and *signal-day* eligibility are US-specific.
"""
import re
import threading
import pandas as pd
import numpy as np
from sqlalchemy import MetaData, Table, select
from core import database
from monitor import seven_system as engine

LOCK = threading.Lock()
MIN_DOLLAR_VOLUME = 5_000_000.0
US_TICKER = re.compile(r'^[A-Z]{1,5}(?:[.-][A-Z])?$')

# Exclude foreign-market suffixes; a single-letter suffix can represent a US share class.
def is_us_symbol(value):
    ticker = str(value or '').strip().upper()
    return bool(US_TICKER.fullmatch(ticker)) and not ticker.endswith(('.CA', '.SR'))


def load_us_stocks(progress=None):
    DB=database()
    frames={}
    with DB() as session:
        table=Table('market_candles_1d', MetaData(),autoload_with=session.get_bind())
        columns={c.name.lower():c for c in table.columns}
        def col(*names):
            return next((columns[n] for n in names if n in columns),None)
        symbol=col('symbol'); date=col('session_date'); adj=col('adj_c','c'); raw=col('c','close')
        opening=col('o','open'); high=col('h','high'); low=col('l','low'); volume=col('v','volume')
        if any(x is None for x in (symbol,date,adj,raw,opening,high,low,volume)):
            raise RuntimeError('Required OHLCV market_candles_1d columns missing')
        syms=[r[0] for r in session.execute(select(symbol).distinct()).all()]
        syms=sorted(set(str(s).upper() for s in syms if s and is_us_symbol(s)))
        if progress: progress('loading', 0, len(syms))
        # Use adjusted close for consistency with the original Egypt engine;
        # unadjusted close is mandatory for candle and dollar-value eligibility.
        cols=[date,adj,opening,high,low,volume,raw]
        for index,ticker in enumerate(syms, 1):
            records=session.execute(select(*cols).where(symbol==ticker).order_by(date)).all()
            if len(records)<60:
                if progress and (index % 20 == 0 or index==len(syms)): progress('loading', index, len(syms))
                continue
            df=pd.DataFrame(records,columns=['date','close','open','high','low','volume','actual_close'])
            df['date']=pd.to_datetime(df['date'],errors='coerce')
            for field in ('close','open','high','low','volume','actual_close'):
                df[field]=pd.to_numeric(df[field],errors='coerce')
            df=df.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
            if len(df)>=60:frames[ticker]=df
            if progress and (index % 20 == 0 or index==len(syms)): progress('loading', index, len(syms))
    return frames


def repeat_price_time_screener(progress=None, **kwargs):
    loaded=load_us_stocks(progress=progress)
    parity = kwargs.get('factor_parity', 'both')
    factors = [n for n in range(int(kwargs.get('factor_min',2)), int(kwargs.get('system_number',7)) + 1)
               if parity == 'both' or (n % 2 == 1 if parity == 'odd' else n % 2 == 0)]
    total = len(loaded) * len(factors)
    if progress: progress('scanning', 0, total)
    completed = [0]
    orig_single = engine._single
    def tracked_single(*args, **opts):
        out = orig_single(*args, **opts)
        completed[0] += 1
        if progress and (completed[0] % 10 == 0 or completed[0] == total):
            progress('scanning', completed[0], total)
        return out
    # Original core algorithm unchanged. Exclusive lock protects the temporary loader
    # override in the *dedicated US process* (never the Rajih web service).
    with LOCK:
        orig=engine._load_stocks
        original_single=engine._single
        try:
            engine._load_stocks=lambda:loaded
            engine._single=tracked_single
            result=engine.repeat_price_time_screener(**kwargs)
        finally:
            engine._load_stocks=orig
            engine._single=original_single
    if progress: progress('filtering', 0, len(loaded))
    eligible={}
    for index, (ticker,frame) in enumerate(loaded.items(), 1):
        by_date={}
        for i in range(1,len(frame)):
            today,prev=frame.iloc[i],frame.iloc[i-1]
            c=today['actual_close']; op=today['open']; v=today['volume']; pv=prev['volume']
            valid=(pd.notna(c) and pd.notna(op) and pd.notna(v) and pd.notna(pv)
                   and np.isfinite([c,op,v,pv]).all()
                   and c>op and v>pv and v>0 and c*v>=MIN_DOLLAR_VOLUME)
            by_date[pd.Timestamp(today['date']).date().isoformat()]=bool(valid)
        eligible[ticker]=by_date
        if progress and (index % 20 == 0 or index == len(loaded)): progress('filtering', index, len(loaded))
    result=dict(result)
    result['rows']=[row for row in result.get('rows',[])
                    if is_us_symbol(row.get('symbol')) and eligible.get(str(row.get('symbol','')).upper(),{}).get(str(row.get('latest_repeat_date','')),False)]
    result['count']=len(result['rows'])
    # Preserve future actual (unadjusted) closes so the viewer can calculate
    # any horizon without rerunning the expensive factor scan.
    for row in result['rows']:
        df = loaded.get(str(row['symbol']).upper())
        if df is None: continue
        dates = pd.to_datetime(df['date'])
        after = df.loc[dates > pd.Timestamp(row['latest_repeat_date'])]
        row['_forward_closes'] = [float(v) if pd.notna(v) and np.isfinite(v) else None
                                   for v in after['close'].iloc[:3000]]
        row['_forward_highs'] = [float(v) if pd.notna(v) and np.isfinite(v) else None
                                  for v in after['high'].iloc[:3000]]
    result['universe_count']=len(loaded)
    return result
