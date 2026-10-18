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
US_TICKER = re.compile(r'^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$')


def load_us_stocks():
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
        syms=sorted(set(str(s).upper() for s in syms if s and US_TICKER.fullmatch(str(s).upper())))
        # Use adjusted close for consistency with the original Egypt engine;
        # unadjusted close is mandatory for candle and dollar-value eligibility.
        cols=[date,adj,opening,high,low,volume,raw]
        for ticker in syms:
            records=session.execute(select(*cols).where(symbol==ticker).order_by(date)).all()
            if len(records)<60:continue
            df=pd.DataFrame(records,columns=['date','close','open','high','low','volume','actual_close'])
            df['date']=pd.to_datetime(df['date'],errors='coerce')
            for field in ('close','open','high','low','volume','actual_close'):
                df[field]=pd.to_numeric(df[field],errors='coerce')
            df=df.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
            if len(df)>=60:frames[ticker]=df
    return frames


def repeat_price_time_screener(**kwargs):
    loaded=load_us_stocks()
    # Original core algorithm unchanged. Exclusive lock protects the temporary loader
    # override in the *dedicated US process* (never the Rajih web service).
    with LOCK:
        orig=engine._load_stocks
        try:
            engine._load_stocks=lambda:loaded
            result=engine.repeat_price_time_screener(**kwargs)
        finally:
            engine._load_stocks=orig
    eligible={}
    for ticker,frame in loaded.items():
        by_date={}
        for i in range(1,len(frame)):
            today,prev=frame.iloc[i],frame.iloc[i-1]
            c=today['actual_close']; op=today['open']; v=today['volume']; pv=prev['volume']
            valid=(pd.notna(c) and pd.notna(op) and pd.notna(v) and pd.notna(pv)
                   and np.isfinite([c,op,v,pv]).all()
                   and c>op and v>pv and v>0 and c*v>=MIN_DOLLAR_VOLUME)
            by_date[pd.Timestamp(today['date']).date().isoformat()]=bool(valid)
        eligible[ticker]=by_date
    result=dict(result)
    result['rows']=[row for row in result.get('rows',[])
                    if eligible.get(str(row.get('symbol','')).upper(),{}).get(str(row.get('latest_repeat_date','')),False)]
    result['count']=len(result['rows'])
    result['universe_count']=len(loaded)
    return result
