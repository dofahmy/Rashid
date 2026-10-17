"""Saudi-market loader using the existing, unchanged Repeat computation.
Only the .SR stock universe is substituted inside this dedicated, separate service.
"""
import pandas as pd
import numpy as np
from sqlalchemy import MetaData, Table, select
from core import database
from monitor import seven_system as _engine


def _load_saudi_stocks():
    DB=database()
    frames={}
    with DB() as session:
        table=Table('market_candles_1d', MetaData(), autoload_with=session.get_bind())
        m={c.name.lower():c for c in table.columns}
        def col(*names):
            return next((m.get(n) for n in names if m.get(n) is not None), None)
        sym=col('symbol','ticker','sym')
        date=col('session_date','date')
        close=col('adj_c','adj_close','c','close')
        opn=col('o','open'); high=col('h','high'); low=col('l','low'); vol=col('v','volume')
        if sym is None or date is None or close is None:
            raise RuntimeError('Saudi source columns missing in market_candles_1d')
        q=select(sym).where(sym.ilike('%.SR')).distinct().order_by(sym)
        symbols=[x[0] for x in session.execute(q) if x[0]]
        selected=[date,close]; names=['date','close']
        for name,c in [('open',opn),('high',high),('low',low),('volume',vol)]:
            if c is not None: selected.append(c);names.append(name)
        for symbol in symbols:
            raw=session.execute(select(*selected).where(sym==symbol).order_by(date)).all()
            if not raw:continue
            df=pd.DataFrame(raw,columns=names)
            df['date']=pd.to_datetime(df['date'],errors='coerce')
            df['close']=pd.to_numeric(df['close'],errors='coerce')
            for name in ('open','high','low','volume'):
                df[name]=pd.to_numeric(df[name],errors='coerce') if name in df else np.nan
            df=df.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
            if len(df)>=60: frames[str(symbol).upper()]=df
    return frames

_engine._load_stocks=_load_saudi_stocks

def repeat_price_time_screener(**kwargs):
    loaded=_load_saudi_stocks()
    # avoid re-reading the database during same run; restore always
    original=_engine._load_stocks
    try:
        _engine._load_stocks=lambda: loaded
        result=_engine.repeat_price_time_screener(**kwargs)
    finally:
        _engine._load_stocks=original
    result['universe_count']=len(loaded)
    return result
