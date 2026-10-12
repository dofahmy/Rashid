"""Independent Yahoo daily quote updater for EGX Time-Price 3D.

Writes only market_candles_1d; no Rajih/Brown code or database schema changes.
Dates are ISO-8601 strings because this project's session_date is VARCHAR.
"""
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
from sqlalchemy import MetaData, Table, create_engine, select, and_, insert, update


def _frame(raw, sym, multiple):
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance may return (ticker, field) OR (field, ticker)
            if sym in raw.columns.get_level_values(0):
                result = raw[sym].copy()
            elif sym in raw.columns.get_level_values(1):
                result = raw.xs(sym, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            result = raw.copy()
        result.columns = [str(x).strip().lower().replace(' ', '_') for x in result.columns]
        return result
    except (KeyError, ValueError):
        return pd.DataFrame()


def update_egx_prices(days_back=21):
    url = os.environ['DATABASE_URL']
    engine = create_engine(url, pool_pre_ping=True)
    table = Table('market_candles_1d', MetaData(), autoload_with=engine)
    cols = set(table.columns.keys())
    required = {'symbol','session_date','o','h','l','c'}
    if not required.issubset(cols):
        raise RuntimeError('market_candles_1d is missing required daily-bar columns')
    with engine.connect() as conn:
        symbols = [str(x).upper() for x in conn.execute(select(table.c.symbol).distinct()).scalars()
                   if str(x).upper().endswith('.CA')]
    symbols.sort()
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime('%Y-%m-%d')
    start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime('%Y-%m-%d')
    report = {'symbols_total':len(symbols),'symbols_with_source_bars':0,'rows_inserted':0,
              'rows_updated':0,'latest_source_date':None,'errors':[], 'source':'Yahoo Finance',
              'start':start,'end_exclusive':end}
    today = datetime.now(timezone.utc)
    for offset in range(0,len(symbols),25):
        chunk = symbols[offset:offset+25]
        try:
            raw = yf.download(' '.join(chunk),start=start,end=end,interval='1d',
                              group_by='ticker',auto_adjust=False,threads=True,progress=False)
        except Exception as exc:
            report['errors'].append(f'{chunk[0]}: download: {str(exc)[:180]}')
            continue
        with engine.begin() as conn:
            existing_rows=conn.execute(select(table.c.symbol,table.c.session_date,table.c.o,
                         table.c.h,table.c.l,table.c.c).where(table.c.symbol.in_(chunk),
                         table.c.session_date>=start)).fetchall()
            existing={(str(r.symbol).upper(),str(r.session_date)):r for r in existing_rows}
            for sym in chunk:
                frame=_frame(raw,sym,len(chunk)>1)
                if frame.empty:continue
                required_price={'open','high','low','close'}
                if not required_price.issubset(frame.columns):
                    report['errors'].append(f'{sym}: missing OHLC columns');continue
                seen_valid=False
                for dt,row in frame.iterrows():
                    try:
                        stamp=pd.Timestamp(dt)
                        date=stamp.strftime('%Y-%m-%d')
                        o,h,l,c=(float(row[x]) for x in ('open','high','low','close'))
                        if not (0<l<=min(o,c)<=max(o,c)<=h):continue
                        vals={'symbol':sym,'session_date':date,'o':o,'h':h,'l':l,'c':c}
                        if 'v' in cols:vals['v']=max(0,int(float(row['volume']))) if 'volume' in row and pd.notna(row['volume']) else 0
                        if 'adj_c' in cols:vals['adj_c']=float(row['adj_close']) if 'adj_close' in row and pd.notna(row['adj_close']) else c
                        if 'source' in cols:vals['source']='yfinance'
                        if 'retrieved_at' in cols:vals['retrieved_at']=today
                        old=existing.get((sym,date))
                        if old is None:
                            conn.execute(insert(table).values(**vals))
                            report['rows_inserted']+=1
                        elif any(abs(float(getattr(old,k))-vals[k])>1e-5 for k in ('o','h','l','c')):
                            conn.execute(update(table).where(and_(table.c.symbol==sym,
                                table.c.session_date==date)).values(**{k:v for k,v in vals.items() if k not in ('symbol','session_date')}))
                            report['rows_updated']+=1
                        report['latest_source_date']=max(report['latest_source_date'] or date,date)
                        seen_valid=True
                    except Exception as exc:
                        report['errors'].append(f'{sym}: {str(exc)[:160]}')
                if seen_valid:report['symbols_with_source_bars']+=1
        print(f"EGX 3D download {min(offset+25,len(symbols))}/{len(symbols)}; inserted={report['rows_inserted']}, updated={report['rows_updated']}",flush=True)
    return report
