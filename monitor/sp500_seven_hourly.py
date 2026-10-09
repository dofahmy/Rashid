"""S&P500 hourly Double-7 Repeat screener. Separately stored 1h bars.
Hourly data is not available from the existing daily database: refresh with CLI.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sqlalchemy import text
from core import database
from monitor.sp500_seven_data import ensure_tables,CONSTIT_TABLE
from monitor.seven_system import _single
from monitor.sp500_seven_system import _repeat_direction_from_candle

TABLE='sp500_seven_hourly'

def ensure_hourly_table():
    ensure_tables()
    with database().begin() as s:
        s.execute(text(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            symbol TEXT NOT NULL, bar_time TIMESTAMP NOT NULL,
            open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
            close DOUBLE PRECISION NOT NULL, volume BIGINT,
            PRIMARY KEY(symbol,bar_time))"""))

def refresh(period='60d', batch_size=20):
    import yfinance as yf
    ensure_hourly_table()
    with database()() as s:
        symbols=[x[0] for x in s.execute(text(f'SELECT symbol FROM {CONSTIT_TABLE} WHERE active=TRUE ORDER BY symbol')).all()]
    if not symbols:raise RuntimeError('No S&P500 constituents; refresh daily S&P500 data first.')
    successes=0; bars=0; errors=[]
    for j in range(0,len(symbols),batch_size):
        batch=symbols[j:j+batch_size]
        try:
            raw=yf.download(tickers=' '.join(x.replace('.','-') for x in batch),period=period,interval='1h',auto_adjust=True,group_by='ticker',threads=True,progress=False)
        except Exception as exc:
            errors.append(f'{j}: {exc}');continue
        records=[]
        for sym in batch:
            feed=sym.replace('.','-')
            try:
                data=raw[feed] if isinstance(raw.columns,pd.MultiIndex) else (raw if len(batch)==1 else pd.DataFrame())
                if data is None or data.empty:continue
                data=data.rename(columns=str.lower)
                for stamp,r in data.iterrows():
                    if pd.isna(r.get('close')):continue
                    ts=pd.Timestamp(stamp)
                    if ts.tzinfo is not None:ts=ts.tz_convert('UTC').tz_localize(None)
                    records.append(dict(symbol=sym,bar_time=ts.to_pydatetime(),open=float(r.open) if pd.notna(r.get('open')) else None,high=float(r.high) if pd.notna(r.get('high')) else None,low=float(r.low) if pd.notna(r.get('low')) else None,close=float(r.close),volume=int(r.volume) if pd.notna(r.get('volume')) else None))
                successes+=1
            except Exception as exc:errors.append(f'{sym}: {exc}')
        if records:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from sqlalchemy import Table, MetaData
            with database().begin() as s:
                table=Table(TABLE,MetaData(),autoload_with=s.get_bind())
                for k in range(0,len(records),1000):
                    stmt=pg_insert(table).values(records[k:k+1000]);stmt=stmt.on_conflict_do_update(index_elements=['symbol','bar_time'],set_={col:getattr(stmt.excluded,col) for col in ('open','high','low','close','volume')})
                    s.execute(stmt)
            bars+=len(records)
        print(f'Hourly refresh {min(j+batch_size,len(symbols))}/{len(symbols)}; saved={bars}',flush=True)
    return dict(symbols_with_data=successes,bars_saved=bars,errors=errors[:30])

def _frames():
    ensure_hourly_table()
    with database()() as s:
        rows=s.execute(text(f'SELECT symbol,bar_time,open,high,low,close,volume FROM {TABLE} ORDER BY symbol,bar_time')).mappings().all()
    out={}
    for sym,group in pd.DataFrame(rows).groupby('symbol') if rows else []:
        df=group.rename(columns={'bar_time':'date'}).copy().reset_index(drop=True)
        df['date']=pd.to_datetime(df['date'],utc=True).dt.tz_localize(None)
        for c in ['open','high','low','close','volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
        out[sym]=df.sort_values('date').reset_index(drop=True)
    return out

def _hourly_clusters(signals,tolerance,max_gap):
    sig=signals[signals.price_signal7.fillna(False)&signals.volume_signal7.fillna(False)]
    clusters=[]; current=[]
    for i,r in sig.iterrows():
        val=float(r.raw_close)
        near=(current and i-current[-1][0]<=max_gap and abs(val-current[-1][2])/((val+current[-1][2])/2)*100 <= tolerance)
        if not near:
            if len(current)>=2:clusters.append(current)
            current=[]
        current.append((i,pd.Timestamp(r.date),val))
    if len(current)>=2:clusters.append(current)
    return clusters

def _rule(frame,cluster):
    gap=max(cluster[i][0]-cluster[i-1][0] for i in range(1,len(cluster)))
    touches=len(cluster)
    idx=cluster[-1][0]
    if idx<59:return gap,touches,None,'NO_DATA'
    lows=pd.to_numeric(frame.iloc[idx-59:idx+1]['low'],errors='coerce')
    if len(lows)!=60 or lows.isna().any():return gap,touches,None,'NO_DATA'
    higher=bool(lows.iloc[30:].min()>lows.iloc[:30].min())
    return gap,touches,higher,'MATCH' if (11<=gap<=20 and touches==2 and higher) else 'NO_MATCH'

def directional_profit_exits(df, idx, price, direction, targets=(3, 5)):
    """First post-signal hourly High (BUY) or Low (SELL) touching profit target.

    No stop-loss. OPEN means an observed, valid price history has not hit the
    target yet; absence of complete OHLC records is NO_DATA. No entry fill implied.
    """
    result = {}
    for target in targets:
        result.update({f'exit_{target}_status': 'NO_DATA',
                       f'exit_{target}_session': None,
                       f'exit_{target}_date': None})
    result['exit_observed_sessions'] = 0
    result['exit_max_favorable_pct'] = None
    result['directional_current_return_pct'] = None
    result['directional_gap_return_pct'] = None
    if direction not in ('BUY', 'SELL') or price is None or price <= 0:
        for target in targets: result[f'exit_{target}_status'] = 'NO_SIGNAL'
        return result
    future = df.iloc[idx + 1:]
    if not future.empty:
        col = 'high' if direction == 'BUY' else 'low'
        values = pd.to_numeric(future[col], errors='coerce')
        result['exit_observed_sessions'] = len(future)
        if values.notna().all() and (values > 0).all():
            favorable = float(values.max()) if direction == 'BUY' else float(values.min())
            result['exit_max_favorable_pct'] = round((favorable / price - 1) * 100 * (1 if direction == 'BUY' else -1), 2)
            for target in targets:
                threshold = price * (1 + target / 100) if direction == 'BUY' else price * (1 - target / 100)
                hits = values >= threshold if direction == 'BUY' else values <= threshold
                if hits.any():
                    pos = int(np.flatnonzero(hits.to_numpy())[0])
                    result[f'exit_{target}_status'] = 'EXIT'
                    result[f'exit_{target}_session'] = pos + 1
                    result[f'exit_{target}_date'] = pd.Timestamp(future.iloc[pos]['date']).strftime('%Y-%m-%d %H:%M')
                else:
                    result[f'exit_{target}_status'] = 'OPEN'
        # No gaps in the hourly OHLC record are assumed when calculating first hit.
    else:
        for target in targets: result[f'exit_{target}_status'] = 'OPEN'
    last_close = pd.to_numeric(df.iloc[-1]['close'], errors='coerce')
    if pd.notna(last_close):
        result['directional_current_return_pct'] = round((float(last_close)/price-1)*100*(1 if direction=='BUY' else -1), 2)
    return result

def screener(price_tolerance_pct=1.0,max_gap_bars=20,recent_bars=300):
    frames=_frames();out=[];latest=None
    for sym,df in frames.items():
        if df.empty:continue
        latest=max(latest,pd.Timestamp(df.date.max())) if latest is not None else pd.Timestamp(df.date.max())
        s,_=_single(df,True)
        clusters=_hourly_clusters(s,price_tolerance_pct,max_gap_bars)
        if not clusters:continue
        c=clusters[-1];last=c[-1];idx=last[0]
        age=len(df)-idx-1
        if recent_bars is not None and age>recent_bars:continue
        gap,touches,higher,status=_rule(df,c)
        price=round(float(np.mean([p for _,_,p in c])),4)
        spread=round((max(p for _,_,p in c)-min(p for _,_,p in c))/price*100,3) if price else None
        candle=df.iloc[idx].to_dict();candle['raw_close']=candle['close']
        direction,reason=_repeat_direction_from_candle(candle)
        exits=directional_profit_exits(df,idx,price,direction)
        all_future=df.iloc[idx+1:]
        peak_high=None; peak_bars=None
        if len(all_future):
            hvals=pd.to_numeric(all_future.high,errors='coerce')
            if hvals.notna().any():
                peak_pos=int(np.argmax(hvals.fillna(float('-inf')).to_numpy()))
                peak_high=round(float(hvals.iloc[peak_pos]),4)
                peak_bars=peak_pos+1
        future=df.iloc[idx+1:idx+21]
        maxrise=round((future.high.max()/price-1)*100,2) if len(future) and future.high.notna().any() else None
        maxdown=round((future.low.min()/price-1)*100,2) if len(future) and future.low.notna().any() else None
        anchor=float(df.iloc[idx]['close'])
        elapsed=min(gap,age)
        gapraw=round((float(df.iloc[idx+elapsed].close)/anchor-1)*100,2) if anchor>0 and elapsed>0 else None
        exits["directional_gap_return_pct"] = round(gapraw * (1 if direction == "BUY" else -1), 2) if gapraw is not None and direction in ("BUY", "SELL") else None
        out.append(dict(symbol=sym,latest_repeat_date=last[1].strftime('%Y-%m-%d %H:%M UTC'),repeat_price=price,
            gap=gap,touches=touches,repeat_peak_high=peak_high,repeat_peak_bars=peak_bars,price_spread_pct=spread,breakout_pre60_higher_lows=higher,
            breakout_rule_match=status=='MATCH',breakout_rule_status=status,
            gap_elapsed_bars=elapsed,gap_complete=elapsed>=gap,gap_target_bars=gap,gap_raw_return_pct=gapraw,
            max_rise_20h_pct=maxrise,max_drawdown_20h_pct=maxdown,
            repeat_direction=direction,direction_reason=reason,
            candle_open=round(float(candle['open']),4) if pd.notna(candle['open']) else None,
            candle_high=round(float(candle['high']),4) if pd.notna(candle['high']) else None,
            candle_low=round(float(candle['low']),4) if pd.notna(candle['low']) else None,
            candle_close=round(float(candle['close']),4),
            observed_20h_bars=len(future),complete_20h=len(future)==20,
            current_price=round(float(df.iloc[-1].close),4),
            dates=' | '.join(t.strftime('%Y-%m-%d %H:%M') for _,t,_ in c),
            prices=' | '.join(str(p) for _,_,p in c),**exits))
    out.sort(key=lambda r:r['latest_repeat_date'],reverse=True)
    return dict(rows=out,count=len(out),latest_market_date=latest.strftime('%Y-%m-%d %H:%M UTC') if latest is not None else None)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--period',default='60d');parser.add_argument('--batch-size',type=int,default=20)
    opts=parser.parse_args();print(refresh(opts.period,opts.batch_size))
