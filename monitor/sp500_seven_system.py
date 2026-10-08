# monitor/sp500_seven_system.py
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import text

from core import database
from monitor.sp500_seven_data import ensure_tables, DAILY_TABLE, CONSTIT_TABLE, INDEX_TABLE
from monitor.seven_forward_20d import forward_20d
from monitor.seven_breakout_rule import classify_repeat
from monitor.seven_exit_targets import repeat_exit_targets
from monitor.seven_system import (
    _single, _repeat_double7_levels, _filter, _chart_payload
)


def _load_stocks():
    ensure_tables()
    DB = database()
    with DB() as s:
        syms = [r[0] for r in s.execute(text(
            f"SELECT symbol FROM {CONSTIT_TABLE} WHERE active=TRUE ORDER BY symbol"
        )).all()]
        rows = s.execute(text(f"""
            SELECT d.symbol, d.session_date, d.open, d.high, d.low, d.close, d.adj_close, d.volume
            FROM {DAILY_TABLE} d
            JOIN {CONSTIT_TABLE} c ON c.symbol=d.symbol
            WHERE c.active=TRUE
            ORDER BY d.symbol, d.session_date
        """)).mappings().all()

    grouped = defaultdict(list)
    for r in rows:
        grouped[str(r["symbol"])].append(r)

    out = {}
    for sym in syms:
        rr = grouped.get(sym, [])
        if not rr:
            continue
        df = pd.DataFrame(rr)
        df["date"] = pd.to_datetime(df["session_date"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["raw_close"] = pd.to_numeric(df["close"], errors="coerce")
        df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        # Adjust all tradeable OHLC consistently with adj_close for backtests.
        factor = (df["close"] / df["raw_close"]).replace([np.inf, -np.inf], np.nan)
        for col in ("open", "high", "low"):
            df[col] = df[col] * factor
        df = df[["date","open","high","low","raw_close","close","volume"]].dropna(subset=["date","close"])
        out[sym] = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def list_symbols():
    ensure_tables()
    DB = database()
    with DB() as s:
        return [r[0] for r in s.execute(text(
            f"SELECT symbol FROM {CONSTIT_TABLE} WHERE active=TRUE ORDER BY symbol"
        )).all()]


def constituent_count():
    ensure_tables()
    DB = database()
    with DB() as s:
        return int(s.scalar(text(
            f"SELECT COUNT(*) FROM {CONSTIT_TABLE} WHERE active=TRUE"
        )) or 0)


def _load_index():
    ensure_tables()
    DB = database()
    with DB() as s:
        rows = s.execute(text(
            f"SELECT session_date, close, volume FROM {INDEX_TABLE} ORDER BY session_date"
        )).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["date","close","volume"]), "YAHOO_^GSPC"
    x = pd.DataFrame(rows)
    x["date"] = pd.to_datetime(x["session_date"])
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce")
    return x[["date","close","volume"]].dropna(subset=["date","close"]).reset_index(drop=True), "S&P 500 ^GSPC"


def _market(frames):
    ps=defaultdict(int);vs=defaultdict(int);pc=defaultdict(int);vc=defaultdict(int)
    for df in frames.values():
        for _,r in df.iterrows():
            d=pd.Timestamp(r.date).normalize()
            p=int(round(float(r.close)*100))
            ps[d]+=p;pc[d]+=1
            if pd.notna(r.volume):
                vs[d]+=int(round(float(r.volume)));vc[d]+=1
    rows=[]
    for d in sorted(set(ps)|set(vs)):
        p=ps.get(d,0);v=vs.get(d,0)
        rows.append(dict(
            date=d, raw_close=np.nan,
            price_value=p/100,
            price_remainder=p%7 if pc[d] else None,
            price_signal7=bool(pc[d] and p%7==0),
            volume_value=v if vc[d] else None,
            volume_remainder=v%7 if vc[d] else None,
            volume_signal7=bool(vc[d] and v%7==0),
            price_count=pc[d],volume_count=vc[d],
        ))
    return pd.DataFrame(rows),{}


def _index_map():
    idx,source=_load_index()
    return {pd.Timestamp(r.date).normalize():float(r.close) for _,r in idx.iterrows()},source


def _stock_signal_map(frames):
    by_date=defaultdict(lambda:{'price_symbols':[],'volume_symbols':[],'double_symbols':[]})
    for sym,df in frames.items():
        s,_=_single(df,True)
        for _,r in s.iterrows():
            d=pd.Timestamp(r.date).normalize()
            p=bool(r.get('price_signal7',False));v=bool(r.get('volume_signal7',False))
            det={
                'symbol':sym,
                'close':round(float(r.raw_close),4) if pd.notna(r.raw_close) else None,
                'cum_price':round(float(r.price_value),2) if pd.notna(r.price_value) else None,
                'cum_volume':int(round(float(r.volume_value))) if pd.notna(r.volume_value) else None,
            }
            if p:by_date[d]['price_symbols'].append(det)
            if v:by_date[d]['volume_symbols'].append(det)
            if p and v:by_date[d]['double_symbols'].append(det)
    return by_date



def _repeat_direction_from_candle(row):
    vals=[row.get("open"),row.get("high"),row.get("low"),row.get("raw_close")]
    if any(v is None or pd.isna(v) for v in vals):
        return "UNKNOWN","OHLC unavailable"
    o=float(row["open"]);h=float(row["high"]);l=float(row["low"]);c=float(row["raw_close"])
    body=abs(c-o)
    upper=max(0.0,h-max(o,c))
    lower=max(0.0,min(o,c)-l)
    if c<o:
        if lower>body and lower>upper:
            return "BUY","negative candle but lower wick > body"
        return "SELL","negative candle"
    if c>o:
        if upper>body:
            return "SELL","positive candle but upper wick > body"
        return "BUY","positive candle"
    if upper>lower:return "SELL","flat candle + dominant upper wick"
    if lower>upper:return "BUY","flat candle + dominant lower wick"
    return "NEUTRAL","flat balanced candle"

def _gap_matched_return(df, signal_date, gap_sessions, direction):
    x=df.sort_values("date").reset_index(drop=True)
    target=pd.Timestamp(signal_date).normalize()
    m=x.index[pd.to_datetime(x["date"]).dt.normalize()==target].tolist()
    if not m:return {}
    i=int(m[-1]);n=max(1,int(gap_sessions));j=min(i+n,len(x)-1)
    elapsed=j-i
    start=float(x.iloc[i]["raw_close"]);end=float(x.iloc[j]["raw_close"])
    raw=(end/start-1.0)*100.0 if start else None
    aligned=(-raw if direction=="SELL" else raw) if raw is not None else None
    return {"gap_target_sessions":n,"gap_elapsed_sessions":elapsed,"gap_complete":elapsed>=n,"gap_end_date":pd.Timestamp(x.iloc[j]["date"]).date().isoformat(),"gap_end_price":round(end,4),"gap_raw_return_pct":round(raw,2) if raw is not None else None,"gap_direction_return_pct":round(aligned,2) if aligned is not None else None}

def repeat_price_time_screener(price_tolerance_pct=1.0,max_gap_sessions=20,recent_days=90):
    frames=_load_stocks();rows=[];latest=None
    for df in frames.values():
        if len(df):
            d=pd.Timestamp(df.date.max()).normalize();latest=d if latest is None or d>latest else latest
    for sym,df in frames.items():
        single,_=_single(df,True)
        levels=_repeat_double7_levels(single,price_tolerance_pct,max_gap_sessions)
        if not levels:continue
        z=max(levels,key=lambda a:pd.Timestamp(a["end_date"]))
        end=pd.Timestamp(z["end_date"]).normalize();age=(latest-end).days if latest is not None else None
        if recent_days is not None and age is not None and age>int(recent_days):continue
        rr=df[pd.to_datetime(df["date"]).dt.normalize()==end]
        if len(rr):
            candle=rr.iloc[-1];direction,reason=_repeat_direction_from_candle(candle)
            o=round(float(candle["open"]),4) if pd.notna(candle["open"]) else None
            h=round(float(candle["high"]),4) if pd.notna(candle["high"]) else None
            l=round(float(candle["low"]),4) if pd.notna(candle["low"]) else None
            c=round(float(candle["raw_close"]),4) if pd.notna(candle["raw_close"]) else None
        else:
            direction,reason="UNKNOWN","OHLC unavailable";o=h=l=c=None
        dates=[pd.Timestamp(x.strip()).normalize() for x in str(z["dates"]).split("|") if x.strip()]
        if len(dates)>=2:
            pos={pd.Timestamp(d).normalize():i for i,d in enumerate(pd.to_datetime(df["date"]))}
            actual_gap=pos.get(dates[-1],0)-pos.get(dates[-2],0)
            if actual_gap<=0:actual_gap=int(z["max_session_gap"])
        else:actual_gap=int(z["max_session_gap"])
        perf=_gap_matched_return(df,z["end_date"],actual_gap,direction)
        rows.append({"symbol":sym,"latest_repeat_date":z["end_date"],"first_repeat_date":z["start_date"],"repeat_price":z["avg_price"],"touches":z["touches"],"price_spread_pct":z["price_spread_pct"],"dates":z["dates"],"prices":z["prices"],"days_ago":age,"repeat_direction":direction,"direction_reason":reason,"candle_open":o,"candle_high":h,"candle_low":l,"candle_close":c,"actual_repeat_gap_sessions":actual_gap,**classify_repeat(df,z["end_date"],z["max_session_gap"],z["touches"]),**perf,**forward_20d(df,z["end_date"],z["avg_price"]),**repeat_exit_targets(df,z["end_date"],z["avg_price"])})
    rows.sort(key=lambda r:(pd.Timestamp(r["latest_repeat_date"]),r["touches"]),reverse=True)
    return {"rows":rows,"count":len(rows),"latest_market_date":latest.date().isoformat() if latest is not None else None,"price_tolerance_pct":float(price_tolerance_pct),"max_gap_sessions":int(max_gap_sessions),"recent_days":recent_days}


def run(scope='market',symbol=None,metric='both',date_mode='all',day=None,month=None,start=None,end=None,signals_only=False,repeat_price_tolerance_pct=1.0,repeat_max_gap_sessions=20):
    frames=_load_stocks()
    if not frames:
        raise RuntimeError("S&P 500 data table is empty. Run Update S&P500 Data first.")

    source=None;stock_details={};repeat_levels=[]
    index_by_date,index_source=_index_map()

    if scope=='market':
        full,meta=_market(frames);name='S&P 500 constituents';source=index_source
        full['index_value']=full.date.map(lambda d:index_by_date.get(pd.Timestamp(d).normalize(),np.nan))
        stock_details=_stock_signal_map(frames)
        full['stocks_price7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),{}).get('price_symbols',[])))
        full['stocks_volume7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),{}).get('volume_symbols',[])))
        full['stocks_double7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),{}).get('double_symbols',[])))
    elif scope=='stock':
        symbol=(symbol or '').upper()
        if symbol not in frames:raise ValueError('Choose a valid S&P 500 symbol.')
        full,meta=_single(frames[symbol],True);name=symbol
        repeat_levels=_repeat_double7_levels(full,repeat_price_tolerance_pct,repeat_max_gap_sessions)
        full['index_value']=np.nan;full['stocks_price7_count']=np.nan;full['stocks_volume7_count']=np.nan;full['stocks_double7_count']=np.nan
    elif scope=='index':
        idx,source=_load_index()
        if idx.empty:raise RuntimeError("S&P 500 index data is empty. Update data first.")
        full,meta=_single(idx,False);name='S&P 500 Index (^GSPC)'
        full['index_value']=full['raw_close'];full['stocks_price7_count']=np.nan;full['stocks_volume7_count']=np.nan;full['stocks_double7_count']=np.nan
    else:raise ValueError('Invalid scope.')

    f=_filter(full,date_mode,day,month,start,end)
    pm=f.price_signal7.fillna(False)
    vm=f.volume_signal7.fillna(False) if 'volume_signal7' in f else pd.Series(False,index=f.index)
    dm=pm&vm
    mask=pm if metric=='price' else vm if metric=='volume' else dm
    sig=f[mask].copy()
    shown=sig if signals_only else f

    if scope=='stock':chart=_chart_payload(f,dm,'raw_close')
    elif scope=='market':chart=_chart_payload(f,dm,'index_value')
    else:chart=_chart_payload(f,pm,'raw_close')

    def recs(df):
        out=[]
        for _,r in df.iterrows():
            out.append({
                'date':pd.Timestamp(r.date).date().isoformat(),
                'raw_close':None if pd.isna(r.get('raw_close',np.nan)) else round(float(r.get('raw_close')),4),
                'index_value':None if pd.isna(r.get('index_value',np.nan)) else round(float(r.get('index_value')),2),
                'price_value':None if pd.isna(r.get('price_value')) else round(float(r.get('price_value')),2),
                'price_remainder':None if pd.isna(r.get('price_remainder')) else int(r.get('price_remainder')),
                'price_signal7':bool(r.get('price_signal7',False)),
                'volume_value':None if pd.isna(r.get('volume_value')) else int(round(float(r.get('volume_value')))),
                'volume_remainder':None if pd.isna(r.get('volume_remainder')) else int(r.get('volume_remainder')),
                'volume_signal7':bool(r.get('volume_signal7',False)),
                'double_signal7':bool(r.get('price_signal7',False) and r.get('volume_signal7',False)),
                'price_count':int(r.get('price_count')) if pd.notna(r.get('price_count',np.nan)) else None,
                'volume_count':int(r.get('volume_count')) if pd.notna(r.get('volume_count',np.nan)) else None,
                'stocks_price7_count':int(r.get('stocks_price7_count')) if pd.notna(r.get('stocks_price7_count',np.nan)) else None,
                'stocks_volume7_count':int(r.get('stocks_volume7_count')) if pd.notna(r.get('stocks_volume7_count',np.nan)) else None,
                'stocks_double7_count':int(r.get('stocks_double7_count')) if pd.notna(r.get('stocks_double7_count',np.nan)) else None,
            })
        return out

    return {
        'scope':scope,'symbol':symbol,'display_name':name,'source':source,'anchor':meta,
        'rows':recs(shown),'signal_rows':recs(sig),
        'total_rows':len(shown),'signal_count':len(sig),
        'double_signal_count':int(dm.sum()) if scope!='index' and len(f) else 0,
        'price_signal_count':int(pm.sum()) if len(f) else 0,
        'volume_signal_count':int(vm.sum()) if scope!='index' and len(f) else 0,
        'chart':chart,'repeat_levels':repeat_levels,
        'repeat_price_tolerance_pct':float(repeat_price_tolerance_pct),
        'repeat_max_gap_sessions':int(repeat_max_gap_sessions),
    }


def market_day_stock_details(day,kind='double'):
    frames=_load_stocks();m=_stock_signal_map(frames);d=pd.Timestamp(day).normalize()
    row=m.get(d,{'price_symbols':[],'volume_symbols':[],'double_symbols':[]})
    key={'price':'price_symbols','volume':'volume_symbols','double':'double_symbols'}.get(kind,'double_symbols')
    return row.get(key,[])
