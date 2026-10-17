# monitor/egx_data_refresh.py
"""
Refresh EGX daily OHLCV data in the existing project daily table from Yahoo Finance.
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import select, update, insert
from core import database
from monitor.gann_analysis import _daily_table

def _pick_col(t, aliases, required=True):
    m={c.name.lower():c for c in t.c}
    for a in aliases:
        if a.lower() in m:return m[a.lower()]
    if required: raise KeyError(f"Missing {aliases}; available={list(m)}")
    return None

def _scalar(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:return None

def _download_chunk(symbols,start,end):
    import yfinance as yf
    return yf.download(
        tickers=" ".join(symbols),start=start,end=end,interval="1d",
        group_by="ticker",auto_adjust=False,actions=False,threads=True,progress=False
    )

def _extract_symbol_frame(raw,symbol,multi):
    if raw is None or len(raw)==0:return pd.DataFrame()
    if multi:
        if symbol not in raw.columns.get_level_values(0):return pd.DataFrame()
        x=raw[symbol].copy()
    else:x=raw.copy()
    if x.empty:return x
    x=x.reset_index()
    cols={str(c).lower().replace(" ","_"):c for c in x.columns}
    dc=cols.get("date") or cols.get("datetime") or x.columns[0]
    def ser(name):
        c=cols.get(name)
        return pd.to_numeric(x[c],errors="coerce") if c is not None else pd.Series(np.nan,index=x.index)
    out=pd.DataFrame({
        "date":pd.to_datetime(x[dc],errors="coerce"),
        "open":ser("open"),"high":ser("high"),"low":ser("low"),"close":ser("close"),
        "adj_close":ser("adj_close"),"volume":ser("volume"),
    })
    return out.dropna(subset=["date","close"])

def refresh_daily_data(days_back=14):
    DB=database(); now_utc=datetime.now(timezone.utc)
    start=(now_utc.date()-timedelta(days=max(7,int(days_back)))).isoformat()
    end=(now_utc.date()+timedelta(days=1)).isoformat()
    with DB() as s:
        t=_daily_table(s); cs=_pick_col(t,["symbol","ticker","sym"])
        syms=[r[0] for r in s.execute(select(cs).where(cs.ilike("%.CA")).distinct().order_by(cs)).all() if r and r[0]]
    if not syms:raise RuntimeError("No .CA symbols found in the EGX daily table.")
    updated_rows=0;updated_symbols=0;errors=[]
    for off in range(0,len(syms),25):
        chunk=syms[off:off+25]
        try:raw=_download_chunk(chunk,start,end)
        except Exception as exc:
            errors.append(f"chunk {off}: {type(exc).__name__}: {exc}");continue
        multi=isinstance(raw.columns,pd.MultiIndex)
        with DB.begin() as s:
            t=_daily_table(s)
            cs=_pick_col(t,["symbol","ticker","sym"])
            cd=_pick_col(t,["session_date","date","d"])
            cts=_pick_col(t,["ts","timestamp"],False)
            co=_pick_col(t,["o","open"],False);ch=_pick_col(t,["h","high"],False);cl=_pick_col(t,["l","low"],False)
            cc=_pick_col(t,["c","close"]);cv=_pick_col(t,["v","volume"],False)
            cadj=_pick_col(t,["adj_c","adj_close","adjusted_close"],False)
            cfeed=_pick_col(t,["feed_symbol"],False);csource=_pick_col(t,["source"],False)
            cretrieved=_pick_col(t,["retrieved_at"],False);cmeta=_pick_col(t,["metadata_json"],False)
            for sym in chunk:
                x=_extract_symbol_frame(raw,sym,multi)
                if x.empty:continue
                touched=False
                for _,r in x.iterrows():
                    d=pd.Timestamp(r["date"]).date();close=_scalar(r["close"])
                    if close is None or close<=0:continue
                    vals={cs.name:sym,cd.name:d,cc.name:close}
                    if cts is not None:vals[cts.name]=int(pd.Timestamp(d,tz="UTC").timestamp())
                    if co is not None:vals[co.name]=_scalar(r["open"]) or close
                    if ch is not None:vals[ch.name]=_scalar(r["high"]) or close
                    if cl is not None:vals[cl.name]=_scalar(r["low"]) or close
                    if cv is not None:vals[cv.name]=_scalar(r["volume"])
                    if cadj is not None:vals[cadj.name]=_scalar(r["adj_close"]) or close
                    if cfeed is not None:vals[cfeed.name]=sym
                    if csource is not None:vals[csource.name]="YAHOO_EGX_PAGE_REFRESH"
                    if cretrieved is not None:vals[cretrieved.name]=now_utc
                    if cmeta is not None:vals[cmeta.name]=json.dumps({"refresh":"backend_button","source":"yfinance"},ensure_ascii=False)
                    exists=s.scalar(select(cs).where(cs==sym,cd==d).limit(1))
                    if exists is None:s.execute(insert(t).values(**vals))
                    else:
                        upd={k:v for k,v in vals.items() if k not in (cs.name,cd.name)}
                        s.execute(update(t).where(cs==sym,cd==d).values(**upd))
                    updated_rows+=1;touched=True
                updated_symbols+=int(touched)
        print(f"EGX data refresh {min(off+25,len(syms))}/{len(syms)} symbols_updated={updated_symbols} rows={updated_rows}",flush=True)
    return {"symbols_total":len(syms),"symbols_updated":updated_symbols,"rows_upserted":updated_rows,"errors":errors[:20],"start":start,"end_exclusive":end}

if __name__=="__main__":
    print(json.dumps(refresh_daily_data(),ensure_ascii=False,indent=2))
