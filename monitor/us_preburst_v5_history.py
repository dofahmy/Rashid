"""Full as-of-bar screener history. Independent of V4 snapshot tables.
Build once for flexible user-selected filters. Uses same indicator conventions
as monitor.us_preburst_research.features, but vectorized per symbol.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from core import database
from monitor.us_preburst_setup import COLS, FEATURES

TABLE='us_preburst_full_history_v5'

def ensure():
    cols=', '.join(f'"{k}" DOUBLE PRECISION' for k in COLS)
    with database().begin() as con:
        con.execute(text(f'''CREATE TABLE IF NOT EXISTS {TABLE} (
          market TEXT NOT NULL, symbol TEXT NOT NULL, reference_utc TIMESTAMP NOT NULL,
          {cols}, PRIMARY KEY (market,symbol,reference_utc))'''))
        con.execute(text(f'CREATE INDEX IF NOT EXISTS ix_{TABLE}_time ON {TABLE}(market,reference_utc DESC)'))

def calc(sym, data, market):
    d=pd.DataFrame(data,columns=['date','open','high','low','close','volume'])
    if d.empty:return []
    for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c],errors='coerce')
    c=d.close; lo=d.low;hi=d.high;v=d.volume
    # Index i indicates the close of bar i, matching the current-mode semantics.
    diff=c.diff();up=diff.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();down=(-diff.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=up/down.replace(0,np.nan)
    rsi=100-100/(1+rs)
    rsi= rsi.where(down.ne(0),100.0)
    rsi=rsi.where(~((down==0)&(up==0)),np.nan)
    ma20=c.rolling(20,min_periods=20).mean();std20=c.rolling(20,min_periods=20).std(ddof=0)
    tr=pd.concat([hi-lo,(hi-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
    vals={
      'rsi14':rsi.where(np.arange(len(d))>=19),
      'momentum5_pct':100*(c/c.shift(5)-1),
      'momentum20_pct':100*(c/c.shift(20)-1),
      'close_vs_sma20_pct':100*(c/ma20-1),
      'close_vs_sma50_pct':100*(c/c.rolling(50,min_periods=50).mean()-1),
      'bbz20':(c-ma20)/std20.replace(0,np.nan),
      'atr14_pct':(100*tr.rolling(14,min_periods=14).mean()/c).where(np.arange(len(d))>=21),
      'return_20':100*(c/c.shift(19)-1),
      'position_20_pct':100*(c-lo.rolling(20,min_periods=20).min())/(hi.rolling(20,min_periods=20).max()-lo.rolling(20,min_periods=20).min()).replace(0,np.nan),
      'volume5_vs_50':v.rolling(5,min_periods=5).mean()/((v.shift(5).rolling(45,min_periods=45).mean()).replace(0,np.nan))
    }
    # The original features() requires >=21 bars for momentum20,
    # >=22 for ATR. Its return_20 uses first and last CLOSE of last 20.
    vals['return_20']=vals['return_20'].where(np.arange(len(d))>=19)
    repeats=np.full(len(d),np.nan)
    cl=c.to_numpy(float)
    if len(cl)>60:
        repeats[60:]=0
        for gap in range(1,61):
            current=cl[60:]
            previous=cl[60-gap:len(cl)-gap]
            repeats[60:]+=np.where(np.isfinite(current)&(current>0)&np.isfinite(previous),np.abs(previous/current-1)<=0.01,False)
    vals['repeat_count_1pct_60']=pd.Series(repeats,index=d.index)
    out=pd.DataFrame({k:vals[k] for k in COLS})
    out=out.replace([np.inf,-np.inf],np.nan)
    out.insert(0,'reference_utc',pd.to_datetime(d.date,errors='coerce',utc=True).dt.tz_localize(None))
    out.insert(0,'symbol',str(sym));out.insert(0,'market',market)
    out=out.iloc[21:].copy()
    out=out.astype(object).where(pd.notna(out),None)
    return out.to_dict('records')

def build(market='us'):
    ensure();count=0;bars=0;prev=None;buf=[]
    fields=['market','symbol','reference_utc']+COLS
    stmt=text(f'''INSERT INTO {TABLE} ({','.join('"'+x+'"' for x in fields)}) VALUES ({','.join(':'+x for x in fields)})
       ON CONFLICT (market,symbol,reference_utc) DO UPDATE SET {','.join('"'+x+'"=EXCLUDED."'+x+'"' for x in COLS)}''')
    def write(sym, data):
        nonlocal count,bars
        if not data:return
        rows=calc(sym,data,market);count+=1;bars+=len(rows)
        with database().begin() as wr:
            for off in range(0,len(rows),250):wr.execute(stmt,rows[off:off+250])
        if count%250==0:print(f'{market}: processed={count} symbols, rows={bars}',flush=True)
    with database()() as con:
        if market=='us':
            result=con.execute(text('SELECT symbol,bar_time,open,high,low,close,volume FROM us_all_seven_hourly_bars ORDER BY symbol,bar_time'))
        else:
            from monitor.gann_analysis import _daily_table
            from sqlalchemy import select
            tbl=_daily_table(con); names={col.name.lower():col for col in tbl.c}
            def pick(*opts):return next((names[z] for z in opts if z in names),None)
            sy=pick('symbol','ticker','sym');dt=pick('session_date','date','d');cl=pick('c','close');high=pick('h','high');low=pick('l','low');op=pick('o','open');vol=pick('v','volume')
            if any(x is None for x in [sy,dt,cl,high,low]):raise RuntimeError('Missing EGX OHLC columns')
            result=con.execute(select(sy,dt,op if op is not None else cl,high,low,cl,vol if vol is not None else cl).where(sy.ilike('%.CA')).order_by(sy,dt))
        for row in result:
            sym=str(row[0]);
            if prev is not None and prev!=sym:write(prev,buf);buf=[]
            prev=sym;buf.append(tuple(row[1:]))
        if prev is not None:write(prev,buf)
    return {'market':market,'symbols':count,'feature_rows':bars}

def add_signal_prices(rows, market):
    """Enrich UI rows with exact source close at signal bar and latest recorded close.

    Return None only when the corresponding OHLC candle is genuinely unavailable.
    Fetch in batches, with explicit Postgres timestamp casts for reliable joins.
    """
    if not rows:
        return rows
    from sqlalchemy import bindparam, select
    import datetime as _dt

    with database()() as con:
        if market == 'egypt':
            from monitor.gann_analysis import _daily_table
            tbl = _daily_table(con)
            cmap = {x.name.lower(): x for x in tbl.c}
            def pick(*opts): return next((cmap[x] for x in opts if x in cmap), None)
            sy, dt, cl = pick('symbol', 'ticker', 'sym'), pick('session_date','date','d'), pick('c','close')
            if any(x is None for x in (sy,dt,cl)):
                raise RuntimeError('EGX OHLC source missing symbol/date/close')
            fmt = con.get_bind().dialect.identifier_preparer
            table_name = fmt.format_table(tbl)
            sym_col, time_col, price_col = fmt.quote(sy.name),fmt.quote(dt.name),fmt.quote(cl.name)
            time_cast = 'DATE'
        else:
            table_name='us_all_seven_hourly_bars'
            sym_col,time_col,price_col='symbol','bar_time','close'
            time_cast='TIMESTAMP'

        for off in range(0,len(rows),100):
            batch=rows[off:off+100]
            binds={}; values=[]
            for i,r in enumerate(batch):
                key=str(r['symbol']); start=r.get('signal_started_utc') or r.get('reference_utc')
                if isinstance(start,str):
                    start=pd.Timestamp(start).to_pydatetime()
                binds[f's{i}']=key
                binds[f't{i}']=start.date() if time_cast=='DATE' and hasattr(start,'date') else start
                values.append(f'(:s{i}, CAST(:t{i} AS {time_cast}))')
            # Per-symbol last available close uses DISTINCT ON; start uses the exact timestamp.
            q=f"""WITH requested(symbol,signal_time) AS (VALUES {','.join(values)}),
            latest AS (
                SELECT DISTINCT ON (src.{sym_col}) src.{sym_col} AS symbol,
                    src.{price_col} AS current_price
                FROM {table_name} src JOIN requested req ON src.{sym_col}=req.symbol
                ORDER BY src.{sym_col},src.{time_col} DESC
            )
            SELECT req.symbol, first_bar.{price_col} AS signal_price, latest.current_price
            FROM requested req
            LEFT JOIN {table_name} first_bar
                ON first_bar.{sym_col}=req.symbol AND first_bar.{time_col}=req.signal_time
            LEFT JOIN latest ON latest.symbol=req.symbol"""
            result={str(x['symbol']):x for x in con.execute(text(q),binds).mappings()}
            for r in batch:
                hit=result.get(str(r['symbol']))
                r['signal_price']=float(hit['signal_price']) if hit and hit['signal_price'] is not None else None
                r['current_price']=float(hit['current_price']) if hit and hit['current_price'] is not None else None
    return rows

def current_dashboard(storage, selected, settings, market='us', limit=200):
    """Find exact latest matching-run starts from ALL available indicator snapshots."""
    ensure();condition=['market=:market'];params={'market':storage}
    member_filter=''
    from sqlalchemy import bindparam
    if market=='sp500':
        from monitor.sp500_seven_system import list_symbols
        params['members']=list_symbols() or ['___EMPTY___'];member_filter=' AND symbol IN :members'
    for k in selected:
        condition.append(f'"{k}" BETWEEN :min_{k} AND :max_{k}')
        params[f'min_{k}']=settings[k]['min'];params[f'max_{k}']=settings[k]['max']
    where=' AND '.join(condition)
    def sql(s):
        z=text(s)
        if member_filter:z=z.bindparams(bindparam('members',expanding=True))
        return z
    with database()() as con:
        latest=con.execute(sql(f'''SELECT symbol,MAX(reference_utc) reference_utc FROM {TABLE} WHERE market=:market{member_filter} GROUP BY symbol'''),params).mappings().all()
        lmap={r['symbol']:r['reference_utc'] for r in latest}
        # Global market close comes from the stored universe, not from matching rows.
        market_latest=max(lmap.values()) if lmap else None
        # Compare latest row for each symbol only; do not mistake older matching rows for live signals.
        now=con.execute(sql(f'''SELECT h.* FROM {TABLE} h JOIN
          (SELECT symbol,MAX(reference_utc) mx FROM {TABLE} WHERE market=:market{member_filter} GROUP BY symbol) q
          ON h.symbol=q.symbol AND h.reference_utc=q.mx AND h.market=:market
          WHERE {' AND '.join('h.'+x for x in condition)}'''),params).mappings().all()
        symbols=[r['symbol'] for r in now]
        history=[]
        for off in range(0,len(symbols),350):
            group=symbols[off:off+350]
            history+=con.execute(text(f'''SELECT symbol,reference_utc,{','.join('"'+k+'"' for k in selected) if selected else 'market'}
             FROM {TABLE} WHERE market=:market AND symbol IN :syms ORDER BY symbol,reference_utc DESC''').bindparams(bindparam('syms',expanding=True)),{'market':storage,'syms':group}).mappings().all()
    series={}
    for row in history:series.setdefault(row['symbol'],[]).append(row)
    def matches(row):return all(row[k] is not None and settings[k]['min']<=row[k]<=settings[k]['max'] for k in selected)
    output=[];stale_output=[];by_date={};unknown=0
    for current in now:
        symbol=current['symbol'];hist=series.get(symbol,[])
        length=0
        for row in hist:
            if not matches(row):break
            length+=1
        began=hist[length-1]['reference_utc'] if length else current['reference_utc']
        earlier_exists=len(hist)>length
        status=('جديد مؤكد' if length==1 and earlier_exists else 'مستمر' if length>1 and earlier_exists else 'بداية غير مؤكدة')
        if not earlier_exists:unknown+=1
        fresh=(market_latest is not None and current['reference_utc']==market_latest)
        v=dict(current);v.update(signal_started_utc=began,matching_bars=length,signal_status=(status if fresh else 'بيانات قديمة — '+status),is_new=(fresh and status=='جديد مؤكد'),is_stale=not fresh)
        if fresh:
            date=str(began)[:10];by_date[date]=by_date.get(date,0)+1
            output.append(v)
        else:
            stale_output.append(v)
    output.sort(key=lambda r:(0 if r['is_new'] else 1,-pd.Timestamp(r['signal_started_utc']).timestamp(),r['symbol']))
    stale_output.sort(key=lambda r:(-pd.Timestamp(r['reference_utc']).timestamp(),r['symbol']))
    add_signal_prices(output[:limit], storage)
    add_signal_prices(stale_output[:limit], storage)
    return {'rows':output[:limit],'total':len(output),'baseline':len(lmap),'new_confirmed':sum(r['is_new'] for r in output),'ongoing':sum(r['signal_status']=='مستمر' for r in output),'unknown':sum(r['signal_status']=='بداية غير مؤكدة' for r in output),'by_date':sorted(by_date.items(),reverse=True),'history_loaded':len(history),'stale_count':len(stale_output),'stale_rows':stale_output[:limit],'market_latest_utc':market_latest,'fresh_available':sum(t==market_latest for t in lmap.values()) if market_latest else 0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-current',action='store_true');p.add_argument('--build-egypt',action='store_true');a=p.parse_args()
    if a.build_current:print(json.dumps(build('us'),ensure_ascii=False))
    if a.build_egypt:print(json.dumps(build('egypt'),ensure_ascii=False))
