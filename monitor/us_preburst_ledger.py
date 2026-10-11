"""Historical Pre-Burst filter episodes and price outcomes. Build offline, read via UI.

All signals use indicator values known at candle close, and source OHLC values.
No future-data fields are used for signal creation.
"""
import argparse, hashlib, json
from collections import defaultdict
from datetime import datetime
import numpy as np
from sqlalchemy import text, select
from core import database
from monitor.us_preburst_v5_history import TABLE

LEDGER = 'us_preburst_signal_ledger_v6'
HORIZONS = (20, 50, 100, 250)

def fingerprint(selected, settings):
    payload = [[k, float(settings[k]['min']), float(settings[k]['max'])] for k in sorted(selected)]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()[:24]

def ensure():
    with database().begin() as con:
        con.execute(text(f'''CREATE TABLE IF NOT EXISTS {LEDGER} (
          filter_id VARCHAR(24) NOT NULL, market TEXT NOT NULL, symbol TEXT NOT NULL,
          start_utc TIMESTAMP NOT NULL, end_utc TIMESTAMP, status TEXT NOT NULL,
          entry_close DOUBLE PRECISION, exit_close DOUBLE PRECISION,
          return_pct DOUBLE PRECISION, max_rise_during_match_pct DOUBLE PRECISION,
          bars_matched INTEGER NOT NULL, bars_available_after INTEGER NOT NULL,
          max_rise_20_pct DOUBLE PRECISION, max_rise_50_pct DOUBLE PRECISION,
          max_rise_100_pct DOUBLE PRECISION, max_rise_250_pct DOUBLE PRECISION,
          max_rise_20_complete BOOLEAN NOT NULL, max_rise_50_complete BOOLEAN NOT NULL,
          max_rise_100_complete BOOLEAN NOT NULL, max_rise_250_complete BOOLEAN NOT NULL,
          PRIMARY KEY(filter_id, market, symbol, start_utc)
        )'''))
        con.execute(text(f'CREATE INDEX IF NOT EXISTS ix_{LEDGER}_filter ON {LEDGER}(filter_id,market,start_utc DESC)'))

def _stock_events(sym, bars, selected, settings, market, fid):
    if not bars: return []
    # rows (time, open, high, low, close, indicators...) already ordered ascending
    mask=[]
    for row in bars:
        yes=True
        for k in selected:
            val=row.get(k)
            if val is None or not np.isfinite(float(val)) or not settings[k]['min']<=float(val)<=settings[k]['max']:
                yes=False;break
        mask.append(yes)
    out=[]; n=len(bars);i=0
    while i<n:
        if not mask[i] or bars[i]['close'] is None or bars[i]['close'] <= 0:
            i+=1;continue
        start=i
        while i+1<n and mask[i+1] and bars[i+1]['close'] is not None and bars[i+1]['close']>0:i+=1
        end=i
        entry=bars[start]['close']; last_match=bars[end]['close']
        # Signal disappears when next observed bar is nonmatching. Exit uses that bar CLOSE.
        closed=end+1<n
        exit_i=end+1 if closed else end
        exit_price=bars[exit_i]['close']
        status='مغلقة' if closed else 'مفتوحة عند آخر بيانات السهم'
        def pct(p):return (100*(p/entry-1)) if entry and p is not None and np.isfinite(p) else None
        row={'filter_id':fid,'market':market,'symbol':sym,
             'start_utc':bars[start]['time'],'end_utc':bars[exit_i]['time'] if closed else None,
             'status':status,'entry_close':entry,'exit_close':exit_price if closed else None,
             'return_pct':pct(exit_price),'bars_matched':end-start+1,
             'bars_available_after':n-start-1,
             'max_rise_during_match_pct':pct(max((x['high'] for x in bars[start+1:end+1] if x['high'] is not None),default=None))}
        for h in HORIZONS:
            future=[x['high'] for x in bars[start+1:min(n,start+h+1)] if x['high'] is not None and np.isfinite(x['high'])]
            row[f'max_rise_{h}_pct']=pct(max(future)) if future else None
            row[f'max_rise_{h}_complete']=n-start-1>=h
        out.append(row);i+=1
    return out

def _source_sql(con, market, selected):
    indicators=', '.join(f'h."{k}" AS "{k}"' for k in selected)
    extra=(', '+indicators) if indicators else ''
    if market=='us':
        return f'''SELECT h.symbol,h.reference_utc AS time,b.high,b.close{extra}
        FROM {TABLE} h LEFT JOIN us_all_seven_hourly_bars b
        ON b.symbol=h.symbol AND b.bar_time=h.reference_utc
        WHERE h.market=:market ORDER BY h.symbol,h.reference_utc'''
    from monitor.gann_analysis import _daily_table
    tbl=_daily_table(con); fmt=con.get_bind().dialect.identifier_preparer
    names={c.name.lower():c for c in tbl.c}
    def pick(*choices):return next((names[n] for n in choices if n in names),None)
    sy=pick('symbol','ticker','sym');dt=pick('session_date','date','d');hi=pick('h','high');cl=pick('c','close')
    if any(x is None for x in [sy,dt,hi,cl]):raise RuntimeError('EGX source requires symbol,date,high,close')
    tn=fmt.format_table(tbl)
    return f'''SELECT h.symbol,h.reference_utc AS time,b.{fmt.quote(hi.name)} AS high,
        b.{fmt.quote(cl.name)} AS close{extra}
        FROM {TABLE} h LEFT JOIN {tn} b
        ON b.{fmt.quote(sy.name)}=h.symbol
        AND CAST(b.{fmt.quote(dt.name)} AS DATE)=CAST(h.reference_utc AS DATE)
        WHERE h.market=:market ORDER BY h.symbol,h.reference_utc'''

def build(market, selected, settings, progress_every=250):
    ensure(); fid=fingerprint(selected,settings)
    if market=='sp500':raise ValueError('Build --market us, then S&P 500 uses US records with membership filter')
    if market not in ('us','egypt'):raise ValueError(market)
    with database().begin() as con:
        con.execute(text(f'DELETE FROM {LEDGER} WHERE market=:m AND filter_id=:fid'),{'m':market,'fid':fid})
    insert_cols=['filter_id','market','symbol','start_utc','end_utc','status','entry_close','exit_close','return_pct','max_rise_during_match_pct','bars_matched','bars_available_after']+[f'max_rise_{h}_{s}' for h in HORIZONS for s in ('pct','complete')]
    stmt=text(f'INSERT INTO {LEDGER} ({",".join(insert_cols)}) VALUES ({",".join(":"+k for k in insert_cols)}) ON CONFLICT DO NOTHING')
    cnt=0;episodes=0;closed=0;prev=None;rows=[];batch=[]
    def flush_sym(sym, bars):
        nonlocal cnt,episodes,closed,batch
        if sym is None:return
        cnt+=1;events=_stock_events(sym,bars,selected,settings,market,fid)
        episodes+=len(events);closed+=sum(r['status']=='مغلقة' for r in events)
        batch.extend(events)
        if len(batch)>=800:
            with database().begin() as writer:writer.execute(stmt,batch)
            batch=[]
        if cnt%progress_every==0:print(f'{market} symbols={cnt} episodes={episodes} closed={closed}',flush=True)
    # Streaming avoids loading millions of historical snapshots into RAM.
    with database()() as con:
        statement=text(_source_sql(con,market,selected)).execution_options(stream_results=True,yield_per=5000)
        res=con.execute(statement,{'market':market}).mappings()
        for r in res:
            sym=r['symbol']
            if prev is not None and sym!=prev:flush_sym(prev,rows);rows=[]
            prev=sym
            price=r['close'];high=r['high']
            rows.append({'time':r['time'],'close':float(price) if price is not None else None,
                         'high':float(high) if high is not None else None,
                         **{k:r[k] for k in selected}})
        flush_sym(prev,rows)
    if batch:
        with database().begin() as writer:writer.execute(stmt,batch)
    return {'market':market,'filter_id':fid,'symbols':cnt,'episodes':episodes,'closed':closed,'open':episodes-closed}

def report(market, selected, settings, limit=200):
    ensure();fid=fingerprint(selected,settings)
    params={'f':fid,'m':'egypt' if market=='egypt' else 'us'}
    member_where='';membership=[]
    if market=='sp500':
        from monitor.sp500_seven_system import list_symbols
        membership=list_symbols() or ['___EMPTY___']
        member_where=' AND symbol IN :members';params['members']=membership
    from sqlalchemy import bindparam
    def q(sql):
        st=text(sql)
        return st.bindparams(bindparam('members',expanding=True)) if membership else st
    with database()() as con:
        agg=con.execute(q(f'''SELECT COUNT(*) total,
           COUNT(*) FILTER(WHERE status='مغلقة') closed,
           COUNT(*) FILTER(WHERE status<>'مغلقة') opened,
           COUNT(*) FILTER(WHERE status='مغلقة' AND return_pct>0) wins,
           AVG(return_pct) FILTER(WHERE status='مغلقة') avg_return,
           AVG(max_rise_during_match_pct) max_during
           FROM {LEDGER} WHERE filter_id=:f AND market=:m{member_where}'''),params).mappings().one()
        data=con.execute(q(f'''SELECT * FROM {LEDGER} WHERE filter_id=:f AND market=:m{member_where}
           ORDER BY CASE WHEN status<>'مغلقة' THEN 0 ELSE 1 END, start_utc DESC,symbol
           LIMIT {int(limit)}'''),params).mappings().all()
    return {'total':int(agg['total']),'closed':int(agg['closed']),'opened':int(agg['opened']),
            'wins':int(agg['wins']),'avg_return':agg['avg_return'],'avg_during':agg['max_during'],
            'rows':data,'filter_id':fid,'not_built':agg['total']==0}

if __name__=='__main__':
    from monitor.us_preburst_setup import FEATURES, DEFAULT_ACTIVE
    p=argparse.ArgumentParser();p.add_argument('--market',choices=['us','egypt'],required=True);p.add_argument('--build',action='store_true')
    p.add_argument('--query-string',default='',help='Optional URL query string of active setup filters')
    a=p.parse_args()
    if a.query_string:
        from urllib.parse import parse_qs, urlsplit
        from monitor.us_preburst_setup import parse_filters
        raw=a.query_string.split('?',1)[-1]
        args={k:v[-1] for k,v in parse_qs(raw,keep_blank_values=True).items()}
        selected, settings=parse_filters(args)
    else:
        settings={k:{'min':float(v[1]),'max':float(v[2])} for k,v in FEATURES.items()}
        selected=list(DEFAULT_ACTIVE)
    print(json.dumps(build(a.market,selected,settings) if a.build else report(a.market,selected,settings),default=str,ensure_ascii=False),flush=True)
