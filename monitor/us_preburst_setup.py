"""Read-only interactive screener, fed by a cache of as-of-time indicators.

Historical cache is extracted from preburst research CSV using only pre-start bars.
Current cache recomputes features using only candles in PostgreSQL through latest bar.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from core import database

TABLE='us_preburst_setup_snapshot_v2'
HISTORY_BARS=8  # As-of indicators at last eight closes, including latest
FEATURES={
 'rsi14':('RSI 14',27,43),
 'momentum5_pct':('Momentum 5 %',-9,0),
 'momentum20_pct':('Momentum 20 %',-22,-4),
 'close_vs_sma20_pct':('السعر بالنسبة لـ SMA20 %',-13,-3),
 'close_vs_sma50_pct':('السعر بالنسبة لـ SMA50 %',-20,-5),
 'bbz20':('Bollinger Z',-1.9,-0.85),
 'atr14_pct':('ATR 14 %',2.8,7),
 'return_20':('عائد 20 شمعة %',-20,-4),
 'position_20_pct':('الموقع بنطاق 20 شمعة %',2,24),
 'volume5_vs_50':('الفوليوم 5 ÷ السابق 45',0.5,1.7),
 'repeat_count_1pct_60':('تكرار السعر 1% خلال 60 شمعة',0,10),
}
DEFAULT_ACTIVE=['rsi14','momentum5_pct','momentum20_pct','close_vs_sma20_pct','return_20','position_20_pct']
COLS=list(FEATURES)

def ensure():
    with database().begin() as con:
        cols=', '.join(f'"{k}" DOUBLE PRECISION' for k in COLS)
        con.execute(text(f'''CREATE TABLE IF NOT EXISTS {TABLE} (
            mode TEXT NOT NULL, symbol TEXT NOT NULL, reference_utc TIMESTAMP NOT NULL,
            start_utc TIMESTAMP, peak_utc TIMESTAMP, rise_pct DOUBLE PRECISION,
            bars_to_peak INTEGER, {cols},
            PRIMARY KEY(mode,symbol,reference_utc))'''))
        con.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{TABLE}_mode ON {TABLE} (mode)'))

def finite(v):
    try:
        n=float(v)
        return n if pd.notna(n) and abs(n)!=float('inf') else None
    except (ValueError,TypeError): return None

def upsert(rows,mode):
    ensure()
    fields=['mode','symbol','reference_utc','start_utc','peak_utc','rise_pct','bars_to_peak']+COLS
    stmt=text(f'''INSERT INTO {TABLE} ({','.join('"'+x+'"' for x in fields)})
       VALUES ({','.join(':'+x for x in fields)})
       ON CONFLICT (mode,symbol,reference_utc) DO UPDATE SET
       {','.join('"'+x+'"=EXCLUDED."'+x+'"' for x in fields if x not in ('mode','symbol','reference_utc'))}''')
    with database().begin() as con:
        con.execute(text(f'DELETE FROM {TABLE} WHERE mode=:m'),{'m':mode})
        for offset in range(0,len(rows),250):con.execute(stmt,rows[offset:offset+250])

def historical(filepath):
    df=pd.read_csv(filepath)
    df=df[(df.bars_to_peak.between(21,250)) & (df.rise_pct>100)].copy()
    # Multiple episodes can share a start; one row for each stock/start time.
    df=df.drop_duplicates(['symbol','start_utc']).reset_index(drop=True)
    rows=[]
    for _,r in df.iterrows():
        item={'mode':'historical','symbol':str(r.symbol),'reference_utc':str(r.start_utc),
              'start_utc':str(r.start_utc),'peak_utc':str(r.peak_utc),
              'rise_pct':finite(r.rise_pct),'bars_to_peak':int(r.bars_to_peak)}
        for key in COLS:
            v=r.get(key)
            item[key]=finite(v)*100 if key=='position_20_pct' and finite(v) is not None else finite(v)
            if key=='position_20_pct':item[key]=finite(r.get('position_20'))*100 if finite(r.get('position_20')) is not None else None
        rows.append(item)
    upsert(rows,'historical')
    return {'mode':'historical','rows':len(rows),'unique_symbols':len(set(x['symbol'] for x in rows))}

def _features_row(sym, data, mode, offset=0):
    from monitor.us_preburst_research import features
    if len(data) < 22 + offset: return None
    selected=data[:len(data)-offset] if offset else data
    df=pd.DataFrame(selected, columns=['date','open','high','low','close','volume'])
    for col in ('open','high','low','close','volume'):
        df[col]=pd.to_numeric(df[col],errors='coerce')
    vals=features(df,len(df))
    item={'mode':mode,'symbol':sym,'reference_utc':str(selected[-1][0]),
          'start_utc':None,'peak_utc':None,'rise_pct':None,'bars_to_peak':None}
    for key in COLS:
        v=finite(vals.get('position_20' if key=='position_20_pct' else key))
        item[key]=v*100 if key=='position_20_pct' and v is not None else v
    return item

def _build_market(mode, table_query, timeframe):
    """Persist latest eight indicator snapshots to support true transition detection.

    Each observation is computed using only bars available up through its timestamp.
    No assumption is made that any symbol has consecutive UTC hours: consecutive
    observations refer to consecutive *stored market bars*.
    """
    buckets={f'{mode}_hist_{n}':[] for n in range(HISTORY_BARS)}
    count=0; prev=None;buf=[]
    def process(sym,data):
        nonlocal count
        count+=1
        for n in range(HISTORY_BARS):
            row=_features_row(sym,data,f'{mode}_hist_{n}',offset=n)
            if row:buckets[f'{mode}_hist_{n}'].append(row)
        if count%500==0:print(f'{mode} preburst history: {count} symbols',flush=True)
    with database()() as con:
        for r in con.execute(text(table_query)):
            sym=str(r[0])
            if prev is not None and prev!=sym:process(prev,buf);buf=[]
            prev=sym;buf.append(tuple(r[1:]))
        if prev is not None:process(prev,buf)
    # Keep V3 mode names for backwards compatibility with historical data/UI.
    # Replace all snapshots transactionally within the upsert function.
    for n in range(HISTORY_BARS):
        key=f'{mode}_hist_{n}'
        compat_mode=mode if n==0 else (f'{mode}_prev' if n==1 else key)
        for row in buckets[key]:row['mode']=compat_mode
        upsert(buckets[key],compat_mode)
    return {'market':mode,'rows':len(buckets[f'{mode}_hist_0']),
            'previous_rows':len(buckets[f'{mode}_hist_1']),
            'history_depth':HISTORY_BARS,'timeframe':timeframe}

def current():
    return _build_market('current','SELECT symbol,bar_time,open,high,low,close,volume FROM us_all_seven_hourly_bars ORDER BY symbol,bar_time','hourly')

def egypt_current():
    from monitor.gann_analysis import _daily_table
    from sqlalchemy import select
    buckets={f'egypt_hist_{n}':[] for n in range(HISTORY_BARS)}
    with database()() as con:
        tbl=_daily_table(con);names={c.name.lower():c for c in tbl.c}
        def pick(*opts):
            for opt in opts:
                if opt in names:return names[opt]
            return None
        cs=pick('symbol','ticker','sym'); cd=pick('session_date','date','d')
        cc=pick('c','close');ch=pick('h','high');cl=pick('l','low')
        co=pick('o','open');cv=pick('v','volume')
        if not all(x is not None for x in [cs,cd,cc,ch,cl]):
            raise RuntimeError('EGX daily table is missing required symbol/date/OHLC columns')
        fields=[cs,cd,co if co is not None else cc,ch,cl,cc,cv if cv is not None else cc]
        result=con.execute(select(*fields).where(cs.ilike('%.CA')).order_by(cs,cd))
        count=0;prev=None;buf=[]
        def process(sym,data):
            nonlocal count
            count+=1
            for n in range(HISTORY_BARS):
                row=_features_row(sym,data,f'egypt_hist_{n}',offset=n)
                if row:buckets[f'egypt_hist_{n}'].append(row)
            if count%500==0:print(f'egypt preburst history: {count} symbols',flush=True)
        for r in result:
            sym=str(r[0])
            if prev is not None and prev!=sym:process(prev,buf);buf=[]
            prev=sym;buf.append(tuple(r[1:]))
        if prev is not None:process(prev,buf)
    for n in range(HISTORY_BARS):
        key=f'egypt_hist_{n}';compat_mode='egypt' if n==0 else ('egypt_prev' if n==1 else key)
        for row in buckets[key]:row['mode']=compat_mode
        upsert(buckets[key],compat_mode)
    return {'market':'egypt','rows':len(buckets['egypt_hist_0']),
            'previous_rows':len(buckets['egypt_hist_1']),
            'history_depth':HISTORY_BARS,'timeframe':'daily'}

def parse_filters(args):
    selected=[];settings={}
    for key,(label,lo,hi) in FEATURES.items():
        active=args.get('use_'+key,'1' if key in DEFAULT_ACTIVE else '0')=='1'
        mn=float(args.get('min_'+key,lo));mx=float(args.get('max_'+key,hi))
        if mn>mx:raise ValueError(f'الحد الأدنى أكبر من الأعلى: {label}')
        settings[key]={'label':label,'min':mn,'max':mx,'active':active}
        if active:selected.append(key)
    return selected,settings

def dashboard(args):
    ensure()
    mode=args.get('mode','historical')
    market=args.get('market','us')
    if mode not in ('historical','current'):mode='historical'
    if market not in ('us','sp500','egypt'):market='us'
    selected,settings=parse_filters(args)
    if mode=='current':
        from monitor.us_preburst_v5_history import current_dashboard
        engine_mode='egypt' if market=='egypt' else 'us'
        result=current_dashboard(engine_mode,selected,settings,market)
        from monitor.us_preburst_ledger import report as historical_ledger_report
        ledger = historical_ledger_report(market, selected, settings)
        # Keep the same return structure used by the existing Flask view/template.
        return {'mode':mode,'market':market,'timeframe':'يومي' if market=='egypt' else 'ساعة',
          'settings':settings,'selected':selected,
          'baseline':{'rows':result['baseline'],'stocks':result['baseline']},
          'stats':{'rows':result['total'],'stocks':result['total']},
          'rows':result['rows'],'limited':result['total']>200,
          'new_count':result['new_confirmed'],
          'diagnostics':{'new_confirmed':result['new_confirmed'],'ongoing':result['ongoing'],
             'unknown':result['unknown'],'history_bars':'كل الشموع المتاحة','old_snapshot_rows':result['history_loaded']},
          'ledger':ledger,'by_date':result['by_date'],'stale_rows':result['stale_rows'],'stale_count':result['stale_count'], 'market_latest_utc':result['market_latest_utc'],'fresh_available':result['fresh_available'], 'note':''}
    storage='historical' if mode=='historical' else ('egypt' if market=='egypt' else 'current')
    prev_storage=storage+'_prev' if mode=='current' else None
    params={'mode':storage};where=['mode=:mode']
    for key in selected:
        where.append(f'"{key}" BETWEEN :lo_{key} AND :hi_{key}')
        params['lo_'+key]=settings[key]['min']; params['hi_'+key]=settings[key]['max']
    # S&P500 universe: membership from the project's database, not a hard-coded ticker list.
    if market=='sp500':
        from monitor.sp500_seven_system import list_symbols
        members=list_symbols()
        from sqlalchemy import bindparam
        member_clause='symbol IN :members'
        params['members']=members or ['___EMPTY___']
    elif market=='egypt' and mode=='historical':
        # The 471-stock retrospective study contains US stocks only.
        member_clause="symbol LIKE '%.CA'"
    else:member_clause='1=1'
    from sqlalchemy import bindparam
    def query(sql, bind_members=True):
        stmt=text(sql)
        if market=='sp500' and bind_members:
            stmt=stmt.bindparams(bindparam('members',expanding=True))
        return stmt
    filtered=' AND '.join(where)+' AND '+member_clause
    baseline_clause='mode=:mode AND '+member_clause
    # For current modes, rows are sorted with genuinely NEW matches at the top.
    with database()() as con:
        baseline=con.execute(query(f'SELECT COUNT(*) rows,COUNT(DISTINCT symbol) stocks FROM {TABLE} WHERE {baseline_clause}'),params).mappings().one()
        stat=con.execute(query(f'SELECT COUNT(*) rows,COUNT(DISTINCT symbol) stocks FROM {TABLE} WHERE {filtered}'),params).mappings().one()
        all_rows=con.execute(query(f"""SELECT symbol,reference_utc,start_utc,rise_pct,bars_to_peak,{','.join('"'+x+'"' for x in COLS)}
          FROM {TABLE} WHERE {filtered} ORDER BY reference_utc DESC,symbol ASC LIMIT 10000"""),params).mappings().all()
        history={}
        if prev_storage and all_rows:
            syms=list({r['symbol'] for r in all_rows})
            for n in range(1,HISTORY_BARS):
                hist_mode=f'{storage}_prev' if n==1 else f'{storage}_hist_{n}'
                older=con.execute(text(f"""SELECT symbol,reference_utc,{','.join('"'+x+'"' for x in COLS)} FROM {TABLE}
                  WHERE mode=:snapshot AND symbol IN :symbols""").bindparams(bindparam('symbols',expanding=True)),
                  {'snapshot':hist_mode,'symbols':syms}).mappings().all()
                history[n]={r['symbol']:r for r in older}
    def matches(r):
        return r is not None and all(r[k] is not None and settings[k]['min']<=r[k]<=settings[k]['max'] for k in selected)
    rows=[]
    for item in all_rows:
        r=dict(item)
        r['signal_started_utc']=None
        r['matching_bars']=None
        if prev_storage:
            old=history.get(1,{}).get(r['symbol'])
            if old is None:
                r['signal_status']='غير مؤكد — لا توجد شمعة سابقة'
                r['is_new']=False
            elif not matches(old):
                r['signal_status']='جديد مؤكد'
                r['is_new']=True
                r['signal_started_utc']=r['reference_utc']
                r['matching_bars']=1
            else:
                # Trace back contiguous matches through available snapshots.
                n_matches=1;earliest=r['reference_utc'];enough_history=True
                for n in range(1,HISTORY_BARS):
                    snapshot=history.get(n,{}).get(r['symbol'])
                    if snapshot is None:
                        enough_history=False;break
                    if not matches(snapshot):break
                    n_matches+=1;earliest=snapshot['reference_utc']
                r['is_new']=False
                r['signal_started_utc']=earliest
                r['matching_bars']=n_matches
                r['signal_status']=('مستمر (بدأ قبل نافذة التتبع)' if n_matches==HISTORY_BARS else
                                    'مستمر (بداية ضمن النافذة)' if enough_history else 'مستمر (تاريخ ناقص)')
        else:
            r['signal_status']='تاريخي';r['is_new']=False
        rows.append(r)
    if prev_storage:
        rows.sort(key=lambda r:(0 if r['is_new'] else (1 if r['signal_status'].startswith('مستمر') else 2),
                                -pd.Timestamp(r['signal_started_utc'] or r['reference_utc']).timestamp(),r['symbol']))
    diagnostics={'new_confirmed':sum(r['is_new'] for r in rows),
                 'ongoing':sum(r['signal_status'].startswith('مستمر') for r in rows),
                 'unknown':sum(r['signal_status'].startswith('غير مؤكد') for r in rows),
                 'old_snapshot_rows':len(history.get(1,{})) if prev_storage else 0,
                 'history_bars':HISTORY_BARS}
    return {'mode':mode,'market':market,'timeframe':'يومي' if market=='egypt' else 'ساعة',
      'settings':settings,'selected':selected,'baseline':dict(baseline),'stats':dict(stat),
      'rows':rows[:200],'limited':int(stat['rows'])>200,
      'new_count':sum(r['is_new'] for r in rows),'diagnostics':diagnostics,
      'note':'الوضع التاريخي مبني على الأسهم الأمريكية فقط' if mode=='historical' and market=='egypt' else ''}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--historical-features',default='',help='preburst_episodes_features.csv from retrospective run')
    p.add_argument('--build-current',action='store_true')
    p.add_argument('--build-egypt',action='store_true')
    a=p.parse_args()
    if not a.historical_features and not a.build_current and not a.build_egypt:p.error('specify --historical-features and/or --build-current')
    if a.historical_features:print(json.dumps(historical(a.historical_features),ensure_ascii=False),flush=True)
    if a.build_current:print(json.dumps(current(),ensure_ascii=False),flush=True)
    if a.build_egypt:print(json.dumps(egypt_current(),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
