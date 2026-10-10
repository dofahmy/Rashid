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

def current():
    # Use the same technical-indicator formulas as the historical preburst study.
    from monitor.us_preburst_research import features
    rows=[];prev=None;buf=[];count=0
    def process(sym,data):
        nonlocal count
        count+=1
        if len(data)<21:return
        df=pd.DataFrame(data,columns=['date','open','high','low','close','volume'])
        for col in ('open','high','low','close','volume'):df[col]=pd.to_numeric(df[col],errors='coerce')
        feats=features(df,len(df))
        item={'mode':'current','symbol':sym,'reference_utc':str(data[-1][0]),'start_utc':None,'peak_utc':None,'rise_pct':None,'bars_to_peak':None}
        for key in COLS:
            if key=='position_20_pct':
                v=finite(feats.get('position_20'))
                item[key]=v*100 if v is not None else None
            else:item[key]=finite(feats.get(key))
        rows.append(item)
        if count%500==0:print(f'Current preburst features: {count} stocks',flush=True)
    with database()() as con:
        for r in con.execute(text('SELECT symbol,bar_time,open,high,low,close,volume FROM us_all_seven_hourly_bars ORDER BY symbol,bar_time')):
            sym=str(r[0]);
            if prev is not None and prev!=sym:process(prev,buf);buf=[]
            prev=sym;buf.append(tuple(r[1:]))
        if prev is not None:process(prev,buf)
    upsert(rows,'current')
    return {'mode':'current','rows':len(rows),'unique_symbols':len(rows)}

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
    if mode not in ('historical','current'):mode='historical'
    selected,settings=parse_filters(args)
    where=['mode=:mode'];params={'mode':mode}
    for key in selected:
        # Missing feature means the row cannot meet the enabled requirement.
        where.append(f'"{key}" BETWEEN :lo_{key} AND :hi_{key}')
        params['lo_'+key]=settings[key]['min'];params['hi_'+key]=settings[key]['max']
    clause=' AND '.join(where)
    with database()() as con:
        baseline=con.execute(text(f'SELECT COUNT(*) rows, COUNT(DISTINCT symbol) stocks FROM {TABLE} WHERE mode=:mode'),{'mode':mode}).mappings().one()
        stat=con.execute(text(f'SELECT COUNT(*) rows, COUNT(DISTINCT symbol) stocks FROM {TABLE} WHERE {clause}'),params).mappings().one()
        rows=con.execute(text(f'''SELECT symbol,reference_utc,start_utc,rise_pct,bars_to_peak,{','.join('"'+x+'"' for x in COLS)}
          FROM {TABLE} WHERE {clause} ORDER BY {'rise_pct DESC NULLS LAST,' if mode=='historical' else ''} symbol,reference_utc DESC LIMIT 200'''),params).mappings().all()
    return {'mode':mode,'settings':settings,'selected':selected,'baseline':dict(baseline),'stats':dict(stat),
      'rows':[dict(r) for r in rows],'limited':int(stat['rows'])>len(rows)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--historical-features',default='',help='preburst_episodes_features.csv from retrospective run')
    p.add_argument('--build-current',action='store_true')
    a=p.parse_args()
    if not a.historical_features and not a.build_current:p.error('specify --historical-features and/or --build-current')
    if a.historical_features:print(json.dumps(historical(a.historical_features),ensure_ascii=False),flush=True)
    if a.build_current:print(json.dumps(current(),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
