from __future__ import annotations
from collections import defaultdict
import numpy as np, pandas as pd
from sqlalchemy import select
from core import database
from monitor.gann_analysis import _daily_table, _load_egx_panel
PRICE_SCALE=100

def _pick(t,names,required=True):
    m={c.name.lower():c for c in t.c}
    for n in names:
        if n.lower() in m:return m[n.lower()]
    if required:raise KeyError(f'Missing {names}; available={list(m)}')
    return None

def _units(v):
    try:
        x=float(v)
        return int(round(x*PRICE_SCALE)) if np.isfinite(x) else None
    except Exception:return None

def _load_stocks():
    DB=database();out={}
    with DB() as s:
        t=_daily_table(s);cs=_pick(t,['symbol','ticker','sym']);cd=_pick(t,['session_date','date','d','datetime','timestamp','ts'])
        cc=_pick(t,['adj_c','adj_close','adjusted_close'],False)
        if cc is None:
            cc=_pick(t,['c','close'])
        cv=_pick(t,['v','volume'],False)
        syms=[r[0] for r in s.execute(select(cs).where(cs.ilike('%.CA')).distinct().order_by(cs)).all() if r and r[0]]
        for sym in syms:
            cols=[cd,cc];names=['date','close']
            if cv is not None:cols.append(cv);names.append('volume')
            rows=s.execute(select(*cols).where(cs==sym).order_by(cd)).all()
            if not rows:continue
            df=pd.DataFrame(rows,columns=names);df['date']=pd.to_datetime(df['date'],errors='coerce');df['close']=pd.to_numeric(df['close'],errors='coerce')
            df['volume']=pd.to_numeric(df['volume'],errors='coerce') if 'volume' in df else np.nan
            df=df.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
            if len(df):out[str(sym).upper()]=df
    return out

def list_symbols():
    DB=database()
    with DB() as s:
        t=_daily_table(s)
        cs=_pick(t,['symbol','ticker','sym'])
        return sorted([
            r[0] for r in s.execute(
                select(cs).where(cs.ilike('%.CA')).distinct().order_by(cs)
            ).all() if r and r[0]
        ])

def _load_index():
    DB=database();idx,panel,source=_load_egx_panel(DB);x=idx[['d','c']].copy();x.columns=['date','close'];x['date']=pd.to_datetime(x['date']);x['close']=pd.to_numeric(x['close'],errors='coerce');x['volume']=np.nan
    return x.dropna(subset=['date','close']).sort_values('date').reset_index(drop=True),str(source)

def _market(frames):
    ps=defaultdict(int);vs=defaultdict(int);pc=defaultdict(int);vc=defaultdict(int)
    for df in frames.values():
        for _,r in df.iterrows():
            d=pd.Timestamp(r.date).normalize();u=_units(r.close)
            if u is not None:ps[d]+=u;pc[d]+=1
            if pd.notna(r.volume):vs[d]+=int(round(float(r.volume)));vc[d]+=1
    rows=[]
    for d in sorted(set(ps)|set(vs)):
        p=ps.get(d,0);v=vs.get(d,0)
        rows.append(dict(date=d,price_value=p/100,price_remainder=p%7 if pc[d] else None,price_signal7=bool(pc[d] and p%7==0),volume_value=v if vc[d] else None,volume_remainder=v%7 if vc[d] else None,volume_signal7=bool(vc[d] and v%7==0),price_count=pc[d],volume_count=vc[d]))
    return pd.DataFrame(rows),{}

def _single(df,volume=True):
    x=df.copy().sort_values('date').reset_index(drop=True);x['price_units']=x.close.map(_units)
    pa=next((i for i,v in enumerate(x.price_units) if v is not None and v%7==0),None)
    va=None
    if volume:
        va=next((i for i,v in enumerate(x.volume) if pd.notna(v) and int(round(float(v)))%7==0),None)
    x['price_value']=np.nan;x['price_remainder']=np.nan;x['price_signal7']=False;x['volume_value']=np.nan;x['volume_remainder']=np.nan;x['volume_signal7']=False
    if pa is not None:
        c=x.loc[pa:,'price_units'].fillna(0).astype('int64').cumsum();x.loc[pa:,'price_value']=c.to_numpy()/100;x.loc[pa:,'price_remainder']=(c%7).to_numpy();x.loc[pa:,'price_signal7']=(c%7==0).to_numpy()
    if volume and va is not None:
        c=x.loc[va:,'volume'].fillna(0).round().astype('int64').cumsum();x.loc[va:,'volume_value']=c.to_numpy();x.loc[va:,'volume_remainder']=(c%7).to_numpy();x.loc[va:,'volume_signal7']=(c%7==0).to_numpy()
    meta={'price_anchor_date':pd.Timestamp(x.iloc[pa].date).date().isoformat() if pa is not None else None,'price_anchor_close':float(x.iloc[pa].close) if pa is not None else None,'volume_anchor_date':pd.Timestamp(x.iloc[va].date).date().isoformat() if va is not None else None,'volume_anchor_volume':int(round(float(x.iloc[va].volume))) if va is not None else None}
    return x,meta

def _filter(x,mode,day,month,start,end):
    if x.empty:return x
    d=pd.to_datetime(x.date)
    if mode=='day' and day:x=x[d.dt.normalize()==pd.Timestamp(day).normalize()]
    elif mode=='month' and month:x=x[d.dt.to_period('M')==pd.Period(month,freq='M')]
    elif mode=='range':
        if start:x=x[d>=pd.Timestamp(start)];d=pd.to_datetime(x.date)
        if end:x=x[d<=pd.Timestamp(end)]
    return x.reset_index(drop=True)

def run(scope='market',symbol=None,metric='both',date_mode='all',day=None,month=None,start=None,end=None,signals_only=False):
    frames=_load_stocks();source=None
    if scope=='market':full,meta=_market(frames);name='السوق المصري كله'
    elif scope=='stock':
        symbol=(symbol or '').upper()
        if symbol not in frames:raise ValueError('اختاري سهمًا صحيحًا.')
        full,meta=_single(frames[symbol],True);name=symbol
    elif scope=='index':
        idx,source=_load_index();full,meta=_single(idx,False);name='المؤشر / Market Proxy'
    else:raise ValueError('نوع التحليل غير صحيح.')
    f=_filter(full,date_mode,day,month,start,end)
    pm=f.price_signal7.fillna(False);vm=f.volume_signal7.fillna(False) if 'volume_signal7' in f else pd.Series(False,index=f.index)
    mask=pm if metric=='price' else vm if metric=='volume' else (pm&vm)
    sig=f[mask].copy()
    shown=sig if signals_only else f
    def recs(df):
        o=[]
        for _,r in df.iterrows():
            o.append({'date':pd.Timestamp(r.date).date().isoformat(),'price_value':None if pd.isna(r.get('price_value')) else round(float(r.get('price_value')),2),'price_remainder':None if pd.isna(r.get('price_remainder')) else int(r.get('price_remainder')),'price_signal7':bool(r.get('price_signal7',False)),'volume_value':None if pd.isna(r.get('volume_value')) else int(round(float(r.get('volume_value')))),'volume_remainder':None if pd.isna(r.get('volume_remainder')) else int(r.get('volume_remainder')),'volume_signal7':bool(r.get('volume_signal7',False)),'double_signal7':bool(r.get('price_signal7',False) and r.get('volume_signal7',False)),'price_count':int(r.get('price_count')) if pd.notna(r.get('price_count',np.nan)) else None,'volume_count':int(r.get('volume_count')) if pd.notna(r.get('volume_count',np.nan)) else None})
        return o
    return {'scope':scope,'symbol':symbol,'display_name':name,'source':source,'anchor':meta,'rows':recs(shown),'signal_rows':recs(sig),'total_rows':len(shown),'signal_count':len(sig),'double_signal_count':int((pm&vm).sum()) if scope!='index' and len(f) else 0,'price_signal_count':int(pm.sum()) if len(f) else 0,'volume_signal_count':int(vm.sum()) if scope!='index' and len(f) else 0}
