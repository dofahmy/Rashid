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
        t=_daily_table(s)
        cs=_pick(t,['symbol','ticker','sym'])
        cd=_pick(t,['session_date','date','d','datetime','timestamp','ts'])
        cc=_pick(t,['adj_c','adj_close','adjusted_close'],False)
        if cc is None:cc=_pick(t,['c','close'])
        cv=_pick(t,['v','volume'],False)
        syms=[r[0] for r in s.execute(select(cs).where(cs.ilike('%.CA')).distinct().order_by(cs)).all() if r and r[0]]
        for sym in syms:
            cols=[cd,cc];names=['date','close']
            if cv is not None:cols.append(cv);names.append('volume')
            rows=s.execute(select(*cols).where(cs==sym).order_by(cd)).all()
            if not rows:continue
            df=pd.DataFrame(rows,columns=names)
            df['date']=pd.to_datetime(df['date'],errors='coerce')
            df['close']=pd.to_numeric(df['close'],errors='coerce')
            df['volume']=pd.to_numeric(df['volume'],errors='coerce') if 'volume' in df else np.nan
            df=df.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
            if len(df):out[str(sym).upper()]=df
    return out

def list_symbols():
    DB=database()
    with DB() as s:
        t=_daily_table(s);cs=_pick(t,['symbol','ticker','sym'])
        return sorted([r[0] for r in s.execute(select(cs).where(cs.ilike('%.CA')).distinct().order_by(cs)).all() if r and r[0]])

def _load_index():
    DB=database();idx,panel,source=_load_egx_panel(DB)
    x=idx[['d','c']].copy();x.columns=['date','close']
    x['date']=pd.to_datetime(x['date']);x['close']=pd.to_numeric(x['close'],errors='coerce');x['volume']=np.nan
    return x.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True),str(source)

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
        rows.append(dict(
            date=d,
            raw_close=np.nan,
            price_value=p/100,
            price_remainder=p%7 if pc[d] else None,
            price_signal7=bool(pc[d] and p%7==0),
            volume_value=v if vc[d] else None,
            volume_remainder=v%7 if vc[d] else None,
            volume_signal7=bool(vc[d] and v%7==0),
            price_count=pc[d],volume_count=vc[d]
        ))
    return pd.DataFrame(rows),{}

def _single(df,volume=True):
    x=df.copy().sort_values('date').reset_index(drop=True)
    x['raw_close']=x['close'].astype(float)
    x['price_units']=x.close.map(_units)

    pa=next((i for i,v in enumerate(x.price_units) if v is not None and v%7==0),None)
    va=None
    if volume:
        va=next((i for i,v in enumerate(x.volume) if pd.notna(v) and int(round(float(v)))%7==0),None)

    x['price_value']=np.nan;x['price_remainder']=np.nan;x['price_signal7']=False
    x['volume_value']=np.nan;x['volume_remainder']=np.nan;x['volume_signal7']=False

    if pa is not None:
        c=x.loc[pa:,'price_units'].fillna(0).astype('int64').cumsum()
        x.loc[pa:,'price_value']=c.to_numpy()/100
        x.loc[pa:,'price_remainder']=(c%7).to_numpy()
        x.loc[pa:,'price_signal7']=(c%7==0).to_numpy()

    if volume and va is not None:
        c=x.loc[va:,'volume'].fillna(0).round().astype('int64').cumsum()
        x.loc[va:,'volume_value']=c.to_numpy()
        x.loc[va:,'volume_remainder']=(c%7).to_numpy()
        x.loc[va:,'volume_signal7']=(c%7==0).to_numpy()

    meta={
        'price_anchor_date':pd.Timestamp(x.iloc[pa].date).date().isoformat() if pa is not None else None,
        'price_anchor_close':float(x.iloc[pa].close) if pa is not None else None,
        'volume_anchor_date':pd.Timestamp(x.iloc[va].date).date().isoformat() if va is not None else None,
        'volume_anchor_volume':int(round(float(x.iloc[va].volume))) if va is not None else None
    }
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

def _index_map():
    idx,source=_load_index()
    return {pd.Timestamp(r.date).normalize():float(r.close) for _,r in idx.iterrows()},source

def _stock_double_map(frames):
    """
    For market mode, count how many INDIVIDUAL stocks produced a strict
    price+volume Double 7 on each date using each stock's own anchor logic.
    """
    by_date=defaultdict(list)
    for sym,df in frames.items():
        s,_=_single(df,True)
        mask=s.price_signal7.fillna(False)&s.volume_signal7.fillna(False)
        for _,r in s[mask].iterrows():
            by_date[pd.Timestamp(r.date).normalize()].append({
                'symbol':sym,
                'close':round(float(r.raw_close),4) if pd.notna(r.raw_close) else None,
                'cum_price':round(float(r.price_value),2) if pd.notna(r.price_value) else None,
                'cum_volume':int(round(float(r.volume_value))) if pd.notna(r.volume_value) else None,
            })
    return by_date

def _chart_payload(price_df,signal_mask,value_col):
    """
    Server-prepared SVG coordinates. Markers are placed:
      - UP arrow below price when current Double-7 level > previous Double-7 level
      - DOWN arrow above price when current level < previous Double-7 level
      - diamond for first/equal level
    """
    z=price_df[['date',value_col]].copy().dropna()
    if z.empty:return {'points':[],'markers':[],'min':0,'max':1}
    z=z.reset_index(drop=True)
    vals=z[value_col].astype(float).to_numpy()
    mn=float(np.min(vals));mx=float(np.max(vals))
    if mx<=mn:mx=mn+1.0
    date_to_i={pd.Timestamp(d).normalize():i for i,d in enumerate(z.date)}
    points=[{'i':i,'date':pd.Timestamp(r.date).date().isoformat(),'value':float(r[value_col])} for i,r in z.iterrows()]
    markers=[];prev=None
    sigdates=set(pd.Timestamp(d).normalize() for d in price_df.loc[signal_mask,'date'])
    for d in sorted(sigdates):
        i=date_to_i.get(d)
        if i is None:continue
        v=float(z.iloc[i][value_col])
        direction='flat' if prev is None or abs(v-prev)<1e-12 else ('up' if v>prev else 'down')
        markers.append({'i':i,'date':d.date().isoformat(),'value':v,'direction':direction})
        prev=v
    return {'points':points,'markers':markers,'min':mn,'max':mx}

def run(scope='market',symbol=None,metric='both',date_mode='all',day=None,month=None,start=None,end=None,signals_only=False):
    frames=_load_stocks();source=None;stock_details={}
    index_by_date,index_source=_index_map()

    if scope=='market':
        full,meta=_market(frames);name='السوق المصري كله';source=index_source
        # Attach synthetic index level for every market session.
        full['index_value']=full.date.map(lambda d:index_by_date.get(pd.Timestamp(d).normalize(),np.nan))
        stock_details=_stock_double_map(frames)
        full['stocks_double7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),[])))
    elif scope=='stock':
        symbol=(symbol or '').upper()
        if symbol not in frames:raise ValueError('اختاري سهمًا صحيحًا.')
        full,meta=_single(frames[symbol],True);name=symbol
        full['index_value']=np.nan;full['stocks_double7_count']=np.nan
    elif scope=='index':
        idx,source=_load_index();full,meta=_single(idx,False);name='المؤشر / Market Proxy'
        full['index_value']=full['raw_close'];full['stocks_double7_count']=np.nan
    else:raise ValueError('نوع التحليل غير صحيح.')

    f=_filter(full,date_mode,day,month,start,end)
    pm=f.price_signal7.fillna(False)
    vm=f.volume_signal7.fillna(False) if 'volume_signal7' in f else pd.Series(False,index=f.index)
    dm=pm&vm
    mask=pm if metric=='price' else vm if metric=='volume' else dm
    sig=f[mask].copy()
    shown=sig if signals_only else f

    # chart uses the same selected date window, but plots actual price/index,
    # not cumulative 7 arithmetic.
    if scope=='stock':
        chart=_chart_payload(f,dm,'raw_close')
    elif scope=='market':
        chart=_chart_payload(f,dm,'index_value')
    elif scope=='index':
        chart=_chart_payload(f,pm,'raw_close')
    else:
        chart={'points':[],'markers':[],'min':0,'max':1}

    def recs(df):
        o=[]
        for _,r in df.iterrows():
            d=pd.Timestamp(r.date).normalize()
            o.append({
                'date':d.date().isoformat(),
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
                'stocks_double7_count':int(r.get('stocks_double7_count')) if pd.notna(r.get('stocks_double7_count',np.nan)) else None,
            })
        return o

    market_day_details={
        d.date().isoformat():rows for d,rows in stock_details.items()
    } if scope=='market' else {}

    return {
        'scope':scope,'symbol':symbol,'display_name':name,'source':source,'anchor':meta,
        'rows':recs(shown),'signal_rows':recs(sig),
        'total_rows':len(shown),'signal_count':len(sig),
        'double_signal_count':int(dm.sum()) if scope!='index' and len(f) else 0,
        'price_signal_count':int(pm.sum()) if len(f) else 0,
        'volume_signal_count':int(vm.sum()) if scope!='index' and len(f) else 0,
        'chart':chart,
        'market_day_details':market_day_details,
    }

def market_day_stock_details(day):
    frames=_load_stocks()
    by_date=_stock_double_map(frames)
    d=pd.Timestamp(day).normalize()
    return by_date.get(d,[])
