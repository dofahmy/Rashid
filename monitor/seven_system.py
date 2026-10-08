from __future__ import annotations
from collections import defaultdict
import numpy as np, pandas as pd
from sqlalchemy import select
from core import database
from monitor.gann_analysis import _daily_table, _load_egx_panel
from monitor.seven_forward_20d import forward_20d

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
        co=_pick(t,['adj_o','adjusted_open','adj_open','o','open'],False)
        cv=_pick(t,['v','volume'],False)
        ch=_pick(t,['adj_h','adjusted_high','adj_high','h','high'],False)
        cl=_pick(t,['adj_l','adjusted_low','adj_low','l','low'],False)
        syms=[r[0] for r in s.execute(select(cs).where(cs.ilike('%.CA')).distinct().order_by(cs)).all() if r and r[0]]
        for sym in syms:
            cols=[cd,cc];names=['date','close']
            if co is not None:cols.append(co);names.append('open')
            if cv is not None:cols.append(cv);names.append('volume')
            if ch is not None:cols.append(ch);names.append('high')
            if cl is not None:cols.append(cl);names.append('low')
            rows=s.execute(select(*cols).where(cs==sym).order_by(cd)).all()
            if not rows:continue
            df=pd.DataFrame(rows,columns=names)
            df['date']=pd.to_datetime(df['date'],errors='coerce')
            df['close']=pd.to_numeric(df['close'],errors='coerce')
            df['volume']=pd.to_numeric(df['volume'],errors='coerce') if 'volume' in df else np.nan
            for ohlc_name in ('open','high','low'):
                if ohlc_name in df:df[ohlc_name]=pd.to_numeric(df[ohlc_name],errors='coerce')
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


def _repeat_double7_levels(df, price_tolerance_pct=1.0, max_gap_sessions=20):
    """
    Find SAME-STOCK repeated Double-7 price levels that are close in BOTH
    price and time.

    Default:
      price within 1.0%
      next Double-7 occurrence within 20 stock sessions
    """
    if df is None or df.empty:
        return []

    x=df.copy().sort_values('date').reset_index(drop=True)
    mask=x.price_signal7.fillna(False)&x.volume_signal7.fillna(False)
    sig=x[mask].copy()
    if len(sig)<2:
        return []

    rows=[]
    for i,r in sig.iterrows():
        close=float(r.raw_close if 'raw_close' in r and pd.notna(r.raw_close) else r.close)
        rows.append({'date':pd.Timestamp(r.date),'close':close,'session_pos':int(i)})

    def near_price(a,b):
        if a<=0 or b<=0:return False
        mid=(a+b)/2.0
        return abs(a-b)/mid*100.0 <= float(price_tolerance_pct)

    clusters=[]
    current=[rows[0]]

    for r in rows[1:]:
        prev=current[-1]
        gap=r['session_pos']-prev['session_pos']
        if gap<=int(max_gap_sessions) and near_price(r['close'],prev['close']):
            current.append(r)
        else:
            if len(current)>=2:clusters.append(current)
            current=[r]
    if len(current)>=2:clusters.append(current)

    out=[]
    for n,c in enumerate(clusters,1):
        prices=[z['close'] for z in c]
        dates=[z['date'] for z in c]
        gaps=[c[j]['session_pos']-c[j-1]['session_pos'] for j in range(1,len(c))]
        out.append({
            'cluster_id':n,
            'start_date':dates[0].date().isoformat(),
            'end_date':dates[-1].date().isoformat(),
            'touches':len(c),
            'avg_price':round(float(np.mean(prices)),4),
            'min_price':round(float(np.min(prices)),4),
            'max_price':round(float(np.max(prices)),4),
            'price_spread_pct':round((max(prices)-min(prices))/np.mean(prices)*100.0,3) if np.mean(prices) else None,
            'max_session_gap':max(gaps) if gaps else 0,
            'dates':' | '.join(z['date'].date().isoformat() for z in c),
            'prices':' | '.join(f"{z['close']:.4f}" for z in c),
        })
    return out

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

def _stock_signal_map(frames):
    """
    Per market date, track INDIVIDUAL-stock 7 signals:
      price_symbols  = stocks whose own cumulative price signal is 7
      volume_symbols = stocks whose own cumulative volume signal is 7
      double_symbols = intersection on the same date
    """
    by_date=defaultdict(lambda:{'price_symbols':[],'volume_symbols':[],'double_symbols':[]})
    for sym,df in frames.items():
        s,_=_single(df,True)
        for _,r in s.iterrows():
            d=pd.Timestamp(r.date).normalize()
            price7=bool(r.get('price_signal7',False))
            volume7=bool(r.get('volume_signal7',False))
            detail={
                'symbol':sym,
                'close':round(float(r.raw_close),4) if pd.notna(r.raw_close) else None,
                'cum_price':round(float(r.price_value),2) if pd.notna(r.price_value) else None,
                'cum_volume':int(round(float(r.volume_value))) if pd.notna(r.volume_value) else None,
            }
            if price7: by_date[d]['price_symbols'].append(detail)
            if volume7: by_date[d]['volume_symbols'].append(detail)
            if price7 and volume7: by_date[d]['double_symbols'].append(detail)
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


def repeat_price_time_screener(price_tolerance_pct=1.0, max_gap_sessions=20, recent_days=None):
    """
    Screen the whole EGX universe for the LATEST repeated same-stock Double-7
    price/time cluster.

    Returns ONE latest qualifying cluster per stock, sorted by latest date DESC.

    Optional recent_days:
      if set, keep only stocks whose latest qualifying cluster ended within
      recent_days calendar days from the latest market date in the database.
    """
    frames=_load_stocks()
    rows=[]

    latest_market_date=None
    for df in frames.values():
        if df is None or df.empty:
            continue
        d=pd.to_datetime(df['date'],errors='coerce').dropna()
        if len(d):
            mx=pd.Timestamp(d.max()).normalize()
            latest_market_date=mx if latest_market_date is None or mx>latest_market_date else latest_market_date

    for sym,df in frames.items():
        single,_=_single(df,True)
        levels=_repeat_double7_levels(
            single,
            price_tolerance_pct=price_tolerance_pct,
            max_gap_sessions=max_gap_sessions,
        )
        if not levels:
            continue

        # latest qualifying cluster for this stock
        latest=max(levels,key=lambda z:pd.Timestamp(z['end_date']))

        end_date=pd.Timestamp(latest['end_date']).normalize()
        if recent_days is not None and latest_market_date is not None:
            age=(latest_market_date-end_date).days
            if age>int(recent_days):
                continue
        else:
            age=(latest_market_date-end_date).days if latest_market_date is not None else None

        # current/latest stock price for context
        current_close=None
        current_date=None
        if len(df):
            current_close=float(df.iloc[-1]['close'])
            current_date=pd.Timestamp(df.iloc[-1]['date']).date().isoformat()

        rows.append({
            'symbol':sym,
            'latest_repeat_date':latest['end_date'],
            'first_repeat_date':latest['start_date'],
            'repeat_price':latest['avg_price'],
            'min_price':latest['min_price'],
            'max_price':latest['max_price'],
            'touches':latest['touches'],
            'max_session_gap':latest['max_session_gap'],
            'price_spread_pct':latest['price_spread_pct'],
            'dates':latest['dates'],
            'prices':latest['prices'],
            'days_ago':age,
            **forward_20d(df,latest['end_date'],latest['avg_price']),
            'current_close':round(current_close,4) if current_close is not None else None,
            'current_date':current_date,
            'current_vs_repeat_pct':(
                round((current_close/latest['avg_price']-1.0)*100.0,2)
                if current_close is not None and latest['avg_price'] not in (None,0)
                else None
            ),
        })

    rows.sort(
        key=lambda r:(
            pd.Timestamp(r['latest_repeat_date']),
            int(r['touches']),
            -float(r['price_spread_pct'] or 0),
        ),
        reverse=True
    )

    return {
        'rows':rows,
        'count':len(rows),
        'latest_market_date':latest_market_date.date().isoformat() if latest_market_date is not None else None,
        'price_tolerance_pct':float(price_tolerance_pct),
        'max_gap_sessions':int(max_gap_sessions),
        'recent_days':recent_days,
    }

def run(scope='market',symbol=None,metric='both',date_mode='all',day=None,month=None,start=None,end=None,signals_only=False,repeat_price_tolerance_pct=1.0,repeat_max_gap_sessions=20):
    frames=_load_stocks();source=None;stock_details={};repeat_levels=[]
    index_by_date,index_source=_index_map()

    if scope=='market':
        full,meta=_market(frames);name='السوق المصري كله';source=index_source
        # Attach synthetic index level for every market session.
        full['index_value']=full.date.map(lambda d:index_by_date.get(pd.Timestamp(d).normalize(),np.nan))
        stock_details=_stock_signal_map(frames)
        full['stocks_price7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),{}).get('price_symbols',[])))
        full['stocks_volume7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),{}).get('volume_symbols',[])))
        full['stocks_double7_count']=full.date.map(lambda d:len(stock_details.get(pd.Timestamp(d).normalize(),{}).get('double_symbols',[])))
    elif scope=='stock':
        symbol=(symbol or '').upper()
        if symbol not in frames:raise ValueError('اختاري سهمًا صحيحًا.')
        full,meta=_single(frames[symbol],True);name=symbol
        repeat_levels=_repeat_double7_levels(
            full,
            price_tolerance_pct=repeat_price_tolerance_pct,
            max_gap_sessions=repeat_max_gap_sessions
        )
        full['index_value']=np.nan;full['stocks_price7_count']=np.nan;full['stocks_volume7_count']=np.nan;full['stocks_double7_count']=np.nan
    elif scope=='index':
        idx,source=_load_index();full,meta=_single(idx,False);name='المؤشر / Market Proxy'
        full['index_value']=full['raw_close'];full['stocks_price7_count']=np.nan;full['stocks_volume7_count']=np.nan;full['stocks_double7_count']=np.nan
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
                'stocks_price7_count':int(r.get('stocks_price7_count')) if pd.notna(r.get('stocks_price7_count',np.nan)) else None,
                'stocks_volume7_count':int(r.get('stocks_volume7_count')) if pd.notna(r.get('stocks_volume7_count',np.nan)) else None,
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
        'repeat_levels':repeat_levels,
        'repeat_price_tolerance_pct':float(repeat_price_tolerance_pct),
        'repeat_max_gap_sessions':int(repeat_max_gap_sessions),
        'market_day_details':market_day_details,
    }

def market_day_stock_details(day,kind='double'):
    frames=_load_stocks()
    by_date=_stock_signal_map(frames)
    d=pd.Timestamp(day).normalize()
    row=by_date.get(d,{'price_symbols':[],'volume_symbols':[],'double_symbols':[]})
    key={'price':'price_symbols','volume':'volume_symbols','double':'double_symbols'}.get(kind,'double_symbols')
    return row.get(key,[])
