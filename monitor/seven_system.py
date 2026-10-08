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

        cadj=_pick(t,['adj_c','adj_close','adjusted_close'],False)
        cc=_pick(t,['c','close'])
        co=_pick(t,['o','open'],False)
        ch=_pick(t,['h','high'],False)
        cl=_pick(t,['l','low'],False)
        cv=_pick(t,['v','volume'],False)

        syms=[r[0] for r in s.execute(
            select(cs).where(cs.ilike('%.CA')).distinct().order_by(cs)
        ).all() if r and r[0]]

        for sym in syms:
            cols=[cd,cc]
            names=['date','raw_close']

            if cadj is not None:
                cols.append(cadj);names.append('adj_close')
            if co is not None:
                cols.append(co);names.append('open')
            if ch is not None:
                cols.append(ch);names.append('high')
            if cl is not None:
                cols.append(cl);names.append('low')
            if cv is not None:
                cols.append(cv);names.append('volume')

            rows=s.execute(select(*cols).where(cs==sym).order_by(cd)).all()
            if not rows:continue

            df=pd.DataFrame(rows,columns=names)
            df['date']=pd.to_datetime(df['date'],errors='coerce')
            df['raw_close']=pd.to_numeric(df['raw_close'],errors='coerce')

            if 'adj_close' in df:
                df['close']=pd.to_numeric(df['adj_close'],errors='coerce')
                df['close']=df['close'].where(df['close'].notna(),df['raw_close'])
            else:
                df['close']=df['raw_close']

            for col in ('open','high','low','volume'):
                if col in df:
                    df[col]=pd.to_numeric(df[col],errors='coerce')
                else:
                    df[col]=np.nan

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
    if 'raw_close' not in x.columns:
        x['raw_close']=x['close'].astype(float)
    else:
        x['raw_close']=pd.to_numeric(x['raw_close'],errors='coerce')
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



def _repeat_direction_from_candle(row):
    """
    Direction rule requested for Repeat Price+Time.

    SELL:
      1) negative candle: Close < Open
      2) positive candle, but weak close:
         (High - Close) > candle body abs(Close - Open)

    BUY:
      - positive candle not meeting the weak-close SELL rule.

    For a flat/doji candle, use wick dominance as a tie-breaker.
    """
    vals=[row.get('open'),row.get('high'),row.get('low'),row.get('raw_close')]
    if any(v is None or pd.isna(v) for v in vals):
        return 'UNKNOWN','OHLC غير متاح'

    o=float(row['open']);h=float(row['high']);l=float(row['low']);c=float(row['raw_close'])
    body=abs(c-o)
    upper=max(0.0,h-c)
    lower=max(0.0,c-l)

    if c < o:
        return 'SELL','شمعة سلبية'

    if c > o:
        if upper > body:
            return 'SELL','شمعة إيجابية لكن High-Close أكبر من جسم الشمعة'
        return 'BUY','شمعة إيجابية وإغلاق غير ضعيف'

    # Doji / flat candle: use the stronger rejection side.
    if upper > lower:
        return 'SELL','Doji والـ upper wick أكبر'
    if lower > upper:
        return 'BUY','Doji والـ lower wick أكبر'
    return 'NEUTRAL','Doji متعادل'


def _actual_latest_repeat_gap(df, level):
    """
    Number of trading sessions between the LAST TWO Double-7 repeat dates
    in the selected price/time cluster.
    """
    dates=[pd.Timestamp(x.strip()).normalize() for x in str(level.get('dates','')).split('|') if x.strip()]
    if len(dates)<2:
        return max(1,int(level.get('max_session_gap') or 1))

    calendar=[pd.Timestamp(d).normalize() for d in pd.to_datetime(df['date'])]
    pos={d:i for i,d in enumerate(calendar)}
    d1,d2=dates[-2],dates[-1]
    if d1 in pos and d2 in pos and pos[d2]>pos[d1]:
        return int(pos[d2]-pos[d1])
    return max(1,int(level.get('max_session_gap') or 1))


def _gap_matched_return(df, signal_date, gap_sessions, direction):
    """
    Evaluate the Repeat signal over the SAME number of sessions as its
    latest repeat gap.

    Example:
      latest repeat gap = 20 sessions
      -> evaluate from Repeat date close to close 20 sessions later.

    If the full horizon has not elapsed, return a pending partial result.
    Uses adjusted close for return calculations where available.
    """
    if df is None or df.empty:
        return {}

    x=df.sort_values('date').reset_index(drop=True)
    target=pd.Timestamp(signal_date).normalize()
    normalized=pd.to_datetime(x['date']).dt.normalize()
    matches=x.index[normalized==target].tolist()
    if not matches:
        return {}

    i=int(matches[-1])
    n=max(1,int(gap_sessions))
    j=min(i+n,len(x)-1)
    elapsed=j-i

    # 'close' is the existing adjusted-close series used by the EGX Seven logic.
    start=float(x.iloc[i]['close'])
    finish=float(x.iloc[j]['close'])
    raw=(finish/start-1.0)*100.0 if start else None

    if raw is None:
        aligned=None
    elif direction=='SELL':
        aligned=-raw
    else:
        aligned=raw

    return {
        'gap_target_sessions':n,
        'gap_elapsed_sessions':elapsed,
        'gap_complete':bool(elapsed>=n),
        'gap_end_date':pd.Timestamp(x.iloc[j]['date']).date().isoformat(),
        'gap_end_price':round(float(x.iloc[j]['raw_close']),4) if pd.notna(x.iloc[j].get('raw_close')) else round(finish,4),
        'gap_raw_return_pct':round(raw,2) if raw is not None else None,
        'gap_direction_return_pct':round(aligned,2) if aligned is not None else None,
    }


def repeat_price_time_screener(price_tolerance_pct=1.0, max_gap_sessions=20, recent_days=None):
    """
    Whole-EGX screening for the latest Repeat Price+Time per stock,
    plus candle-derived direction and gap-matched performance.
    """
    frames=_load_stocks()
    rows=[]

    latest_market_date=None
    for df in frames.values():
        if df is None or df.empty:continue
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
        if not levels:continue

        latest=max(levels,key=lambda z:pd.Timestamp(z['end_date']))
        end_date=pd.Timestamp(latest['end_date']).normalize()
        age=(latest_market_date-end_date).days if latest_market_date is not None else None
        if recent_days is not None and age is not None and age>int(recent_days):
            continue

        rr=df[pd.to_datetime(df['date']).dt.normalize()==end_date]
        if len(rr):
            candle=rr.iloc[-1]
            direction,reason=_repeat_direction_from_candle(candle)
            candle_open=round(float(candle['open']),4) if pd.notna(candle['open']) else None
            candle_high=round(float(candle['high']),4) if pd.notna(candle['high']) else None
            candle_low=round(float(candle['low']),4) if pd.notna(candle['low']) else None
            candle_close=round(float(candle['raw_close']),4) if pd.notna(candle['raw_close']) else None
        else:
            direction,reason='UNKNOWN','OHLC غير متاح'
            candle_open=candle_high=candle_low=candle_close=None

        actual_gap=_actual_latest_repeat_gap(df,latest)
        perf=_gap_matched_return(df,latest['end_date'],actual_gap,direction)

        current_close=round(float(df.iloc[-1]['raw_close']),4) if len(df) and pd.notna(df.iloc[-1]['raw_close']) else None
        current_date=pd.Timestamp(df.iloc[-1]['date']).date().isoformat() if len(df) else None

        rows.append({
            'symbol':sym,
            'latest_repeat_date':latest['end_date'],
            'first_repeat_date':latest['start_date'],
            'repeat_price':latest['avg_price'],
            'min_price':latest['min_price'],
            'max_price':latest['max_price'],
            'touches':latest['touches'],
            'max_session_gap':latest['max_session_gap'],
            'actual_repeat_gap_sessions':actual_gap,
            'price_spread_pct':latest['price_spread_pct'],
            'dates':latest['dates'],
            'prices':latest['prices'],
            'days_ago':age,
            'current_close':current_close,
            'current_date':current_date,

            'repeat_direction':direction,
            'direction_reason':reason,
            'candle_open':candle_open,
            'candle_high':candle_high,
            'candle_low':candle_low,
            'candle_close':candle_close,

            **perf,
        })

    rows.sort(
        key=lambda r:(pd.Timestamp(r['latest_repeat_date']),int(r['touches'])),
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
