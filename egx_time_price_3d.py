#!/usr/bin/env python3
"""EGX Time–Price historic touch dashboard, PostgreSQL-backed and paginated.
Numeric convention: 1x1 = confirmed pivot ATR14 / unit_days per trading session.
The FIRST confirmed high-low contact is recorded for each symbol / pivot / angle.
Three distinct signal types: two different angles on the same session, same angle from different confirmed anchors on the same session, or a second visit to the identical line after a non-touch session. No future pivot information is used.
"""
import argparse
import hmac
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from sqlalchemy import create_engine, text

RATIOS = {'1x2': .5, '1x1': 1., '2x1': 2.}
STATUS_AR = {'CROSS_UP_CLOSE':'اختراق صاعد','CROSS_DOWN_CLOSE':'كسر هابط',
             'TOUCH_ABOVE':'لمس وإغلاق فوق','TOUCH_BELOW':'لمس وإغلاق تحت'}
ENGINE = None
LOCK = threading.Lock()
STATE = {'busy':False,'message':'جاهز — اضغطي إعادة حساب التاريخ لبناء تصنيفات V10.1','error':None,'version':None}
OUTPUT = '/tmp/egx-time-price-3d'
UNIT_DAYS = 20.


def engine():
    global ENGINE
    if ENGINE is None:
        url=os.environ.get('DATABASE_URL')
        if not url: raise RuntimeError('DATABASE_URL not configured')
        ENGINE=create_engine(url,pool_pre_ping=True,pool_size=3,max_overflow=2)
    return ENGINE


def ensure_tables():
    with engine().begin() as con:
        con.execute(text('''CREATE TABLE IF NOT EXISTS egx_tp3d_runs (
            run_id BIGSERIAL PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ, market_day TEXT, stock_count INT DEFAULT 0,
            touch_count BIGINT DEFAULT 0, invalid_bars BIGINT DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'building')'''))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS raw_touch_count BIGINT DEFAULT 0'))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS first_touch_before_pair_filter BIGINT DEFAULT 0'))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS qualifying_symbol_days BIGINT DEFAULT 0'))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS retouch_symbol_days BIGINT DEFAULT 0'))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS build_version TEXT'))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS different_angle_symbol_days BIGINT DEFAULT 0'))
        con.execute(text('ALTER TABLE egx_tp3d_runs ADD COLUMN IF NOT EXISTS same_angle_anchor_symbol_days BIGINT DEFAULT 0'))
        con.execute(text('''CREATE TABLE IF NOT EXISTS egx_tp3d_signals_v10 (
            run_id BIGINT NOT NULL, signal_type TEXT NOT NULL,
            symbol TEXT NOT NULL, touch_date TEXT NOT NULL,
            first_touch_date TEXT, anchor_type TEXT NOT NULL, anchor_date TEXT NOT NULL,
            angle TEXT NOT NULL, status TEXT NOT NULL, level DOUBLE PRECISION NOT NULL,
            candle_low DOUBLE PRECISION NOT NULL, candle_high DOUBLE PRECISION NOT NULL,
            touch_close DOUBLE PRECISION NOT NULL, after10_close DOUBLE PRECISION,
            after10_date TEXT, return10_pct DOUBLE PRECISION,
            PRIMARY KEY(run_id,signal_type,symbol,touch_date,anchor_type,anchor_date,angle))'''))
        for col,typ in [('after20_close','DOUBLE PRECISION'),('after20_date','TEXT'),('return20_pct','DOUBLE PRECISION'),('current_close','DOUBLE PRECISION'),('latest_price_date','TEXT'),('return_current_pct','DOUBLE PRECISION')]:
            con.execute(text(f'ALTER TABLE egx_tp3d_signals_v10 ADD COLUMN IF NOT EXISTS {col} {typ}'))
        con.execute(text('''CREATE TABLE IF NOT EXISTS egx_tp3d_v11_checkpoint(symbol TEXT PRIMARY KEY,last_day TEXT NOT NULL,bar_count BIGINT NOT NULL,updated_at TIMESTAMPTZ DEFAULT now())'''))
        con.execute(text('''CREATE INDEX IF NOT EXISTS idx_egx_tp3d_v10_groups
                         ON egx_tp3d_signals_v10(run_id,signal_type,touch_date,anchor_type,status)'''))
        con.execute(text('''CREATE INDEX IF NOT EXISTS idx_egx_tp3d_v10_symbols
                         ON egx_tp3d_signals_v10(run_id,signal_type,touch_date,symbol)'''))


def active_run():
    with engine().connect() as con:
        r=con.execute(text("SELECT run_id,market_day,stock_count,touch_count,raw_touch_count,first_touch_before_pair_filter,qualifying_symbol_days,retouch_symbol_days,different_angle_symbol_days,same_angle_anchor_symbol_days,invalid_bars,finished_at FROM egx_tp3d_runs WHERE status='complete' AND build_version='v11' ORDER BY run_id DESC LIMIT 1")).mappings().first()
    return dict(r) if r else None


def read_bars():
    q=text('''SELECT symbol,session_date,o,h,l,c FROM market_candles_1d
              WHERE UPPER(symbol) LIKE '%.CA' AND session_date IS NOT NULL
              ORDER BY symbol,session_date''')
    with engine().connect() as con: df=pd.read_sql(q,con)
    df['session_date']=pd.to_datetime(df.session_date,errors='coerce').dt.normalize()
    for k in ('o','h','l','c'): df[k]=pd.to_numeric(df[k],errors='coerce')
    valid=(df.session_date.notna() & df[['o','h','l','c']].notna().all(axis=1) &
           (df[['o','h','l','c']]>0).all(axis=1) &
           (df.h>=df[['o','c','l']].max(axis=1)-1e-7) &
           (df.l<=df[['o','c','h']].min(axis=1)+1e-7))
    bad_count=int((~valid).sum())
    return df.loc[valid].drop_duplicates(['symbol','session_date'],keep='last'),bad_count


def pivots(a,kind,wing=10,min_atr=5.0):
    """Yield index, anchor-price, anchor-ATR; confirmed no earlier than idx+wing."""
    low=a['l']; high=a['h']; atr=a['atr']; n=len(low)
    for i in range(wing,n-wing):
        av=atr[i]
        if not np.isfinite(av) or av<=0:continue
        low_range=low[i-wing:i+wing+1];hi_range=high[i-wing:i+wing+1]
        if kind=='LOW':
            if low[i]!=low_range.min() or hi_range.max()-low[i]<min_atr*av or hi_range[wing+1:].max()-low[i]<2.5*av:continue
            yield i,float(low[i]),float(av)
        else:
            if high[i]!=hi_range.max() or high[i]-low_range.min()<min_atr*av or high[i]-low_range[wing+1:].min()<2.5*av:continue
            yield i,float(high[i]),float(av)


def tr_atr(g):
    pc=g.c.shift(1)
    tr=pd.concat((g.h-g.l,(g.h-pc).abs(),(g.l-pc).abs()),axis=1).max(axis=1)
    return tr.rolling(14,min_periods=14).mean().to_numpy(dtype=float)


def contact_event(sym,dates,a,kind,idx,angle,j,level,prev_level,first_date=None):
    close=float(a['c'][j]); prevclose=float(a['c'][j-1]);
    if prevclose<prev_level and close>level:status='CROSS_UP_CLOSE'
    elif prevclose>prev_level and close<level:status='CROSS_DOWN_CLOSE'
    elif close>=level:status='TOUCH_ABOVE'
    else:status='TOUCH_BELOW'
    n=len(dates)
    after=float(a['c'][j+10]) if j+10<n else None
    after20=float(a['c'][j+20]) if j+20<n else None
    current=float(a['c'][-1])
    return {'symbol':sym,'touch_date':str(dates[j]),'first_touch_date':first_date,
            'anchor_type':kind,'anchor_date':str(dates[idx]),'angle':angle,
            'status':status,'level':float(level),'candle_low':float(a['l'][j]),
            'candle_high':float(a['h'][j]),'touch_close':close,
            'after10_close':after,'after10_date':str(dates[j+10]) if after is not None else None,
            'return10_pct':(after/close-1)*100 if after is not None else None,
            'after20_close':after20,'after20_date':str(dates[j+20]) if after20 is not None else None,
            'return20_pct':(after20/close-1)*100 if after20 is not None else None,
            'current_close':current,'latest_price_date':str(dates[-1]),
            'return_current_pct':(current/close-1)*100}


def symbol_events(sym,g,market_day,unit_days,wing=10,diagnostics=None):
    """Return two independent candidate collections: first contacts and second separate contacts.

    Retest requires at least one full non-touch session between first and second contact.
    Both events are causal: pivot must be confirmed before either contact.
    """
    g=g[g.session_date<=pd.Timestamp(market_day)].sort_values('session_date').reset_index(drop=True)
    if len(g)<60 or g.session_date.iloc[-1].strftime('%Y-%m-%d')!=market_day:return [],[]
    n=len(g); dates=g.session_date.dt.strftime('%Y-%m-%d').to_numpy()
    a={k:g[k].to_numpy(dtype=float) for k in ('o','h','l','c')};a['atr']=tr_atr(g)
    firsts=[];retouches=[]
    idx_all=np.arange(n)
    for kind,sign in (('LOW',1),('HIGH',-1)):
        for idx,base,anchor_atr in pivots(a,kind,wing):
            first=idx+wing+1
            if first>=n:continue
            t=idx_all[first:]; lows=a['l'][first:]; highs=a['h'][first:]
            for angle,ratio in RATIOS.items():
                slope=sign*anchor_atr/unit_days*ratio
                levels=base+(t-idx)*slope
                touched=(levels>0)&(lows-1e-8<=levels)&(levels<=highs+1e-8)
                offsets=np.flatnonzero(touched)
                if diagnostics is not None:diagnostics['raw_contacts']+=int(len(offsets))
                if len(offsets)==0:continue
                fst=int(offsets[0]);j=int(t[fst]);first_date=str(dates[j])
                firsts.append(contact_event(sym,dates,a,kind,idx,angle,j,float(levels[fst]),float(levels[fst]-slope)))
                # Second visit MUST follow a non-contact day; contiguous contact is one episode.
                latter=offsets[(offsets>fst+1) & (~touched[offsets-1])]
                if len(latter):
                    off=int(latter[0]);k=int(t[off]);
                    retouches.append(contact_event(sym,dates,a,kind,idx,angle,k,float(levels[off]),float(levels[off]-slope),first_date=first_date))
    return firsts,retouches


INSERT = text('''INSERT INTO egx_tp3d_signals_v10
 (run_id,signal_type,symbol,touch_date,first_touch_date,anchor_type,anchor_date,angle,status,level,candle_low,candle_high,
 touch_close,after10_close,after10_date,return10_pct,after20_close,after20_date,return20_pct,current_close,latest_price_date,return_current_pct)
 VALUES (:run_id,:signal_type,:symbol,:touch_date,:first_touch_date,:anchor_type,:anchor_date,:angle,:status,:level,
 :candle_low,:candle_high,:touch_close,:after10_close,:after10_date,:return10_pct,:after20_close,:after20_date,:return20_pct,:current_close,:latest_price_date,:return_current_pct)
 ON CONFLICT(run_id,signal_type,symbol,touch_date,anchor_type,anchor_date,angle) DO NOTHING''')


def market_snapshot():
    'Fast DB checkpoint query.'
    with engine().connect() as con:
        return {r[0]:(str(r[1]),int(r[2])) for r in con.execute(text("""
            SELECT symbol,MAX(session_date),COUNT(*) FROM market_candles_1d
            WHERE UPPER(symbol) LIKE '%.CA' AND session_date IS NOT NULL
            GROUP BY symbol"""))}


def update_outcomes(con, run_id, sym, g):
    'Update outcomes.'
    dates=g.session_date.dt.strftime('%Y-%m-%d').tolist()
    mapping={d:i for i,d in enumerate(dates)}
    close=g.c.to_numpy(dtype=float)
    rows=con.execute(text("SELECT touch_date FROM egx_tp3d_signals_v10 WHERE run_id=:r AND symbol=:s GROUP BY touch_date"),{'r':run_id,'s':sym}).scalars().all()
    for day in rows:
        j=mapping.get(day)
        if j is None:continue
        base=close[j]
        a10=float(close[j+10]) if j+10<len(close) else None
        a20=float(close[j+20]) if j+20<len(close) else None
        cur=float(close[-1])
        con.execute(text("""UPDATE egx_tp3d_signals_v10 SET
            after10_close=:a10,after10_date=:d10,return10_pct=:r10,
            after20_close=:a20,after20_date=:d20,return20_pct=:r20,
            current_close=:cur,latest_price_date=:dc,return_current_pct=:rc
            WHERE run_id=:run AND symbol=:sym AND touch_date=:day"""),
            {'run':run_id,'sym':sym,'day':day,'a10':a10,'d10':dates[j+10] if a10 is not None else None,
             'r10':(a10/base-1)*100 if a10 is not None else None,
             'a20':a20,'d20':dates[j+20] if a20 is not None else None,
             'r20':(a20/base-1)*100 if a20 is not None else None,
             'cur':cur,'dc':dates[-1],'rc':(cur/base-1)*100})


def daily_update():
    'Only changed symbols are recalculated.'
    with LOCK:
        if STATE['busy']:raise RuntimeError('هناك تحديث جار بالفعل')
        STATE.update(busy=True,error=None,message='التحقق من جلسات السوق الجديدة ...')
    try:
        run=active_run()
        if not run:
            with LOCK:STATE['message']='لا يوجد بناء تاريخي V11؛ شغّلي البناء مرة واحدة بالأمر --full-build'
            return
        snap=market_snapshot()
        with engine().connect() as con:
            checkpoint={r[0]:(r[1],int(r[2])) for r in con.execute(text('SELECT symbol,last_day,bar_count FROM egx_tp3d_v11_checkpoint'))}
        changed=[sym for sym,v in snap.items() if checkpoint.get(sym)!=v]
        if not changed:
            with LOCK:STATE['message']='لا توجد شموع جديدة؛ لم يُنفذ أي فحص تاريخي أو كتابة إشارات'
            return
        processed=0
        for sym in changed:
            with engine().connect() as con:
                g=pd.read_sql(text('SELECT symbol,session_date,o,h,l,c FROM market_candles_1d WHERE symbol=:sym ORDER BY session_date'),con,params={'sym':sym})
            g['session_date']=pd.to_datetime(g.session_date,errors='coerce').dt.normalize()
            for c in ('o','h','l','c'):g[c]=pd.to_numeric(g[c],errors='coerce')
            good=(g.session_date.notna() & g[['o','h','l','c']].notna().all(axis=1) &(g[['o','h','l','c']]>0).all(axis=1) & (g.h>=g[['o','c','l']].max(axis=1)-1e-7)&(g.l<=g[['o','c','h']].min(axis=1)+1e-7))
            g=g[good].drop_duplicates('session_date',keep='last').reset_index(drop=True)
            if len(g)<60:continue
            d=g.session_date.iloc[-1].strftime('%Y-%m-%d')
            firsts,retouches=symbol_events(sym,g,d,UNIT_DAYS)
            events=qualified_events(firsts,retouches)
            with engine().begin() as con:
                # Per-stock atomic replacement: never exposes partially-written history for that stock.
                con.execute(text('DELETE FROM egx_tp3d_signals_v10 WHERE run_id=:r AND symbol=:s'),{'r':run['run_id'],'s':sym})
                for j in range(0,len(events),1000):
                    con.execute(INSERT,[{**e,'run_id':run['run_id']} for e in events[j:j+1000]])
                con.execute(text("""INSERT INTO egx_tp3d_v11_checkpoint(symbol,last_day,bar_count)
                  VALUES (:s,:d,:n) ON CONFLICT(symbol) DO UPDATE SET
                  last_day=excluded.last_day,bar_count=excluded.bar_count,updated_at=now()"""),
                  {'s':sym,'d':snap[sym][0],'n':snap[sym][1]})
            processed+=1
            with LOCK:STATE['message']=f'تم تحديث {processed}/{len(changed)} سهم تغيرت بياناته'
        with engine().begin() as con:
            con.execute(text("""UPDATE egx_tp3d_runs SET market_day=:day,
                touch_count=(SELECT COUNT(*) FROM egx_tp3d_signals_v10 WHERE run_id=:r),finished_at=NOW()
                WHERE run_id=:r"""),{'r':run['run_id'],'day':max((v[0] for v in snap.values()),default=run['market_day'])})
        with engine().begin() as con:
            grp=con.execute(text("""SELECT signal_type,COUNT(DISTINCT (symbol,touch_date)) AS ct FROM egx_tp3d_signals_v10
                       WHERE run_id=:r GROUP BY signal_type"""),{'r':run['run_id']}).all()
            counts={k:int(v) for k,v in grp}
            con.execute(text("""UPDATE egx_tp3d_runs SET
                 different_angle_symbol_days=:diff,same_angle_anchor_symbol_days=:same,retouch_symbol_days=:re
                 WHERE run_id=:r"""),{'r':run['run_id'],'diff':counts.get('DIFFERENT_ANGLES',0),
                   'same':counts.get('SAME_ANGLE_DIFF_ANCHORS',0),'re':counts.get('SAME_LINE_TWICE',0)})
        with LOCK:STATE['message']=f'اكتمل تحديث {processed} سهم تغيّر تاريخه من أصل {len(snap)}؛ لم نُعد فحص الأسهم غير المتغيرة'
    except Exception as e:
        with LOCK:STATE.update(error=str(e)[:500],message='فشل التحديث: '+str(e)[:180])
    finally:
        with LOCK:STATE['busy']=False


def qualified_events(firsts,retouches):
    ans=[];by_day={}
    for event in firsts:by_day.setdefault(event['touch_date'],[]).append(event)
    for items in by_day.values():
        unique={(r['anchor_type'],r['anchor_date'],r['angle']):r for r in items}
        if len(unique)!=2:continue
        pair=list(unique.values());typ='DIFFERENT_ANGLES' if len({x['angle'] for x in pair})==2 else 'SAME_ANGLE_DIFF_ANCHORS'
        ans.extend([{**x,'signal_type':typ} for x in pair])
    ans.extend([{**x,'signal_type':'SAME_LINE_TWICE'} for x in retouches])
    return ans


def rebuild():
    with LOCK:
        if STATE['busy']:raise RuntimeError('حساب آخر جارٍ بالفعل')
        STATE.update(busy=True,error=None,message='تحميل الشموع من PostgreSQL...')
    rid=None
    try:
        data,bad=read_bars()
        dates=data.groupby('symbol').session_date.max().dt.strftime('%Y-%m-%d')
        if dates.empty:raise RuntimeError('No valid Egyptian daily bars')
        market_day=str(dates.mode().iloc[0])
        with engine().begin() as con:
            rid=con.execute(text("INSERT INTO egx_tp3d_runs(market_day,invalid_bars,build_version) VALUES (:d,:bad,'v11') RETURNING run_id"),{'d':market_day,'bad':bad}).scalar_one()
        stock_count=0; event_count=0; raw_first_touch=0; pair_days=0;retouch_days=0;diff_angle_days=0;same_angle_anchor_days=0
        buffer=[]; diagnostics={'raw_contacts':0}
        for sym,g in data.groupby('symbol',sort=True):
            if len(g)<60 or g.session_date.max().strftime('%Y-%m-%d')!=market_day:continue
            stock_count+=1
            # Gather only this stock's first-touch events in memory. Avoid storing
            # non-qualifying days in PostgreSQL (large savings vs deleting later).
            stock_firsts,stock_retouches=symbol_events(sym,g,market_day,UNIT_DAYS,diagnostics=diagnostics)
            raw_first_touch+=len(stock_firsts)
            by_day={}
            for event in stock_firsts:by_day.setdefault(event['touch_date'],[]).append(event)
            for day,items in by_day.items():
                # Unique line identity includes pivot type/date and angular ratio.
                # A line can have the same angle as another line but a different anchor.
                line_items={}
                for event in items:
                    identity=(event['anchor_type'],event['anchor_date'],event['angle'])
                    line_items.setdefault(identity,event)
                if len(line_items)!=2:continue
                pair=list(line_items.values())
                angles={r['angle'] for r in pair}
                signal_type=('DIFFERENT_ANGLES' if len(angles)==2 else 'SAME_ANGLE_DIFF_ANCHORS')
                pair_days+=1
                if signal_type=='DIFFERENT_ANGLES':diff_angle_days+=1
                else:same_angle_anchor_days+=1
                for event in pair:
                    event['run_id']=rid;event['signal_type']=signal_type
                    buffer.append(event)
                    if len(buffer)>=4000:
                        with engine().begin() as con:con.execute(INSERT,buffer)
                        event_count+=len(buffer);buffer=[]
            retouch_days+=len({r['touch_date'] for r in stock_retouches})
            for event in stock_retouches:
                event['run_id']=rid;event['signal_type']='SAME_LINE_TWICE'
                buffer.append(event)
                if len(buffer)>=4000:
                    with engine().begin() as con:con.execute(INSERT,buffer)
                    event_count+=len(buffer);buffer=[]
            with LOCK:
                STATE['message']=(f'فحص {stock_count} سهم | زاويتان مختلفتان: {diff_angle_days:,} | نفس الزاوية من ارتكازين: {same_angle_anchor_days:,} '
                                  f'| الإشارات بعد الشرط: {event_count+len(buffer):,} '
                                  f'| إعادة لمس نفس الخط (سهم/يوم): {retouch_days:,} | أول لمسات قبل الشرط: {raw_first_touch:,}')
        if buffer:
            with engine().begin() as con:con.execute(INSERT,buffer)
            event_count+=len(buffer)
        with engine().begin() as con:
            con.execute(text("UPDATE egx_tp3d_runs SET status='complete',finished_at=NOW(),stock_count=:n,touch_count=:c,raw_touch_count=:raw,first_touch_before_pair_filter=:first,qualifying_symbol_days=:pairs,retouch_symbol_days=:retouch,different_angle_symbol_days=:diff,same_angle_anchor_symbol_days=:same WHERE run_id=:id"),{'n':stock_count,'c':event_count,'raw':diagnostics['raw_contacts'],'first':raw_first_touch,'pairs':pair_days,'retouch':retouch_days,'diff':diff_angle_days,'same':same_angle_anchor_days,'id':rid})
            # Keep previous successful run as rollback point, not all historical duplicates.
            old=[] # preserve older runs for safe rollback
            if old:
                con.execute(text('DELETE FROM egx_tp3d_signals_v10 WHERE run_id = ANY(:ids)'),{'ids':old})
                con.execute(text('DELETE FROM egx_tp3d_runs WHERE run_id = ANY(:ids)'),{'ids':old})
        snap=market_snapshot()
        with engine().begin() as con:
            for sym,(day,n) in snap.items():
                con.execute(text('''INSERT INTO egx_tp3d_v11_checkpoint(symbol,last_day,bar_count) VALUES (:s,:d,:n)
                  ON CONFLICT(symbol) DO UPDATE SET last_day=excluded.last_day,bar_count=excluded.bar_count,updated_at=now()'''),{'s':sym,'d':day,'n':n})
        with LOCK:STATE.update(message=f'اكتمل الفحص: {diff_angle_days:,} حالات زاويتين مختلفتين، {same_angle_anchor_days:,} حالات نفس الزاوية من ارتكازين، {retouch_days:,} حالات إعادة لمس، {event_count:,} سطر حتى {market_day}',version=rid)
    except Exception as exc:
        if rid is not None:
            with engine().begin() as con:con.execute(text("UPDATE egx_tp3d_runs SET status='failed' WHERE run_id=:id"),{'id':rid})
        with LOCK:STATE.update(error=str(exc)[:500],message='فشل الحساب: '+str(exc)[:200])
    finally:
        with LOCK:STATE['busy']=False


def auth(request):
    token=os.getenv('EGX_3D_REFRESH_TOKEN')
    return bool(token and hmac.compare_digest(request.headers.get('X-Refresh-Token',''),token))


def filters(args):
    period=args.get('period','30')
    if period not in ('30','90','180','365','all'):period='30'
    side=args.get('side','BOTH').upper()
    if side not in ('LOW','HIGH','BOTH'):side='BOTH'
    status=args.get('status','ALL').upper()
    if status not in (*STATUS_AR.keys(),'ALL'):status='ALL'
    angle=args.get('angle','ALL')
    if angle not in (*RATIOS.keys(),'ALL'):angle='ALL'
    signal=args.get('signal','ALL').upper()
    if signal not in ('ALL','DIFFERENT_ANGLES','SAME_ANGLE_DIFF_ANCHORS','SAME_LINE_TWICE'):signal='ALL'
    return period,side,status,angle,signal


def where_filters(run,args):
    period,side,status,angle,signal=filters(args)
    where=['run_id=:run'];params={'run':run['run_id']}
    if period!='all':
        cutoff=(pd.Timestamp(run['market_day'])-pd.Timedelta(days=int(period))).strftime('%Y-%m-%d')
        where.append('touch_date>=:cutoff');params['cutoff']=cutoff
    if side!='BOTH':where.append('anchor_type=:side');params['side']=side
    if status!='ALL':where.append('status=:status');params['status']=status
    if angle!='ALL':where.append('angle=:angle');params['angle']=angle
    if signal!='ALL':where.append('signal_type=:signal');params['signal']=signal
    return ' AND '.join(where),params


def bounded_int(s,default,lo,hi):
    try:return max(lo,min(hi,int(s)))
    except (ValueError,TypeError):return default


HTML = r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Time–Price Squaring — السوق المصري</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{margin:0;background:#f4f7fb;color:#172f46;font-family:Tahoma,Arial,sans-serif}header{background:#112b46;color:#fff;padding:18px 4%}main{max-width:1500px;margin:auto;padding:20px}article{background:#fff;border:1px solid #dce6ee;border-radius:13px;padding:18px;margin-bottom:16px}button,select,input{padding:10px;border:1px solid #b9c9d5;border-radius:8px;font:inherit}button{background:#137a82;color:#fff;cursor:pointer}button:disabled{opacity:.5;cursor:wait}.toolbar{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:10px}label{font-size:13px}label select{display:block;margin-top:6px}.muted{color:#637689;font-size:12px;line-height:1.9}.stat{display:inline-block;background:#e7f3ef;padding:9px;margin:4px;border-radius:7px}table{width:100%;border-collapse:collapse}td,th{text-align:right;border-bottom:1px solid #e2e9f0;padding:10px;white-space:nowrap}th{background:#edf3f7}.scroll{overflow:auto}.detail{background:#eef7fb;border:1px solid #a6cbd5;border-radius:10px;padding:12px;display:none;margin-bottom:15px}.pager{display:flex;align-items:center;gap:12px;justify-content:center;margin:12px 0}#chart{height:500px}</style></head><body>
<header><h2>Time–Price Squaring — السوق المصري</h2><p>تاريخ اللمس الحقيقي وعوائد 10 و20 جلسة وحتى الآن</p></header><main>
<article><div id="stats"></div><p class="muted">الزوايا بمعايرة ثابتة: 1×1 = ATR14 للارتكاز / 20 جلسة. لا يُستخدم الارتكاز قبل تأكيده بعشر جلسات. عمق 3D تمثيلي.</p><div class="toolbar"><button id="update">تحديث البيانات</button><button id="rebuild">فحص التحديثات</button><button onclick="location.reload()">تحديث الصفحة</button></div><div id="job" class="muted"></div></article>
<article><div class="toolbar"><label>مدة اللمسات<select id="period"><option value="30">30 يوم</option><option value="90">90 يوم</option><option value="180">6 شهور</option><option value="365">سنة</option><option value="all">كل التاريخ</option></select></label><label>نوع الإشارة<select id="signal"><option value="ALL">الأنواع الثلاثة</option><option value="DIFFERENT_ANGLES">زاويتان مختلفتان في نفس الجلسة</option><option value="SAME_ANGLE_DIFF_ANCHORS">نفس الزاوية من ارتكازين مختلفين</option><option value="SAME_LINE_TWICE">نفس الخط مرتين في يومين مختلفين</option></select></label><label>القمة / القاع<select id="side"><option value="BOTH">LOW و HIGH</option><option value="LOW">LOW — قاع</option><option value="HIGH">HIGH — قمة</option></select></label><label>حالة السهم<select id="status"><option value="ALL">كل الحالات</option><option value="CROSS_UP_CLOSE">اختراق صاعد</option><option value="CROSS_DOWN_CLOSE">كسر هابط</option><option value="TOUCH_ABOVE">لمس وإغلاق فوق</option><option value="TOUCH_BELOW">لمس وإغلاق تحت</option></select></label><label>نوع الخط الهندسي<select id="angle"><option value="ALL">كل الخطوط</option><option value="1x1">1×1</option><option value="1x2">1×2</option><option value="2x1">2×1</option></select></label><label>الحد الأدنى لعدد الأسهم في المجموعة<input id="minStocks" type="number" min="1" max="216" step="1" value="5" style="display:block;margin-top:6px;width:105px"></label><label>ترتيب المجموعات<select id="sort"><option value="date_desc">الأحدث أولًا</option><option value="date_asc">الأقدم أولًا</option><option value="count_desc">الأكثر أسهم أولًا</option><option value="count_asc">الأقل أسهم أولًا</option></select></label></div>
<div id="detail" class="detail"><div class="toolbar"><h3 id="detailTitle" style="flex:1"></h3><button id="closeDetail">إغلاق التفاصيل</button></div><div class="scroll"><table><thead><tr><th>السهم</th><th>نوع الإشارة</th><th>الخط</th><th>اللمسة الأولى</th><th>تاريخ الارتكاز (مرجع)</th><th>سعر الخط</th><th>إغلاق يوم اللمس</th><th>بعد 10 جلسات</th><th>عائد 10 جلسات</th><th>بعد 20 جلسة</th><th>عائد 20 جلسة</th><th>آخر إغلاق</th><th>تاريخه</th><th>العائد حتى الآن</th><th>حالة اللمس</th></tr></thead><tbody id="detailRows"></tbody></table></div><div class="pager"><button id="prevDetail">السابق</button><span id="detailPages"></span><button id="nextDetail">التالي</button></div></div>
<div class="toolbar" style="margin-top:15px"><h3 style="flex:1">أسهم عائد أول 20 جلسة أكبر من صفر وأقل من 10%</h3><button id="toggleAfter20" type="button" aria-expanded="false" aria-controls="after20Panel">إظهار نتائج 20 جلسة</button></div>
<div id="after20Panel" style="display:none">
<div class="toolbar"><label>ترتيب العائد من +20 للآن<select id="after20Sort"><option value="desc">الأعلى أولًا</option><option value="asc">الأقل أولًا</option><option value="date">الإشارة الأحدث</option></select></label></div>
<p class="muted">النتائج تتبع مدة اللمسات وباقي الفلاتر أعلاه. كل سهم يظهر مرة لكل تاريخ إشارة ونوع إشارة، حتى إذا لمس أكثر من خط. لا تظهر الإشارات التي لم تكتمل لها 20 جلسة.</p>
<div id="after20Count" class="muted"></div><div class="scroll"><table><thead><tr><th>السهم</th><th>تاريخ الإشارة</th><th>نوع الإشارة</th><th>المصدر</th><th>الحالة</th><th>إغلاق الإشارة</th><th>تاريخ +20</th><th>إغلاق +20</th><th>عائد 20 جلسة %</th><th>آخر إغلاق</th><th>تاريخ آخر سعر</th><th>العائد من +20 حتى الآن %</th></tr></thead><tbody id="after20Rows"></tbody></table></div>
<div class="pager"><button id="prevAfter20">السابق</button><span id="after20Pages"></span><button id="nextAfter20">التالي</button></div>
</div>
<div id="performance" class="scroll"></div><div id="resultCount" class="muted"></div><p class="muted">نوع 1: أول لمس لخطين بزاويتين مختلفتين في نفس الجلسة. نوع 2: أول لمس لخطين بنفس الزاوية من ارتكازين مختلفين في نفس الجلسة. نوع 3: ثاني لمس لنفس الخط بعد جلسة واحدة على الأقل دون لمس. عوائد 10 و20 جلسة وحتى الآن تبدأ من يوم الإشارة. قد يظهر السهم في أكثر من نوع.</p><div class="scroll"><table><thead><tr><th>تاريخ الإشارة</th><th>نوع الإشارة</th><th>مصدر الخط</th><th>الحالة</th><th>عدد الأسهم (اضغطي للتفاصيل)</th></tr></thead><tbody id="groups"></tbody></table></div><div class="pager"><button id="prevGroup">السابق</button><span id="groupPages"></span><button id="nextGroup">التالي</button></div></article>
<article><div class="toolbar"><h3 style="flex:1">شارت السهم — آخر 100 جلسة</h3><input id="ticker" placeholder="COMI.CA"><button id="loadChart">عرض الشارت 3D</button></div><div id="chart"></div></article></main>
<script>
const $=id=>document.getElementById(id),signalNames={DIFFERENT_ANGLES:'زاويتان مختلفتان',SAME_ANGLE_DIFF_ANCHORS:'نفس الزاوية من ارتكازين',SAME_LINE_TWICE:'نفس الخط مرتين'},names={CROSS_UP_CLOSE:'اختراق صاعد',CROSS_DOWN_CLOSE:'كسر هابط',TOUCH_ABOVE:'لمس وإغلاق فوق',TOUCH_BELOW:'لمس وإغلاق تحت'};
let page=1,detailPage=1,selected=null, groupTotal=0, detailTotal=0;
let after20Page=1,after20TotalPages=0;
let after20Visible=false;
const filt=()=>new URLSearchParams({period:$('period').value,signal:$('signal').value,side:$('side').value,status:$('status').value,angle:$('angle').value,sort:$('sort').value,min_stocks:$('minStocks').value});
const safe=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw Error(d.error||r.status);return d;}
async function groups(){try{performance();if(after20Visible)loadAfter20();let q=filt();q.set('page',page);const d=await api('/api/groups?'+q);groupTotal=d.pages;
$('resultCount').textContent=`${d.total} مجموعة | ${d.unique_stocks} سهم فريد ضمن الفلتر`;
$('groups').innerHTML=d.items.map(g=>`<tr><td>${g.touch_date}</td><td>${signalNames[g.signal_type]}</td><td>${g.anchor_type==='LOW'?'LOW — قاع':'HIGH — قمة'}</td><td>${names[g.status]}</td><td><button class="showDetail" data-day="${g.touch_date}" data-side="${g.anchor_type}" data-status="${g.status}" data-signal="${g.signal_type}">${g.stocks} سهم</button></td></tr>`).join('')||'<tr><td colspan="5">لا توجد نتائج بهذه الشروط.</td></tr>';
$('groupPages').textContent=`${page} / ${d.pages||1}`;$('prevGroup').disabled=page<=1;$('nextGroup').disabled=page>=d.pages;
document.querySelectorAll('.showDetail').forEach(b=>b.onclick=()=>{selected={day:b.dataset.day,side:b.dataset.side,status:b.dataset.status,signal:b.dataset.signal};detailPage=1;details();});
}catch(e){$('groups').innerHTML='<tr><td>'+safe(e.message)+'</td></tr>';}}
async function loadAfter20(){if(!after20Visible)return;try{
let q=filt();q.set('page',after20Page);q.set('after_sort',$('after20Sort').value);
const d=await api('/api/after20?'+q);after20TotalPages=d.pages;
$('after20Count').textContent=`${d.total} حالة سهم/يوم مطابقة | ${d.unique_stocks} سهم فريد | العائد من +20 حتى الآن يُحسب من إغلاق الجلسة العشرين`;
const pct=v=>v==null?'—':`${v>0?'+':''}${Number(v).toFixed(2)}%`;
$('after20Rows').innerHTML=d.items.map(x=>`<tr><td><a href="#chart" class="after20Chart" data-symbol="${safe(x.symbol)}">${safe(x.symbol)}</a></td><td>${safe(x.touch_date)}</td><td>${safe(signalNames[x.signal_type]||x.signal_type)}</td><td>${x.anchor_type==='LOW'?'LOW — قاع':'HIGH — قمة'}</td><td>${safe(names[x.status]||x.status)}</td><td>${Number(x.touch_close).toFixed(4)}</td><td>${safe(x.after20_date)}</td><td>${Number(x.after20_close).toFixed(4)}</td><td>${pct(x.return20_pct)}</td><td>${Number(x.current_close).toFixed(4)}</td><td>${safe(x.latest_price_date)}</td><td>${pct(x.return_since20_pct)}</td></tr>`).join('')||'<tr><td colspan="12">لا توجد أسهم تطابق هذه الشروط في المدة المختارة.</td></tr>';
$('after20Pages').textContent=`${after20Page} / ${d.pages||1}`;$('prevAfter20').disabled=after20Page<=1;$('nextAfter20').disabled=after20Page>=d.pages;
document.querySelectorAll('.after20Chart').forEach(x=>x.onclick=()=>{$('ticker').value=x.dataset.symbol;chart();});
}catch(e){$('after20Count').textContent='خطأ في عرض الأسهم: '+e.message;}}
$('toggleAfter20').onclick=()=>{
 after20Visible=!after20Visible;
 $('after20Panel').style.display=after20Visible?'block':'none';
 $('toggleAfter20').setAttribute('aria-expanded',String(after20Visible));
 $('toggleAfter20').textContent=after20Visible?'إخفاء نتائج 20 جلسة':'إظهار نتائج 20 جلسة';
 if(after20Visible)loadAfter20();
};
$('after20Sort').onchange=()=>{after20Page=1;if(after20Visible)loadAfter20();};
$('prevAfter20').onclick=()=>{if(after20Page>1){after20Page--;loadAfter20();}};
$('nextAfter20').onclick=()=>{if(after20Page<after20TotalPages){after20Page++;loadAfter20();}};
async function performance(){try{let d=await api('/api/performance?'+filt()); $('performance').innerHTML='<table><thead><tr><th>نوع الإشارة</th><th>سهم/يوم</th><th>10 جلسات: عينة / متوسط %</th><th>20 جلسة: عينة / متوسط %</th><th>حتى الآن: عينة / متوسط %</th></tr></thead><tbody>'+d.items.map(x=>`<tr><td>${signalNames[x.signal_type]}</td><td>${x.stock_days}</td><td>${x.complete10} / ${x.avg10??'—'}</td><td>${x.complete20} / ${x.avg20??'—'}</td><td>${x.current_count} / ${x.avg_current??'—'}</td></tr>`).join('')+'</tbody></table>'; }catch(e){$('performance').textContent=e.message;}}
async function details(){if(!selected)return;try{let q=new URLSearchParams({day:selected.day,side:selected.side,status:selected.status,signal:selected.signal,angle:$('angle').value,page:detailPage});let d=await api('/api/details?'+q);
$('detail').style.display='block';$('detailTitle').textContent=`${selected.day} | ${signalNames[selected.signal]} | ${selected.side==='LOW'?'LOW — قاع':'HIGH — قمة'} | ${names[selected.status]} | ${d.stocks} سهم`;
$('detailRows').innerHTML=d.items.map(x=>`<tr><td><a href="#chart" class="chartLink" data-sym="${safe(x.symbol)}">${safe(x.symbol)}</a></td><td>${signalNames[x.signal_type]}</td><td>${x.angle}</td><td>${x.first_touch_date||'—'}</td><td>${x.anchor_date}</td><td>${x.level.toFixed(4)}</td><td>${x.touch_close.toFixed(4)}</td><td>${x.after10_close==null?'قيد الانتظار':x.after10_close.toFixed(4)}</td><td>${x.return10_pct==null?'—':x.return10_pct.toFixed(2)+'%'}</td><td>${x.after20_close==null?'قيد الانتظار':x.after20_close.toFixed(4)}</td><td>${x.return20_pct==null?'—':x.return20_pct.toFixed(2)+'%'}</td><td>${x.current_close==null?'—':x.current_close.toFixed(4)}</td><td>${x.latest_price_date||'—'}</td><td>${x.return_current_pct==null?'—':x.return_current_pct.toFixed(2)+'%'}</td><td>${names[x.status]}</td></tr>`).join('');
$('detailPages').textContent=`${detailPage} / ${d.pages||1}`;detailTotal=d.pages;$('prevDetail').disabled=detailPage<=1;$('nextDetail').disabled=detailPage>=d.pages;
document.querySelectorAll('.chartLink').forEach(x=>x.onclick=()=>{$('ticker').value=x.dataset.sym;chart();});$('detail').scrollIntoView({behavior:'smooth',block:'start'});
}catch(e){alert(e.message);}}
$('closeDetail').onclick=()=>{$('detail').style.display='none';selected=null;};
for(let key of ['period','signal','side','status','angle','sort','minStocks'])$(key).onchange=()=>{after20Page=1;page=1;selected=null;$('detail').style.display='none';groups();};
$('prevGroup').onclick=()=>{page--;groups();};$('nextGroup').onclick=()=>{page++;groups();};$('prevDetail').onclick=()=>{detailPage--;details();};$('nextDetail').onclick=()=>{detailPage++;details();};
async function chart(){try{let d=await api('/api/chart?symbol='+encodeURIComponent($('ticker').value));let candles=d.candles, angles=d.angles;
if(typeof Plotly==='undefined')throw Error('مكتبة Plotly لم يتم تحميلها');
let start=candles[0].t,end=candles[candles.length-1].t,scale=angles.find(x=>x.angle==='1x1')?.slope||1,base=candles[0].c;
let traces=[{type:'scatter3d',mode:'lines',name:'الإغلاق',x:candles.map(x=>x.t-start),y:candles.map(x=>(x.c-base)/scale),z:candles.map(x=>0),line:{color:'#122e4d',width:6}}];
for(let a of angles){let first=Math.max(start,a.anchor_t);traces.push({type:'scatter3d',mode:'lines',name:`${a.kind} ${a.angle}`,x:[first-start,end-start],y:[first,end].map(t=>(a.base+a.sign*a.slope*a.ratio*(t-a.anchor_t)-base)/scale),z:[1,1]});}
let ys=traces.flatMap(x=>x.y),ymin=Math.min(...ys),ymax=Math.max(...ys),span=Math.max(end-start,ymax-ymin,1);
Plotly.react('chart',traces,{scene:{aspectmode:'manual',aspectratio:{x:(end-start)/span,y:(ymax-ymin)/span,z:.18},dragmode:false,camera:{eye:{x:1.5,y:1.5,z:1.05},projection:{type:'orthographic'}},xaxis:{title:'جلسات',range:[0,end-start]},yaxis:{title:'سعر بمعايرة ثابتة',range:[ymin,ymax]},zaxis:{range:[0,1.5]}},height:490,margin:{l:0,r:0,t:0,b:0}},{scrollZoom:false,displayModeBar:false,responsive:true});
}catch(e){$('chart').textContent=e.message;}}
$('loadChart').onclick=chart;
async function job(){try{let d=await api('/api/job');$('stats').innerHTML=d.run?`<span class="stat">آخر جلسة: ${d.run.market_day}</span><span class="stat">الأسهم: ${d.run.stock_count}</span><span class="stat">حالات زاويتين مختلفتين (سهم/يوم): ${(d.run.different_angle_symbol_days??0).toLocaleString()}</span><span class="stat">حالات نفس الزاوية من ارتكازين (سهم/يوم): ${(d.run.same_angle_anchor_symbol_days??0).toLocaleString()}</span><span class="stat">حالات إعادة لمس نفس الخط (سهم/يوم): ${(d.run.retouch_symbol_days??0).toLocaleString()}</span><span class="stat">سطور الإشارات المعتمدة: ${d.run.touch_count.toLocaleString()}</span><span class="stat">أول اللمسات قبل شرط الخطين: ${(d.run.first_touch_before_pair_filter??0).toLocaleString()}</span><span class="stat">كل التماسات قبل أول لمس: ${(d.run.raw_touch_count??0).toLocaleString()}</span><span class="stat">الشموع المستبعدة: ${d.run.invalid_bars.toLocaleString()}</span>`:'<span class="stat">السجل التاريخي لم يُبنَ بعد</span>';
$('job').textContent=d.message+(d.error?' | '+d.error:'');if(d.busy){setTimeout(job,4000);}else if(window.refreshAfterJob){window.refreshAfterJob=false;groups();}}catch(e){$('job').textContent=e.message;}}
async function requestJob(endpoint){let token=prompt('رمز التحديث EGX_3D_REFRESH_TOKEN:');if(!token)return;try{await api(endpoint,{method:'POST',headers:{'X-Refresh-Token':token}});window.refreshAfterJob=true;job();}catch(e){alert(e.message);}}
$('update').onclick=()=>requestJob('/update-data');$('rebuild').onclick=()=>requestJob('/refresh');groups();job();
</script></body></html>'''


def create_app():
    app=Flask(__name__)
    @app.get('/')
    def home():return Response(HTML,mimetype='text/html')

    @app.get('/api/job')
    def job():
        with LOCK: st=dict(STATE)
        return jsonify(run=active_run(),**st)

    @app.get('/api/groups')
    def groups():
        run=active_run()
        if not run:return jsonify(items=[],total=0,pages=0,unique_stocks=0)
        where,params=where_filters(run,request.args)
        pg=bounded_int(request.args.get('page'),1,1,100000)
        sort=request.args.get('sort','date_desc')
        order={'date_desc':'touch_date DESC,stocks DESC','date_asc':'touch_date ASC,stocks DESC',
               'count_desc':'stocks DESC,touch_date DESC','count_asc':'stocks ASC,touch_date DESC'}.get(sort,'touch_date DESC,stocks DESC')
        # Filter AFTER all other filters and count DISTINCT stocks per group.
        # Leave the historical signals untouched: this is a lightweight read-only filter.
        minimum=bounded_int(request.args.get('min_stocks'),5,1,216)
        base=f'FROM egx_tp3d_signals_v10 WHERE {where}'
        grouped=f'''SELECT touch_date,signal_type,anchor_type,status,
                        COUNT(DISTINCT symbol) AS stocks
                        {base} GROUP BY 1,2,3,4
                        HAVING COUNT(DISTINCT symbol)>=:min_stocks'''
        parameters={**params,'min_stocks':minimum}
        with engine().connect() as con:
            count=con.execute(text(f'SELECT COUNT(*) FROM ({grouped}) q'),parameters).scalar_one()
            # Only count symbols present in qualifying groups, never the removed groups.
            unique_sql=f'''SELECT COUNT(DISTINCT s.symbol)
                FROM (SELECT symbol,touch_date,signal_type,anchor_type,status {base}) s
                JOIN ({grouped}) q
                  ON s.touch_date=q.touch_date AND s.signal_type=q.signal_type
                 AND s.anchor_type=q.anchor_type AND s.status=q.status
                '''
            unique=con.execute(text(unique_sql),parameters).scalar_one()
            rows=[dict(r) for r in con.execute(text(
                f'SELECT * FROM ({grouped}) q ORDER BY {order} LIMIT 75 OFFSET :offset'),
                {**parameters,'offset':(pg-1)*75}).mappings()]
        return jsonify(items=rows,total=count,pages=math.ceil(count/75),unique_stocks=unique)

    @app.get('/api/after20')
    def after20_candidates():
        """Filter existing results only: no historical recomputation or writes.

        Collapse multiple line rows into one per stock/date/type. Only completed
        20-session outcomes in (0,10) qualify. For multi-line signals, use a
        deterministic representative row; return values belong to the stock.
        """
        run=active_run()
        if not run:return jsonify(items=[],total=0,pages=0,unique_stocks=0)
        where,params=where_filters(run,request.args)
        where+=' AND after20_close IS NOT NULL AND current_close IS NOT NULL AND touch_close > 0 AND after20_close > 0 AND return20_pct > 0 AND return20_pct < 10'
        pg=bounded_int(request.args.get('page'),1,1,100000)
        sort=request.args.get('after_sort','desc')
        order={'asc':'return_since20_pct ASC, touch_date DESC',
               'desc':'return_since20_pct DESC, touch_date DESC',
               'date':'touch_date DESC, return_since20_pct DESC'}.get(sort,'return_since20_pct DESC, touch_date DESC')
        # ROW_NUMBER yields exactly one result for a stock and signal on a date,
        # even when two or more qualifying geometric lines generated that signal.
        # Anchor side and status are retained from the first stable line row.
        cte=f"""WITH qualified AS (
           SELECT *, ROW_NUMBER() OVER (
               PARTITION BY symbol,touch_date,signal_type
               ORDER BY anchor_type,angle,anchor_date
           ) AS rn
           FROM egx_tp3d_signals_v10 WHERE {where}
        ), selected AS (
           SELECT symbol,touch_date,signal_type,anchor_type,status,
                  touch_close,after20_date,after20_close,return20_pct,
                  current_close,latest_price_date,
                  100.0*(current_close/NULLIF(after20_close,0)-1) AS return_since20_pct
           FROM qualified WHERE rn=1
        )"""
        with engine().connect() as con:
            summary=con.execute(text(cte+" SELECT COUNT(*) AS total,COUNT(DISTINCT symbol) AS unique_stocks FROM selected"),params).mappings().one()
            data=con.execute(text(cte+f" SELECT * FROM selected ORDER BY {order} LIMIT 75 OFFSET :offset"),{**params,'offset':(pg-1)*75}).mappings().all()
        total=int(summary['total'])
        return jsonify(items=[dict(r) for r in data],total=total,unique_stocks=int(summary['unique_stocks']),pages=math.ceil(total/75))

    @app.get('/api/details')
    def details():
        run=active_run()
        if not run:return jsonify(error='No completed run'),404
        day=request.args.get('day','');kind=request.args.get('side','');status=request.args.get('status','');signal=request.args.get('signal','')
        if not (len(day)==10 and day[:4].isdigit() and kind in ('LOW','HIGH') and status in STATUS_AR and signal in ('DIFFERENT_ANGLES','SAME_ANGLE_DIFF_ANCHORS','SAME_LINE_TWICE')):return jsonify(error='Invalid group'),400
        pg=bounded_int(request.args.get('page'),1,1,100000)
        params={'run':run['run_id'],'day':day,'side':kind,'status':status,'signal':signal,'offset':(pg-1)*75}
        w='run_id=:run AND touch_date=:day AND anchor_type=:side AND status=:status AND signal_type=:signal'
        angle=request.args.get('angle','ALL')
        if angle not in (*RATIOS.keys(),'ALL'):return jsonify(error='Invalid angle filter'),400
        if angle!='ALL':w+=' AND angle=:angle';params['angle']=angle
        with engine().connect() as con:
            count=con.execute(text(f'SELECT COUNT(*) FROM egx_tp3d_signals_v10 WHERE {w}'),params).scalar_one()
            syms=con.execute(text(f'SELECT COUNT(DISTINCT symbol) FROM egx_tp3d_signals_v10 WHERE {w}'),params).scalar_one()
            items=[dict(r) for r in con.execute(text(f'''SELECT symbol,signal_type,angle,first_touch_date,anchor_date,status,level,touch_close,after10_close,after10_date,return10_pct,after20_close,after20_date,return20_pct,current_close,latest_price_date,return_current_pct
                                FROM egx_tp3d_signals_v10 WHERE {w} ORDER BY symbol,angle,anchor_date LIMIT 75 OFFSET :offset'''),params).mappings()]
        return jsonify(items=items,total=count,stocks=syms,pages=math.ceil(count/75))

    @app.get('/api/performance')
    def performance():
        run=active_run()
        if not run:return jsonify(items=[])
        where,params=where_filters(run,request.args)
        with engine().connect() as con:
            rows=[dict(r) for r in con.execute(text(f"""
                SELECT signal_type, COUNT(*) AS line_rows,
                  COUNT(DISTINCT(symbol || '|' || touch_date)) AS stock_days,
                  COUNT(return10_pct) AS complete10, ROUND(AVG(return10_pct)::numeric,2) AS avg10,
                  COUNT(return20_pct) AS complete20, ROUND(AVG(return20_pct)::numeric,2) AS avg20,
                  COUNT(return_current_pct) AS current_count,
                  ROUND(AVG(return_current_pct)::numeric,2) AS avg_current
                FROM egx_tp3d_signals_v10 WHERE {where}
                GROUP BY signal_type ORDER BY signal_type
            """),params).mappings()]
        return jsonify(items=rows)

    @app.get('/api/chart')
    def chart():
        sym=request.args.get('symbol','').strip().upper()[:25]
        if not sym.endswith('.CA') or not all(c.isalnum() or c in '._' for c in sym):return jsonify(error='Invalid EGX symbol'),400
        with engine().connect() as con:
            g=pd.read_sql(text('''SELECT session_date,o,h,l,c FROM market_candles_1d WHERE symbol=:s ORDER BY session_date'''),con,params={'s':sym})
        if g.empty:return jsonify(error='Symbol not found'),404
        g['session_date']=pd.to_datetime(g.session_date,errors='coerce')
        for k in ('o','h','l','c'):g[k]=pd.to_numeric(g[k],errors='coerce')
        g=g[g[['o','h','l','c']].notna().all(axis=1)&(g.l>0)&(g.h>=g[['o','c','l']].max(axis=1))&(g.l<=g[['o','c','h']].min(axis=1))].reset_index(drop=True)
        run=active_run()
        if run:g=g[g.session_date<=pd.Timestamp(run['market_day'])].reset_index(drop=True)
        if len(g)<60:return jsonify(error='Insufficient valid candles'),404
        a={k:g[k].to_numpy(dtype=float) for k in ('h','l','c')};a['atr']=tr_atr(g)
        bars=[{'t':i,'c':float(g.c.iloc[i])} for i in range(max(0,len(g)-100),len(g))]
        angles=[]
        for kind,sign in (('LOW',1),('HIGH',-1)):
            pv=list(pivots(a,kind))
            if not pv:continue
            ix,base,atr=pv[-1]
            for label,ratio in RATIOS.items():angles.append({'kind':kind,'sign':sign,'angle':label,'ratio':ratio,'anchor_t':ix,'base':base,'slope':atr/UNIT_DAYS})
        return jsonify(candles=bars,angles=angles)

    @app.post('/refresh')
    def refresh():
        if not auth(request):return jsonify(error='رمز التحديث غير صحيح أو غير مضبوط في Railway'),403
        with LOCK:
            if STATE['busy']:return jsonify(error='تحديث جارٍ بالفعل'),409
        threading.Thread(target=daily_update,daemon=True).start()
        return jsonify(status='started'),202

    @app.post('/update-data')
    def update_data():
        if not auth(request):return jsonify(error='رمز التحديث غير صحيح أو غير مضبوط في Railway'),403
        with LOCK:
            if STATE['busy']:return jsonify(error='تحديث جارٍ بالفعل'),409
            STATE['busy']=True;STATE['message']='جلب أحدث الأسعار من Yahoo Finance...'
        def task():
            try:
                from egx_3d_market_update import update_egx_prices
                report=update_egx_prices()
                with LOCK:STATE['message']='تم تحميل الأسعار؛ التحقق من الأسهم الجديدة فقط ...';STATE['busy']=False
                daily_update()
                with LOCK:
                    if not STATE['error']:STATE['message']='تحديث الأسعار وإعادة الحساب اكتمل. '+str(report)[:250]
            except Exception as exc:
                with LOCK:STATE['error']=str(exc)[:500];STATE['message']='فشل تحديث الأسعار';STATE['busy']=False
        threading.Thread(target=task,daemon=True).start()
        return jsonify(status='started'),202
    return app


def main():
    global UNIT_DAYS
    p=argparse.ArgumentParser()
    p.add_argument('--full-build',action='store_true');p.add_argument('--serve',action='store_true');p.add_argument('--unit-days',type=float,default=20.)
    p.add_argument('--output-dir',default=OUTPUT);p.add_argument('--port',type=int,default=8091)
    args=p.parse_args();UNIT_DAYS=args.unit_days
    if UNIT_DAYS<=0:p.error('--unit-days must be positive')
    ensure_tables()
    if args.serve:
        # Serve immediately; initial heavy computation runs asynchronously.
        # Full build is manual to avoid expensive repeated rebuilds on Railway redeploy.
        create_app().run(host='0.0.0.0',port=int(os.environ.get('PORT') or args.port),debug=False,threaded=True)
    elif args.full_build:
        rebuild()
        print(json.dumps(active_run(),default=str,ensure_ascii=False,indent=2))
    else:
        daily_update()
        print(json.dumps(active_run(),default=str,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
