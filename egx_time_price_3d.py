#!/usr/bin/env python3
"""EGX Time–Price historic touch dashboard, PostgreSQL-backed and paginated.
Numeric convention: 1x1 = confirmed pivot ATR14 / unit_days per trading session.
Only the FIRST confirmed high-low contact is recorded for each symbol / pivot / angle.
Two distinct signal types: exactly two different first-touch lines on the same day, OR the same line touched again after an intervening non-touch session. No future pivot information is used.
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
STATE = {'busy':False,'message':'جاهز — اضغطي إعادة الحساب لتطبيق قاعدة أول لمس على النتائج القديمة','error':None,'version':None}
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
        con.execute(text('''CREATE TABLE IF NOT EXISTS egx_tp3d_signals_v10 (
            run_id BIGINT NOT NULL, signal_type TEXT NOT NULL,
            symbol TEXT NOT NULL, touch_date TEXT NOT NULL,
            first_touch_date TEXT, anchor_type TEXT NOT NULL, anchor_date TEXT NOT NULL,
            angle TEXT NOT NULL, status TEXT NOT NULL, level DOUBLE PRECISION NOT NULL,
            candle_low DOUBLE PRECISION NOT NULL, candle_high DOUBLE PRECISION NOT NULL,
            touch_close DOUBLE PRECISION NOT NULL, after10_close DOUBLE PRECISION,
            after10_date TEXT, return10_pct DOUBLE PRECISION,
            PRIMARY KEY(run_id,signal_type,symbol,touch_date,anchor_type,anchor_date,angle))'''))
        con.execute(text('''CREATE INDEX IF NOT EXISTS idx_egx_tp3d_v10_groups
                         ON egx_tp3d_signals_v10(run_id,signal_type,touch_date,anchor_type,status)'''))
        con.execute(text('''CREATE INDEX IF NOT EXISTS idx_egx_tp3d_v10_symbols
                         ON egx_tp3d_signals_v10(run_id,signal_type,touch_date,symbol)'''))


def active_run():
    with engine().connect() as con:
        r=con.execute(text("SELECT run_id,market_day,stock_count,touch_count,raw_touch_count,first_touch_before_pair_filter,qualifying_symbol_days,retouch_symbol_days,invalid_bars,finished_at FROM egx_tp3d_runs WHERE status='complete' AND build_version='v10' ORDER BY run_id DESC LIMIT 1")).mappings().first()
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


def pivots(a,kind,wing=5,min_atr=1.5):
    """Yield index, anchor-price, anchor-ATR; confirmed no earlier than idx+wing."""
    low=a['l']; high=a['h']; atr=a['atr']; n=len(low)
    for i in range(wing,n-wing):
        av=atr[i]
        if not np.isfinite(av) or av<=0:continue
        low_range=low[i-wing:i+wing+1];hi_range=high[i-wing:i+wing+1]
        if kind=='LOW':
            if low[i]!=low_range.min() or hi_range.max()-low[i]<min_atr*av:continue
            yield i,float(low[i]),float(av)
        else:
            if high[i]!=hi_range.max() or high[i]-low_range.min()<min_atr*av:continue
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
    return {'symbol':sym,'touch_date':str(dates[j]),'first_touch_date':first_date,
            'anchor_type':kind,'anchor_date':str(dates[idx]),'angle':angle,
            'status':status,'level':float(level),'candle_low':float(a['l'][j]),
            'candle_high':float(a['h'][j]),'touch_close':close,
            'after10_close':after,'after10_date':str(dates[j+10]) if after is not None else None,
            'return10_pct':(after/close-1)*100 if after is not None else None}


def symbol_events(sym,g,market_day,unit_days,wing=5,diagnostics=None):
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
 touch_close,after10_close,after10_date,return10_pct)
 VALUES (:run_id,:signal_type,:symbol,:touch_date,:first_touch_date,:anchor_type,:anchor_date,:angle,:status,:level,
 :candle_low,:candle_high,:touch_close,:after10_close,:after10_date,:return10_pct)
 ON CONFLICT(run_id,signal_type,symbol,touch_date,anchor_type,anchor_date,angle) DO NOTHING''')


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
            rid=con.execute(text("INSERT INTO egx_tp3d_runs(market_day,invalid_bars,build_version) VALUES (:d,:bad,'v10') RETURNING run_id"),{'d':market_day,'bad':bad}).scalar_one()
        stock_count=0; event_count=0; raw_first_touch=0; pair_days=0;retouch_days=0
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
                distinct={(r['anchor_type'],r['anchor_date'],r['angle']) for r in items}
                if len(distinct)!=2:continue
                pair_days+=1
                for event in items:
                    event['run_id']=rid;event['signal_type']='TWO_LINES'
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
                STATE['message']=(f'فحص {stock_count} سهم | أيام لمس خطين بالضبط: {pair_days:,} '
                                  f'| الإشارات بعد الشرط: {event_count+len(buffer):,} '
                                  f'| إعادة لمس نفس الخط (سهم/يوم): {retouch_days:,} | أول لمسات قبل الشرط: {raw_first_touch:,}')
        if buffer:
            with engine().begin() as con:con.execute(INSERT,buffer)
            event_count+=len(buffer)
        with engine().begin() as con:
            con.execute(text("UPDATE egx_tp3d_runs SET status='complete',finished_at=NOW(),stock_count=:n,touch_count=:c,raw_touch_count=:raw,first_touch_before_pair_filter=:first,qualifying_symbol_days=:pairs,retouch_symbol_days=:retouch WHERE run_id=:id"),{'n':stock_count,'c':event_count,'raw':diagnostics['raw_contacts'],'first':raw_first_touch,'pairs':pair_days,'retouch':retouch_days,'id':rid})
            # Keep previous successful run as rollback point, not all historical duplicates.
            old=con.execute(text("SELECT run_id FROM egx_tp3d_runs WHERE status='complete' AND build_version='v10' ORDER BY run_id DESC OFFSET 2")).scalars().all()
            if old:
                con.execute(text('DELETE FROM egx_tp3d_signals_v10 WHERE run_id = ANY(:ids)'),{'ids':old})
                con.execute(text('DELETE FROM egx_tp3d_runs WHERE run_id = ANY(:ids)'),{'ids':old})
        with LOCK:STATE.update(message=f'اكتمل الفحص: {pair_days:,} حالات خطين في يوم واحد، {retouch_days:,} حالات إعادة لمس نفس الخط، {event_count:,} سطر معتمد حتى {market_day}',version=rid)
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
    if signal not in ('ALL','TWO_LINES','SAME_LINE_TWICE'):signal='ALL'
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
<header><h2>Time–Price Squaring — السوق المصري</h2><p>تاريخ اللمس الحقيقي، حالة الشمعة، والعائد بعد 10 جلسات تداول</p></header><main>
<article><div id="stats"></div><p class="muted">الزوايا بمعايرة ثابتة: 1×1 = ATR14 للارتكاز / 20 جلسة. لا يُستخدم الارتكاز قبل تأكيده بخمس جلسات. عمق 3D تمثيلي.</p><div class="toolbar"><button id="update">تحديث البيانات</button><button id="rebuild">إعادة حساب التاريخ</button><button onclick="location.reload()">تحديث الصفحة</button></div><div id="job" class="muted"></div></article>
<article><div class="toolbar"><label>مدة اللمسات<select id="period"><option value="30">30 يوم</option><option value="90">90 يوم</option><option value="180">6 شهور</option><option value="365">سنة</option><option value="all">كل التاريخ</option></select></label><label>نوع الإشارة<select id="signal"><option value="ALL">النوعان معًا</option><option value="TWO_LINES">خطين مختلفين في نفس اليوم</option><option value="SAME_LINE_TWICE">نفس الخط مرتين في يومين مختلفين</option></select></label><label>القمة / القاع<select id="side"><option value="BOTH">LOW و HIGH</option><option value="LOW">LOW — قاع</option><option value="HIGH">HIGH — قمة</option></select></label><label>حالة السهم<select id="status"><option value="ALL">كل الحالات</option><option value="CROSS_UP_CLOSE">اختراق صاعد</option><option value="CROSS_DOWN_CLOSE">كسر هابط</option><option value="TOUCH_ABOVE">لمس وإغلاق فوق</option><option value="TOUCH_BELOW">لمس وإغلاق تحت</option></select></label><label>نوع الخط الهندسي<select id="angle"><option value="ALL">كل الخطوط</option><option value="1x1">1×1</option><option value="1x2">1×2</option><option value="2x1">2×1</option></select></label><label>ترتيب المجموعات<select id="sort"><option value="date_desc">الأحدث أولًا</option><option value="date_asc">الأقدم أولًا</option><option value="count_desc">الأكثر أسهم أولًا</option><option value="count_asc">الأقل أسهم أولًا</option></select></label></div>
<div id="detail" class="detail"><div class="toolbar"><h3 id="detailTitle" style="flex:1"></h3><button id="closeDetail">إغلاق التفاصيل</button></div><div class="scroll"><table><thead><tr><th>السهم</th><th>نوع الإشارة</th><th>الخط</th><th>اللمسة الأولى</th><th>تاريخ الارتكاز (مرجع)</th><th>سعر الخط</th><th>إغلاق يوم اللمس</th><th>بعد 10 جلسات</th><th>عائد 10 جلسات</th><th>حالة اللمس</th></tr></thead><tbody id="detailRows"></tbody></table></div><div class="pager"><button id="prevDetail">السابق</button><span id="detailPages"></span><button id="nextDetail">التالي</button></div></div>
<div id="resultCount" class="muted"></div><p class="muted">نوع الإشارة الأول: أول لمس لخطين مختلفين بالضبط في نفس الجلسة. النوع الثاني: ثاني لمس لنفس الخط بعد جلسة واحدة على الأقل دون لمس؛ يتم تقييم العائد بعد 10 جلسات من اللمسة الثانية. قد يظهر السهم في النوعين.</p><div class="scroll"><table><thead><tr><th>تاريخ الإشارة</th><th>نوع الإشارة</th><th>مصدر الخط</th><th>الحالة</th><th>عدد الأسهم (اضغطي للتفاصيل)</th></tr></thead><tbody id="groups"></tbody></table></div><div class="pager"><button id="prevGroup">السابق</button><span id="groupPages"></span><button id="nextGroup">التالي</button></div></article>
<article><div class="toolbar"><h3 style="flex:1">شارت السهم — آخر 100 جلسة</h3><input id="ticker" placeholder="COMI.CA"><button id="loadChart">عرض الشارت 3D</button></div><div id="chart"></div></article></main>
<script>
const $=id=>document.getElementById(id),signalNames={TWO_LINES:'خطان مختلفان',SAME_LINE_TWICE:'نفس الخط مرتين'},names={CROSS_UP_CLOSE:'اختراق صاعد',CROSS_DOWN_CLOSE:'كسر هابط',TOUCH_ABOVE:'لمس وإغلاق فوق',TOUCH_BELOW:'لمس وإغلاق تحت'};
let page=1,detailPage=1,selected=null, groupTotal=0, detailTotal=0;
const filt=()=>new URLSearchParams({period:$('period').value,signal:$('signal').value,side:$('side').value,status:$('status').value,angle:$('angle').value,sort:$('sort').value});
const safe=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw Error(d.error||r.status);return d;}
async function groups(){try{let q=filt();q.set('page',page);const d=await api('/api/groups?'+q);groupTotal=d.pages;
$('resultCount').textContent=`${d.total} مجموعة | ${d.unique_stocks} سهم فريد ضمن الفلتر`;
$('groups').innerHTML=d.items.map(g=>`<tr><td>${g.touch_date}</td><td>${signalNames[g.signal_type]}</td><td>${g.anchor_type==='LOW'?'LOW — قاع':'HIGH — قمة'}</td><td>${names[g.status]}</td><td><button class="showDetail" data-day="${g.touch_date}" data-side="${g.anchor_type}" data-status="${g.status}" data-signal="${g.signal_type}">${g.stocks} سهم</button></td></tr>`).join('')||'<tr><td colspan="5">لا توجد نتائج بهذه الشروط.</td></tr>';
$('groupPages').textContent=`${page} / ${d.pages||1}`;$('prevGroup').disabled=page<=1;$('nextGroup').disabled=page>=d.pages;
document.querySelectorAll('.showDetail').forEach(b=>b.onclick=()=>{selected={day:b.dataset.day,side:b.dataset.side,status:b.dataset.status,signal:b.dataset.signal};detailPage=1;details();});
}catch(e){$('groups').innerHTML='<tr><td>'+safe(e.message)+'</td></tr>';}}
async function details(){if(!selected)return;try{let q=new URLSearchParams({day:selected.day,side:selected.side,status:selected.status,signal:selected.signal,angle:$('angle').value,page:detailPage});let d=await api('/api/details?'+q);
$('detail').style.display='block';$('detailTitle').textContent=`${selected.day} | ${signalNames[selected.signal]} | ${selected.side==='LOW'?'LOW — قاع':'HIGH — قمة'} | ${names[selected.status]} | ${d.stocks} سهم`;
$('detailRows').innerHTML=d.items.map(x=>`<tr><td><a href="#chart" class="chartLink" data-sym="${safe(x.symbol)}">${safe(x.symbol)}</a></td><td>${signalNames[x.signal_type]}</td><td>${x.angle}</td><td>${x.first_touch_date||'—'}</td><td>${x.anchor_date}</td><td>${x.level.toFixed(4)}</td><td>${x.touch_close.toFixed(4)}</td><td>${x.after10_close==null?'قيد الانتظار':x.after10_close.toFixed(4)}</td><td>${x.return10_pct==null?'—':x.return10_pct.toFixed(2)+'%'}</td><td>${names[x.status]}</td></tr>`).join('');
$('detailPages').textContent=`${detailPage} / ${d.pages||1}`;detailTotal=d.pages;$('prevDetail').disabled=detailPage<=1;$('nextDetail').disabled=detailPage>=d.pages;
document.querySelectorAll('.chartLink').forEach(x=>x.onclick=()=>{$('ticker').value=x.dataset.sym;chart();});$('detail').scrollIntoView({behavior:'smooth',block:'start'});
}catch(e){alert(e.message);}}
$('closeDetail').onclick=()=>{$('detail').style.display='none';selected=null;};
for(let key of ['period','signal','side','status','angle','sort'])$(key).onchange=()=>{page=1;selected=null;$('detail').style.display='none';groups();};
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
async function job(){try{let d=await api('/api/job');$('stats').innerHTML=d.run?`<span class="stat">آخر جلسة: ${d.run.market_day}</span><span class="stat">الأسهم: ${d.run.stock_count}</span><span class="stat">حالات خطين مختلفين (سهم/يوم): ${(d.run.qualifying_symbol_days??0).toLocaleString()}</span><span class="stat">حالات إعادة لمس نفس الخط (سهم/يوم): ${(d.run.retouch_symbol_days??0).toLocaleString()}</span><span class="stat">سطور الإشارات المعتمدة: ${d.run.touch_count.toLocaleString()}</span><span class="stat">أول اللمسات قبل شرط الخطين: ${(d.run.first_touch_before_pair_filter??0).toLocaleString()}</span><span class="stat">كل التماسات قبل أول لمس: ${(d.run.raw_touch_count??0).toLocaleString()}</span><span class="stat">الشموع المستبعدة: ${d.run.invalid_bars.toLocaleString()}</span>`:'<span class="stat">السجل التاريخي لم يُبنَ بعد</span>';
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
        base=f'FROM egx_tp3d_signals_v10 WHERE {where}'
        with engine().connect() as con:
            count=con.execute(text(f'''SELECT count(*) FROM (SELECT touch_date,signal_type,anchor_type,status {base} GROUP BY 1,2,3,4) g'''),params).scalar_one()
            unique=con.execute(text(f'SELECT COUNT(DISTINCT symbol) {base}'),params).scalar_one()
            sql=f'''SELECT touch_date,signal_type,anchor_type,status,COUNT(DISTINCT symbol) AS stocks
                      {base} GROUP BY 1,2,3,4 ORDER BY {order} LIMIT 75 OFFSET :offset'''
            rows=[dict(r) for r in con.execute(text(sql),{**params,'offset':(pg-1)*75}).mappings()]
        return jsonify(items=rows,total=count,pages=math.ceil(count/75),unique_stocks=unique)

    @app.get('/api/details')
    def details():
        run=active_run()
        if not run:return jsonify(error='No completed run'),404
        day=request.args.get('day','');kind=request.args.get('side','');status=request.args.get('status','');signal=request.args.get('signal','')
        if not (len(day)==10 and day[:4].isdigit() and kind in ('LOW','HIGH') and status in STATUS_AR and signal in ('TWO_LINES','SAME_LINE_TWICE')):return jsonify(error='Invalid group'),400
        pg=bounded_int(request.args.get('page'),1,1,100000)
        params={'run':run['run_id'],'day':day,'side':kind,'status':status,'signal':signal,'offset':(pg-1)*75}
        w='run_id=:run AND touch_date=:day AND anchor_type=:side AND status=:status AND signal_type=:signal'
        angle=request.args.get('angle','ALL')
        if angle not in (*RATIOS.keys(),'ALL'):return jsonify(error='Invalid angle filter'),400
        if angle!='ALL':w+=' AND angle=:angle';params['angle']=angle
        with engine().connect() as con:
            count=con.execute(text(f'SELECT COUNT(*) FROM egx_tp3d_signals_v10 WHERE {w}'),params).scalar_one()
            syms=con.execute(text(f'SELECT COUNT(DISTINCT symbol) FROM egx_tp3d_signals_v10 WHERE {w}'),params).scalar_one()
            items=[dict(r) for r in con.execute(text(f'''SELECT symbol,signal_type,angle,first_touch_date,anchor_date,status,level,touch_close,after10_close,after10_date,return10_pct
                                FROM egx_tp3d_signals_v10 WHERE {w} ORDER BY symbol,angle,anchor_date LIMIT 75 OFFSET :offset'''),params).mappings()]
        return jsonify(items=items,total=count,stocks=syms,pages=math.ceil(count/75))

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
        threading.Thread(target=rebuild,daemon=True).start()
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
                with LOCK:STATE['message']='تم تحميل الأسعار؛ جارٍ إعادة الحساب...';STATE['busy']=False
                rebuild()
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
    p.add_argument('--serve',action='store_true');p.add_argument('--unit-days',type=float,default=20.)
    p.add_argument('--output-dir',default=OUTPUT);p.add_argument('--port',type=int,default=8091)
    args=p.parse_args();UNIT_DAYS=args.unit_days
    if UNIT_DAYS<=0:p.error('--unit-days must be positive')
    ensure_tables()
    if args.serve:
        # Serve immediately; initial heavy computation runs asynchronously.
        if not active_run():threading.Thread(target=rebuild,daemon=True).start()
        create_app().run(host='0.0.0.0',port=int(os.environ.get('PORT') or args.port),debug=False,threaded=True)
    else:
        rebuild()
        print(json.dumps(active_run(),default=str,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
