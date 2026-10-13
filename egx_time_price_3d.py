#!/usr/bin/env python3
"""Independent EGX angle-touch dashboard. Does not modify the Rajih/Brown engine.

Angles are NUMERIC Gann-style calibrated lines (not a canonical mathematical 45°
without a declared price-to-time conversion). Interactive 3D rendering fixes that
conversion and disables user zoom and camera rotation.
"""
import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

RATIOS = {'1x2': .5, '1x1': 1., '2x1': 2.}
state = {'data': None, 'error': None, 'busy': False, 'job': {'status':'idle','message':'لم يبدأ تحديث الأسعار'}}
state_lock = threading.Lock()

def read_bars():
    db = os.environ.get('DATABASE_URL')
    if not db: raise RuntimeError('DATABASE_URL is missing')
    sql = text('''SELECT symbol,session_date,o,h,l,c FROM market_candles_1d
        WHERE UPPER(symbol) LIKE '%.CA' AND session_date IS NOT NULL
        ORDER BY symbol, session_date''')
    with create_engine(db, pool_pre_ping=True).connect() as con:
        df=pd.read_sql(sql,con)
    df['session_date']=pd.to_datetime(df.session_date, errors='coerce').dt.normalize()
    for col in ('o','h','l','c'):df[col]=pd.to_numeric(df[col],errors='coerce')
    valid=(df[['o','h','l','c']].notna().all(axis=1) &
           (df[['o','h','l','c']]>0).all(axis=1) &
           (df.h >= df[['o','c','l']].max(axis=1)-1e-7) &
           (df.l <= df[['o','c','h']].min(axis=1)+1e-7))
    bad=df.loc[~valid,['symbol','session_date','o','h','l','c']].copy()
    df=df.loc[valid].drop_duplicates(['symbol','session_date'],keep='last')
    return df,bad

def calc_atr(g):
    pc=g.c.shift()
    tr=pd.concat([g.h-g.l,(g.h-pc).abs(),(g.l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(14,min_periods=14).mean()

def last_pivot(g,kind,wing=5,lookback=180,min_atr=1.5):
    n=len(g)
    # last bar excluded, plus `wing` confirmation sessions
    end=n-wing-1
    for i in range(end-1,max(wing-1,n-lookback)-1,-1):
        row=g.iloc[i];v=float(row.atr)
        if not np.isfinite(v) or v<=0:continue
        w=g.iloc[i-wing:i+wing+1]
        if kind=='LOW':
            extreme=float(row.l)
            if extreme!=float(w.l.min()) or float(w.h.max())-extreme < min_atr*v:continue
        else:
            extreme=float(row.h)
            if extreme!=float(w.h.max()) or extreme-float(w.l.min()) < min_atr*v:continue
        return i,extreme,v
    return None

def historical_pivots(g, kind, wing=5, min_atr=1.5):
    """All locally confirmed pivots up to the final completed bar."""
    v=g[['l','h','atr']].to_numpy(dtype=float)
    n=len(v)
    for i in range(wing,n-wing-1):
        lo,hi,at=v[i]
        if not np.isfinite(at) or at<=0:continue
        window=v[i-wing:i+wing+1]
        if kind=='LOW':
            if lo!=np.min(window[:,0]) or np.max(window[:,1])-lo<min_atr*at:continue
            yield i,lo,at
        else:
            if hi!=np.max(window[:,1]) or hi-np.min(window[:,0])<min_atr*at:continue
            yield i,hi,at

def build(out_dir,unit_days=20.,near_atr=.15,wing=5,lookback=180,min_atr=1.5):
    df,bad=read_bars()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    bad.to_csv(out/'invalid_bars.csv',index=False,encoding='utf-8-sig')
    # newest MODE rather than maximum: avoids one isolated symbol with a future quote
    dates=df.groupby('symbol').session_date.max().dt.strftime('%Y-%m-%d')
    mode_day=str(dates.mode().iloc[0]) if len(dates) else None
    levels=[];viz={};skip=[];historical_hits=[];historical_near=[]
    for sym,g in df.groupby('symbol',sort=True):
        g=g.sort_values('session_date').reset_index(drop=True)
        # Backtest through the latest complete common market session only.
        if mode_day: g=g[g.session_date <= pd.Timestamp(mode_day)].reset_index(drop=True)
        if len(g)<60:
            skip.append({'symbol':sym,'reason':'short_history'});continue
        g['atr']=calc_atr(g)
        last=g.iloc[-1];close=float(last.c);lo=float(last.l);hi=float(last.h)
        if not np.isfinite(last.atr) or last.atr<=0:
            skip.append({'symbol':sym,'reason':'missing_atr'});continue
        if last.session_date.strftime('%Y-%m-%d')!=mode_day:
            skip.append({'symbol':sym,'reason':'stale_session','last_session':last.session_date.strftime('%Y-%m-%d')});continue
        chart=[]
        for i in range(max(0,len(g)-100),len(g)):
            r=g.iloc[i]
            chart.append({'t':int(i),'date':r.session_date.strftime('%Y-%m-%d'),
                          'o':round(float(r.o),6),'h':round(float(r.h),6),
                          'l':round(float(r.l),6),'c':round(float(r.c),6)})
        angles=[]
        for kind,sign in [('LOW',1),('HIGH',-1)]:
            anchor=last_pivot(g,kind,wing,lookback,min_atr)
            if anchor is None:continue
            idx,base,anchor_atr=anchor
            one_day_price=anchor_atr/unit_days
            age=len(g)-1-idx
            for label,mul in RATIOS.items():
                level=base+sign*age*one_day_price*mul
                if level<=0:continue
                touched=lo-1e-8<=level<=hi+1e-8
                near=abs(close-level)<=float(last.atr)*near_atr
                # yesterday's projected level, used for cross state
                prev_level=base+sign*(age-1)*one_day_price*mul
                prevclose=float(g.iloc[-2].c)
                if touched and prevclose<prev_level and close>level:status='CROSS_UP_CLOSE'
                elif touched and prevclose>prev_level and close<level:status='CROSS_DOWN_CLOSE'
                elif touched:status='TOUCH_ABOVE' if close>=level else 'TOUCH_BELOW'
                elif near:status='NEAR'
                else:status='NONE'
                item={'symbol':sym,'session_date':mode_day,'anchor_type':kind,
                      'anchor_date':g.iloc[idx].session_date.strftime('%Y-%m-%d'),
                      'anchor_t':int(idx),'anchor_price':round(base,6),
                      'angle':label,'level':round(level,6),'ratio':mul,
                      'unit_per_day':round(one_day_price,10),'age_sessions':int(age),
                      'last_low':round(lo,6),'last_high':round(hi,6),
                      'last_close':round(close,6),'touch':bool(touched),
                      'near':bool(near),'status':status}
                levels.append(item); angles.append(item)
        # Historical *touch events*, not historical pivot dates shown as events.
        # An anchor formed on bar idx becomes KNOWN only after idx+wing bars.
        # Start evaluations at idx+wing+1, so no pivot is used before confirmation.
        # Use each session's close/low/high and the projected line at that session.
        close_a=g.c.to_numpy(dtype=float); low_a=g.l.to_numpy(dtype=float)
        high_a=g.h.to_numpy(dtype=float); atr_a=g.atr.to_numpy(dtype=float)
        dates_a=g.session_date.dt.strftime('%Y-%m-%d').to_numpy()
        for kind,sign in [('LOW',1),('HIGH',-1)]:
            for idx,base,anchor_atr in historical_pivots(g,kind,wing,min_atr):
                begin=idx+wing+1
                if begin>=len(g):continue
                positions=np.arange(begin,len(g),dtype=int)
                age=positions-idx
                one_day_price=anchor_atr/unit_days
                for label,mul in RATIOS.items():
                    line=base+sign*age*one_day_price*mul
                    prev_line=base+sign*(age-1)*one_day_price*mul
                    touch=(line>0)&(low_a[positions]-1e-8<=line)&(line<=high_a[positions]+1e-8)
                    near=(line>0)&(np.abs(close_a[positions]-line)<=atr_a[positions]*near_atr)
                    candidates=np.flatnonzero(touch|near)
                    for ix in candidates:
                        t=int(positions[ix]);lvl=float(line[ix]);is_touch=bool(touch[ix]);is_near=bool(near[ix])
                        prevclose=float(close_a[t-1]); curclose=float(close_a[t])
                        if is_touch and prevclose<float(prev_line[ix]) and curclose>lvl:status='CROSS_UP_CLOSE'
                        elif is_touch and prevclose>float(prev_line[ix]) and curclose<lvl:status='CROSS_DOWN_CLOSE'
                        elif is_touch:status='TOUCH_ABOVE' if curclose>=lvl else 'TOUCH_BELOW'
                        else:status='NEAR'
                        # Strictly the 10th *subsequent available trading session*.
                        forward=t+10
                        f_close=float(close_a[forward]) if forward<len(g) else None
                        item={'symbol':sym,'session_date':str(dates_a[t]),'anchor_type':kind,
                              'anchor_date':str(dates_a[idx]),'anchor_t':int(idx),
                              'anchor_price':round(base,6),'angle':label,
                              'level':round(lvl,6),'ratio':mul,
                              'unit_per_day':round(one_day_price,10),'age_sessions':int(age[ix]),
                              'last_low':round(float(low_a[t]),6),'last_high':round(float(high_a[t]),6),
                              'last_close':round(curclose,6),'touch':is_touch,'near':is_near,'status':status,
                              'plus_10_close':round(f_close,6) if f_close is not None else None,
                              'plus_10_date':str(dates_a[forward]) if forward<len(g) else None,
                              'return_10_pct':round((f_close/curclose-1)*100,4) if f_close is not None else None}
                        if is_touch:historical_hits.append(item)
                        historical_near.append(item)
        viz[sym]={'candles':chart,'angles':angles,'last_close':close}
    allrows=pd.DataFrame(levels)
    if allrows.empty:raise RuntimeError('No levels found; check market bars and pivots')
    # Do not discard future-performance fields by forcing allrows.columns.
    hits=pd.DataFrame(historical_hits)
    nearby=pd.DataFrame(historical_near)
    unique_cols=['symbol','session_date','anchor_type','anchor_date','angle','status']
    if not hits.empty:hits=hits.drop_duplicates(unique_cols)
    if not nearby.empty:nearby=nearby.drop_duplicates(unique_cols)
    allrows.to_csv(out/'all_levels.csv',index=False,encoding='utf-8-sig')
    hits.to_csv(out/'touched_levels.csv',index=False,encoding='utf-8-sig')
    nearby.to_csv(out/'touched_or_near.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(skip).to_csv(out/'skipped.csv',index=False,encoding='utf-8-sig')
    summary={'version':'EGX-ANGLE-3D-v6-touch-event','market_day':mode_day,
             'source_stocks':int(df.symbol.nunique()),'analyzed_stocks':len(viz),
             'touching_stocks':int(hits.symbol.nunique()) if not hits.empty else 0,'touched_levels':len(hits),'low_touching_stocks':int(hits.loc[hits.anchor_type=='LOW','symbol'].nunique()) if not hits.empty else 0,'high_touching_stocks':int(hits.loc[hits.anchor_type=='HIGH','symbol'].nunique()) if not hits.empty else 0,
             'invalid_bars':len(bad),'calibration':'1x1 = anchor ATR14 / '+str(unit_days)+' sessions',
             'generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'note':'Historical touch events use confirmed anchors only after wing confirmation; 10-session performance starts from touch-session close. Depth is illustrative.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'summary':summary,'viz':viz,'hits':hits.to_dict(orient='records'),
            'nearby':nearby.to_dict(orient='records')}

HTML = '''<!DOCTYPE html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EGX — Time Price 3D</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{margin:0;background:#f4f7fb;font-family:Tahoma,Arial,sans-serif;color:#183047}header{background:#112b46;color:#fff;padding:18px 4%}main{max-width:1500px;margin:auto;padding:20px}article{background:white;border:1px solid #dce6ee;border-radius:14px;padding:18px;margin-bottom:16px}.toolbar{display:flex;align-items:end;gap:10px;flex-wrap:wrap}button,select,input{padding:11px;border:1px solid #bccbd8;border-radius:8px;font:inherit}button{background:#137a82;color:white;cursor:pointer}button:disabled{opacity:.5}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #e7ebf0;text-align:right}th{background:#edf3f7}.scroll{overflow:auto}#chart{height:570px;width:100%}.small{color:#68798a;font-size:12px;line-height:1.8}.stat{display:inline-block;background:#e7f3ef;border-radius:8px;padding:10px;margin:6px}.sortable{cursor:pointer;user-select:none;white-space:nowrap}.sortable:hover{background:#dae7ef}.sort-hint{font-size:12px;color:#526b7c;margin:7px 0}</style></head><body>
<header><h2>Time–Price Squaring — السوق المصري</h2><div>لمسات الخطوط الهندسية 1×1 و1×2 و2×1 — من قمم وقيعان مؤكدة</div></header>
<main><article><div id="stats"></div><div class="small">المقياس الهندسي الرقمي ثابت: 1×1 = ATR14 عند القمة/القاع ÷ 20 جلسة. بُعد العمق للعرض فقط. تم تعطيل تدوير الرسم والـZoom لمنع تغيير الزاوية بصريًا. هذا نموذج معايرة تجريبي وليس زاوية جان مطلقة.</div></article>
<article><div class="toolbar"><label>نطاق العرض<br><select id="scope"><option value="market">كل السوق — قائمة الأسهم</option><option value="symbol">سهم واحد — شارت 3D</option></select></label><label>القمة / القاع<br><select id="side"><option value="BOTH">LOW و HIGH</option><option value="LOW">LOW — خطوط من القيعان</option><option value="HIGH">HIGH — خطوط من القمم</option></select></label><label>نوع اللمس<br><select id="filter"><option value="touch">لمس فعلي في آخر شمعة</option><option value="near">لمس أو قرب من الخط</option><option value="all">كل اللمسات وحالات القرب تاريخيًا</option></select></label><label id="symbolwrap" style="display:none">السهم<br><select id="symbol"></select></label><button id="fetchdata" style="background:#277047">تحديث البيانات</button><button id="refresh">إعادة الحساب من قاعدة البيانات</button><button onclick="location.reload()">تحديث عرض الصفحة</button></div><p id="updateStatus" class="small" role="status" aria-live="polite"></p><p id="note" class="small"></p><div id="marketSummary"></div><div id="chart" style="display:none"></div></article>
<article><div class="toolbar"><h3 id="tableTitle" style="margin:0;flex:1">لمسات الخطوط التاريخية حسب تاريخ اللمس والحالة</h3><label>المدة من تاريخ اللمس<br><select id="period"><option value="30">30 يوم</option><option value="90">90 يوم</option><option value="180">6 شهور</option><option value="365">سنة</option><option value="all" selected>كل التاريخ</option></select></label></div><div class="toolbar" style="margin:10px 0"><label>فلتر الحالة<br><select id="statusFilter"><option value="ALL">كل الحالات</option><option value="CROSS_UP_CLOSE">اختراق صاعد</option><option value="CROSS_DOWN_CLOSE">كسر هابط</option><option value="TOUCH_ABOVE">لمس وإغلاق فوق</option><option value="TOUCH_BELOW">لمس وإغلاق تحت</option><option value="NEAR">قريب من الخط</option></select></label><label>ترتيب المجموعات<br><select id="groupSort"><option value="date_desc">الأحدث تاريخًا</option><option value="date_asc">الأقدم تاريخًا</option><option value="count_desc">عدد الأسهم: الأكبر أولًا</option><option value="count_asc">عدد الأسهم: الأصغر أولًا</option><option value="status">الحالة</option></select></label></div><div class="small" id="groupCounts"></div><p class="small">التاريخ هو يوم اللمس الفعلي، وليس تاريخ القمة أو القاع. يتم احتساب الأداء من إغلاق يوم اللمس إلى إغلاق عاشر جلسة تداول تالية. الارتكازات لا تُستخدم قبل تأكيدها بخمس جلسات. عدد الأسهم فريد داخل المجموعة.</p><div id="groupDetail" style="display:none;margin:14px 0 20px;padding:14px;background:#f1f7fb;border:1px solid #a6cbd5;border-radius:12px"><div class="toolbar"><h3 id="detailTitle" style="flex:1"></h3><button id="closeDetail" type="button">إغلاق التفاصيل</button></div><div class="scroll"><table><thead><tr><th>السهم</th><th>الزاوية</th><th>سعر الخط</th><th>Low / High يوم اللمس</th><th>إغلاق يوم اللمس</th><th>إغلاق بعد 10 جلسات من اللمس</th><th>تاريخ الجلسة +10</th><th>العائد بعد 10 جلسات %</th><th>الحالة</th></tr></thead><tbody id="rows"></tbody></table></div></div><div class="scroll"><table><thead><tr><th>تاريخ اللمس</th><th>النوع</th><th>الحالة</th><th>عدد الأسهم (اضغطي للتفاصيل)</th></tr></thead><tbody id="groups"></tbody></table></div></article></main>
<script>
const dataset=__DATA__;
const $=x=>document.getElementById(x);
const statusAr={CROSS_UP_CLOSE:'اختراق صاعد',CROSS_DOWN_CLOSE:'كسر هابط',TOUCH_ABOVE:'لمس وإغلاق فوق',TOUCH_BELOW:'لمس وإغلاق تحت',NEAR:'قريب من الخط',NONE:'لم يلمس'};
const s=dataset.summary;
// Sorting never changes computed touches or levels; priority is a display convention only.
let sortColumn='priority',sortDirection='asc';
const statusRank={CROSS_UP_CLOSE:0,CROSS_DOWN_CLOSE:0,TOUCH_ABOVE:1,TOUCH_BELOW:1,NEAR:2,NONE:3};
function closeDistancePct(x){return Math.abs(x.last_close-x.level)/Math.max(Math.abs(x.level),1e-12)*100;}
function byPriority(a,b){return (statusRank[a.status]??4)-(statusRank[b.status]??4)||closeDistancePct(a)-closeDistancePct(b)||b.anchor_date.localeCompare(a.anchor_date)||a.symbol.localeCompare(b.symbol)||a.angle.localeCompare(b.angle);}
function sortRows(a,b){
 if(sortColumn==='priority')return byPriority(a,b);
 let x,y;
 if(sortColumn==='status'){x=statusRank[a.status]??4;y=statusRank[b.status]??4;}
 else if(['level','last_low','last_close'].includes(sortColumn)){x=Number(a[sortColumn]);y=Number(b[sortColumn]);}
 else {x=String(a[sortColumn]??'');y=String(b[sortColumn]??'');}
 let cmp=typeof x==='number'?(x-y):x.localeCompare(y,'en');
 return sortDirection==='asc'?(cmp||byPriority(a,b)):(-cmp||byPriority(a,b));
}
function initSorting(){
 document.querySelectorAll('th[data-sort]').forEach(th=>th.addEventListener('click',()=>{
  const col=th.dataset.sort;
  if(sortColumn===col)sortDirection=sortDirection==='asc'?'desc':'asc';
  else {sortColumn=col;sortDirection=['anchor_date','level','last_close','last_low'].includes(col)?'desc':'asc';}
  renderTable();
 }));
}

$('stats').innerHTML=`<span class="stat">آخر جلسة: ${s.market_day}</span><span class="stat">أسهم مفحوصة: ${s.analyzed_stocks}</span><span class="stat">أسهم لمست: ${s.touching_stocks}</span><span class="stat">مستويات ملموسة: ${s.touched_levels}</span><span class="stat">شموع غير سليمة: ${s.invalid_bars}</span>`;
function shown(){
 let f=$('filter').value;
 let arr=f==='touch'?dataset.hits:dataset.nearby;
 let side=$('side').value;
 return side==='BOTH'?arr:arr.filter(x=>x.anchor_type===side);
}
let activeKey=null;
let groupedCache=new Map();
function periodRows(rows){
 const period=$('period').value;
 if(period==='all')return rows;
 const d=new Date(s.market_day+'T12:00:00Z');d.setUTCDate(d.getUTCDate()-Number(period));
 const cutoff=d.toISOString().slice(0,10);
 return rows.filter(r=>r.session_date>=cutoff);
}
function showDetail(key){
 const group=groupedCache.get(key);if(!group)return;
 activeKey=key;
 $('groupDetail').style.display='block';
 $('detailTitle').textContent=`${group.date} | ${group.kind==='LOW'?'LOW — قاع':'HIGH — قمة'} | ${statusAr[group.status]} | ${group.symbols.size} سهم`;
 $('rows').innerHTML=group.rows.slice().sort((a,b)=>a.symbol.localeCompare(b.symbol)||a.angle.localeCompare(b.angle)).map(x=>`<tr><td><a href="#chart" class="sym-link" data-symbol="${x.symbol}">${x.symbol}</a></td><td>${x.angle}</td><td>${x.level.toFixed(4)}</td><td>${x.last_low.toFixed(4)} / ${x.last_high.toFixed(4)}</td><td>${x.last_close.toFixed(4)}</td><td>${x.plus_10_close==null?'قيد الانتظار':Number(x.plus_10_close).toFixed(4)}</td><td>${x.plus_10_date||'—'}</td><td>${x.return_10_pct==null?'—':Number(x.return_10_pct).toFixed(2)+'%'}</td><td>${statusAr[x.status]}</td></tr>`).join('');
 document.querySelectorAll('.sym-link').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();$('scope').value='symbol';$('symbol').value=a.dataset.symbol;renderTable();window.scrollTo({top:0,behavior:'smooth'});}));
}
function renderTable(){
 let unfiltered=shown();let syms=[...new Set(Object.keys(dataset.viz))].sort();let last=$('symbol').value;
 $('symbol').innerHTML=syms.map(x=>`<option value="${x}">${x}</option>`).join('');
 $('symbol').value=syms.includes(last)?last:(syms[0]||'');
 let inMarket=$('scope').value==='market';
 $('symbolwrap').style.display=inMarket?'none':'';
 $('chart').style.display=inMarket?'none':'block';
 $('marketSummary').style.display=inMarket?'block':'none';
 const statusChoice=$('statusFilter').value;const periodFiltered=periodRows(inMarket?unfiltered:unfiltered.filter(x=>x.symbol===$('symbol').value));const rows=statusChoice==='ALL'?periodFiltered:periodFiltered.filter(x=>x.status===statusChoice);
 let lows=new Set(rows.filter(x=>x.anchor_type==='LOW').map(x=>x.symbol));
 let highs=new Set(rows.filter(x=>x.anchor_type==='HIGH').map(x=>x.symbol));
 let both=[...lows].filter(x=>highs.has(x));
 $('marketSummary').innerHTML=`<span class="stat">LOW — ${lows.size} سهم</span><span class="stat">HIGH — ${highs.size} سهم</span><span class="stat">LOW + HIGH — ${both.length} سهم</span>`;
 $('groupCounts').textContent=`${rows.length} خط مطابق | ${new Set(rows.map(x=>x.symbol)).size} سهم فريد | آخر جلسة متاحة: ${s.market_day}`;
 groupedCache=new Map();
 for(const r of rows){const key=[r.session_date,r.anchor_type,r.status].join('|');
  if(!groupedCache.has(key))groupedCache.set(key,{date:r.session_date,kind:r.anchor_type,status:r.status,rows:[],symbols:new Set()});
  const g=groupedCache.get(key);g.rows.push(r);g.symbols.add(r.symbol);
 }
 const groupSort=$('groupSort').value;
 const groups=[...groupedCache.entries()].sort((a,b)=>{
   const A=a[1],B=b[1];
   if(groupSort==='count_desc')return B.symbols.size-A.symbols.size||B.date.localeCompare(A.date);
   if(groupSort==='count_asc')return A.symbols.size-B.symbols.size||B.date.localeCompare(A.date);
   if(groupSort==='date_asc')return A.date.localeCompare(B.date)||A.kind.localeCompare(B.kind);
   if(groupSort==='status')return (statusRank[A.status]??4)-(statusRank[B.status]??4)||B.symbols.size-A.symbols.size;
   return B.date.localeCompare(A.date)||A.kind.localeCompare(B.kind)||A.status.localeCompare(B.status);
 });
 $('groups').innerHTML=groups.length?groups.map(([key,g])=>`<tr><td>${g.date}</td><td>${g.kind==='LOW'?'LOW — قاع':'HIGH — قمة'}</td><td>${statusAr[g.status]}</td><td><button class="group-open" data-key="${key}" type="button">${g.symbols.size} سهم</button></td></tr>`).join(''):'<tr><td colspan="4">لا توجد مجموعات مطابقة لهذه المدة والتصفية.</td></tr>';
 document.querySelectorAll('.group-open').forEach(b=>b.addEventListener('click',()=>{showDetail(b.dataset.key);$('groupDetail').scrollIntoView({behavior:'smooth',block:'start'});}));
 if(activeKey && groupedCache.has(activeKey))showDetail(activeKey);else{$('groupDetail').style.display='none';activeKey=null;}
 $('tableTitle').textContent=inMarket?'مجموعات السوق حسب تاريخ اللمس والحالة':'مجموعات السهم المحدد حسب تاريخ اللمس والحالة';
 $('note').textContent=`التواريخ المعروضة هي تواريخ لمس الخطوط، حتى آخر جلسة سوق مشتركة (${s.market_day}). عائد +10 جلسات يُحتسب من إغلاق اللمس، وليس من تاريخ ارتكاز الخط.`;
 if(!inMarket)renderChart();
}
$('closeDetail').addEventListener('click',()=>{$('groupDetail').style.display='none';activeKey=null;});
$('period').addEventListener('change',()=>{activeKey=null;renderTable();});
$('statusFilter').addEventListener('change',()=>{activeKey=null;renderTable();});
$('groupSort').addEventListener('change',()=>{activeKey=null;renderTable();});
function renderChart(){let sym=$('symbol').value;let set=dataset.viz[sym];if(!set){$('note').textContent='لا توجد مستويات للشارت المحدد';return;}
let candles=set.candles;let angles=set.angles;let timeStart=candles[0].t;let tEnd=candles[candles.length-1].t;
// Fixed common scale in *normalized* price units: one x session = anchor ATR/20 as y unit.
let slope=angles.find(a=>a.angle==='1x1')?.unit_per_day||1; let base=candles[0].c;
let pts=candles.map(x=>({x:x.t-timeStart,y:(x.c-base)/slope}));
let yVals=pts.map(p=>p.y);let ySpan=Math.max(...yVals)-Math.min(...yVals);let minY=Math.min(...yVals)-3;let maxY=Math.max(...yVals)+3;
let traces=[{type:'scatter3d',mode:'lines+markers',name:'الإغلاق',x:pts.map(p=>p.x),y:pts.map(p=>p.y),z:pts.map(()=>0),line:{color:'#172a46',width:6},marker:{size:2}}];
const colors={'1x1':'#d97706','1x2':'#2975b7','2x1':'#8b43b0'};
for(let a of angles){let first=Math.max(a.anchor_t,timeStart);let xs=[first,tEnd];let ys=xs.map(t=>(a.anchor_price+(a.anchor_type==='LOW'?1:-1)*(t-a.anchor_t)*a.unit_per_day*a.ratio-base)/slope);
traces.push({type:'scatter3d',mode:'lines',name:`${a.anchor_type} ${a.angle} (${a.level.toFixed(2)})`,x:xs.map(t=>t-timeStart),y:ys,z:[1,1],line:{color:colors[a.angle],width:a.touch?10:4,dash:a.touch?'solid':'dot'}});}
let xSpan=tEnd-timeStart;
let showSpan=Math.max(xSpan,Math.abs(minY),Math.abs(maxY),1);
// Equal x/y units within a fixed camera box; fixed axes and no interactions.
let x0=0,x1=Math.max(xSpan,1),yy0=minY,yy1=maxY;let aspect=Math.max(x1-x0,yy1-yy0);
let layout={margin:{l:0,r:0,t:20,b:0},height:560,paper_bgcolor:'#fff',scene:{dragmode:false,aspectmode:'manual',aspectratio:{x:(x1-x0)/aspect,y:(yy1-yy0)/aspect,z:.18},camera:{eye:{x:1.55,y:1.5,z:1.05},projection:{type:'orthographic'}},xaxis:{title:'Trading sessions',range:[x0,x1],fixedrange:true},yaxis:{title:'Price in fixed units',range:[yy0,yy1],fixedrange:true},zaxis:{title:'visual depth',range:[0,1.5],fixedrange:true}},legend:{orientation:'h',y:-.1}};
$('note').textContent=`${sym} — تحويل السعر المعتمد للرسم: ${slope.toFixed(6)} وحدة سعر/جلسة. الرسم ثابت بلا Zoom. البيانات حتى ${s.market_day}.`;
if(typeof Plotly==='undefined'){$('chart').textContent='مكتبة الرسم لم تتحمل؛ اتصال المتصفح بالإنترنت مطلوب.';return;}
Plotly.react('chart',traces,layout,{displayModeBar:false,scrollZoom:false,responsive:true});
}
$('symbol').addEventListener('change',renderTable);$('filter').addEventListener('change',renderTable);$('side').addEventListener('change',renderTable);$('scope').addEventListener('change',renderTable);
$('fetchdata').addEventListener('click',async()=>{
 const b=$('fetchdata');const key=prompt('رمز تحديث الأسعار (EGX_3D_REFRESH_TOKEN):');if(!key)return;b.disabled=true;b.textContent='جارٍ تحديث الأسعار...';
 try{
  const r=await fetch('/update-data',{method:'POST',headers:{'X-Refresh-Token':key}});const x=await r.json();
  if(!r.ok)throw Error(x.error||'فشل بدء التحديث');
  $('updateStatus').textContent='جارٍ جلب الأسعار من Yahoo Finance ثم إعادة حساب Time–Price. قد يستغرق هذا عدة دقائق.';
  while(true){
    await new Promise(resolve=>setTimeout(resolve,3000));
    const q=await fetch('/update-status',{cache:'no-store'});const job=await q.json();
    $('updateStatus').textContent=job.message||'جارٍ التحديث...';
    if(job.status==='complete'){alert(job.message);location.reload();return;}
    if(job.status==='failed')throw Error(job.message||'تعذر التحديث');
  }
 }catch(e){alert('فشل تحديث البيانات: '+e.message);b.disabled=false;b.textContent='تحديث البيانات';}
});
$('refresh').addEventListener('click',async()=>{let b=$('refresh');let token=prompt('رمز تحديث الأسعار (إن تم ضبطه في Railway):')||'';b.disabled=true;b.textContent='جارٍ إعادة الحساب...';try{let r=await fetch('/refresh',{method:'POST',headers:{'X-Refresh-Token':token}});let x=await r.json();if(!r.ok)throw Error(x.error||'فشل');location.reload();}catch(e){alert('تعذر الحساب: '+e.message);b.disabled=false;b.textContent='إعادة الحساب من قاعدة البيانات';}});
renderTable();
</script></body></html>'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='/tmp/egx-time-price-3d')
    ap.add_argument('--serve',action='store_true')
    ap.add_argument('--port',type=int,default=8091)
    ap.add_argument('--unit-days',type=float,default=20)
    a=ap.parse_args()
    if a.unit_days<=0:ap.error('--unit-days must be positive')
    kwargs={'unit_days':a.unit_days}
    state['data']=build(a.output_dir,**kwargs)
    print(json.dumps(state['data']['summary'],ensure_ascii=False,indent=2),flush=True)
    html=HTML.replace('__DATA__',json.dumps(state['data'],ensure_ascii=False,default=str).replace('</','<\\/'))
    Path(a.output_dir,'index.html').write_text(html,encoding='utf-8')
    print('Saved:',Path(a.output_dir,'index.html'),flush=True)
    if not a.serve:return
    from flask import Flask,Response,request,jsonify
    app=Flask(__name__)
    @app.get('/')
    def home():
        with state_lock:data=state['data']
        return Response(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False,default=str).replace('</','<\\/')),mimetype='text/html')
    @app.post('/refresh')
    def refresh():
        # Optional private service: callers must hold REFRESH_TOKEN to refresh if set.
        expected=os.getenv('EGX_3D_REFRESH_TOKEN')
        if expected and request.headers.get('X-Refresh-Token')!=expected:
            return jsonify(error='Refresh access requires configured token'),403
        with state_lock:
            if state['busy']:return jsonify(error='Another refresh is running'),409
            state['busy']=True
        try:
            d=build(a.output_dir,**kwargs)
            with state_lock:state['data']=d
            return jsonify(d['summary'])
        except Exception as e:return jsonify(error=str(e)),500
        finally:
            with state_lock:state['busy']=False
    @app.post('/update-data')
    def update_data():
        # Only one update/recompute at once; explicit configured token is REQUIRED to write prices.
        expected=os.getenv('EGX_3D_REFRESH_TOKEN')
        if not expected:
            return jsonify(error='Set EGX_3D_REFRESH_TOKEN in Railway for write access'),403
        if request.headers.get('X-Refresh-Token') != expected:
            return jsonify(error='Access denied; update requires authorization'),403
        with state_lock:
            if state['busy']:return jsonify(error='هناك تحديث أو حساب جارٍ بالفعل'),409
            state['busy']=True
            state['job']={'status':'running','message':'جارٍ تنزيل أحدث شموع السوق المصري...'}
        def task():
            try:
                from egx_3d_market_update import update_egx_prices
                report=update_egx_prices()
                with state_lock:state['job']={'status':'running','message':'اكتمل جلب الأسعار، جارٍ حساب المستويات الهندسية...'}
                result=build(a.output_dir,**kwargs)
                with state_lock:
                    previous=state['data']['summary']['market_day'] if state['data'] else None
                    current=result['summary']['market_day']
                    state['data']=result
                    message=(f"انتهى التحديث. آخر جلسة مشتركة: {current} (كانت {previous}). "
                             f"تم حفظ {report['rows_inserted']} شمعة جديدة وتحديث {report['rows_updated']} شمعة، "
                             f"وآخر تاريخ من المصدر: {report['latest_source_date'] or 'غير متاح'}. "
                             + ('المصدر لم يقدّم جلسة أحدث للسوق كله.' if current==previous else 'تم تحديث عرض السوق.'))
                    state['job']={'status':'complete','message':message,'report':report}
            except Exception as exc:
                with state_lock:state['job']={'status':'failed','message':str(exc)[:500]}
            finally:
                with state_lock:state['busy']=False
        threading.Thread(target=task,daemon=True).start()
        return jsonify(status='running'),202
    @app.get('/update-status')
    def update_status():
        with state_lock: return jsonify(state['job'])
    app.run(host='0.0.0.0',port=int(os.getenv('PORT') or a.port),debug=False)
if __name__=='__main__':main()
