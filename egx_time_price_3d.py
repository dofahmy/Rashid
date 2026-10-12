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
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

RATIOS = {'1x2': .5, '1x1': 1., '2x1': 2.}
state = {'data': None, 'error': None, 'busy': False}
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

def build(out_dir,unit_days=20.,near_atr=.15,wing=5,lookback=180,min_atr=1.5):
    df,bad=read_bars()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    bad.to_csv(out/'invalid_bars.csv',index=False,encoding='utf-8-sig')
    # newest MODE rather than maximum: avoids one isolated symbol with a future quote
    dates=df.groupby('symbol').session_date.max().dt.strftime('%Y-%m-%d')
    mode_day=str(dates.mode().iloc[0]) if len(dates) else None
    levels=[];viz={};skip=[]
    for sym,g in df.groupby('symbol',sort=True):
        g=g.sort_values('session_date').reset_index(drop=True)
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
        viz[sym]={'candles':chart,'angles':angles,'last_close':close}
    allrows=pd.DataFrame(levels)
    if allrows.empty:raise RuntimeError('No levels found; check market bars and pivots')
    hits=allrows[allrows.touch].copy()
    nearby=allrows[allrows.touch|allrows.near].copy()
    allrows.to_csv(out/'all_levels.csv',index=False,encoding='utf-8-sig')
    hits.to_csv(out/'touched_levels.csv',index=False,encoding='utf-8-sig')
    nearby.to_csv(out/'touched_or_near.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(skip).to_csv(out/'skipped.csv',index=False,encoding='utf-8-sig')
    summary={'version':'EGX-ANGLE-3D-v1','market_day':mode_day,
             'source_stocks':int(df.symbol.nunique()),'analyzed_stocks':len(viz),
             'touching_stocks':int(hits.symbol.nunique()),'touched_levels':len(hits),
             'invalid_bars':len(bad),'calibration':'1x1 = anchor ATR14 / '+str(unit_days)+' sessions',
             'generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'note':'Geometric visualization uses a fixed price/time conversion, no zoom; depth is illustrative, not an independently measured third variable. Refresh recalculates stored DB prices only.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'summary':summary,'viz':viz,'hits':hits.to_dict(orient='records'),
            'nearby':nearby.to_dict(orient='records')}

HTML = '''<!DOCTYPE html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EGX — Time Price 3D</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{margin:0;background:#f4f7fb;font-family:Tahoma,Arial,sans-serif;color:#183047}header{background:#112b46;color:#fff;padding:18px 4%}main{max-width:1500px;margin:auto;padding:20px}article{background:white;border:1px solid #dce6ee;border-radius:14px;padding:18px;margin-bottom:16px}.toolbar{display:flex;align-items:end;gap:10px;flex-wrap:wrap}button,select,input{padding:11px;border:1px solid #bccbd8;border-radius:8px;font:inherit}button{background:#137a82;color:white;cursor:pointer}button:disabled{opacity:.5}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #e7ebf0;text-align:right}th{background:#edf3f7}.scroll{overflow:auto}#chart{height:570px;width:100%}.small{color:#68798a;font-size:12px;line-height:1.8}.stat{display:inline-block;background:#e7f3ef;border-radius:8px;padding:10px;margin:6px}</style></head><body>
<header><h2>Time–Price Squaring — السوق المصري</h2><div>لمسات الخطوط الهندسية 1×1 و1×2 و2×1 — من قمم وقيعان مؤكدة</div></header>
<main><article><div id="stats"></div><div class="small">المقياس الهندسي الرقمي ثابت: 1×1 = ATR14 عند القمة/القاع ÷ 20 جلسة. بُعد العمق للعرض فقط. تم تعطيل تدوير الرسم والـZoom لمنع تغيير الزاوية بصريًا. هذا نموذج معايرة تجريبي وليس زاوية جان مطلقة.</div></article>
<article><div class="toolbar"><label>السهم<br><select id="symbol"></select></label><label>النتائج<br><select id="filter"><option value="touch">لمس فعلي</option><option value="near">لمس أو قريب</option><option value="all">كل الأسهم</option></select></label><button id="refresh">إعادة الحساب من قاعدة البيانات</button><button onclick="location.reload()">تحديث عرض الصفحة</button></div><p id="note" class="small"></p><div id="chart"></div></article>
<article><h3>تفاصيل المستويات — آخر جلسة</h3><div class="scroll"><table><thead><tr><th>السهم</th><th>القمة/القاع</th><th>تاريخ الارتكاز</th><th>الزاوية</th><th>سعر الخط</th><th>Low / High</th><th>الإغلاق</th><th>الحالة</th></tr></thead><tbody id="rows"></tbody></table></div></article></main>
<script>
const dataset=__DATA__;
const $=x=>document.getElementById(x);
const statusAr={CROSS_UP_CLOSE:'اختراق صاعد',CROSS_DOWN_CLOSE:'كسر هابط',TOUCH_ABOVE:'لمس وإغلاق فوق',TOUCH_BELOW:'لمس وإغلاق تحت',NEAR:'قريب من الخط',NONE:'لم يلمس'};
const s=dataset.summary;
$('stats').innerHTML=`<span class="stat">آخر جلسة: ${s.market_day}</span><span class="stat">أسهم مفحوصة: ${s.analyzed_stocks}</span><span class="stat">أسهم لمست: ${s.touching_stocks}</span><span class="stat">مستويات ملموسة: ${s.touched_levels}</span><span class="stat">شموع غير سليمة: ${s.invalid_bars}</span>`;
function shown(){let f=$('filter').value;if(f==='all')return Object.values(dataset.viz).flatMap(x=>x.angles);return f==='near'?dataset.nearby:dataset.hits;}
function renderTable(){let rows=shown();let syms=[...new Set(rows.map(x=>x.symbol))].sort();let last=$('symbol').value;$('symbol').innerHTML=syms.map(x=>`<option value="${x}">${x}</option>`).join('');$('symbol').value=syms.includes(last)?last:(syms[0]||''); renderChart();}
function renderChart(){let sym=$('symbol').value;let set=dataset.viz[sym];if(!set){$('rows').innerHTML='';$('note').textContent='لا توجد مستويات';return;}
let lines=shown().filter(x=>x.symbol===sym);$('rows').innerHTML=lines.map(x=>`<tr><td>${x.symbol}</td><td>${x.anchor_type}</td><td>${x.anchor_date}</td><td>${x.angle}</td><td>${x.level.toFixed(4)}</td><td>${x.last_low.toFixed(4)} / ${x.last_high.toFixed(4)}</td><td>${x.last_close.toFixed(4)}</td><td>${statusAr[x.status]}</td></tr>`).join('');
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
$('symbol').addEventListener('change',renderChart);$('filter').addEventListener('change',renderTable);
$('refresh').addEventListener('click',async()=>{let b=$('refresh');b.disabled=true;b.textContent='جارٍ إعادة الحساب...';try{let r=await fetch('/refresh',{method:'POST'});let x=await r.json();if(!r.ok)throw Error(x.error||'فشل');location.reload();}catch(e){alert('تعذر الحساب: '+e.message);b.disabled=false;b.textContent='إعادة الحساب من قاعدة البيانات';}});
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
    app.run(host='0.0.0.0',port=int(os.getenv('PORT') or a.port),debug=False)
if __name__=='__main__':main()
