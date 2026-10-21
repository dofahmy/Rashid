"""USA dynamic screener backed by persistent PostgreSQL data."""
import csv, io, os, re
from flask import Flask, render_template, request, jsonify, Response
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

app = Flask(__name__, template_folder='templates')
FIELDS = {
 'rp1':'أداء اليوم مقابل مؤشر السوق (نقطة مئوية)',
 'rp5':'RS آخر 5 جلسات','rp10':'RS آخر 10 جلسات','rp20':'RS آخر 20 جلسة',
 'rp60':'RS آخر 60 جلسة','rp120':'RS آخر 120 جلسة','rp252':'RS آخر 252 جلسة',
 'avg_rp5':'متوسط التفوق اليومي 5 جلسات','avg_rp10':'متوسط التفوق اليومي 10 جلسات',
 'avg_rp20':'متوسط التفوق اليومي 20 جلسة','avg_rp60':'متوسط التفوق اليومي 60 جلسة',
 'avg_rp120':'متوسط التفوق اليومي 120 جلسة',
 'stock_day_pct':'تغير السهم في يوم','sp500_day_pct':'تغير مؤشر السوق في يوم',
 'adx14':'ADX14','plus_di14':'+DI14','minus_di14':'-DI14',
 'relative_volume20':'Relative Volume 20','relative_volatility14_60':'Relative Volatility 14/60',
 'close':'الإغلاق'
}
OPS={'gt':'>','ge':'>=','lt':'<','le':'<=','eq':'=','between':'BETWEEN'}
ORDERABLE=set(FIELDS)|{'symbol','session_date'}
_engine=None

def engine():
 global _engine
 if _engine is None:
  url=os.getenv('DATABASE_URL')
  if not url: raise RuntimeError('DATABASE_URL غير موجود')
  _engine=create_engine(url,pool_pre_ping=True,pool_recycle=300)
 return _engine

def chosen_market():
 market=request.args.get('market','USA').upper()
 if market not in ('USA','EG','SA'):raise ValueError('سوق غير معروف')
 return market

def table_for(market):
 return 'public.rs_screen_features' if market=='USA' else 'public.rs_market_features'

def table_status(market='USA'):
 try:
  with engine().connect() as cn:
   table=table_for(market)
   exists=cn.execute(text('SELECT to_regclass(:table) IS NOT NULL'),{'table':table}).scalar()
   if not exists:return False,None
   total=cn.execute(text(f'SELECT COUNT(*) FROM {table}'+(' WHERE market=:m' if market!='USA' else '')),{'m':market} if market!='USA' else {}).scalar()
   return total>0,total
 except (SQLAlchemyError,RuntimeError):return False,None

def parse():
 from datetime import date
 a=request.args
 day=a.get('day','').strip();start=a.get('start','').strip();end=a.get('end','').strip()
 if day and (start or end):raise ValueError('اختاري يوم واحد أو فترة، مش الاتنين معًا')
 for value in (day,start,end):
  if value:
   try:date.fromisoformat(value)
   except ValueError:raise ValueError('التاريخ غير صحيح')
 if start and end and start>end:raise ValueError('تاريخ البداية بعد النهاية')
 fields=a.getlist('field');ops=a.getlist('op');vals=a.getlist('value');vals2=a.getlist('value2')
 if len(fields)>12:raise ValueError('الحد الأقصى 12 شرط')
 conditions=[]
 for i,field in enumerate(fields):
  if not field:continue
  op=ops[i] if i<len(ops) else ''
  if field not in FIELDS or op not in OPS:raise ValueError('شرط غير معروف')
  try:
   val=float(vals[i]); val2=float(vals2[i]) if op=='between' else None
  except (ValueError,IndexError):raise ValueError('أدخلي قيمة رقمية لكل شرط')
  import math
  if not math.isfinite(val) or abs(val)>1e10 or (val2 is not None and (not math.isfinite(val2) or abs(val2)>1e10)):raise ValueError('قيمة خارج النطاق')
  if op=='between' and val2<val:raise ValueError('حدود النطاق معكوسة')
  conditions.append((field,op,val,val2))
 logic=a.get('logic','AND').upper()
 if logic not in ('AND','OR'):logic='AND'
 sort=a.get('sort','rp1')
 if sort not in ORDERABLE:sort='rp1'
 try:limit=int(a.get('limit','500'))
 except ValueError:limit=500
 limit=max(1,min(2000,limit))
 return day,start,end,conditions,logic,sort,limit

# Benchmark-sensitive metrics; technical indicators remain ticker-specific.
RS_FIELDS = {k for k in FIELDS if k.startswith('rp') or k.startswith('avg_rp')}
RS_FIELDS |= {'sp500_day_pct'}
STOCK_FIELDS = ['close','stock_day_pct','adx14','plus_di14','minus_di14',
                'relative_volume20','relative_volatility14_60']
VIEW_FIELDS = ['symbol','session_date'] + list(FIELDS)


def benchmark_choice(market):
 choice=request.args.get('benchmark','EGX30').upper()
 if market!='EG':return 'DEFAULT'
 if choice not in ('EGX30','EGXALL','OR','AND'):
  raise ValueError('المؤشر المرجعي غير معروف')
 return choice


def conditions_sql(conditions,logic,params):
 parts=[]
 for i,(field,op,val,val2) in enumerate(conditions):
  params[f'v{i}']=val
  if op=='between':
   params[f'v2_{i}']=val2
   parts.append(f'(q.{field} BETWEEN :v{i} AND :v2_{i})')
  else:parts.append(f'(q.{field} {OPS[op]} :v{i})')
 return '('+(f' {logic} ').join(parts)+')' if parts else 'TRUE'


def available_egxall_columns(cn):
 exists=cn.execute(text("SELECT to_regclass('public.rs_egxall_features')")).scalar()
 if exists is None:raise ValueError('جدول RS الخاص بـ EGXALL غير موجود؛ شغّلي build_egxall_rs.py --write أولاً')
 columns={x['name'] for x in inspect(cn).get_columns('rs_egxall_features',schema='public')}
 required={'symbol','session_date'} | (RS_FIELDS-{'sp500_day_pct'})
 absent=required-columns
 if absent:raise ValueError('أعمدة ناقصة في جدول EGXALL: '+', '.join(sorted(absent)))
 return columns


def eg_sources(cn, choice):
 def projection(prefix,override={}):
  expr=['symbol','session_date']
  for fld in FIELDS:
   expr.append(f"{override.get(fld,prefix+'.'+fld)} AS {fld}")
  return ', '.join(expr)
 sql30='SELECT g.symbol, g.session_date, '+', '.join(f'g.{k} AS {k}' for k in FIELDS)+" FROM public.rs_market_features g WHERE g.market='EG'"
 # project EGXALL RS values directly from its own table, not recycled EGX30 RS.
 cols=available_egxall_columns(cn) if choice!='EGX30' else set()
 if choice=='EGX30':return {'EGX30':sql30}
 overrides={}
 for fld in FIELDS:
  if fld in RS_FIELDS:
   if fld=='sp500_day_pct':
    overrides[fld]=('a.sp500_day_pct' if fld in cols else 'ix.market_return_pct')
   else:overrides[fld]='a.'+fld
  elif fld in STOCK_FIELDS:
   if fld in ('close','stock_day_pct') and fld in cols:
    overrides[fld]=f'COALESCE(a.{fld},g.{fld})'
   else:overrides[fld]='g.'+fld
 # EGXALL has only exact benchmark dates; technical indicators are shared by symbol-date.
 sqla='SELECT '+', '.join(['a.symbol AS symbol','a.session_date AS session_date']+
      [f'{overrides[k]} AS {k}' for k in FIELDS])+""" FROM public.rs_egxall_features a
 LEFT JOIN public.rs_market_features g ON g.market='EG' AND g.symbol=a.symbol AND g.session_date=a.session_date
 LEFT JOIN public.eg_internal_index_draft ix ON ix.session_date=a.session_date"""
 return {'EGX30':sql30,'EGXALL':sqla} if choice in ('AND','OR') else {'EGXALL':sqla}


def run_query(opts):
 day,start,end,conditions,logic,sort,limit=opts
 market=chosen_market();choice=benchmark_choice(market)
 params={'lim':limit}
 with engine().connect() as cn:
  if market!='EG' or choice=='EGX30':
   table=table_for(market)
   if cn.execute(text('SELECT to_regclass(:t)'),{'t':table}).scalar() is None:
    raise ValueError('جدول فحص السوق غير موجود')
   if market!='EG':
    source=f'SELECT * FROM {table}'+(' WHERE market=:market' if market=='SA' else '')
    if market=='SA':params['market']='SA'
   else:source='SELECT symbol, session_date, '+', '.join(FIELDS)+" FROM public.rs_market_features WHERE market='EG'"
   sources={'DEFAULT':source} if market!='EG' else {'EGX30':source}
  else:
   if cn.execute(text("SELECT to_regclass('public.rs_market_features')")).scalar() is None:
    raise ValueError('جدول مصر الأصلي غير موجود')
   sources=eg_sources(cn,choice)
  clauses=[]
  if day:clauses.append('q.session_date=:day');params['day']=day
  else:
   if start:clauses.append('q.session_date>=:start');params['start']=start
   if end:clauses.append('q.session_date<=:end');params['end']=end
  filt=conditions_sql(conditions,logic,params)
  arms=[]
  for name,src in sources.items():
   label=name if name!='DEFAULT' else ('USA' if market=='USA' else 'TASI')
   arms.append(f"SELECT q.*, '{label}'::text AS source_benchmark FROM ({src}) q WHERE {filt}")
  combined=' UNION ALL '.join(arms)
  # Filter first, then combine. Thus all conditions must pass on a single benchmark.
  cte='WITH hits AS ('+combined+'), marked AS (SELECT hits.*, '
  if market=='EG' and choice in ('OR','AND'):
   cte+="""BOOL_OR(source_benchmark='EGX30') OVER (PARTITION BY symbol,session_date) AS hit30,
 BOOL_OR(source_benchmark='EGXALL') OVER (PARTITION BY symbol,session_date) AS hitall,
 ROW_NUMBER() OVER (PARTITION BY symbol,session_date ORDER BY CASE WHEN source_benchmark='EGX30' THEN 0 ELSE 1 END) AS seq
 FROM hits), picked AS (SELECT *, CASE WHEN hit30 AND hitall THEN 'EGX30 + EGXALL'
 WHEN hit30 THEN 'EGX30' ELSE 'EGXALL' END AS matched_benchmarks
 FROM marked WHERE seq=1"""
   if choice=='AND':cte+=' AND hit30 AND hitall'
   cte+=')'
  else:
   cte+="FALSE AS hit30,FALSE AS hitall,1 AS seq FROM hits), picked AS (SELECT *,source_benchmark AS matched_benchmarks FROM marked)"
  outer=clauses[:]
  if not day and not start and not end:
   # For mixed mode use last date with qualifying source data, not latest for each arm.
   outer.append('q.session_date=(SELECT MAX(session_date) FROM picked)')
  where=' AND '.join(outer) or 'TRUE'
  total=cn.execute(text(cte+f' SELECT COUNT(*) FROM picked q WHERE {where}'),params).scalar()
  rows=cn.execute(text(cte+f""" SELECT q.* FROM picked q WHERE {where}
   ORDER BY q.{sort} DESC NULLS LAST, q.session_date DESC, q.symbol ASC LIMIT :lim"""),params).mappings().all()
  return [dict(x) for x in rows],total

@app.after_request
def no_cache(resp):resp.headers['Cache-Control']='no-store';return resp
@app.get('/health')
def health():
 ready,n=table_status(request.args.get('market','USA').upper())
 return jsonify(ok=True,service='interactive-rs-screening-postgres',features_db_exists=ready,backend='postgres',rows=n)
@app.get('/')
@app.get('/screening')
def screening():
 market=chosen_market()
 ready,_=table_status(market);rows=[];total=0;error=None
 if request.args.get('run')=='1':
  try:rows,total=run_query(parse())
  except (ValueError,RuntimeError,SQLAlchemyError) as ex:error=str(ex)
 return render_template('rs_dynamic.html',market=market,rows=rows,total=total,error=error,fields=FIELDS,ops=OPS,args=request.args,ready=ready,ran=request.args.get('run')=='1')
@app.get('/screening.json')
def screen_json():
 try:
  rows,total=run_query(parse());return jsonify(total_matches=total,shown=len(rows),rows=rows)
 except (ValueError,RuntimeError,SQLAlchemyError) as ex:return jsonify(error=str(ex)),400
@app.get('/screening.csv')
def screen_csv():
 try:rows,total=run_query(parse())
 except (ValueError,RuntimeError,SQLAlchemyError) as ex:return jsonify(error=str(ex)),400
 out=io.StringIO()
 if rows:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return Response('\ufeff'+out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=screening_results.csv','X-Total-Matches':str(total)})
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.getenv('PORT','8080')))
