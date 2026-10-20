"""USA dynamic screener backed by persistent PostgreSQL data."""
import csv, io, os, re
from flask import Flask, render_template, request, jsonify, Response
from sqlalchemy import create_engine, text
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

def run_query(opts):
 day,start,end,conditions,logic,sort,limit=opts
 market=chosen_market();table=table_for(market)
 predicates=[]; params={}
 if market!='USA':predicates.append('market=:market');params['market']=market
 if day:predicates.append('session_date = :day');params['day']=day
 else:
  if start:predicates.append('session_date >= :start');params['start']=start
  if end:predicates.append('session_date <= :end');params['end']=end
  if not start and not end:predicates.append('session_date = (SELECT MAX(session_date) FROM '+table+(' WHERE market=:market' if market!='USA' else '')+')')
 fl=[]
 for i,(field,op,val,val2) in enumerate(conditions):
  # field and operator can only originate from allowlists
  params[f'v{i}']=val
  if op=='between':
   params[f'v2_{i}']=val2;fl.append(f'({field} BETWEEN :v{i} AND :v2_{i})')
  else:fl.append(f'({field} {OPS[op]} :v{i})')
 if fl:predicates.append('('+(f' {logic} ').join(fl)+')')
 where=' AND '.join(predicates) if predicates else 'TRUE'
 with engine().connect() as cn:
  if cn.execute(text('SELECT to_regclass(:table)'),{'table':table}).scalar() is None:
   raise ValueError('جدول فحص السوق غير موجود. ابنِي بيانات السوق المختار أولاً باستخدام rs_multi_build.py لمصر والسعودية.')
  total=cn.execute(text(f'SELECT COUNT(*) FROM {table} WHERE {where}'),params).scalar()
  rows=cn.execute(text(f'SELECT * FROM {table} WHERE {where} ORDER BY {sort} DESC NULLS LAST LIMIT :lim'),{**params,'lim':limit}).mappings().all()
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
