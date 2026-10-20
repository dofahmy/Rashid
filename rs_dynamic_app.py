"""Interactive full-market technical/relative-performance screener over an indexed SQLite feature store."""
import csv, io, json, os, sqlite3
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
app=Flask(__name__,template_folder='templates')
DB=Path(os.getenv('RS_DYNAMIC_DB','/data/rs_dynamic/market_features.sqlite'))
FIELDS={
 'rp1':'أداء اليوم مقابل S&P500 (نقطة مئوية)',
 'rp5':'RS آخر 5 جلسات','rp10':'RS آخر 10 جلسات','rp20':'RS آخر 20 جلسة',
 'rp60':'RS آخر 60 جلسة','rp120':'RS آخر 120 جلسة','rp252':'RS آخر 252 جلسة',
 'avg_rp5':'متوسط التفوق اليومي 5 جلسات','avg_rp10':'متوسط التفوق اليومي 10 جلسات',
 'avg_rp20':'متوسط التفوق اليومي 20 جلسة','avg_rp60':'متوسط التفوق اليومي 60 جلسة',
 'avg_rp120':'متوسط التفوق اليومي 120 جلسة',
 'stock_day_pct':'تغير السهم في يوم','sp500_day_pct':'تغير المؤشر في يوم',
 'adx14':'ADX14','plus_di14':'+DI14','minus_di14':'-DI14',
 'relative_volume20':'Relative Volume 20','relative_volatility14_60':'Relative Volatility 14/60',
 'close':'الإغلاق'}
OPERATORS={'gt':'>','ge':'>=','lt':'<','le':'<=','eq':'=','between':'BETWEEN'}
ORDERABLE=set(FIELDS)|{'symbol','session_date'}

def conn():
 if not DB.is_file():return None
 c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True,timeout=10)
 c.row_factory=sqlite3.Row
 return c

def parse():
 a=request.args
 day=a.get('day','').strip()
 start=a.get('start','').strip();end=a.get('end','').strip()
 if day and (start or end): raise ValueError('اختاري يوم واحد أو فترة، مش الاتنين معًا')
 import datetime
 for v in [day,start,end]:
  if v:
   try:datetime.date.fromisoformat(v)
   except ValueError:raise ValueError('التاريخ غير صحيح')
 if start and end and start>end:raise ValueError('تاريخ البداية بعد النهاية')
 conditions=[]
 fields=a.getlist('field');ops=a.getlist('op');values=a.getlist('value');values2=a.getlist('value2')
 if len(fields)>12:raise ValueError('الحد الأقصى 12 شرط')
 for i,f in enumerate(fields):
  if not f:continue
  op=ops[i] if i<len(ops) else ''
  if f not in FIELDS or op not in OPERATORS:raise ValueError('شرط غير معروف')
  try:
   v=float(values[i]);v2=float(values2[i]) if op=='between' else None
  except (ValueError,IndexError):raise ValueError('أدخلي قيمة رقمية لكل شرط')
  if not (-1e10<v<1e10) or (v2 is not None and not -1e10<v2<1e10):raise ValueError('قيمة خارج النطاق')
  if op=='between' and v2<v: raise ValueError('حدود النطاق معكوسة')
  conditions.append((f,op,v,v2))
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
 c=conn()
 if c is None:raise FileNotFoundError('قاعدة فحص السوق غير موجودة. شغّلي rs_dynamic_build.py مرة أولًا.')
 try:
  wh=[];params=[]
  if day:wh.append('session_date=?');params.append(day)
  else:
   if start:wh.append('session_date>=?');params.append(start)
   if end:wh.append('session_date<=?');params.append(end)
   if not start and not end:
    latest=c.execute('SELECT max(session_date) FROM screen').fetchone()[0]
    wh.append('session_date=?');params.append(latest)
  inner=[];inner_p=[]
  for field,op,v,v2 in conditions:
   if op=='between':inner.append(f'({field} BETWEEN ? AND ?)');inner_p.extend([v,v2])
   else:inner.append(f'({field} {OPERATORS[op]} ?)');inner_p.append(v)
  if inner:wh.append('('+(' '+logic+' ').join(inner)+')');params+=inner_p
  where=' AND '.join(wh) if wh else '1=1'
  count=c.execute('SELECT count(*) FROM screen WHERE '+where,params).fetchone()[0]
  rows=c.execute(f'SELECT * FROM screen WHERE {where} ORDER BY {sort} DESC LIMIT ?',params+[limit]).fetchall()
  return [dict(x) for x in rows],count
 finally:c.close()

@app.after_request
def headers(r):r.headers['Cache-Control']='no-store';return r
@app.get('/health')
def health():return jsonify(ok=True,service='interactive-rs-screening',features_db_exists=DB.is_file())
@app.get('/')
@app.get('/screening')
def screening():
 opts=None;rows=[];total=0;err=None
 if request.args.get('run')=='1':
  try:opts=parse();rows,total=run_query(opts)
  except (ValueError,FileNotFoundError,sqlite3.Error) as e:err=str(e)
 return render_template('rs_dynamic.html',rows=rows,total=total,error=err,fields=FIELDS,ops=OPERATORS,args=request.args,ready=DB.is_file(),ran=request.args.get('run')=='1')
@app.get('/screening.json')
def screening_json():
 try:
  rows,total=run_query(parse())
  return jsonify({'total_matches':total,'shown':len(rows),'rows':rows})
 except (ValueError,FileNotFoundError,sqlite3.Error) as e:return jsonify({'error':str(e)}),400
@app.get('/screening.csv')
def screening_csv():
 try:rows,total=run_query(parse())
 except (ValueError,FileNotFoundError,sqlite3.Error) as e:return jsonify(error=str(e)),400
 out=io.StringIO()
 if rows:
  w=csv.DictWriter(out,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 return Response('\ufeff'+out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=screening_results.csv','X-Total-Matches':str(total)})
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.getenv('PORT',8080)))
