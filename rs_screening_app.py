"""Public read-only web replacement for the Repeat screening UI. No DB writes."""
import json, os
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app=Flask(__name__,template_folder='templates')
SNAPSHOT=Path(os.getenv('RS_SCREENING_SNAPSHOT','/data/rs_screening/snapshot.json'))
@app.after_request
def headers(resp):
    resp.headers['Cache-Control']='no-store';resp.headers['X-Content-Type-Options']='nosniff'
    return resp
@app.get('/health')
def health():return {'ok':True,'service':'seven-rs-screening','snapshot_exists':SNAPSHOT.exists()}
@app.get('/')
@app.get('/screening')
def screening():
    try: d=json.loads(SNAPSHOT.read_text(encoding='utf8'))
    except (OSError,ValueError):d={'rows':[],'status':'snapshot_missing','benchmark_last_date':'—','stock_groups':0,'total_rs_matches':0}
    mode=request.args.get('stage2','minmax')
    if mode not in ('minmax','trimmed','rs_only'):mode='minmax'
    reference=request.args.get('strategy','all').upper()
    if reference not in ('ALL','BGMS','YDKG','FROG','MU','IMNM','ZIM','IMSR'):reference='ALL'
    rows=d['rows']
    if reference!='ALL':rows=[r for r in rows if r['reference']==reference]
    if mode!='rs_only':rows=[r for r in rows if r[mode+'_pass']]
    rows=rows[:2000]
    return render_template('rs_screening.html',d=d,rows=rows,mode=mode,reference=reference)
@app.get('/screening.json')
def screening_json():
    if not SNAPSHOT.exists():return jsonify({'error':'snapshot_missing'}),503
    return jsonify(json.loads(SNAPSHOT.read_text(encoding='utf8')))
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.getenv('PORT',8080)))
