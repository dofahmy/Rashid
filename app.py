import os, secrets, csv, io, math, time
from functools import wraps
from datetime import timedelta, datetime, timezone
from urllib.parse import quote
from flask import Flask, render_template, request, session, redirect, url_for, abort, Response, flash
from sqlalchemy import select, func, or_, case
from werkzeug.security import check_password_hash
from core import database, Lead, Activity, Outbox, Setting, MARKETS, STATUSES, BRAND, record, queue, now

def create_app(db=None,test_config=None):
    app=Flask(__name__)
    app.config.update(SECRET_KEY=os.getenv('SECRET_KEY'),MAX_CONTENT_LENGTH=32*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Lax',SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE','1')=='1',PERMANENT_SESSION_LIFETIME=timedelta(hours=8),ADMIN_PASSWORD_HASH=os.getenv('ADMIN_PASSWORD_HASH',''))
    if test_config: app.config.update(test_config)
    if not app.config['SECRET_KEY'] or not app.config['ADMIN_PASSWORD_HASH']: raise RuntimeError('Set SECRET_KEY and ADMIN_PASSWORD_HASH before starting administration.')
    DB=db or database(); failures={}
    def csrf():
        if 'csrf' not in session: session['csrf']=secrets.token_urlsafe(32)
        return session['csrf']
    @app.context_processor
    def context(): return dict(csrf=csrf,brand=BRAND,markets=MARKETS,statuses=STATUSES)
    @app.before_request
    def protect():
        if request.method=='POST' and not secrets.compare_digest(str(session.get('csrf','')),str(request.form.get('csrf',''))) or request.method=='POST' and not session.get('csrf'):
            abort(403)
    @app.after_request
    def headers(r):
        r.headers['X-Content-Type-Options']='nosniff'; r.headers['X-Frame-Options']='DENY'; r.headers['Referrer-Policy']='no-referrer'
        r.headers['Content-Security-Policy']="default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; base-uri 'self'"
        if request.path!='/health': r.headers['Cache-Control']='no-store'
        return r
    def auth(fn):
        @wraps(fn)
        def wrapped(*a,**kw):
            if not session.get('admin'): return redirect(url_for('login'))
            return fn(*a,**kw)
        return wrapped
    @app.get('/health')
    def health():
        with DB() as s: s.execute(select(1))
        return {'ok':True,'version':'rajih-us-m15-v5'}
    @app.route('/login',methods=['GET','POST'])
    def login():
        if request.method=='POST':
            key=request.remote_addr; t=time.monotonic(); attempts,until=failures.get(key,(0,0))
            if until>t: flash('محاولات كثيرة. حاول بعد 15 دقيقة.'); return render_template('login.html'),429
            if check_password_hash(app.config['ADMIN_PASSWORD_HASH'],request.form.get('password','')):
                failures.pop(key,None); session.clear(); session['admin']=True; session.permanent=True; csrf(); return redirect(url_for('dashboard'))
            attempts=attempts+1 if until>t-900 else 1
            failures[key]=(attempts,t+900 if attempts>=5 else t)
            flash('كلمة المرور غير صحيحة.'); return render_template('login.html'),401
        return render_template('login.html')
    @app.post('/logout')
    @auth
    def logout(): session.clear(); return redirect(url_for('login'))
    def filters():
        conditions=[]; q=request.args.get('q','').strip()[:100]
        if q:
            pattern='%'+q.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
            conditions.append(or_(*[c.ilike(pattern,escape='\\') for c in [Lead.name,Lead.phone,Lead.username,Lead.telegram_name]],Lead.telegram_id==int(q) if q.isdecimal() and len(q)<=18 else False))
        for field,allowed in [('market',MARKETS),('status',STATUSES)]:
            v=request.args.get(field,'')
            if v in allowed: conditions.append(getattr(Lead,field)==v)
        if request.args.get('complete')=='yes': conditions.append(Lead.completed_at!='')
        elif request.args.get('complete')=='no': conditions.append(Lead.completed_at=='')
        for key,op in [('from',True),('to',False)]:
            v=request.args.get(key,'')
            if v:
                try: datetime.strptime(v,'%Y-%m-%d')
                except ValueError: abort(400)
                conditions.append(Lead.created_at>=v if op else Lead.created_at<v+'T23:59:59.999999+00:00')
        if request.args.get('source'): conditions.append(Lead.source==request.args['source'][:100])
        if request.args.get('due')=='1': conditions.extend([Lead.follow_up!='',Lead.follow_up<=datetime.now(timezone.utc).date().isoformat()])
        return conditions
    @app.get('/')
    @auth
    def dashboard():
        cond=filters()
        try: page=max(1,int(request.args.get('page','1')))
        except ValueError: page=1
        sorts={'created':Lead.created_at,'name':Lead.name,'market':Lead.market,'status':Lead.status,'source':Lead.source,'follow_up':Lead.follow_up}
        sort=request.args.get('sort','created'); col=sorts.get(sort,Lead.created_at); direction=request.args.get('direction','desc')
        with DB() as s:
            total=s.scalar(select(func.count()).select_from(Lead).where(*cond)); pages=max(1,math.ceil(total/30)); page=min(page,pages)
            rows=s.scalars(select(Lead).where(*cond).order_by(col.asc() if direction=='asc' else col.desc(),Lead.id.desc()).offset((page-1)*30).limit(30)).all()
            stats={'all':s.scalar(select(func.count()).select_from(Lead)),'complete':s.scalar(select(func.count()).select_from(Lead).where(Lead.completed_at!='')),'new':s.scalar(select(func.count()).select_from(Lead).where(Lead.completed_at!='',Lead.status=='new')),'trial':s.scalar(select(func.count()).select_from(Lead).where(Lead.status=='trial'))}
            markets_stats=dict(s.execute(select(Lead.market,func.count()).where(Lead.completed_at!='').group_by(Lead.market)).all())
            sources=s.execute(select(Lead.source,func.count(),func.sum(case((Lead.completed_at!='',1),else_=0))).group_by(Lead.source).order_by(func.count().desc())).all()
            failed=s.scalar(select(func.count()).select_from(Outbox).where(Outbox.status=='failed',Outbox.method.in_(['sendMessage','sendPhoto'])))
        def link(**kw): return url_for('dashboard',**{**request.args.to_dict(),**kw})
        return render_template('dashboard.html',rows=rows,stats=stats,market_stats=markets_stats,sources=sources,total=total,page=page,pages=pages,link=link,failed=failed)
    @app.route('/leads/<int:lead_id>',methods=['GET','POST'])
    @auth
    def detail(lead_id):
        with DB.begin() as s:
            l=s.get(Lead,lead_id)
            if not l: abort(404)
            if request.method=='POST':
                status=request.form.get('status'); owner=request.form.get('owner','').strip()[:100]; follow=request.form.get('follow_up','')
                if status not in STATUSES: abort(400)
                if follow:
                    try: datetime.strptime(follow,'%Y-%m-%d')
                    except ValueError: abort(400)
                changes=[]
                if l.status!=status: changes.append(f'الحالة: {STATUSES[l.status]} ← {STATUSES[status]}')
                if l.owner!=owner: changes.append(f'المسؤول: {owner or "بدون"}')
                if l.follow_up!=follow: changes.append(f'المتابعة: {follow or "بدون موعد"}')
                previous_status=l.status
                l.status=status; l.owner=owner; l.follow_up=follow; l.updated_at=now()
                if previous_status not in ('trial','subscribed') and status in ('trial','subscribed'):
                    from monitor.customer import eligible, buttons
                    if eligible(s,l):queue(s,'us-enabled:'+secrets.token_hex(16),l.telegram_id,
                        'تم تفعيل خدمة توصيات الأسهم الأمريكية على فريم 15 دقيقة ✅\nستصلك التوصيات الجديدة عند اكتمال شروطها. تابع المراكز المفتوحة ونتائج الشهر من الأزرار. لإيقاف التنبيهات: /stop_us',buttons())
                note=request.form.get('note','').strip()[:4000]
                if note: record(s,l,'note',note)
                if changes: record(s,l,'admin',' | '.join(changes))
                flash('تم حفظ المتابعة.'); return redirect(url_for('detail',lead_id=lead_id))
            history=s.scalars(select(Activity).where(Activity.lead_id==lead_id).order_by(Activity.id.desc())).all()
            duplicate=s.scalar(select(func.count()).select_from(Lead).where(Lead.phone==l.phone,Lead.id!=l.id)) if l.phone else 0
            outbox=s.scalars(select(Outbox).where(Outbox.chat_id==l.telegram_id,Outbox.method.in_(['sendMessage','sendPhoto'])).order_by(Outbox.id.desc()).limit(10)).all()
            return render_template('detail.html',l=l,history=history,duplicate=duplicate,outbox=outbox,telegram='https://t.me/'+l.username if l.username else f'tg://user?id={l.telegram_id}',whatsapp='https://wa.me/'+l.phone.lstrip('+') if l.phone else '')
    @app.post('/leads/<int:lead_id>/message')
    @auth
    def message(lead_id):
        text=request.form.get('message','').strip()
        if not 1<=len(text)<=4000: abort(400)
        with DB.begin() as s:
            l=s.get(Lead,lead_id)
            if not l: abort(404)
            if not l.completed_at or not l.consent_at: flash('العميل لم يؤكد التسجيل والموافقة على التواصل.'); return redirect(url_for('detail',lead_id=lead_id))
            queue(s,'manual:'+secrets.token_hex(16),l.telegram_id,text)
            record(s,l,'message',text)
        flash('أضيفت الرسالة لطابور الإرسال. تابع حالة التسليم أدناه.'); return redirect(url_for('detail',lead_id=lead_id))
    @app.get('/export.csv')
    @auth
    def export():
        with DB() as s: rows=s.scalars(select(Lead).where(*filters()).order_by(Lead.id.desc())).all()
        stream=io.StringIO(); writer=csv.writer(stream)
        writer.writerow(['ID','Telegram ID','Username','الاسم','واتساب','السوق','الحالة','المسؤول','المتابعة','المصدر','اكتمال التسجيل','موافقة التواصل','تاريخ الدخول'])
        def safe(v):
            v=str(v or '')
            return "'"+v if v.lstrip().startswith(('=','+','-','@','\t','\r')) else v
        for l in rows: writer.writerow([safe(v) for v in [l.id,l.telegram_id,l.username,l.name,l.phone,MARKETS.get(l.market,''),STATUSES[l.status],l.owner,l.follow_up,l.source,l.completed_at,l.consent_at,l.created_at]])
        return Response('\ufeff'+stream.getvalue(),mimetype='text/csv; charset=utf-8',headers={'Content-Disposition':'attachment; filename=stock-leads.csv'})
    @app.post('/stocks/settings')
    @auth
    def stock_settings():
        from monitor.customer import MIN_SCORE_KEY
        try:threshold=float(request.form.get('minimum_score',''))
        except (ValueError,TypeError):abort(400)
        if not math.isfinite(threshold) or not 0<=threshold<=100:abort(400)
        with DB.begin() as s:
            row=s.get(Setting,MIN_SCORE_KEY)
            if row is None:row=Setting(key=MIN_SCORE_KEY);s.add(row)
            row.value=str(threshold)
        flash(f'تم حفظ الحد الأدنى لإرسال توصيات الأمريكي: {threshold:g}/100')
        return redirect(url_for('stocks'))

    @app.get('/stocks')
    @auth
    def stocks():
        from monitor.models import Stock, Plan, Scan, LABELS
        from monitor.strategy import local
        from monitor.customer import minimum_score
        conditions=[]
        market=request.args.get('market','')
        state=request.args.get('state','')
        if market in ('SA','US'): conditions.append(Plan.market==market)
        if state in LABELS: conditions.append(Plan.state==state)
        symbol=request.args.get('symbol','').strip().upper()[:40]
        if symbol: conditions.append(Plan.symbol==symbol)
        try: page=max(1,int(request.args.get('page','1')))
        except ValueError: page=1
        with DB() as s:
            send_minimum_score=minimum_score(s)
            total=s.scalar(select(func.count()).select_from(Plan).where(*conditions))
            pages=max(1,math.ceil(total/50));page=min(page,pages)
            rows=s.scalars(select(Plan).where(*conditions).order_by(Plan.score.desc(),Plan.id.desc()).offset((page-1)*50).limit(50)).all()
            counts=dict(s.execute(select(Plan.state,func.count()).group_by(Plan.state)).all())
            universe=dict(s.execute(select(Stock.market,func.count()).group_by(Stock.market)).all())
            errors=s.scalar(select(func.count()).select_from(Stock).where(Stock.error!=''))
            scans=s.scalars(select(Scan).order_by(Scan.id.desc()).limit(10)).all()
            stocks_by_symbol={r.symbol:r for r in s.scalars(select(Stock).where(Stock.symbol.in_([p.symbol for p in rows])))}
        def link(**kw): return url_for('stocks',**{**request.args.to_dict(),**kw})
        return render_template('stocks.html',rows=rows,counts=counts,universe=universe,errors=errors,scans=scans,
            send_minimum_score=send_minimum_score,labels=LABELS,stock_map=stocks_by_symbol,total=total,page=page,pages=pages,link=link,local=local)

    @app.get('/stocks/<int:plan_id>')
    @auth
    def stock_plan(plan_id):
        import json
        from monitor.models import Plan, Event, Stock, LABELS
        from monitor.strategy import local
        with DB() as s:
            p=s.get(Plan,plan_id)
            if not p: abort(404)
            stock=s.get(Stock,p.symbol)
            events=s.scalars(select(Event).where(Event.plan_id==p.id).order_by(Event.id)).all()
        return render_template('stock_plan.html',p=p,stock=stock,events=events,labels=LABELS,local=local,
            context=json.loads(p.context_json),rules=json.loads(p.policy_json))

    @app.get('/stocks/feed')
    @auth
    def stock_feed():
        from monitor.models import Stock
        market=request.args.get('market','');conditions=[Stock.error!='']
        if market in ('SA','US'): conditions.append(Stock.market==market)
        with DB() as s:
            rows=s.scalars(select(Stock).where(*conditions).order_by(Stock.market,Stock.symbol).limit(250)).all()
        return render_template('stock_feed.html',rows=rows)
    return app
