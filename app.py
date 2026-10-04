import os, secrets, csv, io, math, time
from functools import wraps
from datetime import timedelta, datetime, timezone
from urllib.parse import quote
from flask import Flask, render_template, request, session, redirect, url_for, abort, Response, flash
from sqlalchemy import select, func, or_, case, delete, update
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
            l=s.scalar(select(Lead).where(Lead.id==lead_id).with_for_update())
            if not l: abort(404)
            from monitor.limits import limits,save_limits,occupied,LIMIT_FIELDS
            if request.method=='POST':
                status=request.form.get('status'); owner=request.form.get('owner','').strip()[:100]; follow=request.form.get('follow_up','')
                if status not in STATUSES: abort(400)
                if follow:
                    try: datetime.strptime(follow,'%Y-%m-%d')
                    except ValueError: abort(400)
                changes=[]
                if any('limit_'+k in request.form for k in LIMIT_FIELDS):
                    try:
                        cap={k:int(request.form.get('limit_'+k,'')) for k in LIMIT_FIELDS}
                        save_limits(s,l.telegram_id,cap)
                    except (ValueError,TypeError):
                        abort(400,description='الحدود أعداد صحيحة من 0 إلى 10000، ومجموع الفئات يجب أن يساوي الإجمالي.')
                    changes.append('حدود التوصيات: '+str(cap))
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
            return render_template('detail.html',limits=limits(s,l.telegram_id),usage=occupied(s,l.telegram_id),l=l,history=history,duplicate=duplicate,outbox=outbox,telegram='https://t.me/'+l.username if l.username else f'tg://user?id={l.telegram_id}',whatsapp='https://wa.me/'+l.phone.lstrip('+') if l.phone else '')
    @app.post('/leads/<int:lead_id>/reset-recommendations')
    @auth
    def reset_customer_recommendations(lead_id):
        if request.form.get('confirm_reset')!='1':
            abort(400,description='يجب تأكيد تصفير توصيات العميل.')
        with DB.begin() as s:
            l=s.scalar(select(Lead).where(Lead.id==lead_id).with_for_update())
            if not l: abort(404)

            from monitor.customer import Recipient

            # Recipient is the customer-specific recommendation ledger for both
            # US and commodity recommendations.  Removing only this customer's
            # rows clears their current/results views without changing the
            # global Plan state for any other customer.
            recommendation_count=s.scalar(
                select(func.count()).select_from(Recipient)
                .where(Recipient.telegram_id==l.telegram_id)
            ) or 0

            s.execute(
                delete(Recipient).where(Recipient.telegram_id==l.telegram_id)
            )

            # Prevent recommendation alerts already queued for this customer
            # from being sent after the reset.  Sent messages remain as audit
            # history because Telegram messages cannot be reliably retracted.
            queued_rows=s.scalars(
                select(Outbox).where(
                    Outbox.chat_id==l.telegram_id,
                    Outbox.status.in_(('pending','sending','uncertain')),
                    Outbox.payload.contains('"_us_plan_id"'),
                ).with_for_update()
            ).all()
            for row in queued_rows:
                row.status='cancelled'
                row.error='Cancelled by admin customer recommendation reset'

            record(
                s,l,'admin',
                f'تم تصفير توصيات العميل: أُلغي ارتباط {recommendation_count} توصية '
                f'وتم إلغاء {len(queued_rows)} رسالة توصيات معلقة. '
                'لم تتغير حدود التوصيات أو أهلية استقبال توصيات جديدة.'
            )

        flash(
            f'تم تصفير توصيات العميل. أزيلت {recommendation_count} توصية '
            f'وأُلغي {len(queued_rows)} إرسال معلّق. التوصيات الجديدة ستصل طبيعيًا حسب حالة العميل وحدوده.'
        )
        return redirect(url_for('detail',lead_id=lead_id))

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
    @app.get('/recommendations/current/<token>')
    def customer_current_table(token):
        from monitor.customer_table import authorized_lead,current_rows
        from monitor.customer import price
        with DB() as s:
            lead=authorized_lead(s,token)
            if lead is None:
                return render_template('customer_current.html',expired=True,rows=[],price=price),403
            rows=current_rows(s,lead)
        return render_template('customer_current.html',expired=False,rows=rows,price=price)

    @app.get('/recommendations/results/<token>')
    def customer_results_table(token):
        from monitor.customer_table import authorized_lead,monthly_rows
        from monitor.customer import price,LABELS
        with DB() as s:
            lead=authorized_lead(s,token)
            if lead is None:
                return render_template('customer_results.html',expired=True,rows=[],price=price,labels=LABELS,month=''),403
            rows,month=monthly_rows(s,lead)
        return render_template('customer_results.html',expired=False,rows=rows,price=price,labels=LABELS,month=month)

    @app.post('/stocks/reset')
    @auth
    def reset_stock_history():
        from monitor.worker import exclusive,LOCK_KEY
        from monitor.reset import reset_recommendations
        # The bot holds its lock for its whole lifetime. Refuse rather than
        # delete while a Telegram request or scanner transaction is in flight.
        with exclusive(DB,72617368696416) as bot_free:
            if not bot_free:
                flash('أوقفي خدمة البوت مؤقتًا ثم اضغطي تصفير سجل التطوير.');return redirect(url_for('stocks'))
            with exclusive(DB,LOCK_KEY) as monitor_free:
                if not monitor_free:
                    flash('الفحص يعمل الآن. أوقفي خدمة Monitor مؤقتًا ثم أعيدي المحاولة.');return redirect(url_for('stocks'))
                with DB.begin() as s:reset_recommendations(s)
        flash('تم مسح التوصيات والنتائج القديمة وطابور تنبيهاتها. العملاء وإعداداتهم محفوظون. شغلي البوت وMonitor لبدء السجل الجديد.')
        return redirect(url_for('stocks'))

    @app.post('/stocks/settings')
    @auth
    def stock_settings():
        from monitor.customer import MIN_SCORE_KEY
        from monitor.limits import HOLD_DAYS_KEY,HOLD_PROFIT_KEY,holding_settings
        try:threshold=float(request.form.get('minimum_score',''))
        except (ValueError,TypeError):abort(400)
        if not math.isfinite(threshold) or not 0<=threshold<=100:abort(400)
        with DB.begin() as s:
            if 'hold_days' in request.form or 'hold_min_profit' in request.form:
                try:
                    days=int(request.form.get('hold_days',''))
                    profit=float(request.form.get('hold_min_profit',''))
                except (ValueError,TypeError):abort(400)
                if not 1<=days<=365 or not math.isfinite(profit) or not 0<=profit<=100:abort(400)
                for key,value in ((HOLD_DAYS_KEY,days),(HOLD_PROFIT_KEY,profit)):
                    setting=s.get(Setting,key)
                    if setting is None:setting=Setting(key=key);s.add(setting)
                    setting.value=str(value)
            row=s.get(Setting,MIN_SCORE_KEY)
            if row is None:row=Setting(key=MIN_SCORE_KEY);s.add(row)
            row.value=str(threshold)
        flash(f'تم حفظ الحد الأدنى لإرسال توصيات الأمريكي: {threshold:g}/100')
        return redirect(url_for('stocks'))

    @app.get('/stocks')
    @auth
    def stocks():
        from monitor.models import Stock, Plan, Scan, EgxSignal, EgxOpenSignal, LABELS
        from monitor.strategy import local
        from monitor.customer import minimum_score
        conditions=[]
        market=request.args.get('market','')
        state=request.args.get('state','')
        if market in ('US','XA'): conditions.append(Plan.market==market)
        if state in LABELS: conditions.append(Plan.state==state)
        symbol=request.args.get('symbol','').strip().upper()[:40]
        if symbol: conditions.append(Plan.symbol==symbol)
        try: page=max(1,int(request.args.get('page','1')))
        except ValueError: page=1
        EgxSignal.__table__.create(bind=DB.kw['bind'],checkfirst=True)
        EgxOpenSignal.__table__.create(bind=DB.kw['bind'],checkfirst=True)
        with DB() as s:
            from monitor.limits import holding_settings
            hold_days,hold_min_profit=holding_settings(s)
            send_minimum_score=minimum_score(s)
            selected_market=market if market in ('US','XA') else 'US'
            if market not in ('US','XA'):
                conditions.append(Plan.market=='US')
            conditions.append(Plan.score>=send_minimum_score)
            total=s.scalar(select(func.count()).select_from(Plan).where(*conditions))
            pages=max(1,math.ceil(total/50));page=min(page,pages)
            rows=s.scalars(select(Plan).where(*conditions).order_by(Plan.score.desc(),Plan.id.desc()).offset((page-1)*50).limit(50)).all()
            counts=dict(s.execute(select(Plan.state,func.count()).where(Plan.market==selected_market,Plan.score>=send_minimum_score).group_by(Plan.state)).all())
            universe=dict(s.execute(select(Stock.market,func.count()).group_by(Stock.market)).all())
            waiting_errors=s.scalar(
                select(func.count()).select_from(Stock).where(Stock.error=='awaiting_expected_closed_bar')
            ) or 0
            no_complete_bars=s.scalar(
                select(func.count()).select_from(Stock).where(Stock.error=='no_complete_bars')
            ) or 0
            provider_errors=s.scalar(
                select(func.count()).select_from(Stock).where(
                    Stock.error!='',
                    Stock.error!='awaiting_expected_closed_bar',
                    Stock.error!='no_complete_bars',
                )
            ) or 0
            errors=waiting_errors+no_complete_bars+provider_errors
            scans=s.scalars(select(Scan).order_by(Scan.id.desc()).limit(10)).all()
            # Split each historical scan using the diagnostic error_summary saved by the monitor.
            import json as _json
            for scan in scans:
                try:
                    _summary=_json.loads(scan.summary_json or '{}')
                    _errs=_summary.get('error_summary') or {}
                except Exception:
                    _errs={}
                scan.waiting_errors=int(_errs.get('awaiting_expected_closed_bar',0) or 0)
                scan.no_complete_bars=int(_errs.get('no_complete_bars',0) or 0)
                scan.provider_errors=max(0,int(scan.errors or 0)-scan.waiting_errors-scan.no_complete_bars)
            stocks_by_symbol={r.symbol:r for r in s.scalars(select(Stock).where(Stock.symbol.in_([p.symbol for p in rows])))}
            egx_signals=s.scalars(
                select(EgxOpenSignal).order_by(EgxOpenSignal.signal_date.desc(),EgxOpenSignal.id.desc()).limit(250)
            ).all()
            egx_total=s.scalar(select(func.count()).select_from(EgxOpenSignal)) or 0
            egx_positive=s.scalar(select(func.count()).select_from(EgxOpenSignal).where(EgxOpenSignal.current_return_pct>=0)) or 0
            egx_avg_return=s.scalar(select(func.avg(EgxOpenSignal.current_return_pct)))
            egx_scan=s.scalar(select(Scan).where(Scan.market=='EG').order_by(Scan.id.desc()).limit(1))

            import json as _order_json
            order_map={}
            for _p in rows:
                try:
                    _ctx=_order_json.loads(_p.context_json or '{}');_kind=_ctx.get('order_type','')
                    if _kind not in ('MARKET','LIMIT','STOP'):
                        _ref=_ctx.get('signal_bar_close') or _ctx.get('feed_last_price')
                        if _ref and float(_ref)>0:
                            _gap=100*(float(_p.entry)/float(_ref)-1)
                            _kind='MARKET' if abs(_gap)<=.5 else ('LIMIT' if _p.entry<float(_ref) else 'STOP')
                except Exception:_kind=''
                order_map[_p.id]={'MARKET':'سوق','LIMIT':'Limit شراء','STOP':'Stop شراء'}.get(_kind,'أمر شراء')
        def link(**kw): return url_for('stocks',**{**request.args.to_dict(),**kw})
        return render_template('stocks.html',rows=rows,counts=counts,universe=universe,errors=errors,
            waiting_errors=waiting_errors,no_complete_bars=no_complete_bars,provider_errors=provider_errors,scans=scans,
            hold_days=hold_days,hold_min_profit=hold_min_profit,send_minimum_score=send_minimum_score,labels=LABELS,stock_map=stocks_by_symbol,order_map=order_map,
            egx_signals=egx_signals,egx_total=egx_total,egx_positive=egx_positive,egx_avg_return=egx_avg_return,egx_scan=egx_scan,total=total,page=page,pages=pages,link=link,local=local)


    @app.get('/egx-lab')
    @auth
    def egx_lab():
        """Interactive historical EGX R²/Slope signal explorer for one symbol."""
        import html as _html
        symbol=request.args.get('symbol','').strip().upper().replace('.CA','')[:20]
        try:r2_min=float(request.args.get('r2','0.791694'))
        except (TypeError,ValueError):r2_min=0.791694
        try:slope_min=float(request.args.get('slope','67.5062'))
        except (TypeError,ValueError):slope_min=67.5062
        try:cooldown=max(0,min(1000,int(request.args.get('cooldown','126'))))
        except (TypeError,ValueError):cooldown=126
        try:tp_pct=max(0.1,min(1000.0,float(request.args.get('tp','50'))))
        except (TypeError,ValueError):tp_pct=50.0
        tp_target=tp_pct/100.0
        confirm=request.args.get('confirm','1')!='0'
        rows=[];error='';latest_date='';company='';bars_count=0;chart_svg=''
        summary={}

        def _build_svg(data, rows, tp_pct):
            if not data:
                return ''
            width=1180; height=360
            ml=64; mr=20; mt=24; mb=40
            pw=max(10,width-ml-mr); ph=max(10,height-mt-mb)
            closes=[float(x['ac']) for x in data]
            highs=[float(x['ah']) for x in data]
            lows=[float(x['al']) for x in data]
            pmin=min(lows); pmax=max(highs)
            if pmax<=pmin:
                pmax=pmin+1.0
            n=max(1,len(data)-1)
            def px(i): return ml + (i/n)*pw
            def py(v): return mt + (pmax-float(v))/(pmax-pmin)*ph
            path=' '.join(('M' if i==0 else 'L')+f'{px(i):.2f},{py(c):.2f}' for i,c in enumerate(closes))
            parts=[]
            parts.append(f'<svg viewBox="0 0 {width} {height}" width="100%" height="360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="EGX signal chart">')
            parts.append('<rect x="0" y="0" width="100%" height="100%" rx="12" fill="#ffffff"/>')
            for frac in [0,0.25,0.50,0.75,1.0]:
                y=mt+ph*frac
                val=pmax-(pmax-pmin)*frac
                parts.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" stroke="#e9edf5" stroke-width="1"/>')
                parts.append(f'<text x="{ml-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" fill="#60708a">{val:.2f}</text>')
            parts.append(f'<path d="{path}" fill="none" stroke="#0b5ed7" stroke-width="2.2"/>')
            # Mark signals
            for r in rows:
                i=r.get('index')
                if i is None or i<0 or i>=len(data):
                    continue
                x=px(i); y=py(r['price'])
                status=r.get('status')
                entry_color='#0f9d58' if status=='CLOSED_TP' else '#ff8a00'
                parts.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{mt+ph}" stroke="#cfd7e6" stroke-dasharray="3 4" stroke-width="1"/>')
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{entry_color}" stroke="#ffffff" stroke-width="1.5"/>')
                parts.append(f'<text x="{x:.2f}" y="{max(14, y-10):.2f}" text-anchor="middle" font-size="10" font-weight="700" fill="{entry_color}">{_html.escape(r["date"][5:])}</text>')
                if r.get('hit_tp') is not None and r.get('tp_index') is not None:
                    xi=px(r['tp_index']); yi=py(r['price']*(1+tp_pct/100.0))
                    parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{xi:.2f}" y2="{yi:.2f}" stroke="#14a44d" stroke-width="2"/>')
                    parts.append(f'<circle cx="{xi:.2f}" cy="{yi:.2f}" r="5.5" fill="#14a44d" stroke="#ffffff" stroke-width="1.5"/>')
                    parts.append(f'<text x="{xi:.2f}" y="{min(height-6, yi+16):.2f}" text-anchor="middle" font-size="10" fill="#14a44d">TP</text>')
                else:
                    li=len(data)-1
                    x2=px(li); y2=py(float(data[-1]['ac']))
                    parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#ff8a00" stroke-width="2" stroke-dasharray="5 4"/>')
                    parts.append(f'<circle cx="{x2:.2f}" cy="{y2:.2f}" r="5.5" fill="#ff8a00" stroke="#ffffff" stroke-width="1.5"/>')
                    parts.append(f'<text x="{x2:.2f}" y="{max(14, y2-10):.2f}" text-anchor="middle" font-size="10" fill="#ff8a00">OPEN</text>')
            # x-axis labels
            wanted=min(6,len(data))
            if wanted>1:
                step=max(1,(len(data)-1)//(wanted-1))
                idxs=list(range(0,len(data),step))
                if idxs[-1]!=len(data)-1:
                    idxs.append(len(data)-1)
                idxs=idxs[:wanted-1]+[len(data)-1]
                seen=[]
                for i in idxs:
                    if i in seen: continue
                    seen.append(i)
                    x=px(i)
                    parts.append(f'<text x="{x:.2f}" y="{height-12}" text-anchor="middle" font-size="11" fill="#60708a">{_html.escape(str(data[i]["date"]))}</text>')
            # legend
            parts.append('<g font-size="11" font-weight="600">')
            lx=ml; ly=18
            for color,label in [('#0b5ed7','السعر'),('#14a44d','إشارة أغلقت على TP'),('#ff8a00','إشارة مفتوحة حتى الآن')]:
                parts.append(f'<rect x="{lx}" y="{ly-8}" width="12" height="12" rx="2" fill="{color}"/>')
                parts.append(f'<text x="{lx+18}" y="{ly+2}" fill="#203040">{label}</text>')
                lx += 170 if label=='السعر' else 210
            parts.append('</g>')
            parts.append('</svg>')
            return ''.join(parts)

        if symbol:
            try:
                from monitor.worker import _egx_daily_sync, _egx_linreg, _egx_price_confirm
                data=_egx_daily_sync(symbol)
                bars_count=len(data)
                if not data:
                    raise RuntimeError('لم يرجع Yahoo تاريخًا يوميًا لهذا الرمز.')
                latest_date=data[-1]['date']
                company=data[-1].get('name','') if isinstance(data[-1],dict) else ''
                ac=[float(x['ac']) for x in data]
                metrics=[];rule=[]
                for i in range(len(data)):
                    slope,r2=_egx_linreg(ac,i)
                    metrics.append((slope,r2))
                    ok=(slope is not None and r2 is not None and r2>=r2_min and slope>=slope_min)
                    if confirm:ok=ok and _egx_price_confirm(data,i)
                    rule.append(bool(ok))
                activations=[i for i,x in enumerate(rule) if x and (i==0 or not rule[i-1])]
                # Sequential recommendations only: never open a new recommendation while
                # a previous one is still open. A recommendation closes when the selected
                # TP is first touched. Cooldown is still enforced between entry dates.
                kept=[];last_entry=-10**9;position_open_until=-1
                def _first_tp_index(entry_i):
                    entry=float(data[entry_i]['ac'])
                    target=entry*(1+tp_target)
                    for jj in range(entry_i+1,len(data)):
                        if float(data[jj]['ah'])>=target:
                            return jj
                    return None
                for i in activations:
                    if i<=position_open_until:
                        continue
                    if i-last_entry<cooldown:
                        continue
                    tp_i=_first_tp_index(i)
                    kept.append(i)
                    last_entry=i
                    # If TP is never reached, this recommendation remains open to the
                    # latest bar and blocks every later activation.
                    position_open_until=(tp_i if tp_i is not None else len(data)-1)
                def ret_at(i,n):
                    j=i+n
                    if j>=len(data):return None
                    return 100*(float(data[j]['ac'])/float(data[i]['ac'])-1)
                def fmt_hit(i,target):
                    entry=float(data[i]['ac'])
                    for j in range(i+1,len(data)):
                        if float(data[j]['ah'])>=entry*(1+target):
                            return j-i,data[j]['date'],j
                    return None,None,None
                for i in kept:
                    entry=float(data[i]['ac']);future=data[i:]
                    max_gain=100*(max(float(x['ah']) for x in future)/entry-1)
                    max_dd=100*(min(float(x['al']) for x in future)/entry-1)
                    h20,d20,i20=fmt_hit(i,.20);h50,d50,i50=fmt_hit(i,.50);h100,d100,i100=fmt_hit(i,1.00)
                    htp,dtp,itp=fmt_hit(i,tp_target)
                    slope,r2=metrics[i]
                    current_return=100*(float(data[-1]['ac'])/entry-1)
                    status='CLOSED_TP' if htp is not None else 'OPEN'
                    rows.append({
                        'index':i,
                        'date':data[i]['date'],'price':entry,'r2':r2,'slope':slope,
                        'ret_1m':ret_at(i,21),'ret_3m':ret_at(i,63),'ret_6m':ret_at(i,126),'ret_1y':ret_at(i,252),
                        'current_return':current_return,
                        'max_gain':max_gain,'max_dd':max_dd,
                        'hit20':h20,'hit20_date':d20,'hit50':h50,'hit50_date':d50,'hit100':h100,'hit100_date':d100,
                        'hit_tp':htp,'hit_tp_date':dtp,'tp_index':itp,
                        'age':len(data)-1-i,
                        'confirm':_egx_price_confirm(data,i),
                        'status':status,
                    })
                rows.reverse()
                chart_svg=_build_svg(data, list(reversed(rows)), tp_pct)
                n=len(rows)
                if n:
                    hit50=sum(1 for x in rows if x['hit50'] is not None)
                    hit_tp=sum(1 for x in rows if x['hit_tp'] is not None)
                    open_n=sum(1 for x in rows if x['status']=='OPEN')
                    tp_sessions=[x['hit_tp'] for x in rows if x['hit_tp'] is not None]
                    matured6=[x for x in rows if x['ret_6m'] is not None]
                    matured1y=[x for x in rows if x['ret_1y'] is not None]
                    summary={
                        'count':n,
                        'hit50_pct':100*hit50/n,
                        'hit_tp_pct':100*hit_tp/n,
                        'tp_hits':hit_tp,
                        'open_n':open_n,
                        'tp_median_sessions':sorted(tp_sessions)[len(tp_sessions)//2] if tp_sessions else None,
                        'avg3m':sum(x['ret_3m'] for x in rows if x['ret_3m'] is not None)/max(1,sum(x['ret_3m'] is not None for x in rows)),
                        'avg6m':sum(x['ret_6m'] for x in matured6)/len(matured6) if matured6 else None,
                        'avg1y':sum(x['ret_1y'] for x in matured1y)/len(matured1y) if matured1y else None,
                    }
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
        return render_template('egx_lab.html',symbol=symbol,r2_min=r2_min,slope_min=slope_min,cooldown=cooldown,tp_pct=tp_pct,
            confirm=confirm,rows=rows,error=error,latest_date=latest_date,bars_count=bars_count,summary=summary,company=company,chart_svg=chart_svg)


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
        if market in ('SA','US','XA'): conditions.append(Stock.market==market)
        with DB() as s:
            rows=s.scalars(select(Stock).where(*conditions).order_by(Stock.market,Stock.symbol).limit(250)).all()
        return render_template('stock_feed.html',rows=rows)
    return app
