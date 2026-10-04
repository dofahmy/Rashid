import os, secrets, csv, io, math, time, threading
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
    # Whole-market EGX lab cache. The web service runs one gunicorn worker in this
    # deployment, so this avoids re-downloading ~300 symbols on every sort/click.
    egx_market_cache={}
    egx_market_lock=threading.Lock()
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
        """Interactive EGX signal lab: one stock or the whole Egyptian market."""
        import html as _html
        from concurrent.futures import ThreadPoolExecutor, as_completed

        scope=request.args.get('scope','symbol').strip().lower()
        if scope not in ('symbol','market'): scope='symbol'
        year_raw=request.args.get('year','all').strip().lower()
        if year_raw in ('','all','0'):
            analysis_year=None
        else:
            try:
                analysis_year=int(year_raw)
                if analysis_year<2000 or analysis_year>2100: analysis_year=None
            except (TypeError,ValueError):
                analysis_year=None
        current_year=datetime.now(timezone.utc).year
        analysis_years=list(range(current_year,2018,-1))
        symbol=request.args.get('symbol','').strip().upper().replace('.CA','')[:20]
        chart_symbol=request.args.get('chart_symbol','').strip().upper().replace('.CA','')[:20]
        try:r2_min=float(request.args.get('r2','0.791694'))
        except (TypeError,ValueError):r2_min=0.791694
        try:slope_min=float(request.args.get('slope_min',request.args.get('slope','67.5062')))
        except (TypeError,ValueError):slope_min=67.5062
        try:slope_max=float(request.args.get('slope_max','999999'))
        except (TypeError,ValueError):slope_max=999999.0
        if slope_max < slope_min:
            slope_min,slope_max=slope_max,slope_min
        try:cooldown=max(0,min(1000,int(request.args.get('cooldown','126'))))
        except (TypeError,ValueError):cooldown=126
        try:tp_pct=max(0.1,min(1000.0,float(request.args.get('tp','50'))))
        except (TypeError,ValueError):tp_pct=50.0
        tp_target=tp_pct/100.0
        exit_mode=request.args.get('exit_mode','tp').strip().lower()
        if exit_mode not in ('tp','time','both','slope','all'): exit_mode='tp'
        try:time_exit_sessions=max(1,min(2000,int(request.args.get('time_exit','126'))))
        except (TypeError,ValueError):time_exit_sessions=126
        try:exit_slope=float(request.args.get('exit_slope','100'))
        except (TypeError,ValueError):exit_slope=100.0
        exit_slope_op=request.args.get('exit_slope_op','gte').strip().lower()
        if exit_slope_op not in ('gte','lte'): exit_slope_op='gte'

        cci_enabled=request.args.get('cci_enabled','0')=='1'
        try:cci_period=max(2,min(500,int(request.args.get('cci_period','20'))))
        except (TypeError,ValueError):cci_period=20
        try:cci_min=float(request.args.get('cci_min','-200'))
        except (TypeError,ValueError):cci_min=-200.0
        try:cci_max=float(request.args.get('cci_max','200'))
        except (TypeError,ValueError):cci_max=200.0
        if cci_max < cci_min:
            cci_min,cci_max=cci_max,cci_min
        confirm=request.args.get('confirm','1')!='0'
        rows=[];error='';latest_date='';company='';bars_count=0;chart_svg='';market_errors=0;market_symbols=0;market_scan_status='';market_scan_progress=0;market_scan_error=''
        summary={};show_portfolio=request.args.get('portfolio','0')=='1';portfolio_svg='';portfolio_summary={};portfolio_ledger=[]

        from monitor.worker import _egx_daily_sync, _egx_price_confirm
        try:
            from monitor.worker import _egx_discover_sync
        except ImportError:
            _egx_discover_sync=None

        def _fast_metrics(data, window=126):
            """Same previous-126-session linear regression as worker, but O(n)."""
            vals=[float(x['ac']) for x in data]
            n=len(vals); out=[(None,None)]*n
            sy=[0.0]*(n+1); sy2=[0.0]*(n+1); sky=[0.0]*(n+1)
            for k,v in enumerate(vals):
                sy[k+1]=sy[k]+v; sy2[k+1]=sy2[k]+v*v; sky[k+1]=sky[k]+k*v
            xm=(window-1)/2.0
            xss=sum((x-xm)**2 for x in range(window))
            for i in range(window,n):
                a=i-window; b=i
                sumy=sy[b]-sy[a]; sumy2=sy2[b]-sy2[a]
                ym=sumy/window
                if ym==0: continue
                sum_local_xy=(sky[b]-sky[a])-a*sumy
                cov=sum_local_xy-xm*sumy
                beta=cov/xss
                sst=sumy2-(sumy*sumy/window)
                r2=(beta*beta*xss/sst) if sst>0 else 0.0
                r2=max(0.0,min(1.0,r2))
                slope_pct=100*(beta*(window-1))/ym
                out[i]=(slope_pct,r2)
            return out

        def _cci_series(data, period):
            typical=[(float(x['ah'])+float(x['al'])+float(x['ac']))/3.0 for x in data]
            vals=[]
            for i in range(len(typical)):
                if i+1<period:
                    vals.append(None)
                    continue
                w=typical[i-period+1:i+1]
                sma=sum(w)/period
                md=sum(abs(v-sma) for v in w)/period
                vals.append(0.0 if md==0 else (typical[i]-sma)/(0.015*md))
            return vals

        def _analyse(symbol_code, company_name=''):
            data=_egx_daily_sync(symbol_code)
            if not data:return {'symbol':symbol_code,'company':company_name,'data':[],'rows':[]}
            metrics=_fast_metrics(data)
            cci_values=_cci_series(data,cci_period)
            rule=[]
            for i,(slope,r2) in enumerate(metrics):
                ok=(slope is not None and r2 is not None and r2>=r2_min and slope>=slope_min and slope<=slope_max)
                if confirm:ok=ok and _egx_price_confirm(data,i)
                if cci_enabled:
                    cv=cci_values[i]
                    ok=ok and cv is not None and cci_min<=cv<=cci_max
                rule.append(bool(ok))
            activations=[i for i,x in enumerate(rule) if x and (i==0 or not rule[i-1])]

            def first_tp_index(entry_i):
                entry=float(data[entry_i]['ac']);target=entry*(1+tp_target)
                for jj in range(entry_i+1,len(data)):
                    if float(data[jj]['ah'])>=target:return jj
                return None
            def first_slope_exit_index(entry_i):
                for jj in range(entry_i+1,len(data)):
                    sv,_rv=metrics[jj]
                    if sv is None:continue
                    if exit_slope_op=='gte' and sv>=exit_slope:return jj
                    if exit_slope_op=='lte' and sv<=exit_slope:return jj
                return None
            def exit_candidates(entry_i):
                tp_i=first_tp_index(entry_i)
                time_i=entry_i+time_exit_sessions
                time_i=time_i if time_i<len(data) else None
                slope_i=first_slope_exit_index(entry_i)
                if exit_mode=='tp':return [('tp',tp_i)]
                if exit_mode=='time':return [('time',time_i)]
                if exit_mode=='both':return [('tp',tp_i),('time',time_i)]
                if exit_mode=='slope':return [('slope',slope_i)]
                return [('tp',tp_i),('time',time_i),('slope',slope_i)]
            def exit_idx_for(entry_i):
                vals=[idx for _kind,idx in exit_candidates(entry_i) if idx is not None]
                return min(vals) if vals else None

            kept=[];last_entry=-10**9;position_open_until=-1
            for i in activations:
                if i<=position_open_until:continue
                if i-last_entry<cooldown:continue
                ex=exit_idx_for(i)
                kept.append(i);last_entry=i
                position_open_until=ex if ex is not None else len(data)-1

            def ret_at(i,n):
                j=i+n
                if j>=len(data):return None
                return 100*(float(data[j]['ac'])/float(data[i]['ac'])-1)
            def fmt_hit(i,target):
                entry=float(data[i]['ac'])
                for j in range(i+1,len(data)):
                    if float(data[j]['ah'])>=entry*(1+target):return j-i,data[j]['date'],j
                return None,None,None

            out=[]
            for i in kept:
                entry=float(data[i]['ac'])
                h20,d20,i20=fmt_hit(i,.20);h50,d50,i50=fmt_hit(i,.50);h100,d100,i100=fmt_hit(i,1.00)
                htp,dtp,itp=fmt_hit(i,tp_target)
                slope,r2=metrics[i]
                current_return=100*(float(data[-1]['ac'])/entry-1)
                time_i=i+time_exit_sessions;time_i=time_i if time_i<len(data) else None
                slope_i=first_slope_exit_index(i)
                candidates=exit_candidates(i)
                valid=[(kind,idx) for kind,idx in candidates if idx is not None]
                chosen_kind=None;chosen_idx=None
                if valid:
                    # Earliest exit wins. If TP and another exit happen on the same
                    # daily bar, TP gets priority because intraday high can hit it
                    # before the close-based time/slope exit.
                    priority={'tp':0,'slope':1,'time':2}
                    chosen_kind,chosen_idx=min(valid,key=lambda kv:(kv[1],priority.get(kv[0],9)))
                if chosen_kind=='tp':
                    status='CLOSED_TP';exit_index=chosen_idx;exit_date=data[chosen_idx]['date'];exit_price=entry*(1+tp_target);exit_return=tp_pct;trade_end=chosen_idx
                elif chosen_kind=='time':
                    status='CLOSED_TIME';exit_index=chosen_idx;exit_date=data[chosen_idx]['date'];exit_price=float(data[chosen_idx]['ac']);exit_return=100*(exit_price/entry-1);trade_end=chosen_idx
                elif chosen_kind=='slope':
                    status='CLOSED_SLOPE';exit_index=chosen_idx;exit_date=data[chosen_idx]['date'];exit_price=float(data[chosen_idx]['ac']);exit_return=100*(exit_price/entry-1);trade_end=chosen_idx
                else:
                    status='OPEN';exit_index=None;exit_date=None;exit_price=None;exit_return=None;trade_end=len(data)-1
                trade_window=data[i:trade_end+1]
                max_gain=100*(max(float(x['ah']) for x in trade_window)/entry-1)
                max_dd=100*(min(float(x['al']) for x in trade_window)/entry-1)
                tp_before_exit=False;tp_before_exit_sessions=None;tp_before_exit_date=None
                if exit_mode in ('time','both'):
                    target=entry*(1+tp_target)
                    for jj in range(i+1,trade_end+1):
                        if float(data[jj]['ah'])>=target:
                            tp_before_exit=True;tp_before_exit_sessions=jj-i;tp_before_exit_date=data[jj]['date'];break
                out.append({
                    'symbol':symbol_code,'company':company_name,'index':i,'date':data[i]['date'],'price':entry,'r2':r2,'slope':slope,
                    'ret_1m':ret_at(i,21),'ret_3m':ret_at(i,63),'ret_6m':ret_at(i,126),'ret_1y':ret_at(i,252),
                    'current_return':current_return,'max_gain':max_gain,'max_dd':max_dd,
                    'hit20':h20,'hit20_date':d20,'hit50':h50,'hit50_date':d50,'hit100':h100,'hit100_date':d100,
                    'hit_tp':htp,'hit_tp_date':dtp,'tp_index':itp,'exit_index':exit_index,'exit_date':exit_date,
                    'exit_price':exit_price,'exit_return':exit_return,'tp_before_exit':tp_before_exit,
                    'entry_cci':cci_values[i],
                    'exit_cci':(cci_values[exit_index] if exit_index is not None and exit_index<len(cci_values) else None),
                    'exit_slope_value':(metrics[exit_index][0] if exit_index is not None and metrics[exit_index][0] is not None else None),
                    'tp_before_exit_sessions':tp_before_exit_sessions,'tp_before_exit_date':tp_before_exit_date,
                    'trade_duration':(exit_index-i if exit_index is not None else len(data)-1-i),
                    'age':len(data)-1-i,'confirm':_egx_price_confirm(data,i),'status':status,
                })
            return {'symbol':symbol_code,'company':company_name,'data':data,'rows':out}

        def _summarize(items):
            n=len(items)
            if not n:return {}
            hit50=sum(1 for x in items if x['hit50'] is not None);hit_tp=sum(1 for x in items if x['hit_tp'] is not None)
            closed_n=sum(1 for x in items if x['status']!='OPEN');open_n=n-closed_n
            closed_time=[x for x in items if x['status']=='CLOSED_TIME' and x['exit_return'] is not None]
            closed_tp=[x for x in items if x['status']=='CLOSED_TP' and x['exit_return'] is not None]
            closed_slope=[x for x in items if x['status']=='CLOSED_SLOPE' and x['exit_return'] is not None]
            time_returns=[x['exit_return'] for x in closed_time];exit_returns=[x['exit_return'] for x in items if x['status']!='OPEN' and x['exit_return'] is not None]
            tp_sessions=[x['hit_tp'] for x in items if x['hit_tp'] is not None]
            r3=[x['ret_3m'] for x in items if x['ret_3m'] is not None];r6=[x['ret_6m'] for x in items if x['ret_6m'] is not None];r1=[x['ret_1y'] for x in items if x['ret_1y'] is not None]
            return {'count':n,'hit50_pct':100*hit50/n,'hit_tp_pct':100*hit_tp/n,'tp_hits':hit_tp,'closed_n':closed_n,'open_n':open_n,
                'time_avg_return':sum(time_returns)/len(time_returns) if time_returns else None,'time_median_return':sorted(time_returns)[len(time_returns)//2] if time_returns else None,
                'time_win_pct':100*sum(v>0 for v in time_returns)/len(time_returns) if time_returns else None,'tp_exit_n':len(closed_tp),'time_exit_n':len(closed_time),'slope_exit_n':len(closed_slope),
                'exit_avg_return':sum(exit_returns)/len(exit_returns) if exit_returns else None,'exit_median_return':sorted(exit_returns)[len(exit_returns)//2] if exit_returns else None,
                'exit_win_pct':100*sum(v>0 for v in exit_returns)/len(exit_returns) if exit_returns else None,'tp_median_sessions':sorted(tp_sessions)[len(tp_sessions)//2] if tp_sessions else None,
                'avg3m':sum(r3)/len(r3) if r3 else None,'avg6m':sum(r6)/len(r6) if r6 else None,'avg1y':sum(r1)/len(r1) if r1 else None}

        def _build_svg(data, chart_rows):
            if not data:return ''
            width=1180;height=360;ml=64;mr=20;mt=24;mb=40;pw=width-ml-mr;ph=height-mt-mb
            closes=[float(x['ac']) for x in data];highs=[float(x['ah']) for x in data];lows=[float(x['al']) for x in data]
            pmin=min(lows);pmax=max(highs)
            if pmax<=pmin:pmax=pmin+1
            nn=max(1,len(data)-1)
            def px(i):return ml+(i/nn)*pw
            def py(v):return mt+(pmax-float(v))/(pmax-pmin)*ph
            path=' '.join(('M' if i==0 else 'L')+f'{px(i):.2f},{py(c):.2f}' for i,c in enumerate(closes))
            parts=[f'<svg viewBox="0 0 {width} {height}" width="100%" height="360" xmlns="http://www.w3.org/2000/svg">','<rect x="0" y="0" width="100%" height="100%" rx="12" fill="#fff"/>']
            for frac in [0,.25,.5,.75,1]:
                y=mt+ph*frac;val=pmax-(pmax-pmin)*frac
                parts.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" stroke="#e9edf5"/><text x="{ml-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" fill="#60708a">{val:.2f}</text>')
            parts.append(f'<path d="{path}" fill="none" stroke="#0b5ed7" stroke-width="2.2"/>')
            for r in chart_rows:
                i=r['index'];x=px(i);y=py(r['price']);status=r['status'];color='#0f9d58' if status=='CLOSED_TP' else ('#7c3aed' if status=='CLOSED_TIME' else ('#dc2626' if status=='CLOSED_SLOPE' else '#ff8a00'))
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" stroke="#fff" stroke-width="1.5"/><text x="{x:.2f}" y="{max(14,y-10):.2f}" text-anchor="middle" font-size="10" font-weight="700" fill="{color}">{_html.escape(r["date"][5:])}</text>')
                if r['exit_index'] is not None:
                    xi=px(r['exit_index']);yi=py(r['exit_price']);label='TP' if status=='CLOSED_TP' else ('SLOPE' if status=='CLOSED_SLOPE' else 'TIME');ec='#14a44d' if status=='CLOSED_TP' else ('#dc2626' if status=='CLOSED_SLOPE' else '#7c3aed')
                    parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{xi:.2f}" y2="{yi:.2f}" stroke="{ec}" stroke-width="2"/><circle cx="{xi:.2f}" cy="{yi:.2f}" r="5.5" fill="{ec}" stroke="#fff"/><text x="{xi:.2f}" y="{min(height-6,yi+16):.2f}" text-anchor="middle" font-size="10" fill="{ec}">{label}</text>')
                else:
                    x2=px(len(data)-1);y2=py(data[-1]['ac'])
                    parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#ff8a00" stroke-width="2" stroke-dasharray="5 4"/><circle cx="{x2:.2f}" cy="{y2:.2f}" r="5.5" fill="#ff8a00" stroke="#fff"/><text x="{x2:.2f}" y="{max(14,y2-10):.2f}" text-anchor="middle" font-size="10" fill="#ff8a00">OPEN</text>')
            parts.append('</svg>');return ''.join(parts)


        def _portfolio_backtest(signal_rows, compact_data, analysis_year=None):
            """10-slot portfolio. Each slot starts at 10% of capital.
            Natural setup exits release a slot to cash. If a new recommendation
            arrives while all 10 slots are occupied, the currently best-performing
            open position is closed at that day's close and its slot is reused.
            """
            import bisect
            if not signal_rows:
                return {},'',[]

            # One compact adjusted-close series per symbol.
            series={}
            date_union=set()
            for row in signal_rows:
                sym=row['symbol']
                if sym in series:
                    continue
                pts=compact_data.get(sym)
                if not pts:
                    raw=_egx_daily_sync(sym)
                    pts=[(x['date'],float(x['ac'])) for x in raw]
                pts=sorted((str(d),float(p)) for d,p in pts)
                if not pts:
                    continue
                dates=[d for d,_ in pts]; prices=[p for _,p in pts]
                series[sym]=(dates,prices)
                date_union.update(dates)

            signals=[x for x in signal_rows if x.get('symbol') in series]
            if not signals:
                return {},'',[]

            signals.sort(key=lambda x:(x['date'],x['symbol']))
            first_date=signals[0]['date']
            last_date=max((dates[-1] for dates,_prices in series.values()), default=first_date)
            if analysis_year is not None:
                year_start=f'{analysis_year:04d}-01-01'
                year_end=f'{analysis_year:04d}-12-31'
                first_date=max(first_date,year_start)
                last_date=min(last_date,year_end)
            calendar=sorted(d for d in date_union if first_date<=d<=last_date)
            if not calendar:
                return {},'',[]

            def mark(sym, d):
                dates,prices=series[sym]
                k=bisect.bisect_right(dates,d)-1
                return prices[k] if k>=0 else None

            # 10 independent capital slots, each worth 10 at inception.
            slots=[{'cash':10.0,'pos':None} for _ in range(10)]
            banked_profit=0.0
            by_entry={}
            by_exit={}
            for r in signals:
                by_entry.setdefault(r['date'],[]).append(r)
                if r.get('exit_date'):
                    by_exit.setdefault(r['exit_date'],[]).append(r)

            equity_curve=[]
            portfolio_trades=0
            forced_rotations=0
            natural_exits=0
            winners=0
            closed_returns=[]
            ledger=[]
            next_trade_id=1
            action_marks=[]

            def close_slot(slot, d, price, reason):
                nonlocal forced_rotations,natural_exits,winners,banked_profit
                pos=slot.get('pos')
                if not pos:
                    return
                value=pos['shares']*price
                ret=100*(price/pos['entry_price']-1)

                # Position sizing rule:
                # - never compound profits into the next trade;
                # - next trade can use at most 10% of ORIGINAL capital (= 10 units);
                # - if the slot lost money, the next trade uses the smaller remaining amount.
                reusable=min(value,10.0)
                realized_profit=max(0.0,value-10.0)
                banked_profit+=realized_profit
                slot['cash']=reusable
                slot['pos']=None

                closed_returns.append(ret)
                if ret>0:winners+=1
                if reason=='FORCED_ROTATION':forced_rotations+=1
                else:natural_exits+=1
                rec=pos['ledger']
                rec.update({
                    'exit_date':d,'exit_price':price,'exit_return':ret,
                    'exit_reason':reason,'status':'CLOSED','exit_value':value,
                    'banked_profit':realized_profit,'next_entry_capital':reusable
                })
                action_marks.append((d,'EXIT',rec['trade_id'],pos['symbol']))

            for d in calendar:
                # Natural exits first. Exit price follows the selected setup.
                for r in by_exit.get(d,[]):
                    for slot in slots:
                        pos=slot.get('pos')
                        if pos and pos['row'] is r:
                            close_slot(slot,d,float(r['exit_price']),r['status'])
                            break

                # Then process all new recommendations at their signal close.
                for r in sorted(by_entry.get(d,[]),key=lambda x:x['symbol']):
                    entry_price=float(r['price'])
                    empty=next((s for s in slots if s['pos'] is None),None)
                    if empty is None:
                        candidates=[]
                        for s in slots:
                            p=s['pos']
                            cp=mark(p['symbol'],d)
                            if cp is None: continue
                            current_ret=100*(cp/p['entry_price']-1)
                            candidates.append((current_ret,p['entry_date'],p['symbol'],s,cp))
                        if not candidates:
                            continue
                        # "Close the one that gained the most"; if all are losing,
                        # this closes the least-negative one.
                        _ret,_ed,_sym,empty,cp=max(candidates,key=lambda z:(z[0],-len(z[1]),z[2]))
                        close_slot(empty,d,cp,'FORCED_ROTATION')

                    capital=float(empty['cash'])
                    if capital<=0:
                        continue
                    slot_no=slots.index(empty)+1
                    trade_id=next_trade_id
                    next_trade_id+=1
                    rec={
                        'trade_id':trade_id,'slot':slot_no,'symbol':r['symbol'],
                        'company':r.get('company',''),'entry_date':d,'entry_price':entry_price,
                        'entry_value':capital,'exit_date':None,'exit_price':None,
                        'exit_return':None,'exit_reason':None,'exit_value':None,'status':'OPEN'
                    }
                    ledger.append(rec)
                    action_marks.append((d,'ENTRY',trade_id,r['symbol']))
                    empty['cash']=0.0
                    empty['pos']={
                        'symbol':r['symbol'],'entry_date':d,'entry_price':entry_price,
                        'shares':capital/entry_price,'row':r,'ledger':rec
                    }
                    portfolio_trades+=1

                equity=banked_profit
                for slot in slots:
                    if slot['pos'] is None:
                        equity+=slot['cash']
                    else:
                        p=slot['pos'];cp=mark(p['symbol'],d)
                        equity+=p['shares']*(cp if cp is not None else p['entry_price'])
                equity_curve.append((d,equity))

            if not equity_curve:
                return {},''

            end_value=equity_curve[-1][1]
            total_return=100*(end_value/100.0-1)
            peak=equity_curve[0][1];max_dd=0.0
            for _d,v in equity_curve:
                peak=max(peak,v)
                if peak>0:
                    max_dd=min(max_dd,100*(v/peak-1))
            open_positions=sum(1 for s in slots if s['pos'] is not None)
            final_date=equity_curve[-1][0]
            for slot in slots:
                pos=slot.get('pos')
                if not pos:continue
                cp=mark(pos['symbol'],final_date)
                if cp is None:cp=pos['entry_price']
                rec=pos['ledger']
                rec.update({
                    'current_date':final_date,'current_price':cp,
                    'current_return':100*(cp/pos['entry_price']-1),
                    'current_value':pos['shares']*cp
                })
            summary={
                'start_value':100.0,'end_value':end_value,'total_return':total_return,
                'max_dd':max_dd,'trades':portfolio_trades,'forced_rotations':forced_rotations,
                'natural_exits':natural_exits,'open_positions':open_positions,
                'closed_trades':len(closed_returns),
                'banked_profit':banked_profit,
                'win_rate':(100*winners/len(closed_returns)) if closed_returns else None,
                'avg_closed_return':(sum(closed_returns)/len(closed_returns)) if closed_returns else None,
            }

            # SVG performance chart, normalized to 100 at inception.
            width=1180;height=360;ml=64;mr=20;mt=28;mb=44
            pw=width-ml-mr;ph=height-mt-mb
            vals=[v for _,v in equity_curve]
            vmin=min(vals+[100.0]);vmax=max(vals+[100.0])
            pad=max(1.0,(vmax-vmin)*.08)
            vmin-=pad;vmax+=pad
            nn=max(1,len(equity_curve)-1)
            def px(i):return ml+(i/nn)*pw
            def py(v):return mt+(vmax-v)/(vmax-vmin)*ph
            path=' '.join(('M' if i==0 else 'L')+f'{px(i):.2f},{py(v):.2f}' for i,(_d,v) in enumerate(equity_curve))
            parts=[f'<svg viewBox="0 0 {width} {height}" width="100%" height="360" xmlns="http://www.w3.org/2000/svg">',
                   '<rect x="0" y="0" width="100%" height="100%" rx="12" fill="#fff"/>']
            for frac in [0,.25,.5,.75,1]:
                y=mt+ph*frac;val=vmax-(vmax-vmin)*frac
                parts.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" stroke="#e9edf5"/>')
                parts.append(f'<text x="{ml-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" fill="#60708a">{val:.1f}</text>')
            if vmin<=100<=vmax:
                y0=py(100)
                parts.append(f'<line x1="{ml}" y1="{y0:.2f}" x2="{width-mr}" y2="{y0:.2f}" stroke="#9aa8bb" stroke-dasharray="4 4"/>')
            parts.append(f'<path d="{path}" fill="none" stroke="#0b5ed7" stroke-width="2.5"/>')
            eq_by_date={d:(i,v) for i,(d,v) in enumerate(equity_curve)}
            # Mark portfolio transactions on the equity curve. The full detail is
            # listed in the ledger below, while the chart shows numbered entry/exit points.
            per_date={}
            for d,kind,tid,sym in action_marks:
                per_date.setdefault(d,[]).append((kind,tid,sym))
            for d,marks in per_date.items():
                iv=eq_by_date.get(d)
                if not iv:continue
                i,v=iv;x=px(i);y=py(v)
                entries=sum(1 for k,_t,_s in marks if k=='ENTRY')
                exits=sum(1 for k,_t,_s in marks if k=='EXIT')
                if entries:
                    parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#0f9d58" stroke="#fff"/>')
                if exits:
                    parts.append(f'<circle cx="{x:.2f}" cy="{y+11:.2f}" r="5" fill="#dc2626" stroke="#fff"/>')
                label=('+'+str(entries) if entries else '')+('/-'+str(exits) if exits else '')
                parts.append(f'<text x="{x:.2f}" y="{max(12,y-9):.2f}" text-anchor="middle" font-size="9" font-weight="700" fill="#203040">{label}</text>')
            tick_idxs=sorted(set([0,len(equity_curve)//4,len(equity_curve)//2,(3*len(equity_curve))//4,len(equity_curve)-1]))
            for i in tick_idxs:
                x=px(i);label=_html.escape(equity_curve[i][0])
                parts.append(f'<text x="{x:.2f}" y="{height-14}" text-anchor="middle" font-size="11" fill="#60708a">{label}</text>')
            parts.append(f'<text x="{width-mr}" y="18" text-anchor="end" font-size="12" font-weight="700" fill="#203040">Portfolio value · start = 100</text>')
            parts.append('</svg>')
            return summary,''.join(parts),ledger

        try:
            if scope=='symbol':
                if symbol:
                    res=_analyse(symbol,symbol)
                    filtered_rows=[x for x in res['rows'] if analysis_year is None or str(x.get('date','')).startswith(f'{analysis_year:04d}-')]
                    rows=list(reversed(filtered_rows))
                    bars_count=len(res['data'])
                    latest_date=res['data'][-1]['date'] if res['data'] else ''
                    summary=_summarize(rows)
                    chart_svg=_build_svg(res['data'],filtered_rows)
            else:
                if _egx_discover_sync is None:raise RuntimeError('نسخة monitor/worker.py الحالية لا تحتوي على اكتشاف سوق مصر.')

                cache_key=(
                    round(r2_min,8),round(slope_min,6),round(slope_max,6),cooldown,
                    round(tp_pct,6),exit_mode,time_exit_sessions,round(exit_slope,6),
                    exit_slope_op,bool(confirm),bool(cci_enabled),cci_period,round(cci_min,6),round(cci_max,6)
                )
                force=request.args.get('force_scan','0')=='1'

                def _market_scan_job(key):
                    collected=[];errors=0;latest='';lookup={};data_map={}
                    try:
                        universe=_egx_discover_sync()
                        lookup=dict(universe)
                        with egx_market_lock:
                            e=egx_market_cache.get(key,{})
                            e.update({'status':'running','total':len(universe),'progress':0,'errors':0,
                                      'rows':[],'latest_date':'','lookup':lookup,'started_at':time.time()})
                            egx_market_cache[key]=e
                        with ThreadPoolExecutor(max_workers=20) as pool:
                            futs={pool.submit(_analyse,s,c):(s,c) for s,c in universe}
                            done=0
                            for fut in as_completed(futs):
                                done+=1
                                try:
                                    res=fut.result()
                                    collected.extend(res['rows'])
                                    if res['rows'] and res['data']:
                                        data_map[res['symbol']]=[(x['date'],float(x['ac'])) for x in res['data']]
                                    if res['data']:
                                        d=res['data'][-1]['date']
                                        latest=max(latest,d) if latest else d
                                except Exception:
                                    errors+=1
                                # publish partial results every few completions
                                if done % 10 == 0 or done == len(universe):
                                    partial=sorted(collected,key=lambda x:(x['date'],x['symbol']),reverse=True)
                                    with egx_market_lock:
                                        e=egx_market_cache.get(key,{})
                                        e.update({'status':'running','total':len(universe),'progress':done,
                                                  'errors':errors,'rows':partial,'latest_date':latest,
                                                  'lookup':lookup,'data_map':data_map})
                                        egx_market_cache[key]=e
                        final_rows=sorted(collected,key=lambda x:(x['date'],x['symbol']),reverse=True)
                        with egx_market_lock:
                            egx_market_cache[key]={
                                'status':'done','total':len(universe),'progress':len(universe),
                                'errors':errors,'rows':final_rows,'latest_date':latest,
                                'lookup':lookup,'data_map':data_map,'started_at':egx_market_cache.get(key,{}).get('started_at',time.time()),
                                'finished_at':time.time()
                            }
                    except Exception as exc:
                        with egx_market_lock:
                            egx_market_cache[key]={
                                'status':'error','total':0,'progress':0,'errors':1,'rows':[],
                                'latest_date':'','lookup':{},'error':f'{type(exc).__name__}: {exc}',
                                'finished_at':time.time()
                            }

                with egx_market_lock:
                    cached=egx_market_cache.get(cache_key)
                    # Expire completed results after 30 minutes; data is daily.
                    if cached and cached.get('status')=='done' and time.time()-cached.get('finished_at',0)>1800:
                        cached=None
                        egx_market_cache.pop(cache_key,None)
                    if force:
                        cached=None
                        egx_market_cache.pop(cache_key,None)
                    if cached is None:
                        egx_market_cache[cache_key]={'status':'starting','rows':[],'progress':0,'total':0,'errors':0,
                                                     'latest_date':'','lookup':{},'data_map':{},'started_at':time.time()}
                        threading.Thread(target=_market_scan_job,args=(cache_key,),daemon=True,name='egx-lab-market-scan').start()
                        cached=egx_market_cache[cache_key]

                with egx_market_lock:
                    cached=dict(egx_market_cache.get(cache_key,cached))
                rows=list(cached.get('rows') or [])
                if analysis_year is not None:
                    rows=[x for x in rows if str(x.get('date','')).startswith(f'{analysis_year:04d}-')]
                market_symbols=int(cached.get('total') or 0)
                market_errors=int(cached.get('errors') or 0)
                latest_date=cached.get('latest_date') or ''
                market_scan_status=cached.get('status','starting')
                market_scan_progress=int(cached.get('progress') or 0)
                market_scan_error=cached.get('error','')
                lookup=dict(cached.get('lookup') or {})
                compact_data=dict(cached.get('data_map') or {})
                summary=_summarize(rows)
                if show_portfolio and rows:
                    portfolio_summary,portfolio_svg,portfolio_ledger=_portfolio_backtest(rows,compact_data,analysis_year)

                if chart_symbol:
                    cres=_analyse(chart_symbol,lookup.get(chart_symbol,chart_symbol))
                    chart_rows=[x for x in cres['rows'] if analysis_year is None or str(x.get('date','')).startswith(f'{analysis_year:04d}-')]
                    chart_svg=_build_svg(cres['data'],chart_rows)
                    bars_count=len(cres['data'])
                    company=lookup.get(chart_symbol,chart_symbol)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'

        # Sort the results by any visible data column. Sorting is server-side so
        # it works without JavaScript and preserves all current lab filters.
        sort_key=request.args.get('sort','date').strip().lower()
        sort_dir=request.args.get('dir','desc').strip().lower()
        if sort_dir not in ('asc','desc'): sort_dir='desc'
        sort_fields={
            'symbol':lambda x:(x.get('symbol') or '').upper(),
            'company':lambda x:(x.get('company') or '').upper(),
            'status':lambda x:x.get('status') or '',
            'date':lambda x:x.get('date') or '',
            'exit_date':lambda x:x.get('exit_date') or '',
            'exit_return':lambda x:x.get('exit_return'),
            'exit_price':lambda x:x.get('exit_price'),
            'exit_slope':lambda x:x.get('exit_slope_value'),
            'entry_cci':lambda x:x.get('entry_cci'),
            'exit_cci':lambda x:x.get('exit_cci'),
            'price':lambda x:x.get('price'),
            'r2':lambda x:x.get('r2'),
            'slope':lambda x:x.get('slope'),
            'current_return':lambda x:x.get('current_return'),
            'max_gain':lambda x:x.get('max_gain'),
            'max_dd':lambda x:x.get('max_dd'),
            'hit20':lambda x:x.get('hit20'),
            'hit50':lambda x:x.get('hit50'),
            'hit100':lambda x:x.get('hit100'),
            'trade_duration':lambda x:x.get('trade_duration'),
            'age':lambda x:x.get('age'),
        }
        if rows and sort_key in sort_fields:
            keyfn=sort_fields[sort_key]
            nonempty=[x for x in rows if keyfn(x) is not None]
            empty=[x for x in rows if keyfn(x) is None]
            nonempty.sort(key=keyfn,reverse=(sort_dir=='desc'))
            rows=nonempty+empty

        def sort_link(key):
            args=request.args.to_dict()
            current=args.get('sort','date')
            current_dir=args.get('dir','desc')
            args['sort']=key
            args['dir']='asc' if current==key and current_dir=='desc' else 'desc'
            return url_for('egx_lab',**args)

        def sort_mark(key):
            if sort_key!=key:return ''
            return '▲' if sort_dir=='asc' else '▼'

        def chart_link(row):
            args=request.args.to_dict();args['scope']='market';args['chart_symbol']=row['symbol'];args.pop('symbol',None)
            return url_for('egx_lab',**args)

        return render_template('egx_lab.html',scope=scope,symbol=symbol,chart_symbol=chart_symbol,r2_min=r2_min,slope_min=slope_min,slope_max=slope_max,cooldown=cooldown,tp_pct=tp_pct,
            analysis_year=analysis_year,analysis_years=analysis_years,
            confirm=confirm,rows=rows,error=error,latest_date=latest_date,bars_count=bars_count,summary=summary,company=company,chart_svg=chart_svg,
            exit_mode=exit_mode,time_exit_sessions=time_exit_sessions,exit_slope=exit_slope,exit_slope_op=exit_slope_op,
            cci_enabled=cci_enabled,cci_period=cci_period,cci_min=cci_min,cci_max=cci_max,
            market_symbols=market_symbols,market_errors=market_errors,chart_link=chart_link,sort_link=sort_link,sort_mark=sort_mark,sort_key=sort_key,sort_dir=sort_dir,show_portfolio=show_portfolio,portfolio_svg=portfolio_svg,portfolio_summary=portfolio_summary,portfolio_ledger=portfolio_ledger)


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
