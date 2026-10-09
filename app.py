import os, secrets, csv, io, math, time, re
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
        # Attach a consistent right sidebar to existing HTML pages without replacing base.html.
        # Only known registered GET routes become links; no broken Saudi URLs are invented.
        if r.status_code==200 and r.mimetype=='text/html' and session.get('admin'):
            try:
                from markupsafe import escape
                from flask import current_app
                route_labels={
                    'dashboard':('الإدارة','الرئيسية والعملاء'),
                    'egx_seven':('مصر','نظام 7 — مصر'),
                    'egx_seven_screener':('مصر','Repeat — مصر'),
                    'sp500_seven':('أمريكا','نظام 7 — S&P 500'),
                    'sp500_seven_screener':('أمريكا','Repeat — S&P 500'),
                    'sp500_hourly_screener':('أمريكا','S&P 500 — ساعة'),
                    'us_all_seven_hourly':('أمريكا','كل السوق — ساعة BUY 3%'),
                }
                groups={}
                for rule in current_app.url_map.iter_rules():
                    if 'GET' not in rule.methods or rule.arguments or rule.rule.startswith('/static'):
                        continue
                    endpoint=rule.endpoint
                    if endpoint in route_labels:
                        group,label=route_labels[endpoint]
                    elif any(t in (endpoint+' '+rule.rule).lower() for t in ('saudi','tadawul','ksa','السعود')):
                        group,label='السعودية',endpoint.replace('_',' ').title()
                    else:
                        continue
                    groups.setdefault(group,[]).append((url_for(endpoint),label,rule.rule==request.path))
                parts=['<nav id="market-right-nav" aria-label="التنقل بين صفحات الأسواق"><div class="market-nav-head">صفحات الأسواق</div>']
                for group,links in groups.items():
                    parts.append('<div class="market-nav-section">'+str(escape(group))+'</div>')
                    for href,label,active in links:
                        parts.append('<a class="'+('active' if active else '')+'" href="'+str(escape(href))+'">'+str(escape(label))+'</a>')
                parts.append('</nav>')
                parts.append('''<style id="market-right-nav-style">#market-right-nav{position:fixed;right:0;top:0;bottom:0;width:215px;box-sizing:border-box;overflow-y:auto;z-index:10000;background:#10283d;color:#fff;direction:rtl;box-shadow:-3px 0 12px #0002;font-family:inherit;padding:18px 10px}#market-right-nav a{display:block;color:#e8f4f9;text-decoration:none;padding:10px 9px;border-radius:7px;margin:2px 0;font-size:13px}#market-right-nav a:hover,#market-right-nav a.active{background:#16838c;color:#fff}#market-right-nav .market-nav-head{font-weight:800;font-size:18px;margin:7px 8px 22px}#market-right-nav .market-nav-section{font-size:12px;font-weight:bold;color:#82dce0;margin:16px 8px 4px;border-bottom:1px solid #365066;padding-bottom:7px}body{padding-right:225px!important;box-sizing:border-box} @media(max-width:700px){#market-right-nav{width:145px;padding:12px 5px}#market-right-nav a{font-size:11px;padding:8px 4px}body{padding-right:151px!important}}</style>''')
                output=r.get_data(as_text=True)
                if '</body>' in output.lower():
                    output=re.sub(r'</body\s*>',''.join(parts)+'</body>',output,count=1,flags=re.I)
                else:
                    output+=''.join(parts)
                r.set_data(output)
            except Exception:
                app.logger.exception('Market navigation inject failed')
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
        from monitor.models import Stock, Plan, Scan, LABELS
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
            hold_days=hold_days,hold_min_profit=hold_min_profit,send_minimum_score=send_minimum_score,labels=LABELS,stock_map=stocks_by_symbol,order_map=order_map,total=total,page=page,pages=pages,link=link,local=local)

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



    @app.get('/egx-market')
    @auth
    def egx_market():
        from monitor.egx_live import load_state,get_refresh_status,compute_state
        mode=request.args.get('mode','now')
        selected_date=request.args.get('date','').strip()
        with DB() as s:refresh_status=get_refresh_status(s)
        if mode!='date':
            from monitor.egx_live import save_state
            with DB() as s:
                state=load_state(s)
            # Safety fallback: after a deploy/key change there may be a completed
            # refresh status but no cached state under the new key.
            if state is None:
                try:
                    state=compute_state(None)
                    save_state(DB,state)
                except Exception as exc:
                    flash(f'فشل حساب Now: {type(exc).__name__}: {exc}')
                    state=None
            return render_template('egx_market.html',state=state,refresh_status=refresh_status,selected_mode='now',selected_date='')
        if not selected_date:
            flash('اختاري التاريخ أولًا.');return redirect(url_for('egx_market'))
        try:dt=datetime.strptime(selected_date,'%Y-%m-%d').date()
        except ValueError:abort(400,description='تاريخ غير صحيح.')
        try:state=compute_state(dt)
        except Exception as exc:
            flash(f'فشل حساب التاريخ المحدد: {type(exc).__name__}: {exc}');state=None
        return render_template('egx_market.html',state=state,refresh_status=refresh_status,selected_mode='date',selected_date=selected_date)

    @app.post('/egx-market/refresh')
    @auth
    def egx_refresh_market():
        import subprocess,sys
        from monitor.egx_live import set_refresh_status
        set_refresh_status(DB,{'status':'queued','started_at_utc':datetime.now(timezone.utc).isoformat(),'message':'تم طلب التحديث. سيبدأ تحديث الأسعار وإعادة الحساب الآن.'})
        subprocess.Popen([sys.executable,'-m','monitor.egx_refresh_once'],cwd='/app',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        flash('بدأ تحديث بيانات السوق المصري وإعادة الحساب. أعيدي فتح الصفحة بعد قليل.')
        return redirect(url_for('egx_market'))


    @app.get('/egx-seven')
    @auth
    def egx_seven():
        import traceback
        scope=request.args.get('scope','market')
        symbol=request.args.get('symbol','').strip().upper()
        metric=request.args.get('metric','both')
        date_mode=request.args.get('date_mode','all')
        day=request.args.get('day','').strip()
        month=request.args.get('month','').strip()
        start=request.args.get('start','').strip()
        end=request.args.get('end','').strip()
        signals_only=request.args.get('signals_only')=='1'
        try: repeat_price_tolerance_pct=float(request.args.get('repeat_price_tolerance_pct','1.0'))
        except Exception: repeat_price_tolerance_pct=1.0
        try: repeat_max_gap_sessions=int(request.args.get('repeat_max_gap_sessions','20'))
        except Exception: repeat_max_gap_sessions=20
        repeat_price_tolerance_pct=max(0.0,min(repeat_price_tolerance_pct,20.0))
        repeat_max_gap_sessions=max(1,min(repeat_max_gap_sessions,252))
        if metric=='both' and request.args: signals_only=True

        if scope not in ('market','stock','index'): scope='market'
        if metric not in ('both','price','volume'): metric='both'
        if date_mode not in ('all','day','month','range'): date_mode='all'

        result=None
        error=None
        symbols=[]

        try:
            from monitor.seven_system import run,list_symbols
            try:
                symbols=list_symbols()
            except Exception as exc:
                app.logger.exception('EGX Seven symbol list failed')
                error=f'تعذر تحميل قائمة الأسهم: {type(exc).__name__}: {exc}'

            if request.args:
                try:
                    result=run(
                        scope=scope,
                        symbol=symbol or None,
                        metric=metric,
                        date_mode=date_mode,
                        day=day or None,
                        month=month or None,
                        start=start or None,
                        end=end or None,
                        signals_only=signals_only,
                        repeat_price_tolerance_pct=repeat_price_tolerance_pct,
                        repeat_max_gap_sessions=repeat_max_gap_sessions,
                    )
                except Exception as exc:
                    app.logger.exception('EGX Seven calculation failed')
                    error=f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            app.logger.exception('EGX Seven import failed')
            error=f'فشل تحميل نظام الـ7: {type(exc).__name__}: {exc}'

        return render_template(
            'seven_system.html',
            result=result,error=error,symbols=symbols,
            scope=scope,symbol=symbol,metric=metric,date_mode=date_mode,
            day=day,month=month,start=start,end=end,signals_only=signals_only,
            repeat_price_tolerance_pct=repeat_price_tolerance_pct,
            repeat_max_gap_sessions=repeat_max_gap_sessions
        )


    @app.get('/egx-seven/day/<day>')
    @auth
    def egx_seven_day(day):
        from monitor.seven_system import market_day_stock_details
        try:
            dt=datetime.strptime(day,'%Y-%m-%d').date()
        except ValueError:
            abort(400,description='تاريخ غير صحيح.')
        try:
            kind=request.args.get('kind','double')
            rows=market_day_stock_details(dt,kind=kind)
            error=None
        except Exception as exc:
            rows=[]
            error=f'{type(exc).__name__}: {exc}'
        return render_template('seven_day_stocks.html',day=day,rows=rows,error=error,kind=request.args.get('kind','double'))

    @app.get('/egx-seven/screener')
    @auth
    def egx_seven_screener():
        from monitor.seven_system import repeat_price_time_screener

        try:
            price_tolerance_pct=float(request.args.get('price_tolerance_pct','1.0'))
        except Exception:
            price_tolerance_pct=1.0

        try:
            max_gap_sessions=int(request.args.get('max_gap_sessions','20'))
        except Exception:
            max_gap_sessions=20

        recent_raw=request.args.get('recent_days','90').strip()
        if recent_raw.lower() in ('','all','none'):
            recent_days=None
        else:
            try:
                recent_days=int(recent_raw)
            except Exception:
                recent_days=90

        price_tolerance_pct=max(0.0,min(price_tolerance_pct,20.0))
        max_gap_sessions=max(1,min(max_gap_sessions,252))
        if recent_days is not None:
            recent_days=max(1,min(recent_days,3650))

        error=None
        result=None
        try:
            result=repeat_price_time_screener(
                price_tolerance_pct=price_tolerance_pct,
                max_gap_sessions=max_gap_sessions,
                recent_days=recent_days,
            )
        except Exception as exc:
            app.logger.exception('EGX Seven repeat screener failed')
            error=f'{type(exc).__name__}: {exc}'

        rule_only=request.args.get('rule_only','0')=='1'
        rule_match_count=sum(1 for r in result['rows'] if r.get('breakout_rule_match')) if result else 0
        if result and rule_only:
            result=dict(result)
            result['rows']=[r for r in result['rows'] if r.get('breakout_rule_match')]
            result['count']=len(result['rows'])

        return render_template(
            'seven_screener.html',
            result=result,
            rule_only=rule_only, rule_match_count=rule_match_count,
            error=error,
            price_tolerance_pct=price_tolerance_pct,
            max_gap_sessions=max_gap_sessions,
            recent_days=recent_days if recent_days is not None else 'all',
        )


    @app.get('/sp500-seven')
    @auth
    def sp500_seven():
        from monitor.sp500_seven_system import run,list_symbols
        from monitor.sp500_seven_data import get_status

        scope=request.args.get('scope','market')
        symbol=request.args.get('symbol','').strip().upper()
        metric=request.args.get('metric','both')
        date_mode=request.args.get('date_mode','all')
        day=request.args.get('day','').strip()
        month=request.args.get('month','').strip()
        start=request.args.get('start','').strip()
        end=request.args.get('end','').strip()
        signals_only=request.args.get('signals_only')=='1'
        try: repeat_price_tolerance_pct=float(request.args.get('repeat_price_tolerance_pct','1.0'))
        except Exception: repeat_price_tolerance_pct=1.0
        try: repeat_max_gap_sessions=int(request.args.get('repeat_max_gap_sessions','20'))
        except Exception: repeat_max_gap_sessions=20

        repeat_price_tolerance_pct=max(0.0,min(repeat_price_tolerance_pct,20.0))
        repeat_max_gap_sessions=max(1,min(repeat_max_gap_sessions,252))

        result=None;error=None
        try:
            symbols=list_symbols()
        except Exception:
            symbols=[]

        with DB() as s:
            refresh_status=get_status(s)

        if request.args:
            try:
                result=run(
                    scope=scope,symbol=symbol or None,metric=metric,
                    date_mode=date_mode,day=day or None,month=month or None,
                    start=start or None,end=end or None,signals_only=signals_only,
                    repeat_price_tolerance_pct=repeat_price_tolerance_pct,
                    repeat_max_gap_sessions=repeat_max_gap_sessions,
                )
            except Exception as exc:
                app.logger.exception('SP500 Seven calculation failed')
                error=f'{type(exc).__name__}: {exc}'

        return render_template(
            'sp500_seven.html',
            result=result,error=error,symbols=symbols,
            scope=scope,symbol=symbol,metric=metric,date_mode=date_mode,
            day=day,month=month,start=start,end=end,signals_only=signals_only,
            repeat_price_tolerance_pct=repeat_price_tolerance_pct,
            repeat_max_gap_sessions=repeat_max_gap_sessions,
            refresh_status=refresh_status,
        )

    @app.post('/sp500-seven/refresh')
    @auth
    def sp500_seven_refresh():
        import subprocess,sys
        subprocess.Popen(
            [sys.executable,'-m','monitor.sp500_seven_refresh_once'],
            cwd='/app',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        flash('بدأ تحديث قائمة وأسعار S&P 500. أول تحميل قد يأخذ وقتًا لأنه يبني التاريخ.')
        return redirect(url_for('sp500_seven'))

    @app.get('/sp500-seven/screener')
    @auth
    def sp500_seven_screener():
        from monitor.sp500_seven_system import repeat_price_time_screener
        try: price_tolerance_pct=float(request.args.get('price_tolerance_pct','1.0'))
        except Exception: price_tolerance_pct=1.0
        try: max_gap_sessions=int(request.args.get('max_gap_sessions','20'))
        except Exception: max_gap_sessions=20
        recent_raw=request.args.get('recent_days','90').strip()
        if recent_raw.lower() in ('','all','none'): recent_days=None
        else:
            try: recent_days=int(recent_raw)
            except Exception: recent_days=90

        error=None;result=None
        try:
            result=repeat_price_time_screener(
                price_tolerance_pct=price_tolerance_pct,
                max_gap_sessions=max_gap_sessions,
                recent_days=recent_days,
            )
        except Exception as exc:
            app.logger.exception('SP500 repeat screener failed')
            error=f'{type(exc).__name__}: {exc}'

        rule_only=request.args.get('rule_only','0')=='1'
        rule_match_count=sum(1 for r in result['rows'] if r.get('breakout_rule_match')) if result else 0
        if result and rule_only:
            result=dict(result)
            result['rows']=[r for r in result['rows'] if r.get('breakout_rule_match')]
            result['count']=len(result['rows'])

        return render_template(
            'sp500_seven_screener.html',
            result=result,error=error,
            rule_only=rule_only,rule_match_count=rule_match_count,
            price_tolerance_pct=price_tolerance_pct,
            max_gap_sessions=max_gap_sessions,
            recent_days=recent_days if recent_days is not None else 'all',
        )

    @app.get('/sp500-seven/hourly')
    @auth
    def sp500_hourly_screener():
        from monitor.sp500_seven_hourly import screener
        try: tol=float(request.args.get('price_tolerance_pct','1.0'))
        except (TypeError,ValueError):tol=1.0
        try: gap=int(request.args.get('max_gap_bars','20'))
        except (TypeError,ValueError):gap=20
        try:
            recent_raw=request.args.get('recent_bars','300')
            recent=None if recent_raw=='all' else max(0,int(recent_raw))
        except (TypeError,ValueError):recent=300
        rule_only=request.args.get('rule_only','0')=='1'
        result=None;error=None;match_count=0
        try:
            result=screener(max(0,min(tol,20)),max(1,min(gap,250)),recent)
            match_count=sum(1 for r in result['rows'] if r['breakout_rule_match'])
            if rule_only:
                result=dict(result,rows=[r for r in result['rows'] if r['breakout_rule_match']]);result['count']=len(result['rows'])
        except Exception as exc:
            app.logger.exception('SP500 hourly screener failed');error=f'{type(exc).__name__}: {exc}'
        return render_template('sp500_seven_hourly.html',result=result,error=error,
            price_tolerance_pct=tol,max_gap_bars=gap,recent_bars=recent if recent is not None else 'all',rule_only=rule_only,rule_match_count=match_count)

    @app.get('/us-all-seven/hourly')
    @auth
    def us_all_seven_hourly():
        from monitor.us_all_hourly_dashboard import dashboard
        try:
            raw=request.args.get('days','all')
            days=None if raw=='all' else int(raw)
            if days is not None and days not in (7,14,30,45,60):days=60
            page=max(1,int(request.args.get('page','1')))
            raw_min=request.args.get('gap_min','').strip()
            raw_max=request.args.get('gap_max','').strip()
            gap_min=int(raw_min) if raw_min else None
            gap_max=int(raw_max) if raw_max else None
            factor_cap=int(request.args.get('max_factor','7'))
            result=dashboard(max_factor=factor_cap,days=days,status=request.args.get('status','ALL'),
                query=request.args.get('q',''),page=page,gap_min=gap_min,gap_max=gap_max)
            error=None
        except Exception as exc:
            app.logger.exception('US all hourly dashboard failed')
            result=None;error=f'{type(exc).__name__}: {exc}'
        return render_template('us_all_seven_hourly.html',result=result,error=error)

    @app.get('/sp500-seven/day/<day>')
    @auth
    def sp500_seven_day(day):
        from monitor.sp500_seven_system import market_day_stock_details
        try:
            dt=datetime.strptime(day,'%Y-%m-%d').date()
        except ValueError:
            abort(400,description='Invalid date.')
        kind=request.args.get('kind','double')
        try:
            rows=market_day_stock_details(dt,kind=kind);error=None
        except Exception as exc:
            rows=[];error=f'{type(exc).__name__}: {exc}'
        return render_template('sp500_seven_day_stocks.html',day=day,rows=rows,error=error,kind=kind)

    return app
