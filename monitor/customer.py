"""US-only customer publication ledger, eligibility, and paginated bot views."""
import os, time, math, json
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import Column, Integer, BigInteger, String, select, func, or_
from core import Base, Lead, Outbox, Setting, queue, now
from .models import Plan, Stock, Event

class Publication(Base):
    __tablename__ = 'us_recommendation_publications'
    plan_id = Column(Integer, primary_key=True)
    published_ts = Column(BigInteger, nullable=False, index=True)
    activation_end = Column(BigInteger, nullable=False)

class Recipient(Base):
    __tablename__ = 'us_recommendation_recipients'
    plan_id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, primary_key=True)
    entry_key = Column(String(150), nullable=False, unique=True)

class Preference(Base):
    __tablename__ = 'us_recommendation_preferences'
    telegram_id = Column(BigInteger, primary_key=True)
    paused = Column(Integer, default=0, nullable=False)

class GoldPreference(Base):
    __tablename__ = 'gold_recommendation_preferences'
    telegram_id = Column(BigInteger, primary_key=True)
    paused = Column(Integer, default=0, nullable=False)


NY=ZoneInfo('America/New_York')
PAGE_SIZE=6
LABELS={'ACTIVE':'مفتوحة','TARGET':'تحقق الهدف','STOPPED':'وقف خسارة','TIME_EXIT':'خروج بعد مدة الانتظار','DATA_GAP':'المتابعة معلقة — بيانات ناقصة'}

MIN_SCORE_KEY='us_recommendation_min_score'
COMMODITIES={
    'XA':dict(symbol='XAUUSD',label='الذهب',emoji='🟡',prefix='gold',preference=GoldPreference),
}


def minimum_score(s):
    row=s.get(Setting,MIN_SCORE_KEY)
    if row is None:return 0.0
    try:value=float(row.value)
    except (ValueError,TypeError):return 100.0
    return value if math.isfinite(value) and 0<=value<=100 else 100.0

def score_allowed(s,plan):
    return plan.score is not None and math.isfinite(plan.score) and plan.score>=minimum_score(s)

def enabled(): return os.getenv('US_RECOMMENDATIONS_ENABLED','1')=='1'
def eligible(s, lead):
    if not lead or lead.market not in ('us','both') or lead.status not in ('trial','subscribed'):return False
    if not lead.completed_at or not lead.consent_at or lead.telegram_status=='blocked':return False
    pref=s.get(Preference,lead.telegram_id)
    return not pref or not pref.paused

def commodity_eligible(s,lead,market):
    if not lead or lead.status not in ('trial','subscribed'):return False
    if not lead.completed_at or not lead.consent_at or lead.telegram_status=='blocked':return False
    cfg=COMMODITIES[market];pref=s.get(cfg['preference'],lead.telegram_id)
    return not pref or not pref.paused

def gold_eligible(s,lead):return commodity_eligible(s,lead,'XA')

def buttons():
    return {'inline_keyboard':[[{'text':'📈 التوصيات الحالية','callback_data':'us:current:0'}],
        [{'text':'📊 نتائج توصيات الشهر','callback_data':'us:results:0'}],
        [{'text':'🟡 تداول الذهب','callback_data':'gold:menu'}]]}

def commodity_buttons(market):
    cfg=COMMODITIES[market];p=cfg['prefix'];label=cfg['label']
    return {'inline_keyboard':[[{'text':f'📈 توصيات {label} الحالية','callback_data':f'{p}:current:0'}],
        [{'text':f'📊 نتائج {label} الشهرية','callback_data':f'{p}:results:0'}],
        [{'text':'⬅️ القائمة الرئيسية','callback_data':'main:menu'}]]}

def gold_buttons():return commodity_buttons('XA')

def stamp(ts):return datetime.fromtimestamp(ts,NY).strftime('%Y-%m-%d %H:%M')+' نيويورك'
def price(v):return 'غير متاح' if v is None else f'{v:.2f}'


def order_label(plan):
    try:
        ctx=json.loads(plan.context_json or '{}')
        if ctx.get('entry_system')=='SECOND_STOP_RECOVERY':return 'Limit شراء — المستوى الثاني'
        kind=ctx.get('order_type','')
        if kind not in ('MARKET','LIMIT','STOP'):
            ref=ctx.get('signal_bar_close') or ctx.get('feed_last_price')
            if ref and float(ref)>0:
                gap=100*(float(plan.entry)/float(ref)-1)
                kind='MARKET' if abs(gap)<=.5 else ('LIMIT' if plan.entry<float(ref) else 'STOP')
    except Exception:kind=''
    return {'MARKET':'سوق','LIMIT':'Limit شراء','STOP':'Stop شراء'}.get(kind,'أمر شراء')

def second_stop_customer_card(plan,stock,status,price_value=None):
    company=' '.join((stock.company or '').split())[:120] if stock else ''
    heading=f'{plan.symbol} | {company}' if company else plan.symbol
    buy=plan.paper_entry if price_value is None else price_value
    if buy is None:buy=plan.entry
    return (
        f'{heading} — {status}\n'
        f'نوع الأمر: ليمت شراء — سعر الشراء: {price(buy)}$ · الهدف: {price(plan.target)}$ · الوقف: {price(plan.stop)}$'
    )

def _is_second_stop(plan):
    try:return plan.market=='US' and json.loads(plan.context_json or '{}').get('entry_system')=='SECOND_STOP_RECOVERY'
    except Exception:return False


def performance(plan, stock):
    mark=plan.exit_price if plan.exit_price is not None else (stock.last_price if stock else None)
    if plan.paper_entry is None or not mark:return None
    return (mark/plan.paper_entry-1)*100

def card(plan,stock):
    pnl=performance(plan,stock)
    closed=plan.state in ('TARGET','STOPPED','TIME_EXIT')
    company=' '.join((stock.company or '').split())[:120] if stock else ''
    heading=f'{plan.symbol} | {company}' if company else plan.symbol
    lines=[f'{heading} — {LABELS.get(plan.state,plan.state)}',
           f'نوع الأمر: {order_label(plan)} · دخول التوصية: {price(plan.paper_entry)}$ · الهدف: {price(plan.target)}$ · الوقف: {price(plan.stop)}$']
    try:
        ctx=json.loads(plan.context_json or '{}')
        if ctx.get('entry_system')=='SECOND_STOP_RECOVERY':
            lines.append('طريقة الدخول: تفعيل عند المستوى الثاني؛ الهدف هو مستوى الوقف الأول المحسوب من الإشارة الأصلية.')
    except Exception:
        pass
    if closed:lines.append(f'سعر الخروج المرجعي: {price(plan.exit_price)}$')
    elif stock:lines.append(f'آخر إغلاق: {price(stock.last_price)}$ · تحديث {stamp(stock.last_bar+900)}')
    if pnl is not None:lines.append(f'{"نتيجة الإغلاق" if closed else "الأداء غير المحقق"}: {pnl:+.2f}%')
    if stock and stock.error and not closed:lines.append('⚠️ تحديث البيانات متأخر؛ السعر المعروض آخر سعر متاح.')
    return '\n'.join(lines)

def notice(s,plan,kind,ts,clock=None):
    """Customer recommendation publication and updates."""
    clock=int(time.time() if clock is None else clock)
    if plan.market!='US' and plan.market not in COMMODITIES or not enabled():return

    pub=s.get(Publication,plan.id)

    # SECOND_STOP_RECOVERY customer flow:
    # 1) ENTRY_ALERT when first/original stop is touched -> send pending LIMIT order.
    # 2) ACTIVE when second level is actually filled -> send activation update.
    # 3) Later state changes use the exact same two-line compact format.
    if _is_second_stop(plan):
        stock=s.get(Stock,plan.symbol)

        if kind=='ENTRY_ALERT':
            if not score_allowed(s,plan) or pub:return
            if not 0<=clock-(ts+900)<=900:return

            pub=Publication(plan_id=plan.id,published_ts=clock,activation_end=ts+900)
            s.add(pub)

            leads=s.scalars(select(Lead).where(Lead.status.in_(('trial','subscribed')))).all()
            from .limits import available
            for lead in leads:
                lead=s.scalar(select(Lead).where(
                    Lead.id==lead.id
                ).with_for_update().execution_options(populate_existing=True))
                if not eligible(s,lead) or not available(s,lead.telegram_id,plan.entry,clock):
                    continue

                key=f'usrec:{plan.id}:ENTRY_ALERT:{lead.telegram_id}'
                s.add(Recipient(plan_id=plan.id,telegram_id=lead.telegram_id,entry_key=key))
                text=second_stop_customer_card(
                    plan,stock,'مفتوحة',price_value=plan.entry
                )
                _queue(
                    s,key,lead.telegram_id,text,plan.id,'ENTRY_ALERT',
                    ts+1800,reply_markup=None
                )
            return

        if kind=='ACTIVE':
            # Existing plans created before this update can theoretically activate
            # without an ENTRY_ALERT.  Fall back to publishing to eligible clients
            # so no activation is silently lost.
            if not score_allowed(s,plan) or plan.paper_entry is None:return

            if pub is None:
                pub=Publication(plan_id=plan.id,published_ts=clock,activation_end=ts+900)
                s.add(pub)
                leads=s.scalars(select(Lead).where(Lead.status.in_(('trial','subscribed')))).all()
                from .limits import available
                for lead in leads:
                    lead=s.scalar(select(Lead).where(
                        Lead.id==lead.id
                    ).with_for_update().execution_options(populate_existing=True))
                    if not eligible(s,lead) or not available(s,lead.telegram_id,plan.paper_entry,clock):
                        continue
                    entry_key=f'usrec:{plan.id}:ENTRY_ALERT:{lead.telegram_id}'
                    s.add(Recipient(plan_id=plan.id,telegram_id=lead.telegram_id,entry_key=entry_key))
                    # For fallback only, send the order card immediately before activation.
                    order_text=second_stop_customer_card(
                        plan,stock,'مفتوحة',price_value=plan.entry
                    )
                    _queue(
                        s,entry_key,lead.telegram_id,order_text,plan.id,'ENTRY_ALERT',
                        ts+1800,reply_markup=None
                    )
            else:
                pub.activation_end=ts+900

            for recipient in s.scalars(select(Recipient).where(Recipient.plan_id==plan.id)):
                lead=s.scalar(select(Lead).where(Lead.telegram_id==recipient.telegram_id))
                if not eligible(s,lead):continue
                text=second_stop_customer_card(
                    plan,stock,'تم التفعيل',price_value=plan.paper_entry
                )
                _queue(
                    s,f'usrec:{plan.id}:ACTIVE:{recipient.telegram_id}',
                    recipient.telegram_id,text,plan.id,'ACTIVE',None,
                    reply_markup=None
                )
            return

        if pub and kind in ('TARGET','STOPPED','TIME_EXIT','DATA_GAP'):
            status={
                'TARGET':'تحقق الهدف',
                'STOPPED':'وقف خسارة',
                'TIME_EXIT':'تم الخروج',
                'DATA_GAP':'المتابعة معلقة',
            }[kind]
            for recipient in s.scalars(select(Recipient).where(Recipient.plan_id==plan.id)):
                lead=s.scalar(select(Lead).where(Lead.telegram_id==recipient.telegram_id))
                if not eligible(s,lead):continue
                text=second_stop_customer_card(
                    plan,stock,status,price_value=plan.paper_entry or plan.entry
                )
                _queue(
                    s,f'usrec:{plan.id}:{kind}:{recipient.telegram_id}',
                    recipient.telegram_id,text,plan.id,kind,None,
                    reply_markup=None
                )
            return

        return

    # Legacy US / commodity behavior remains unchanged.
    if kind=='ACTIVE':
        if not score_allowed(s,plan):return
        if pub or plan.paper_entry is None or not 0<=clock-(ts+900)<=900:return
        pub=Publication(plan_id=plan.id,published_ts=clock,activation_end=ts+900);s.add(pub)
        leads=s.scalars(select(Lead).where(Lead.status.in_(('trial','subscribed')))).all()
        from .limits import available
        for lead in leads:
            lead=s.scalar(select(Lead).where(Lead.id==lead.id).with_for_update().execution_options(populate_existing=True))
            if plan.market=='US':
                if not eligible(s,lead) or not available(s,lead.telegram_id,plan.paper_entry,clock):continue
            elif not commodity_eligible(s,lead,plan.market):continue
            key=f'usrec:{plan.id}:ACTIVE:{lead.telegram_id}'
            s.add(Recipient(plan_id=plan.id,telegram_id=lead.telegram_id,entry_key=key))
            stock=s.get(Stock,plan.symbol)
            if plan.market in COMMODITIES:
                cfg=COMMODITIES[plan.market];text=f"{cfg['emoji']} راجح | توصية {cfg['label']} {cfg['symbol']} جديدة — 15 دقيقة"+'\n\n'+card(plan,stock)
            else:text='📈 راجح | توصية أمريكية جديدة — 15 دقيقة'+'\n\n'+card(plan,stock)
            text+='\n\nتفعيل الشمعة: '+stamp(ts+900)
            _queue(s,key,lead.telegram_id,text,plan.id,kind,ts+1800)
    elif pub and kind in ('TARGET','STOPPED','TIME_EXIT','DATA_GAP'):
        for recipient in s.scalars(select(Recipient).where(Recipient.plan_id==plan.id)):
            lead=s.scalar(select(Lead).where(Lead.telegram_id==recipient.telegram_id))
            ok=commodity_eligible(s,lead,plan.market) if plan.market in COMMODITIES else eligible(s,lead)
            if ok:
                title=f"📊 تحديث توصية راجح لـ{COMMODITIES[plan.market]['label']}" if plan.market in COMMODITIES else '📊 تحديث توصية راجح الأمريكية'
                text=title+'\n\n'+card(plan,s.get(Stock,plan.symbol))+'\nوقت الحدث: '+stamp(ts+900)+'\nالأداء مرجعي قبل الرسوم.'
                _queue(s,f'usrec:{plan.id}:{kind}:{recipient.telegram_id}',recipient.telegram_id,text,plan.id,kind,None)

def _queue(s,key,tid,text,pid,kind,expires,reply_markup='default'):
    if s.scalar(select(Outbox.id).where(Outbox.key==key)):return
    payload={'chat_id':tid,'text':text,'_us_plan_id':pid,'_us_kind':kind,'_us_expires':expires}
    if reply_markup=='default':payload['reply_markup']=buttons()
    elif reply_markup is not None:payload['reply_markup']=reply_markup
    s.add(Outbox(key=key,chat_id=tid,payload=json.dumps(payload,ensure_ascii=False)))

def delivery_allowed(s,row,payload,clock=None):
    if '_us_plan_id' not in payload:return True
    lead=s.scalar(select(Lead).where(Lead.telegram_id==row.chat_id).with_for_update().execution_options(populate_existing=True))
    if not enabled():return False
    plan=s.get(Plan,payload['_us_plan_id'])
    if not plan or (plan.market!='US' and plan.market not in COMMODITIES):return False
    if plan.market=='US' and not eligible(s,lead):return False
    if plan.market in COMMODITIES and not commodity_eligible(s,lead,plan.market):return False
    if _is_second_stop(plan) and payload['_us_kind']=='ENTRY_ALERT':
        recipient=s.get(Recipient,(plan.id,row.chat_id))
        return bool(
            recipient and score_allowed(s,plan)
            and plan.state in ('WAITING','ACTIVE')
        )
    if _is_second_stop(plan) and payload['_us_kind']=='ACTIVE':
        recipient=s.get(Recipient,(plan.id,row.chat_id))
        entry=s.scalar(select(Outbox).where(Outbox.key==recipient.entry_key)) if recipient else None
        return bool(
            recipient and entry is not None
            and entry.status in ('sent','uncertain')
            and plan.state in ('ACTIVE','TARGET','STOPPED','TIME_EXIT','DATA_GAP')
        )
    if payload['_us_kind']=='ACTIVE':
        if plan.market=='US':
            from .limits import available
            if not available(s,row.chat_id,plan.paper_entry,clock,before_id=row.id):return False
        return score_allowed(s,plan) and plan.state=='ACTIVE' and int(time.time() if clock is None else clock)<=payload['_us_expires']
    recipient=s.get(Recipient,(plan.id,row.chat_id))
    entry=s.scalar(select(Outbox).where(Outbox.key==recipient.entry_key)) if recipient else None
    return entry is not None and entry.status in ('sent','uncertain')

def commodity_view(s,lead,market,kind,page=0,clock=None):
    cfg=COMMODITIES[market];prefix=cfg['prefix'];label=cfg['label'];symbol=cfg['symbol'];emoji=cfg['emoji']
    if not commodity_eligible(s,lead,market):
        return f'تداول {label} متاح للحسابات المفعّلة بعد تفعيل التجربة أو الاشتراك.',commodity_buttons(market)
    clock=time.time() if clock is None else clock
    dt=datetime.fromtimestamp(clock,NY);start=datetime(dt.year,dt.month,1,tzinfo=NY).timestamp()
    end=datetime(dt.year+int(dt.month==12),1 if dt.month==12 else dt.month+1,1,tzinfo=NY).timestamp()
    from .limits import delivered_ids
    query=select(Plan).join(Publication,Publication.plan_id==Plan.id).where(Plan.market==market,Plan.id.in_(delivered_ids(lead.telegram_id,market)))
    if kind=='current':
        query=query.where(Plan.state.in_(('ACTIVE','DATA_GAP')),Plan.score>=minimum_score(s));title=f'{emoji} توصيات {label} الحالية | {symbol} — 15 دقيقة'
    else:
        closed_in_month=select(Event.plan_id).where(Event.kind.in_(('TARGET','STOPPED','TIME_EXIT')),Event.bar_ts+900>=start,Event.bar_ts+900<end)
        query=query.where(or_((Publication.activation_end>=start)&(Publication.activation_end<end),Plan.id.in_(closed_in_month)));title=f'{emoji} نتائج {label} | {dt:%Y-%m} — توقيت نيويورك'
    total=s.scalar(select(func.count()).select_from(query.subquery())) or 0;pages=max(1,math.ceil(total/PAGE_SIZE));page=max(0,min(page,pages-1))
    plans=s.scalars(query.order_by(Publication.activation_end.desc(),Plan.id.desc()).offset(page*PAGE_SIZE).limit(PAGE_SIZE)).all()
    lines=[title,f'العدد: {total} · صفحة {page+1}/{pages}']
    if kind!='current':
        counts=dict(s.execute(select(Plan.state,func.count()).where(Plan.id.in_(query.with_only_columns(Plan.id))).group_by(Plan.state)).all())
        lines.append(f'تحقق الهدف: {counts.get("TARGET",0)} · خروج بعد المدة: {counts.get("TIME_EXIT",0)} · وقف: {counts.get("STOPPED",0)} · مفتوحة: {counts.get("ACTIVE",0)} · بيانات ناقصة: {counts.get("DATA_GAP",0)}')
    for plan in plans:lines.append(card(plan,s.get(Stock,plan.symbol)))
    if not plans:lines.append(f'لا توجد توصيات {label} مرسلة إلى حسابك في هذه القائمة حاليًا.')
    lines.append(f'{symbol} على فريم 15 دقيقة من Twelve Data. نفس شروط الذهب الحالية، من دون شرط الفوليوم. الأداء مرجعي قبل الرسوم.')
    kb=commodity_buttons(market);nav=[]
    if page:nav.append({'text':'السابق','callback_data':f'{prefix}:{kind}:{page-1}'})
    nav.append({'text':'تحديث','callback_data':f'{prefix}:{kind}:{page}'})
    if page+1<pages:nav.append({'text':'التالي','callback_data':f'{prefix}:{kind}:{page+1}'})
    kb['inline_keyboard'].insert(0,nav);return '\n\n'.join(lines),kb

def gold_view(s,lead,kind,page=0,clock=None):return commodity_view(s,lead,'XA',kind,page,clock)

def view(s,lead,kind,page=0,clock=None):
    if not eligible(s,lead):
        return 'خدمة توصيات الأسهم الأمريكية متاحة للعملاء المسجلين في الأمريكي أو السوقين، بعد تفعيل التجربة أو الاشتراك من خدمة العملاء. لو أوقفت التنبيهات، أرسل /resume_us لاستئنافها.',buttons()
    clock=time.time() if clock is None else clock
    dt=datetime.fromtimestamp(clock,NY);start=datetime(dt.year,dt.month,1,tzinfo=NY).timestamp()
    end=datetime(dt.year+int(dt.month==12),1 if dt.month==12 else dt.month+1,1,tzinfo=NY).timestamp()
    from .limits import delivered_ids
    query=select(Plan).join(Publication,Publication.plan_id==Plan.id).where(Plan.market=='US',Plan.id.in_(delivered_ids(lead.telegram_id,'US')))
    if kind=='current':
        query=query.where(Plan.state.in_(('ACTIVE','DATA_GAP')),Plan.score>=minimum_score(s))
        title='📈 التوصيات الحالية | الأمريكي — 15 دقيقة'
    else:
        closed_in_month=select(Event.plan_id).where(Event.kind.in_(('TARGET','STOPPED','TIME_EXIT')),Event.bar_ts+900>=start,Event.bar_ts+900<end)
        query=query.where(or_((Publication.activation_end>=start)&(Publication.activation_end<end),Plan.id.in_(closed_in_month)))
        title=f'📊 نتائج توصياتك أنت | {dt:%Y-%m} — توقيت نيويورك\nتظهر فقط التوصيات التي تم تسليمها إلى حسابك.'
    total=s.scalar(select(func.count()).select_from(query.subquery())) or 0
    pages=max(1,math.ceil(total/PAGE_SIZE));page=max(0,min(page,pages-1))
    plans=s.scalars(query.order_by(Publication.activation_end.desc(),Plan.id.desc()).offset(page*PAGE_SIZE).limit(PAGE_SIZE)).all()
    lines=[title,f'العدد: {total} · صفحة {page+1}/{pages}']
    if kind!='current':
        counts=dict(s.execute(select(Plan.state,func.count()).where(Plan.id.in_(query.with_only_columns(Plan.id))).group_by(Plan.state)).all())
        lines.append(f'تحقق الهدف: {counts.get("TARGET",0)} · خروج بعد المدة: {counts.get("TIME_EXIT",0)} · وقف: {counts.get("STOPPED",0)} · مفتوحة: {counts.get("ACTIVE",0)} · بيانات ناقصة: {counts.get("DATA_GAP",0)}')
    for plan in plans:lines.append(card(plan,s.get(Stock,plan.symbol)))
    if not plans:lines.append('لا توجد توصيات تم تسليمها إلى حسابك في هذه القائمة حاليًا.')
    lines.append('الأداء محسوب من سعر تفعيل التوصية قبل الرسوم؛ ليس عائد محفظة. آخر سعر من شمعة مكتملة وقد تتأخر البيانات.')
    kb=buttons();nav=[]
    if page:nav.append({'text':'السابق','callback_data':f'us:{kind}:{page-1}'})
    nav.append({'text':'تحديث','callback_data':f'us:{kind}:{page}'})
    if page+1<pages:nav.append({'text':'التالي','callback_data':f'us:{kind}:{page+1}'})
    kb['inline_keyboard'].insert(0,nav)
    return '\n\n'.join(lines),kb
