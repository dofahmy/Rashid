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

NY=ZoneInfo('America/New_York')
PAGE_SIZE=6
LABELS={'ACTIVE':'مفتوحة','TARGET':'تحقق الهدف','STOPPED':'وقف خسارة','DATA_GAP':'المتابعة معلقة — بيانات ناقصة'}

MIN_SCORE_KEY='us_recommendation_min_score'

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

def buttons():
    return {'inline_keyboard':[[{'text':'📈 التوصيات الحالية','callback_data':'us:current:0'}],
        [{'text':'📊 نتائج التوصيات','callback_data':'us:results:0'}]]}

def stamp(ts):return datetime.fromtimestamp(ts,NY).strftime('%Y-%m-%d %H:%M')+' نيويورك'
def price(v):return 'غير متاح' if v is None else f'{v:.2f}'

def performance(plan, stock):
    mark=plan.exit_price if plan.exit_price is not None else (stock.last_price if stock else None)
    if plan.paper_entry is None or not mark:return None
    return (mark/plan.paper_entry-1)*100

def card(plan,stock):
    pnl=performance(plan,stock)
    closed=plan.state in ('TARGET','STOPPED')
    company=' '.join((stock.company or '').split())[:120] if stock else ''
    heading=f'{plan.symbol} | {company}' if company else plan.symbol
    lines=[f'{heading} — {LABELS.get(plan.state,plan.state)}',
           f'دخول التوصية: {price(plan.paper_entry)}$ · الهدف: {price(plan.target)}$ · الوقف: {price(plan.stop)}$']
    if closed:lines.append(f'سعر الخروج المرجعي: {price(plan.exit_price)}$')
    elif stock:lines.append(f'آخر إغلاق: {price(stock.last_price)}$ · تحديث {stamp(stock.last_bar+900)}')
    if pnl is not None:lines.append(f'{"نتيجة الإغلاق" if closed else "الأداء غير المحقق"}: {pnl:+.2f}%')
    if stock and stock.error and not closed:lines.append('⚠️ تحديث البيانات متأخر؛ السعر المعروض آخر سعر متاح.')
    return '\n'.join(lines)

def notice(s,plan,kind,ts,clock=None):
    """Called in the same transaction as a state change; no historical bootstrap sends."""
    clock=int(time.time() if clock is None else clock)
    if plan.market!='US' or not enabled():return
    pub=s.get(Publication,plan.id)
    if kind=='ACTIVE':
        if not score_allowed(s,plan):return
        if pub or plan.paper_entry is None or not 0<=clock-(ts+900)<=900:return
        pub=Publication(plan_id=plan.id,published_ts=clock,activation_end=ts+900);s.add(pub)
        leads=s.scalars(select(Lead).where(Lead.market.in_(('us','both')),Lead.status.in_(('trial','subscribed')))).all()
        for lead in leads:
            if not eligible(s,lead):continue
            key=f'usrec:{plan.id}:ACTIVE:{lead.telegram_id}'
            s.add(Recipient(plan_id=plan.id,telegram_id=lead.telegram_id,entry_key=key))
            stock=s.get(Stock,plan.symbol)
            text='📈 راجح | توصية أمريكية جديدة — 15 دقيقة\n\n'+card(plan,stock)
            text+='\n\nتفعيل الشمعة: '+stamp(ts+900)
            _queue(s,key,lead.telegram_id,text,plan.id,kind,ts+1800)
    elif pub and kind in ('TARGET','STOPPED','DATA_GAP'):
        for recipient in s.scalars(select(Recipient).where(Recipient.plan_id==plan.id)):
            lead=s.scalar(select(Lead).where(Lead.telegram_id==recipient.telegram_id))
            if eligible(s,lead):
                text='📊 تحديث توصية راجح الأمريكية\n\n'+card(plan,s.get(Stock,plan.symbol))+'\nوقت الحدث: '+stamp(ts+900)+'\nالأداء مرجعي قبل الرسوم.'
                _queue(s,f'usrec:{plan.id}:{kind}:{recipient.telegram_id}',recipient.telegram_id,text,plan.id,kind,None)

def _queue(s,key,tid,text,pid,kind,expires):
    if s.scalar(select(Outbox.id).where(Outbox.key==key)):return
    payload={'chat_id':tid,'text':text,'reply_markup':buttons(),'_us_plan_id':pid,'_us_kind':kind,'_us_expires':expires}
    s.add(Outbox(key=key,chat_id=tid,payload=json.dumps(payload,ensure_ascii=False)))

def delivery_allowed(s,row,payload,clock=None):
    if '_us_plan_id' not in payload:return True
    lead=s.scalar(select(Lead).where(Lead.telegram_id==row.chat_id))
    if not enabled() or not eligible(s,lead):return False
    plan=s.get(Plan,payload['_us_plan_id'])
    if not plan or plan.market!='US':return False
    if payload['_us_kind']=='ACTIVE':
        return score_allowed(s,plan) and plan.state=='ACTIVE' and int(time.time() if clock is None else clock)<=payload['_us_expires']
    recipient=s.get(Recipient,(plan.id,row.chat_id))
    entry=s.scalar(select(Outbox).where(Outbox.key==recipient.entry_key)) if recipient else None
    return entry is not None and entry.status in ('sent','uncertain')

def view(s,lead,kind,page=0,clock=None):
    if not eligible(s,lead):
        return 'خدمة توصيات الأسهم الأمريكية متاحة للعملاء المسجلين في الأمريكي أو السوقين، بعد تفعيل التجربة أو الاشتراك من خدمة العملاء. لو أوقفت التنبيهات، أرسل /resume_us لاستئنافها.',buttons()
    clock=time.time() if clock is None else clock
    dt=datetime.fromtimestamp(clock,NY);start=datetime(dt.year,dt.month,1,tzinfo=NY).timestamp()
    end=datetime(dt.year+int(dt.month==12),1 if dt.month==12 else dt.month+1,1,tzinfo=NY).timestamp()
    query=select(Plan).join(Publication,Publication.plan_id==Plan.id).where(Plan.market=='US')
    if kind=='current':
        query=query.where(Plan.state.in_(('ACTIVE','DATA_GAP')),Plan.score>=minimum_score(s))
        title='📈 التوصيات الحالية | الأمريكي — 15 دقيقة'
    else:
        closed_in_month=select(Event.plan_id).where(Event.kind.in_(('TARGET','STOPPED')),Event.bar_ts+900>=start,Event.bar_ts+900<end)
        query=query.where(or_((Publication.activation_end>=start)&(Publication.activation_end<end),Plan.id.in_(closed_in_month)))
        title=f'📊 نتائج التوصيات | {dt:%Y-%m} — توقيت نيويورك\nتشمل ما تفعّل أو أُغلق هذا الشهر؛ المفتوح أداؤه غير محقق.'
    total=s.scalar(select(func.count()).select_from(query.subquery())) or 0
    pages=max(1,math.ceil(total/PAGE_SIZE));page=max(0,min(page,pages-1))
    plans=s.scalars(query.order_by(Publication.activation_end.desc(),Plan.id.desc()).offset(page*PAGE_SIZE).limit(PAGE_SIZE)).all()
    lines=[title,f'العدد: {total} · صفحة {page+1}/{pages}']
    if kind!='current':
        counts=dict(s.execute(select(Plan.state,func.count()).where(Plan.id.in_(query.with_only_columns(Plan.id))).group_by(Plan.state)).all())
        lines.append(f'تحقق الهدف: {counts.get("TARGET",0)} · وقف: {counts.get("STOPPED",0)} · مفتوحة: {counts.get("ACTIVE",0)} · بيانات ناقصة: {counts.get("DATA_GAP",0)}')
    for plan in plans:lines.append(card(plan,s.get(Stock,plan.symbol)))
    if not plans:lines.append('لا توجد توصيات منشورة في هذه القائمة حاليًا.')
    lines.append('الأداء محسوب من سعر تفعيل التوصية قبل الرسوم؛ ليس عائد محفظة. آخر سعر من شمعة مكتملة وقد تتأخر البيانات.')
    kb=buttons();nav=[]
    if page:nav.append({'text':'السابق','callback_data':f'us:{kind}:{page-1}'})
    nav.append({'text':'تحديث','callback_data':f'us:{kind}:{page}'})
    if page+1<pages:nav.append({'text':'التالي','callback_data':f'us:{kind}:{page+1}'})
    kb['inline_keyboard'].insert(0,nav)
    return '\n\n'.join(lines),kb
