"""Private, expiring customer links for the current recommendations table."""
import hashlib,secrets,time,os
from sqlalchemy import Column,String,BigInteger,select,delete
from core import Base,Lead
from .customer import eligible,minimum_score,Publication,performance,stamp,price,NY
from .models import Plan,Stock,Event

class TableAccess(Base):
    __tablename__='customer_table_access'
    token_hash=Column(String(64),primary_key=True)
    telegram_id=Column(BigInteger,nullable=False,index=True)
    expires=Column(BigInteger,nullable=False,index=True)

def token_hash(token):return hashlib.sha256(token.encode()).hexdigest()

def table_entry(s,lead):
    from .customer import buttons
    if not eligible(s,lead):
        return 'جدول التوصيات متاح لحسابات الأمريكي المفعّلة. راجعي تفعيل الحساب أو استئناف التنبيهات.',buttons()
    token=secrets.token_urlsafe(32);clock=int(time.time())
    s.execute(delete(TableAccess).where(TableAccess.expires<clock))
    s.add(TableAccess(token_hash=token_hash(token),telegram_id=lead.telegram_id,expires=clock+86400))
    base=os.getenv('PUBLIC_BASE_URL','https://rashid-production-0987.up.railway.app').rstrip('/')
    return ('📈 التوصيات الحالية | راجح\nافتح الجدول لعرض الأسهم والأداء بالألوان. الرابط الخاص بحسابك صالح لمدة24 ساعة.',
        {'inline_keyboard':[[{'text':'📊 فتح جدول التوصيات الحالية','url':base+'/recommendations/current/'+token}],
         [{'text':'📊 نتائج التوصيات','callback_data':'us:results:0'}]]})

def authorized_lead(s,token,clock=None):
    if not 30<=len(token)<=100:return None
    access=s.get(TableAccess,token_hash(token))
    if not access or access.expires<=int(time.time() if clock is None else clock):return None
    lead=s.scalar(select(Lead).where(Lead.telegram_id==access.telegram_id))
    return lead if eligible(s,lead) else None

def current_rows(s,lead):
    from .limits import delivered_ids
    query=(select(Plan,Stock).join(Publication,Publication.plan_id==Plan.id)
        .outerjoin(Stock,Stock.symbol==Plan.symbol).where(Plan.market=='US',
        Plan.state.in_(('ACTIVE','DATA_GAP')),Plan.score>=minimum_score(s),Plan.id.in_(delivered_ids(lead.telegram_id)))
        .order_by(Publication.activation_end.desc(),Plan.id.desc()))
    rows=[]
    for p,stock in s.execute(query):
        pnl=performance(p,stock)
        tone='flat' if pnl is None or (stock and price(stock.last_price)==price(p.paper_entry)) else 'gain' if pnl>0 else 'loss'
        rows.append(dict(symbol=p.symbol,company=stock.company if stock else '',entry=p.paper_entry,
            target=p.target,stop=p.stop,last=stock.last_price if stock else None,pnl=pnl,tone=tone,
            updated=stamp(stock.last_bar+900) if stock and stock.last_bar else 'غير متاح',
            warning=p.state=='DATA_GAP' or bool(stock and stock.error)))
    return rows


def results_entry(s,lead):
    from .customer import buttons
    if not eligible(s,lead):
        return 'نتائج توصيات الشهر متاحة لحسابات الأمريكي المفعّلة فقط.',buttons()
    token=secrets.token_urlsafe(32);clock=int(time.time())
    s.execute(delete(TableAccess).where(TableAccess.expires<clock))
    s.add(TableAccess(token_hash=token_hash(token),telegram_id=lead.telegram_id,expires=clock+86400))
    base=os.getenv('PUBLIC_BASE_URL','https://rashid-production-0987.up.railway.app').rstrip('/')
    return ('📊 نتائج توصيات الشهر الخاصة بحسابك فقط\nالجدول لا يعرض أي توصية لم يتم تسليمها إلى حسابك. الرابط الخاص صالح لمدة 24 ساعة.',
        {'inline_keyboard':[[{'text':'📊 فتح جدول نتائج توصياتي','url':base+'/recommendations/results/'+token}],
         [{'text':'📈 التوصيات الحالية','callback_data':'us:current:0'}],[{'text':'🟡 تداول الذهب','callback_data':'gold:menu'}],[{'text':'⚪ تداول الفضة','callback_data':'silver:menu'}],[{'text':'🛢️ تداول البترول','callback_data':'oil:menu'}]]})

def monthly_rows(s,lead,clock=None):
    from datetime import datetime
    from sqlalchemy import or_
    from .limits import delivered_ids
    clock=time.time() if clock is None else clock
    dt=datetime.fromtimestamp(clock,NY)
    start=datetime(dt.year,dt.month,1,tzinfo=NY).timestamp()
    end=datetime(dt.year+int(dt.month==12),1 if dt.month==12 else dt.month+1,1,tzinfo=NY).timestamp()
    closed=select(Event.plan_id).where(Event.kind.in_(('TARGET','STOPPED','TIME_EXIT')),Event.bar_ts+900>=start,Event.bar_ts+900<end)
    query=(select(Plan,Stock).join(Publication,Publication.plan_id==Plan.id)
        .outerjoin(Stock,Stock.symbol==Plan.symbol).where(Plan.market=='US',
        Plan.id.in_(delivered_ids(lead.telegram_id,'US')),
        or_((Publication.activation_end>=start)&(Publication.activation_end<end),Plan.id.in_(closed)))
        .order_by(Publication.activation_end.desc(),Plan.id.desc()))
    rows=[]
    for plan,stock in s.execute(query):
        pnl=performance(plan,stock);closed_state=plan.state in ('TARGET','STOPPED','TIME_EXIT')
        rows.append(dict(symbol=plan.symbol,company=stock.company if stock else '',state=plan.state,
            entry=plan.paper_entry,target=plan.target,stop=plan.stop,exit=plan.exit_price if closed_state else None,
            last=stock.last_price if stock else None,pnl=pnl,activation=stamp(plan.activation_ts+900) if plan.activation_ts else 'غير متاح'))
    return rows,dt.strftime('%Y-%m')
