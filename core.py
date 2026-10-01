import os, re, json, unicodedata
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Text, select, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

BRAND = os.getenv('BRAND_NAME', 'راجح | رؤية الأسواق').replace('\u0631\u0627\u0634\u062f', 'راجح')
WELCOME = f'''أهلًا بك في {BRAND} 📈

يسعدنا اهتمامك بتجربة توصيات الأسهم.

للتسجيل، اختر السوق الذي تتابعه من الأزرار بالأسفل.

بعد اختيارك، سنطلب اسمك ورقم هاتفك المسجّل على واتساب، ثم يتواصل معك أحد ممثلي خدمة العملاء عبر تليجرام أو الهاتف/واتساب لتفعيل التجربة المجانية.

أي سوق يهمّك؟'''
MARKETS = {'sa':'الأسهم السعودية 🇸🇦','us':'الأسهم الأمريكية 🇺🇸','both':'السعودية والأمريكية 🌍'}
STATUSES = {'new':'جديد','contacted':'تم التواصل','trial':'التجربة مفعّلة','subscribed':'مشترك','uninterested':'غير مهتم'}
CONSENT = 'أوافق على استخدام اسمي ورقمي وبيانات تليجرام للتواصل معي عبر تليجرام أو الهاتف أو واتساب بخصوص التوصيات وتفعيل التجربة المجانية.'
Base = declarative_base()
def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
class Lead(Base):
    __tablename__='leads'
    id=Column(Integer,primary_key=True)
    telegram_id=Column(BigInteger,unique=True,nullable=False)
    username=Column(String(100),default='')
    telegram_name=Column(String(200),default='')
    name=Column(String(150),default='')
    phone=Column(String(25),default='',index=True)
    market=Column(String(10),default='')
    draft_name=Column(String(150),default='')
    draft_phone=Column(String(25),default='')
    draft_market=Column(String(10),default='')
    step=Column(String(20),default='market')
    source=Column(String(100),default='direct',index=True)
    status=Column(String(20),default='new',index=True)
    owner=Column(String(100),default='')
    follow_up=Column(String(20),default='')
    notes=Column(Text,default='')
    consent_at=Column(String(40),default='')
    consent_text=Column(Text,default='')
    completed_at=Column(String(40),default='',index=True)
    created_at=Column(String(40),default=now,index=True)
    updated_at=Column(String(40),default=now)
    last_message_at=Column(String(40),default='')
    telegram_status=Column(String(20),default='unknown')
class Activity(Base):
    __tablename__='activities'
    id=Column(Integer,primary_key=True)
    lead_id=Column(Integer,index=True,nullable=False)
    kind=Column(String(30),nullable=False)
    body=Column(Text,default='')
    created_at=Column(String(40),default=now)
class Outbox(Base):
    __tablename__='outbox'
    id=Column(Integer,primary_key=True)
    key=Column(String(150),unique=True,nullable=False)
    chat_id=Column(BigInteger,nullable=False)
    method=Column(String(40),default='sendMessage')
    payload=Column(Text,nullable=False)
    status=Column(String(20),default='pending',index=True)
    attempts=Column(Integer,default=0)
    next_at=Column(BigInteger,default=0)
    error=Column(String(120),default='')
class Setting(Base):
    __tablename__='settings'
    key=Column(String(80),primary_key=True)
    value=Column(Text,default='')
class Processed(Base):
    __tablename__='processed_updates'
    id=Column(BigInteger,primary_key=True)

def database(url=None):
    from monitor import models, customer, customer_table  # Additive tables; existing leads remain intact.
    url=url or os.getenv('DATABASE_URL','')
    if not url and os.getenv('RAILWAY_PROJECT_ID'):
        raise RuntimeError('DATABASE_URL is required on Railway. Add a reference to the PostgreSQL service.')
    url=url or 'sqlite:///leads.db'
    if os.getenv('RAILWAY_PROJECT_ID') and url.startswith('sqlite'):
        raise RuntimeError('Use PostgreSQL DATABASE_URL for shared Railway services.')
    if url.startswith('postgres://'): url=url.replace('postgres://','postgresql+psycopg://',1)
    elif url.startswith('postgresql://'): url=url.replace('postgresql://','postgresql+psycopg://',1)
    engine=create_engine(url,pool_pre_ping=True,connect_args={'check_same_thread':False,'timeout':30} if url.startswith('sqlite') else {})
    Base.metadata.create_all(engine)
    return sessionmaker(engine,expire_on_commit=False)

def normalize_phone(value):
    value=''.join(str(unicodedata.digit(c)) if c.isdigit() else c for c in str(value))
    if re.search(r'[^0-9+\s()\-]',value): return None
    value=re.sub(r'[\s()\-]','',value)
    if value.startswith('00'): value='+'+value[2:]
    if not value.startswith('+'): return None
    return value if re.fullmatch(r'\+[1-9][0-9]{7,14}',value) else None

def record(s,l,kind,body): s.add(Activity(lead_id=l.id,kind=kind,body=body))
def queue(s,key,chat,text,markup=None):
    if s.scalar(select(Outbox).where(Outbox.key==key)): return
    payload={'chat_id':chat,'text':text}
    if markup: payload['reply_markup']=markup
    s.add(Outbox(key=key,chat_id=chat,payload=json.dumps(payload,ensure_ascii=False)))
def choices():
    return {'inline_keyboard':[[{'text':v,'callback_data':'market:'+k}] for k,v in MARKETS.items()]}
def menu():
    from monitor.customer import buttons
    kb=buttons();kb['inline_keyboard'].append([{'text':'تعديل بياناتي','callback_data':'edit'}]);return kb
def summary(l):
    return f'راجع بياناتك قبل تأكيد التسجيل:\n\nالاسم: {l.draft_name}\nرقم واتساب: {l.draft_phone}\nالسوق: {MARKETS[l.draft_market]}\n\n{CONSENT}'

def handle_update(s,u):
    uid=u['update_id']
    if s.get(Processed,uid): return
    cb=u.get('callback_query'); m=(cb or {}).get('message') or u.get('message')
    if not m or m.get('chat',{}).get('type')!='private':
        s.add(Processed(id=uid)); return
    who=(cb or {}).get('from') or m.get('from',{})
    if who.get('is_bot'): s.add(Processed(id=uid)); return
    tid=who['id']; text=m.get('text','').strip() if not cb else ''; data=cb.get('data','') if cb else ''
    if text=='/monitor':
        ids={v.strip() for v in os.getenv('ADMIN_TELEGRAM_IDS','').split(',') if v.strip()}
        if str(tid) in ids:
            from monitor.models import Plan, OPEN, LABELS
            rows=s.scalars(select(Plan).where(Plan.state.in_(OPEN)).order_by(Plan.score.desc()).limit(10)).all()
            lines=['راجح | متابعة الاستراتيجية التجريبية', 'بيانات شموع مكتملة قد تتأخر؛ لا يوجد تنفيذ شراء.']
            for p in rows:
                lines.append(f'{p.symbol} | {LABELS[p.state]} | {p.score:g}/100\nدخول {p.entry:g} · هدف {p.target:g} · وقف {p.stop:g}')
            if not rows: lines.append('لا توجد خطط مفتوحة حاليًا. راجع صفحة /stocks في الباك إند وحالة عامل الفحص.')
            queue(s,f'monitor-command:{uid}',tid,'\n\n'.join(lines))
        else:
            queue(s,f'monitor-command:{uid}',tid,'خدمة متابعة الأسهم قيد التجهيز. سيتواصل معك فريق خدمة العملاء عند إتاحتها.')
        s.add(Processed(id=uid)); return
    if text=='/id':
        queue(s,f'id:{uid}',tid,f'Telegram ID: {tid}')
        s.add(Processed(id=uid)); return
    l=s.scalar(select(Lead).where(Lead.telegram_id==tid))
    if not l:
        l=Lead(telegram_id=tid); s.add(l); s.flush()
    l.username=who.get('username',''); l.telegram_name=' '.join(filter(None,[who.get('first_name'),who.get('last_name')]))
    l.updated_at=now(); l.telegram_status='active'
    counter=0
    def send(t,markup=None):
        nonlocal counter
        counter+=1; queue(s,f'update:{uid}:{counter}',tid,t,markup)
    if cb:
        s.add(Outbox(key=f'ack:{uid}',chat_id=tid,method='answerCallbackQuery',payload=json.dumps({'callback_query_id':cb['id']})))
    if text in ('/stop_us','/resume_us'):
        from monitor.customer import Preference
        pref=s.get(Preference,tid)
        if not pref:pref=Preference(telegram_id=tid);s.add(pref)
        pref.paused=int(text=='/stop_us')
        send('تم إيقاف تنبيهات الأمريكي.' if pref.paused else 'تم استئناف تنبيهات الأمريكي إذا كانت التجربة أو الاشتراك مفعّلة.',menu())
    elif data.startswith('us:') or text.split('@')[0] in ('/current','/results','/menu'):
        from monitor.customer import view
        command=text.split('@')[0]
        if command=='/menu':send('قائمة راجح:',menu())
        else:
            parts=data.split(':');kind=parts[1] if len(parts)>1 else ('current' if command=='/current' else 'results')
            page=min(100000,int(parts[2])) if len(parts)>2 and parts[2].isdigit() else 0
            if kind=='results':body,kb=view(s,l,'results',page)
            else:
                from monitor.customer_table import table_entry
                body,kb=table_entry(s,l)
            send(body,kb)
    elif text.startswith('/start') or data=='edit':
        parts=text.split(maxsplit=1)
        if not l.completed_at and l.source=='direct' and len(parts)==2 and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',parts[1]): l.source=parts[1]
        if data=='edit' or not l.completed_at:
            l.step='market'
            if data=='edit':
                send('اختر السوق الذي ترغب في متابعة توصياته:',choices())
            else:
                counter+=1
                payload={'chat_id':tid,'caption':WELCOME,'reply_markup':choices(),'_welcome_photo':True}
                s.add(Outbox(key=f'update:{uid}:{counter}',chat_id=tid,method='sendPhoto',payload=json.dumps(payload,ensure_ascii=False)))
        else: send(f'أهلًا {l.name}، طلبك مسجل بالفعل ✅\nسيتواصل معك أحد ممثلي خدمة العملاء لتفعيل التجربة المجانية.',menu())
    elif text=='/cancel' or data=='cancel':
        l.step='done' if l.completed_at else 'cancelled'
        send('تم إلغاء تعديل البيانات.' if l.completed_at else 'تم إيقاف التسجيل. يمكنك البدء مجددًا باستخدام /start',{'remove_keyboard':True})
    elif data.startswith('market:') and l.step=='market' and data[7:] in MARKETS:
        l.draft_market=data[7:]; l.step='name'
        send('اختيار ممتاز ✅\nمن فضلك اكتب اسمك الذي ترغب أن نناديك به.',{'remove_keyboard':True})
    elif l.step=='name' and text and not text.startswith('/'):
        if len(text)<2 or len(text)>100 or any(unicodedata.category(c).startswith('C') for c in text): send('اكتب اسمًا صحيحًا من 2 إلى 100 حرف.')
        else:
            l.draft_name=text; l.step='phone'
            send('من فضلك أرسل رقم هاتفك المسجل على واتساب مع رمز الدولة.\nمثال: +9665XXXXXXXX أو +201XXXXXXXXX\n\nيمكنك كتابة رقم واتساب مختلف عن رقم تليجرام، أو مشاركة رقمك بالزر:',{'keyboard':[[{'text':'📱 مشاركة رقم هاتفي','request_contact':True}]],'resize_keyboard':True,'one_time_keyboard':True})
    elif l.step=='phone' and not cb:
        contact=m.get('contact')
        if contact and contact.get('user_id')!=tid: send('من فضلك أرسل رقمك أنت، أو اكتبه مع رمز الدولة.')
        else:
            raw=contact.get('phone_number','') if contact else text
            if contact and not raw.startswith('+'): raw='+'+raw
            phone=normalize_phone(raw)
            if not phone: send('الرقم غير صحيح. اكتبه مع رمز الدولة مثل +966 أو +20.')
            else:
                l.draft_phone=phone; l.step='confirm'
                send('شكرًا، تم إدخال رقمك.',{'remove_keyboard':True})
                send(summary(l),{'inline_keyboard':[[{'text':'✅ أوافق وأؤكد التسجيل','callback_data':'confirm'}],[{'text':'✏️ تعديل البيانات','callback_data':'edit'},{'text':'إلغاء','callback_data':'cancel'}]]})
    elif data=='confirm' and l.step=='confirm':
        first=not l.completed_at
        l.name=l.draft_name; l.phone=l.draft_phone; l.market=l.draft_market
        l.consent_at=now(); l.consent_text=CONSENT; l.step='done'
        if first: l.completed_at=now()
        record(s,l,'registration' if first else 'update','تم تأكيد بيانات التسجيل والموافقة على التواصل')
        send(f'شكرًا لك يا {l.name} 🌟\n\nتم تسجيل طلبك بنجاح في {BRAND}.\nسيتواصل معك أحد ممثلي خدمة العملاء عبر تليجرام أو الهاتف/واتساب لتفعيل التجربة المجانية.\n\nيسعدنا انضمامك!',menu())
        for admin in filter(None,os.getenv('ADMIN_TELEGRAM_IDS','').split(',')):
            admin=int(admin.strip())
            queue(s,f'admin:{uid}:{admin}',admin,f'🆕 {"عميل جديد" if first else "تحديث بيانات"}\nالاسم: {l.name}\nالسوق: {MARKETS[l.market]}\nواتساب: {l.phone}\nTelegram ID: {tid}\nالمصدر: {l.source}')
    elif l.step=='done': send('طلبك مسجل بالفعل ✅ يمكنك تعديل بياناتك من الزر.',menu())
    elif l.step=='confirm': send(summary(l),{'inline_keyboard':[[{'text':'✅ أوافق وأؤكد التسجيل','callback_data':'confirm'}],[{'text':'تعديل البيانات','callback_data':'edit'}]]})
    elif l.step=='market': send('اختر السوق من الأزرار التالية:',choices())
    else: send('للبدء أرسل /start، ولإلغاء التسجيل أرسل /cancel.')
    s.add(Processed(id=uid))
