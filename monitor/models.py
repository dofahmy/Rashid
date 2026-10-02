from sqlalchemy import Column, Integer, BigInteger, String, Text, Float, UniqueConstraint
from core import Base, now

class Stock(Base):
    __tablename__ = 'market_stocks'
    symbol = Column(String(40), primary_key=True)
    feed_symbol = Column(String(40), nullable=False)
    market = Column(String(2), nullable=False, index=True)
    company = Column(String(250), nullable=False)
    metadata_json = Column(Text, nullable=False)
    sharia_label = Column(String(150), default='غير معلوم')
    last_bar = Column(BigInteger, default=0)
    last_price = Column(Float)
    checked_at = Column(String(40), default='')
    error = Column(String(120), default='')
    evaluation_json = Column(Text, default='{}')

class Candle(Base):
    __tablename__ = 'market_candles_15m'
    symbol = Column(String(40), primary_key=True)
    ts = Column(BigInteger, primary_key=True)
    o = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    l = Column(Float, nullable=False)
    c = Column(Float, nullable=False)
    v = Column(Float, nullable=False)

class Plan(Base):
    __tablename__ = 'market_plans'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(40), nullable=False, index=True)
    market = Column(String(2), nullable=False, index=True)
    strategy_version = Column(String(80), nullable=False)
    state = Column(String(30), nullable=False, index=True, default='WAITING')
    signal_ts = Column(BigInteger, nullable=False)
    last_bar = Column(BigInteger, nullable=False)
    entry = Column(Float, nullable=False)
    stop = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    atr = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    waiting_bars = Column(Integer, default=0)
    retest_bars = Column(Integer, default=0)
    trigger_ts = Column(BigInteger)
    activation_ts = Column(BigInteger)
    paper_entry = Column(Float)
    exit_price = Column(Float)
    context_json = Column(Text, nullable=False)
    policy_json = Column(Text, nullable=False)
    created_at = Column(String(40), default=now)
    updated_at = Column(String(40), default=now)
    __table_args__ = (UniqueConstraint('symbol','strategy_version','signal_ts'),)

class Event(Base):
    __tablename__ = 'market_events'
    id = Column(Integer, primary_key=True)
    key = Column(String(150), unique=True, nullable=False)
    plan_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(40), nullable=False)
    kind = Column(String(40), nullable=False)
    bar_ts = Column(BigInteger, nullable=False)
    details_json = Column(Text, default='{}')
    created_at = Column(String(40), default=now)

class Scan(Base):
    __tablename__ = 'market_scans'
    id = Column(Integer, primary_key=True)
    market = Column(String(2), nullable=False)
    boundary = Column(BigInteger, nullable=False)
    status = Column(String(30), default='running')
    expected_bar = Column(BigInteger)
    total = Column(Integer, default=0)
    ok = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    new_plans = Column(Integer, default=0)
    transitions = Column(Integer, default=0)
    started_at = Column(String(40), default=now)
    finished_at = Column(String(40), default='')
    summary_json = Column(Text, default='{}')

LABELS = {'WAITING':'أمر شراء معلّق','RETEST':'أمر قديم بانتظار التحويل','ACTIVE':'متفعّل افتراضيًا',
          'TIME_EXIT':'خروج بعد مدة الانتظار','TARGET':'وصل للهدف','STOPPED':'وقف خسارة','CANCELLED':'أُلغيت الخطة',
          'EXPIRED':'انتهت مهلة الخطة','MISSED':'تجاوز الهدف قبل الدخول','DATA_GAP':'فجوة بيانات — يحتاج مراجعة'}
OPEN = ('WAITING','RETEST','ACTIVE')
