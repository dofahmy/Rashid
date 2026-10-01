"""Per-customer concurrent slots and global aged-position exit settings."""
import json, math, time
from sqlalchemy import select
from core import Setting, Outbox
from .models import Plan

LIMIT_FIELDS=('total','under_one','one_to_100','over_100')
HOLD_DAYS_KEY='us_hold_days'
HOLD_PROFIT_KEY='us_hold_min_profit_pct'

def limits(s,tid):
    row=s.get(Setting,f'us_limits:{tid}')
    if row:
        try:
            data=json.loads(row.value)
            if all(type(data[k]) is int and 0<=data[k]<=10000 for k in LIMIT_FIELDS):return data
        except (ValueError,KeyError,TypeError):pass
    return dict.fromkeys(LIMIT_FIELDS,0)

def save_limits(s,tid,data):
    if set(data)!=set(LIMIT_FIELDS) or any(type(v) is not int or not 0<=v<=10000 for v in data.values()):
        raise ValueError('Invalid limits')
    if sum(data[k] for k in LIMIT_FIELDS[1:])!=data['total']:
        raise ValueError('Category slots must sum to total')
    key=f'us_limits:{tid}';row=s.get(Setting,key)
    if row is None:row=Setting(key=key);s.add(row)
    row.value=json.dumps(data)

def bucket(price):
    if price is None or not math.isfinite(price) or price<=0:return None
    return 'under_one' if price<1 else 'one_to_100' if price<100 else 'over_100'

def occupied(s,tid,clock=None,before_id=None):
    """Delivered/uncertain alerts reserve slots; fresh queued entries also reserve them.
    During delivery only older pending messages precede this message, so lowering
    limits cancels excess queued messages without dropping already delivered ones.
    """
    from .customer import Recipient
    clock=int(time.time() if clock is None else clock)
    query=(select(Plan,Outbox).join(Recipient,Recipient.plan_id==Plan.id)
        .join(Outbox,Outbox.key==Recipient.entry_key)
        .where(Recipient.telegram_id==tid,Plan.market=='US',Plan.paper_entry.is_not(None),
               Plan.state.in_(('ACTIVE','DATA_GAP'))))
    counts=dict.fromkeys(LIMIT_FIELDS,0)
    for plan,row in s.execute(query):
        reserved=row.status in ('sent','uncertain','sending')
        if row.status=='pending' and (before_id is None or row.id<before_id):
            try:
                payload=json.loads(row.payload)
                reserved=clock<=int(payload['_us_expires'])
            except (ValueError,TypeError,KeyError):reserved=False
        if reserved:
            counts['total']+=1
            category=bucket(plan.paper_entry)
            if category:counts[category]+=1
    return counts

def available(s,tid,price,clock=None,before_id=None):
    cap=limits(s,tid);used=occupied(s,tid,clock,before_id);category=bucket(price)
    return bool(category and used['total']<cap['total'] and used[category]<cap[category])

def delivered_ids(tid):
    from .customer import Recipient
    return (select(Recipient.plan_id).join(Outbox,Outbox.key==Recipient.entry_key)
            .where(Recipient.telegram_id==tid,Outbox.status=='sent'))

def holding_settings(s):
    def number(key,default,low,high):
        row=s.get(Setting,key)
        try:value=float(row.value) if row else default
        except (ValueError,TypeError):return default
        return value if math.isfinite(value) and low<=value<=high else default
    return number(HOLD_DAYS_KEY,3,1,365),number(HOLD_PROFIT_KEY,.5,0,100)
