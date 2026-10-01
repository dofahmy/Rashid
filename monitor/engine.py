"""Closed-bar state transitions. All executions are paper observations."""
import json, os, math
from sqlalchemy import select, or_
from core import queue, now, Setting
from .models import Plan, Event, OPEN, LABELS
from .strategy import local, rounded

VERSION = 'sahm_m15_monitor_v1'
DEFAULT_POLICY = dict(volume_ratio=1.1, retest_max_bars=3, retest_atr_band=.25,
                      waiting_max_bars=20, minimum_rr=1.5)

def policy():
    result = dict(DEFAULT_POLICY)
    path = os.getenv('MONITOR_POLICY_FILE','')
    if path:
        with open(path,encoding='utf-8') as stream: overrides=json.load(stream)
        if set(overrides)-set(result): raise ValueError('Unknown policy key')
        result.update(overrides)
    for key in ('retest_max_bars','waiting_max_bars'):
        if type(result[key]) is not int or result[key]<1: raise ValueError('Invalid bar count')
    for key in ('volume_ratio','retest_atr_band','minimum_rr'):
        if not isinstance(result[key],(int,float)) or not math.isfinite(result[key]) or result[key]<=0:
            raise ValueError('Invalid policy threshold')
    return result

def event(s,p,kind,ts,details=None):
    key=f'market:{p.id}:{kind}:{int(ts)}'
    if s.scalar(select(Event.id).where(Event.key==key)): return
    s.add(Event(key=key,plan_id=p.id,symbol=p.symbol,kind=kind,bar_ts=int(ts),
                details_json=json.dumps(details or {},ensure_ascii=False)))
    from .customer import notice
    notice(s,p,kind,int(ts))
    # Optional administrator alerts in addition to eligible customer delivery.
    ids=os.getenv('MONITOR_ADMIN_ALERT_IDS','')
    if not ids: return
    text=(f'راجح | متابعة تجريبية 📈\n{p.symbol} — {LABELS.get(kind,kind)}\n'
          f'الدخول المرجعي: {p.entry:g}\nالهدف: {p.target:g}\nالوقف: {p.stop:g}\n'
          f'التقييم: {p.score:g}/100\nشمعة: {local(ts,p.market).isoformat()}\n'
          'رصد شموع مكتملة؛ لا يمثل تنفيذ شراء. قد تتأخر بيانات المصدر.')
    for value in ids.split(','):
        chat=int(value.strip())
        queue(s,key+f':{chat}',chat,text)

def new_plan(s,stock,result,ts,settings):
    if stock.market!='US' or not result.get('conditional_plan'): return None
    from .reset import RESET_KEY
    reset=s.get(Setting,RESET_KEY)
    if reset and int(ts)+900<=int(reset.value):return None
    if s.scalar(select(Plan.id).where(Plan.symbol==stock.symbol,or_(Plan.state.in_(OPEN),(Plan.state=='DATA_GAP') & (Plan.paper_entry.is_not(None))))): return None
    if s.scalar(select(Plan.id).where(Plan.symbol==stock.symbol,Plan.strategy_version==VERSION,Plan.signal_ts==int(ts))): return None
    p=Plan(symbol=stock.symbol,market=stock.market,strategy_version=VERSION,signal_ts=int(ts),last_bar=int(ts),
           state='WAITING',entry=result['entry_reference'],stop=result['stop_reference'],
           target=result['selected_target_price'],atr=result['atr14'],score=result['technical_score_100'],
           waiting_bars=0,retest_bars=0,context_json=json.dumps(result,ensure_ascii=False),
           policy_json=json.dumps(settings,sort_keys=True))
    s.add(p);s.flush();event(s,p,'WAITING',ts,{'bootstrap_or_new':True});return p

def advance(s,p,bar,ratio,contiguous=True):
    ts,o,h,l,c,v=bar
    if int(ts)<=p.last_bar or p.state not in OPEN: return False
    p.last_bar=int(ts);p.updated_at=now();rules=json.loads(p.policy_json);before=p.state
    details={}
    if not contiguous:
        p.state='DATA_GAP';details['reason']='Missing market bars; no inferred activation or fill'
    elif p.state=='ACTIVE':
        # OHLC cannot reveal order of touches: record conservative stop first.
        if l<=p.stop:
            p.state='STOPPED';p.exit_price=min(o,p.stop)
            details={'both_target_and_stop':h>=p.target,'paper_exit':p.exit_price}
        elif h>=p.target:
            p.state='TARGET';p.exit_price=p.target;details={'paper_exit':p.exit_price}
        else:
            from .limits import holding_settings
            days,min_profit=holding_settings(s)
            # Calendar time from activation close; evaluate only a new complete,
            # contiguous candle. Stop and target always take precedence.
            if (p.activation_ts is not None and p.paper_entry and
                ts-p.activation_ts>=days*86400 and
                (c/p.paper_entry-1)*100+1e-9>=min_profit):
                p.state='TIME_EXIT';p.exit_price=c
                details={'paper_exit':c,'hold_days':days,'min_profit_pct':min_profit}
    elif l<=p.stop:
        p.state='CANCELLED';details={'reason':'Stop breached before activation'}
    elif p.state=='WAITING':
        p.waiting_bars+=1
        if h>=p.target:
            p.state='MISSED';details={'reason':'Target touched before paper entry'}
        elif c>=p.entry and ratio is not None and ratio>=rules['volume_ratio']:
            p.state='RETEST';p.trigger_ts=int(ts);p.retest_bars=0
            details={'volume_ratio':ratio}
        elif p.waiting_bars>=rules['waiting_max_bars']:
            p.state='EXPIRED'
    elif p.state=='RETEST':
        p.retest_bars+=1
        band=rules['retest_atr_band']*p.atr
        if h>=p.target:
            p.state='MISSED'
        elif c<p.entry-band:
            p.state='CANCELLED';details={'reason':'Retest closed below tolerance'}
        elif p.entry-band<=l<=p.entry+band and c>=p.entry:
            fill,_=rounded(max(c,p.entry),p.market,True)
            rr=(p.target-fill)/(fill-p.stop) if fill>p.stop else 0
            if rr>=rules['minimum_rr']:
                p.state='ACTIVE';p.paper_entry=fill;p.activation_ts=int(ts)
                details={'paper_entry':fill,'rr_after_retest':rr,'execution':'paper_close_only'}
            else:
                p.state='CANCELLED';details={'reason':'Entry after retest no longer meets minimum RR','rr':rr}
        elif p.retest_bars>=rules['retest_max_bars']:
            p.state='EXPIRED'
    if p.state!=before:
        event(s,p,p.state,ts,details);return True
    return False
