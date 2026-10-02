"""Closed-bar state transitions. All executions are paper observations."""
import json, os, math
from sqlalchemy import select, or_
from core import queue, now, Setting
from .models import Plan, Event, OPEN, LABELS
from .strategy import local, rounded, COMMODITY_MARKETS

VERSION = 'sahm_m15_monitor_v1'
DEFAULT_POLICY = dict(volume_ratio=1.1, retest_max_bars=3, retest_atr_band=.25,
                      waiting_max_bars=20, minimum_rr=1.5, market_entry_max_gap_pct=.5,
                      second_stop_atr=1.2, second_stop_waiting_max_bars=78)

def policy():
    result = dict(DEFAULT_POLICY)
    path = os.getenv('MONITOR_POLICY_FILE','')
    if path:
        with open(path,encoding='utf-8') as stream: overrides=json.load(stream)
        if set(overrides)-set(result): raise ValueError('Unknown policy key')
        result.update(overrides)
    for key in ('retest_max_bars','waiting_max_bars','second_stop_waiting_max_bars'):
        if type(result[key]) is not int or result[key]<1: raise ValueError('Invalid bar count')
    for key in ('volume_ratio','retest_atr_band','minimum_rr','market_entry_max_gap_pct','second_stop_atr'):
        if not isinstance(result[key],(int,float)) or not math.isfinite(result[key]) or result[key]<=0:
            raise ValueError('Invalid policy threshold')
    return result



def _context(p):
    try:return json.loads(p.context_json or '{}')
    except Exception:return {}

def _save_context(p,ctx):
    p.context_json=json.dumps(ctx,ensure_ascii=False)

def _second_stop_system(p):
    return p.market=='US' and _context(p).get('entry_system')=='SECOND_STOP_RECOVERY'


def classify_order(entry,reference_price,settings):
    """Choose a real order style from the planned entry vs latest closed price.

    MARKET: planned entry is within the configured tolerance of the latest close.
    LIMIT: planned buy is materially below the latest close.
    STOP: planned buy is materially above the latest close (breakout order).
    """
    if not reference_price or reference_price<=0:return 'STOP',None
    gap_pct=100*(entry/reference_price-1)
    if abs(gap_pct)<=float(settings.get('market_entry_max_gap_pct',.5)):return 'MARKET',gap_pct
    return ('LIMIT' if entry<reference_price else 'STOP'),gap_pct

def order_type(p):
    ctx=_context(p)
    if _second_stop_system(p):
        if ctx.get('order_type')!='LIMIT':
            ctx['order_type']='LIMIT';_save_context(p,ctx)
        return 'LIMIT'
    kind=ctx.get('order_type')
    if kind in ('MARKET','LIMIT','STOP'):return kind
    ref=ctx.get('signal_bar_close') or ctx.get('feed_last_price')
    kind,gap=classify_order(p.entry,ref,json.loads(p.policy_json or '{}'))
    ctx['order_type']=kind;ctx['order_gap_pct']=gap;ctx['order_reference_price']=ref
    _save_context(p,ctx);return kind

def _activate_fill(s,p,fill,ts,kind,details=None):
    rules=json.loads(p.policy_json);fill,_=rounded(fill,p.market,True)
    if fill<=p.stop or fill>=p.target:return False,'invalid_fill_level'
    rr=(p.target-fill)/(fill-p.stop)
    # US second-stop recovery is intentionally a roughly 1:1 setup.
    if not _second_stop_system(p) and rr+1e-9<rules['minimum_rr']:
        return False,'rr_below_minimum_after_fill'
    p.state='ACTIVE';p.paper_entry=fill;p.activation_ts=int(ts)
    info={'paper_entry':fill,'rr_after_fill':rr,'order_type':kind,'execution':'paper_order_simulation'}
    if details:info.update(details)
    event(s,p,'ACTIVE',ts,info);return True,None

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
    if stock.market!='US' and stock.market not in COMMODITY_MARKETS or not result.get('conditional_plan'): return None
    # Development reset clears old plans and stock watermarks.  The worker then
    # re-evaluates the latest complete candle.  Do not reject that candle merely
    # because it closed before the reset button was pressed (for example, a
    # startup diagnostic scan after market close).  Duplicate/open-plan guards
    # below still prevent creating the same signal more than once.
    if s.scalar(select(Plan.id).where(Plan.symbol==stock.symbol,or_(Plan.state.in_(OPEN),(Plan.state=='DATA_GAP') & (Plan.paper_entry.is_not(None))))): return None
    if s.scalar(select(Plan.id).where(Plan.symbol==stock.symbol,Plan.strategy_version==VERSION,Plan.signal_ts==int(ts))): return None
    ctx=dict(result)
    ref=float(result.get('signal_bar_close') or stock.last_price or result['entry_reference'])
    if stock.market=='US':
        atr=float(result['atr14']);mult=float(settings['second_stop_atr'])
        original_entry=float(result['entry_reference'])
        first_stop=float(result['stop_reference'])
        original_target=float(result['selected_target_price'])
        second_stop,_=rounded(first_stop-mult*atr,stock.market,False)
        third_stop,_=rounded(second_stop-mult*atr,stock.market,False)
        if third_stop<=0 or not (third_stop<second_stop<first_stop):
            return None
        kind='LIMIT';gap=100*(second_stop/ref-1) if ref>0 else None
        ctx.update(
            entry_system='SECOND_STOP_RECOVERY',
            original_entry_reference=original_entry,
            original_first_stop=first_stop,
            original_selected_target=original_target,
            second_stop_atr_mult=mult,
            second_stop_entry=second_stop,
            third_stop_protective=third_stop,
            recovery_target_first_stop=first_stop,
            order_type='LIMIT',
            order_gap_pct=gap,
            order_reference_price=ref,
            order_rule='US recovery: BUY LIMIT at original stop - 1.2 ATR; target=original stop; protective stop=another 1.2 ATR lower',
        )
        plan_entry,plan_stop,plan_target=second_stop,third_stop,first_stop
    else:
        kind,gap=classify_order(result['entry_reference'],ref,settings)
        ctx.update(order_type=kind,order_gap_pct=gap,order_reference_price=ref,
                   order_rule='MARKET if abs(entry-reference)<=0.5%; LIMIT below reference; STOP above reference')
        plan_entry,plan_stop,plan_target=result['entry_reference'],result['stop_reference'],result['selected_target_price']
    p=Plan(symbol=stock.symbol,market=stock.market,strategy_version=VERSION,signal_ts=int(ts),last_bar=int(ts),
           state='WAITING',entry=plan_entry,stop=plan_stop,
           target=plan_target,atr=result['atr14'],score=result['technical_score_100'],
           waiting_bars=0,retest_bars=0,context_json=json.dumps(ctx,ensure_ascii=False),
           policy_json=json.dumps(settings,sort_keys=True))
    s.add(p);s.flush();event(s,p,'WAITING',ts,{'bootstrap_or_new':True,'order_type':kind,'order_gap_pct':gap,'reference_price':ref,'entry_system':ctx.get('entry_system')});return p

def advance(s,p,bar,ratio,contiguous=True):
    ts,o,h,l,c,v=bar
    if int(ts)<=p.last_bar or p.state not in OPEN:return False
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
            if (p.activation_ts is not None and p.paper_entry and
                ts-p.activation_ts>=days*86400 and
                (c/p.paper_entry-1)*100+1e-9>=min_profit):
                p.state='TIME_EXIT';p.exit_price=c
                details={'paper_exit':c,'hold_days':days,'min_profit_pct':min_profit}
    else:
        # New order model. Legacy RETEST plans are reclassified into an actual
        # pending order at their pinned entry; there is no extra retest gate.
        if p.state=='RETEST':p.state='WAITING'
        kind=order_type(p);p.waiting_bars+=1
        if kind=='MARKET':
            # A market order created from a completed 15m signal can only be
            # simulated at the first price of the next complete bar.
            fill=o
            if fill>=p.target:
                p.state='MISSED';details={'reason':'Market opened at/above target before fill','order_type':kind}
            elif fill<=p.stop:
                p.state='CANCELLED';details={'reason':'Market opened at/below stop before fill','order_type':kind}
            else:
                ok,reason=_activate_fill(s,p,fill,ts,kind,{'source_price':'next_bar_open'})
                if ok:return True
                p.state='CANCELLED';details={'reason':reason,'order_type':kind,'attempted_fill':fill}
        elif kind=='LIMIT':
            # Buy-limit waits for price to fall to the pinned entry. A gap below
            # the limit receives the better opening price in this paper model.
            if o<=p.stop:
                p.state='CANCELLED';details={'reason':'Gap opened at/below stop before safe limit fill','order_type':kind}
            elif l<=p.entry:
                fill=min(o,p.entry) if o<=p.entry else p.entry
                ok,reason=_activate_fill(s,p,fill,ts,kind,{'limit_price':p.entry,'gap_improvement':fill<p.entry})
                if ok:return True
                p.state='CANCELLED';details={'reason':reason,'order_type':kind,'attempted_fill':fill}
            elif not _second_stop_system(p) and h>=p.target:
                p.state='MISSED';details={'reason':'Target touched before limit order filled','order_type':kind}
        else: # STOP breakout order
            if l<=p.stop:
                p.state='CANCELLED';details={'reason':'Stop invalidated setup before buy-stop fill','order_type':kind}
            elif o>=p.target:
                p.state='MISSED';details={'reason':'Gap opened at/above target before buy-stop fill','order_type':kind}
            elif h>=p.entry:
                fill=max(o,p.entry)
                if fill>=p.target:
                    p.state='MISSED';details={'reason':'Buy-stop fill would be at/above target','order_type':kind,'attempted_fill':fill}
                else:
                    ok,reason=_activate_fill(s,p,fill,ts,kind,{'stop_price':p.entry,'gap_slippage':fill>p.entry})
                    if ok:return True
                    p.state='CANCELLED';details={'reason':reason,'order_type':kind,'attempted_fill':fill}
        wait_limit=rules.get('second_stop_waiting_max_bars',78) if _second_stop_system(p) else rules['waiting_max_bars']
        if p.state=='WAITING' and p.waiting_bars>=wait_limit:
            p.state='EXPIRED';details={'reason':'Pending order expired','order_type':kind,'waiting_limit_bars':wait_limit}
    if p.state!=before:
        event(s,p,p.state,ts,details);return True
    return False

