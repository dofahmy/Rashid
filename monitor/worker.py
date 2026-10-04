"""Railway worker: python -m monitor.worker [--init-only|--once]."""
import argparse, asyncio, contextlib, json, logging, os, time, math
import urllib.request, urllib.error
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import select, delete, func, case, text
from core import database, now
from .models import Stock, Candle, Plan, Scan, OPEN, EgxSignal, EgxOpenSignal
from .engine import new_plan, advance, policy
from .strategy import clean, local, CONFIG, COMMODITY_MARKETS, evaluate
from .provider import YahooProvider, TwelveDataCommodityProvider, FeedError

log=logging.getLogger('rajih.monitor')
DATA=Path(__file__).parent/'data'
LOCK_KEY=72617368696415

COMMODITY_NAMES={'XA':'gold/XAUUSD'}

# Egypt daily standalone R2+Slope scanner (backend only; no Telegram queueing).
EGX_R2_MIN=0.791694
EGX_SLOPE_MIN=67.5062
EGX_LOOKBACK=126
EGX_COOLDOWN=126
EGX_MAX_HOLD_SESSIONS=int(os.getenv('EGX_MAX_HOLD_SESSIONS','252'))
EGX_MIN_PRICE=1.0
EGX_MIN_ADV20=1_000_000.0
EGX_MIN_AVGVOL20=10_000.0
EGX_TZ='Africa/Cairo'
EGX_SCAN_HOUR=15  # after the normal EGX close; startup scan also runs once.

def _http_json(url,payload=None,headers=None,timeout=30):
    body=None
    hdr={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36'}
    if headers:hdr.update(headers)
    if payload is not None:
        body=json.dumps(payload).encode('utf-8')
        hdr['Content-Type']='application/json'
    req=urllib.request.Request(url,data=body,headers=hdr,method='POST' if body is not None else 'GET')
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def _egx_discover_sync():
    payload={
        'filter':[{'left':'type','operation':'equal','right':'stock'}],
        'options':{'lang':'en'},
        'markets':['egypt'],
        'symbols':{'query':{'types':[]},'tickers':[]},
        'columns':['name','description','exchange'],
        'sort':{'sortBy':'name','sortOrder':'asc'},
        'range':[0,1000],
    }
    obj=_http_json('https://scanner.tradingview.com/egypt/scan',payload)
    out=[];seen=set()
    for item in obj.get('data',[]):
        full=item.get('s','');symbol=full.split(':',1)[1] if ':' in full else full
        vals=item.get('d') or []
        company=(vals[1] if len(vals)>1 and vals[1] else (vals[0] if vals else symbol))
        if symbol and symbol not in seen:
            seen.add(symbol);out.append((symbol,str(company or symbol)))
    return out

def _egx_daily_sync(symbol):
    start=int(datetime(2019,1,1,tzinfo=ZoneInfo('UTC')).timestamp())
    end=int(time.time())+86400
    url=(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.CA'
         f'?period1={start}&period2={end}&interval=1d&events=div%2Csplits&includeAdjustedClose=true')
    obj=_http_json(url)
    res=((obj.get('chart') or {}).get('result') or [])
    if not res:return []
    z=res[0];ts=z.get('timestamp') or []
    q=(((z.get('indicators') or {}).get('quote') or [{}])[0])
    opens=q.get('open') or [];highs=q.get('high') or [];lows=q.get('low') or []
    closes=q.get('close') or [];vols=q.get('volume') or []
    n=min(len(ts),len(opens),len(highs),len(lows),len(closes),len(vols))
    raw=[]
    for i in range(n):
        vals=(opens[i],highs[i],lows[i],closes[i])
        if any(v is None for v in vals):continue
        try:o,h,l,c=map(float,vals);v=float(vols[i] or 0)
        except (TypeError,ValueError):continue
        if min(o,h,l,c)<=0:continue
        raw.append({
            'ts':int(ts[i]),'ro':o,'rh':h,'rl':l,'rc':c,'v':v,
            'date':datetime.fromtimestamp(int(ts[i]),ZoneInfo(EGX_TZ)).date().isoformat(),
        })
    raw.sort(key=lambda r:r['ts'])
    dedup={}
    for r in raw:dedup[r['date']]=r
    raw=list(dedup.values())
    if not raw:return []

    # Yahoo EGX history can keep pre-corporate-action nominal prices while
    # TradingView shows a continuous adjusted chart. Build a split/bonus-share
    # continuous price series ourselves by stitching only very large overnight
    # discontinuities. This keeps historical signal prices on today's scale.
    common_factors=(0.10,0.20,0.25,1/3,0.40,0.50,2/3,0.75,0.80,
                    1.25,4/3,1.50,2.0,2.5,3.0,4.0,5.0,10.0)
    scales=[1.0]*len(raw)
    cumulative=1.0
    for i in range(len(raw)-2,-1,-1):
        nxt=raw[i+1];cur=raw[i]
        gap=float(nxt['ro'])/float(cur['rc']) if cur['rc'] else 1.0
        # EGX ordinary daily gaps are much smaller; a >28% mechanical jump/drop
        # is treated as a corporate-action boundary for continuity purposes.
        if gap < 0.72 or gap > 1.38:
            nearest=min(common_factors,key=lambda f:abs(gap-f)/f)
            if abs(gap-nearest)/nearest <= 0.10:
                gap=nearest
            cumulative*=gap
        scales[i]=cumulative

    rows=[]
    for r,scale in zip(raw,scales):
        o=r['ro']*scale;h=r['rh']*scale;l=r['rl']*scale;c=r['rc']*scale
        rows.append({
            'ts':r['ts'],'date':r['date'],'v':r['v'],
            # Continuous split/bonus-adjusted OHLC used by signal logic/display.
            'o':o,'h':h,'l':l,'c':c,'ac':c,'ah':h,'al':l,
            # Preserve actual Yahoo nominal prices for turnover diagnostics.
            'raw_o':r['ro'],'raw_h':r['rh'],'raw_l':r['rl'],'raw_c':r['rc'],
            'corp_scale':scale,
        })
    return rows

def _egx_linreg(values,end_idx,window=EGX_LOOKBACK):
    # Previous window only; current bar excluded, exactly like the research script.
    if end_idx<window:return None,None
    y=values[end_idx-window:end_idx]
    if len(y)!=window or any(not math.isfinite(v) for v in y):return None,None
    xm=(window-1)/2.0;ym=sum(y)/window
    if ym==0:return None,None
    xss=sum((x-xm)**2 for x in range(window))
    b=sum((x-xm)*(y[x]-ym) for x in range(window))/xss
    sst=sum((v-ym)**2 for v in y)
    ssr=sum((y[x]-(ym+b*(x-xm)))**2 for x in range(window))
    r2=1-ssr/sst if sst>0 else 0.0
    slope_pct=100*(b*(window-1))/ym
    return slope_pct,r2

def _egx_price_confirm(rows, i):
    if i <= 0:
        return False
    r=rows[i]; prev=rows[i-1]
    return bool(
        r['c'] > prev['h']
        and r['c'] > r['o']
        and (r['c'] - r['o']) > (r['h'] - r['c'])
    )

def _egx_trade_exit(rows, metrics, signal_idx):
    """Return the first approved exit after an EGX entry, or None if still open.

    Approved exits, in chronological order:
    1) +50% target touched by adjusted daily high.
    2) Failure Exit during first 60 sessions:
       R2 <= entry R2 - 0.05, slope <= 45% of entry slope,
       and adjusted close below the prior 10-session adjusted low.
    3) Peak Exit after at least 10 sessions:
       while near the post-entry high, slope has fallen 57.5% from its
       post-entry peak and R2 has fallen 0.04 from its post-entry peak;
       the setup stays armed for 30 sessions and exits when adjusted close
       breaks below the prior 5-session adjusted low.
    4) Time Exit at EGX_MAX_HOLD_SESSIONS (default 252 sessions).

    Target is checked first on a day because a touched intraday target means
    the +50% objective was reached even if the same day's close later weakens.
    """
    entry=float(rows[signal_idx]['ac'])
    if entry<=0:
        return None
    entry_slope,entry_r2=metrics[signal_idx]
    if entry_slope is None or entry_r2 is None:
        return None

    peak_price=-math.inf
    peak_slope=-math.inf
    peak_r2=-math.inf
    peak_armed=False
    peak_armed_idx=None

    last=len(rows)-1
    fail_end=min(last,signal_idx+60)

    for j in range(signal_idx+1,last+1):
        held=j-signal_idx
        row=rows[j]
        s,r=metrics[j]
        ah=float(row['ah']);al=float(row['al']);ac=float(row['ac'])

        # 1) Fixed +50% target.
        if ah >= entry*1.50:
            return {'idx':j,'reason':'TARGET_50','date':row['date']}

        # 2) Failure Exit: fixed research rule, only first 60 sessions.
        if j<=fail_end and held>=5 and s is not None and r is not None:
            r2_fail = r <= entry_r2 - 0.05
            slope_fail = entry_slope>0 and s <= entry_slope*0.45
            if j>=10:
                prior10=min(float(x['al']) for x in rows[j-10:j])
                if ac < prior10 and r2_fail and slope_fail:
                    return {'idx':j,'reason':'FAILURE_EXIT','date':row['date']}

        # 3) Peak Exit: fixed research rule.
        if math.isfinite(ah):
            peak_price=max(peak_price,ah)
        if s is not None and math.isfinite(s):
            peak_slope=max(peak_slope,s)
        if r is not None and math.isfinite(r):
            peak_r2=max(peak_r2,r)

        if held<10:
            continue

        near_high = math.isfinite(peak_price) and ah >= 0.98*peak_price
        slope_roll = (
            s is not None and math.isfinite(s) and math.isfinite(peak_slope)
            and peak_slope>0 and s <= peak_slope*0.425
        )
        r2_roll = (
            r is not None and math.isfinite(r) and math.isfinite(peak_r2)
            and r <= peak_r2-0.04
        )

        if (not peak_armed) and near_high and slope_roll and r2_roll:
            peak_armed=True
            peak_armed_idx=j

        if peak_armed and j-peak_armed_idx>30:
            peak_armed=False
            peak_armed_idx=None

        if peak_armed and j>=5:
            prior5=min(float(x['al']) for x in rows[j-5:j])
            if ac < prior5:
                return {'idx':j,'reason':'PEAK_EXIT','date':row['date']}

        # 4) Operational time exit: do not keep stale positions open forever.
        # 252 sessions ~= one trading year and is configurable in Railway.
        if held >= EGX_MAX_HOLD_SESSIONS:
            return {'idx':j,'reason':'TIME_EXIT','date':row['date']}

    return None


def _egx_analyze_signals(rows):
    """Rebuild historical R2+Slope entries and return positions open *now*.

    A historical entry is not considered open merely because it missed +50%.
    It is removed once any approved exit has happened: +50% target, Failure
    Exit, or Peak Exit.  Only the latest still-open position per symbol is
    returned to the backend table.
    """
    if len(rows)<EGX_LOOKBACK+21:
        return {'open':[], 'latest_activation':None}

    ac=[r['ac'] for r in rows]
    rule=[];metrics=[]
    for i,r in enumerate(rows):
        slope,r2=_egx_linreg(ac,i)
        metrics.append((slope,r2))
        # Entry confirmation added to the standalone R2+Slope signal:
        # 1) today's corporate-action-adjusted close must break above yesterday's adjusted high;
        # 2) today's candle must be positive (close > open);
        # 3) the real body must be larger than the upper wick:
        #    (close - open) > (high - close).
        price_confirm = _egx_price_confirm(rows, i)
        rule.append(bool(
            slope is not None and r2 is not None
            and r2 >= EGX_R2_MIN and slope >= EGX_SLOPE_MIN
            and price_confirm
        ))

    new_rule=[rule[i] and (i==0 or not rule[i-1]) for i in range(len(rule))]
    last_kept=-10**9;kept=[]
    for i,is_new in enumerate(new_rule):
        if not is_new:continue
        if i-last_kept<EGX_COOLDOWN:continue
        r=rows[i]
        if r['c']<EGX_MIN_PRICE or i<21:continue
        prior=rows[i-20:i]
        adv20=sum(x.get('raw_c',x['c'])*x['v'] for x in prior)/20
        avgvol20=sum(x['v'] for x in prior)/20
        if adv20<EGX_MIN_ADV20 or avgvol20<EGX_MIN_AVGVOL20:continue
        last_kept=i
        slope,r2=metrics[i]
        kept.append((i,adv20,avgvol20,slope,r2))

    latest=rows[-1]
    open_candidates=[]
    for i,adv20,avgvol20,slope,r2 in kept:
        # Defensive revalidation: an OPEN row can never survive unless the
        # signal candle itself still satisfies every price-confirmation rule.
        if not _egx_price_confirm(rows, i):
            continue
        exit_info=_egx_trade_exit(rows,metrics,i)
        if exit_info is not None:
            continue

        sig=rows[i];entry_adj=float(sig['ac'])
        future=rows[i:]
        open_candidates.append({
            'signal_date':sig['date'],'signal_ts':sig['ts'],
            'signal_price':sig['c'],'signal_adj_price':entry_adj,
            'r2':r2,'slope':slope,'adv20':adv20,'avgvol20':avgvol20,
            'latest_date':latest['date'],'current_price':latest['c'],
            'current_return_pct':100*(float(latest['ac'])/entry_adj-1),
            'max_gain_pct':100*(max(float(x['ah']) for x in future)/entry_adj-1),
            'max_drawdown_pct':100*(min(float(x['al']) for x in future)/entry_adj-1),
            'age_sessions':len(rows)-1-i,
        })

    # One current position per symbol. If independent historical cooldown logic
    # produced more than one unresolved entry, show only the latest live one.
    open_rows=[max(open_candidates,key=lambda x:x['signal_ts'])] if open_candidates else []

    latest_activation=None
    if kept and kept[-1][0]==len(rows)-1 and _egx_price_confirm(rows, kept[-1][0]):
        i,adv20,avgvol20,slope,r2=kept[-1];r=rows[i]
        latest_activation={
            'symbol_date':r['date'],'signal_ts':r['ts'],'signal_price':r['c'],
            'r2':r2,'slope':slope,'adv20':adv20,'avgvol20':avgvol20,
        }
    return {'open':open_rows,'latest_activation':latest_activation}

async def scan_egx_daily(DB,clock):
    """Daily Egypt scan + historical OPEN backfill for the backend page."""
    boundary=int(clock)//86400*86400
    with DB.begin() as s:
        scan=Scan(market='EG',boundary=boundary,status='running');s.add(scan);s.flush();scan_id=scan.id
        # Clear stale rows immediately. The page stays empty while the fresh
        # backfill runs rather than showing positions calculated by older rules.
        s.execute(delete(EgxOpenSignal))
    log.info('EGX v7 strict rebuild started; old open rows cleared')
    try:
        symbols=await asyncio.to_thread(_egx_discover_sync)
        sem=asyncio.Semaphore(max(2,min(16,int(os.getenv('EGX_SCAN_CONCURRENCY','8')))))
        async def one(item):
            symbol,company=item
            async with sem:
                try:
                    rows=await asyncio.to_thread(_egx_daily_sync,symbol)
                    return symbol,company,_egx_analyze_signals(rows),None
                except Exception as exc:
                    return symbol,company,None,type(exc).__name__
        tasks=[asyncio.create_task(one(item)) for item in symbols]
        counts={'total':len(tasks),'ok':0,'errors':0,'new_plans':0,'transitions':0}
        open_total=0
        rebuilt_open_rows=[]
        latest_activations=[]
        for n,task in enumerate(asyncio.as_completed(tasks),1):
            symbol,company,analysis,error=await task
            if error:
                counts['errors']+=1
            else:
                counts['ok']+=1
                opens=analysis['open']
                open_total+=len(opens)
                for e in opens:
                    rebuilt_open_rows.append((symbol,company,e))
                signal=analysis['latest_activation']
                if signal:
                    latest_activations.append((symbol,company,signal))
            if n%25==0:log.info('EGX daily progress %s/%s open=%s',n,len(tasks),open_total)

        # Replace the OPEN table in one transaction. This also removes stale
        # rows left by older code or symbols whose previous record no longer
        # qualifies as a live position.
        with DB.begin() as s:
            s.execute(delete(EgxOpenSignal))
            for symbol,company,e in rebuilt_open_rows:
                s.add(EgxOpenSignal(
                    symbol=symbol,company=company,signal_date=e['signal_date'],
                    signal_ts=e['signal_ts'],signal_price=e['signal_price'],
                    signal_adj_price=e['signal_adj_price'],pre_trend_r2=e['r2'],
                    pre_trend_slope_pct=e['slope'],adv20=e['adv20'],avgvol20=e['avgvol20'],
                    latest_date=e['latest_date'],current_price=e['current_price'],
                    current_return_pct=e['current_return_pct'],max_gain_pct=e['max_gain_pct'],
                    max_drawdown_pct=e['max_drawdown_pct'],age_sessions=e['age_sessions'],updated_at=now()))
            for symbol,company,signal in latest_activations:
                exists=s.scalar(select(func.count()).select_from(EgxSignal).where(
                    EgxSignal.symbol==symbol,EgxSignal.signal_date==signal['symbol_date']))
                if not exists:
                    s.add(EgxSignal(
                        symbol=symbol,company=company,signal_date=signal['symbol_date'],
                        signal_ts=signal['signal_ts'],signal_price=signal['signal_price'],
                        pre_trend_r2=signal['r2'],pre_trend_slope_pct=signal['slope'],
                        adv20=signal['adv20'],avgvol20=signal['avgvol20']))
                    counts['new_plans']+=1
        with DB.begin() as s:
            scan=s.get(Scan,scan_id)
            for k,v in counts.items():setattr(scan,k,v)
            scan.status='partial' if counts['errors'] else 'complete'
            scan.finished_at=now()
            scan.summary_json=json.dumps({'scanner':'EGX_R2_SLOPE_DAILY_V7_STRICT_CANDLE','new_signals':counts['new_plans'],'open_signals':open_total})
        log.info('EGX daily scan %s open_signals=%s',counts,open_total)
    except Exception as exc:
        with DB.begin() as s:
            scan=s.get(Scan,scan_id)
            scan.status='waiting_feed';scan.finished_at=now()
            scan.summary_json=json.dumps({'scanner':'EGX_R2_SLOPE_DAILY_V7_STRICT_CANDLE','reason':type(exc).__name__})
        log.exception('EGX daily scanner failed')

def egx_due(clock):
    dt=datetime.fromtimestamp(clock,ZoneInfo(EGX_TZ))
    # Sunday-Thursday, after close. Worker keeps a once-per-date guard.
    return dt.weekday() in (6,0,1,2,3) and dt.hour>=EGX_SCAN_HOUR


def initialize(DB):
    EgxSignal.__table__.create(bind=DB.kw['bind'],checkfirst=True)
    EgxOpenSignal.__table__.create(bind=DB.kw['bind'],checkfirst=True)
    # Never expose OPEN rows produced by an older filter after a worker restart.
    with DB.begin() as s:
        s.execute(delete(EgxOpenSignal))
    log.info('EGX v7 startup purge complete; open table will be rebuilt')
    universe=json.loads((DATA/'universe.json').read_text(encoding='utf-8'))
    counts={m:sum(r['market_key']==m for r in universe) for m in ('SA','US')}
    if counts!={'SA':375,'US':5691} or len({r['symbol'] for r in universe})!=6066:
        raise RuntimeError('Universe count or unique symbol mismatch')
    with DB.begin() as s:
        existing={r.symbol:r for r in s.scalars(select(Stock))}
        for r in universe:
            if r['market_key']!='US':continue
            stock=existing.get(r['symbol'])
            if stock is None:
                stock=Stock(symbol=r['symbol'],last_bar=0);s.add(stock)
            stock.feed_symbol=r['feed_symbol'];stock.market=r['market_key']
            stock.company=r['name_ar'] or r['name'];stock.sharia_label=r['sharia_label']
            stock.metadata_json=json.dumps(r,ensure_ascii=False)
        # Spot commodities are separate single-instrument markets sourced from Twelve Data.
        commodities=[
            ('XAUUSD','XA','الذهب مقابل الدولار الأمريكي','Gold / US Dollar','disabled_for_spot_commodity_v2'),
        ]
        for symbol,market,name_ar,name,version in commodities:
            row=s.get(Stock,symbol)
            if row is None:
                row=Stock(symbol=symbol,last_bar=0);s.add(row)
            row.feed_symbol=symbol;row.market=market;row.company=name_ar;row.sharia_label='غير مطبق'
            meta={'symbol':symbol,'feed_symbol':symbol,'market_key':market,'name':name,'name_ar':name_ar,
                  'sharia_label':'غير مطبق','sharia_code':'NA','reported_price':None,'volume_rule':version}
            try: old_meta=json.loads(row.metadata_json or '{}')
            except Exception: old_meta={}
            if old_meta.get('volume_rule')!=version:
                row.last_bar=0;row.error='';row.checked_at=None
                log.info('%s spot commodity volume rule disabled; watermark reset for one full re-evaluation',market)
            row.metadata_json=json.dumps(meta,ensure_ascii=False)
    CONFIG['XA']['ref']='XAUUSD'
    return {'US':counts['US'],'XA':1}

@contextlib.contextmanager
def exclusive(DB,lock_key=LOCK_KEY):
    """One scanner across replicas; PostgreSQL session lock on a dedicated connection."""
    engine=DB.kw['bind']
    if engine.dialect.name=='postgresql':
        conn=engine.connect()
        try:
            locked=conn.scalar(text('SELECT pg_try_advisory_lock(:key)'),{'key':lock_key})
            conn.commit()
            yield bool(locked)
        finally:
            try:
                conn.execute(text('SELECT pg_advisory_unlock(:key)'),{'key':lock_key});conn.commit()
            finally: conn.close()
    else:
        # Local development only; production must use PostgreSQL, not shared SQLite.
        lock_path=Path(engine.url.database or 'monitor').absolute().with_suffix(f'.{lock_key}.lock')
        with lock_path.open('a+b') as stream:
            if os.name=='nt':
                import msvcrt
                stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
                acquire=lambda:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                def release():
                    stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                acquire=lambda:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                release=lambda:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
            try:acquire()
            except OSError:yield False;return
            try:yield True
            finally:release()

def due(m,clock):
    dt=local(clock,m);cfg=CONFIG[m];minute=dt.hour*60+dt.minute
    weekdays=(6,0,1,2,3) if m=='SA' else (0,1,2,3,4)
    return dt.weekday() in weekdays and cfg['start']+15<=minute<=cfg['end']+60

def expected_slot(m,clock):
    dt=local(clock,m);cfg=CONFIG[m]
    end_min=min(cfg['end'],(dt.hour*60+dt.minute)//15*15)
    start_min=end_min-15
    if start_min<cfg['start']:return None
    return int(datetime(dt.year,dt.month,dt.day,start_min//60,start_min%60,
                        tzinfo=ZoneInfo(cfg['tz'])).timestamp())

def to_raw(rows,meta):
    names=['open','high','low','close','volume']
    return {'chart':{'result':[{'meta':meta,'timestamp':[r[0] for r in rows],
           'indicators':{'quote':[{k:[r[i+1] for r in rows] for i,k in enumerate(names)}]}}]}}

def validate(raw,stock):
    meta=raw['chart']['result'][0]['meta']
    conditions=[(meta.get('symbol')==stock.feed_symbol,'symbol_mismatch'),
                (meta.get('dataGranularity')=='15m','wrong_granularity'),
                (meta.get('currency')==CONFIG[stock.market]['currency'],'currency_mismatch'),
                (meta.get('instrumentType')==('CURRENCY' if stock.market in COMMODITY_MARKETS else 'EQUITY'),'wrong_instrument_type')]
    for ok,reason in conditions:
        if not ok:raise FeedError(reason)
    return meta

def store_bars(s,symbol,bars):
    dialect=s.get_bind().dialect.name
    if dialect=='postgresql':from sqlalchemy.dialects.postgresql import insert
    else:from sqlalchemy.dialects.sqlite import insert
    for offset in range(0,len(bars),100):
        rows=[dict(symbol=symbol,ts=int(a[0]),o=a[1],h=a[2],l=a[3],c=a[4],v=a[5]) for a in bars[offset:offset+100]]
        if not rows:continue
        stmt=insert(Candle).values(rows)
        stmt=stmt.on_conflict_do_update(index_elements=['symbol','ts'],
             set_={k:getattr(stmt.excluded,k) for k in ('o','h','l','c','v')})
        s.execute(stmt)
    cutoff=s.scalar(select(Candle.ts).where(Candle.symbol==symbol).order_by(Candle.ts.desc()).offset(2200).limit(1))
    if cutoff is not None:s.execute(delete(Candle).where(Candle.symbol==symbol,Candle.ts<=cutoff))

def volume_ratio(bars,ts,market,dates):
    dt=local(ts,market)
    prior=set([day for day in dates if day<dt.date()][-20:])
    same=[b[5] for b in bars if local(b[0],market).date() in prior
          and local(b[0],market).strftime('%H:%M')==dt.strftime('%H:%M')]
    current=next((b[5] for b in bars if b[0]==ts),None)
    if len(same)!=20 or sum(same)<=0 or current is None:return None
    return current/(sum(same)/20)

def apply_stock(DB,symbol,raw,expected,reference_bars,settings,clock,strategy_diag=None):
    """One symbol, one transaction: bars, watermark, plans and events commit together."""
    with DB.begin() as s:
        stock=s.get(Stock,symbol);meta=validate(raw,stock)
        if stock.market!='US' and stock.market not in COMMODITY_MARKETS:return 0,0,False
        incoming=[b for b in clean(raw,stock.market,clock) if b[0]<=expected]
        stock.checked_at=now()
        if not incoming:stock.error='no_complete_bars';return 0,0,False
        store_bars(s,symbol,incoming)
        bars=[list(r) for r in s.execute(select(Candle.ts,Candle.o,Candle.h,Candle.l,Candle.c,Candle.v)
              .where(Candle.symbol==symbol).order_by(Candle.ts)).all()]
        old=stock.last_bar or 0;dates=sorted({local(b[0],stock.market).date() for b in reference_bars})
        changes=0;created=0
        expected_times=[b[0] for b in reference_bars]
        open_plans=s.scalars(select(Plan).where(Plan.symbol==symbol,Plan.state.in_(OPEN))).all()
        pending=[b for b in bars if old<b[0]<=expected] if old else []
        last=old
        for b in pending:
            stock.last_price=b[4];stock.last_bar=int(b[0])
            intervening=[t for t in expected_times if last<t<b[0]]
            contiguous=not intervening
            ratio=volume_ratio(bars,b[0],stock.market,dates)
            for plan in open_plans:changes+=int(advance(s,plan,b,ratio,contiguous))
            last=b[0]
        # No backdated candidate/activation on first load. Only current expected slot may create plans.
        latest=bars[-1][0];stock.last_price=bars[-1][4];stock.last_bar=max(old,int(latest))
        if latest!=expected:
            stock.error='awaiting_expected_closed_bar';return created,changes,False
        result=evaluate(json.loads(stock.metadata_json),dates,to_raw(bars,meta),clock,expected_times)
        flags=json.loads((DATA/'corporate_flags.json').read_text(encoding='utf-8'))
        if stock.market=='US' and stock.symbol in flags:
            result['corporate_event']=flags[stock.symbol];result['conditional_plan']=False
            result['blockers'].append('known_corporate_event_manual_exclusion')
        stock.evaluation_json=json.dumps(result,ensure_ascii=False)
        stock.error=''
        if strategy_diag is not None:
            strategy_diag['evaluated'] += 1
            strategy_diag['eligible'] += int(bool(result.get('eligible')))
            strategy_diag['conditional_plan'] += int(bool(result.get('conditional_plan')))
            blockers=result.get('blockers') or []
            if blockers:
                strategy_diag['blocked_symbols'] += 1
                for reason in blockers:
                    strategy_diag['blocker_counts'][reason] += 1
                    examples=strategy_diag['blocker_examples'][reason]
                    if len(examples) < 5:
                        examples.append(symbol)
            elif not result.get('conditional_plan'):
                reason='conditional_plan_false_without_blocker'
                strategy_diag['blocked_symbols'] += 1
                strategy_diag['blocker_counts'][reason] += 1
                examples=strategy_diag['blocker_examples'][reason]
                if len(examples) < 5:
                    examples.append(symbol)
        # Existing WAITING/RETEST/ACTIVE levels stay pinned. A later closed bar may create a fresh plan.
        if not old or latest>old:
            created=int(new_plan(s,stock,result,latest,settings) is not None)
        return created,changes,True

async def scan_market(DB,provider,m,clock,settings):
    boundary=int(clock)//900*900
    expected=expected_slot(m,clock) if m not in COMMODITY_MARKETS else None
    scan_id=None
    try:
        reference_raw=await provider.fetch(CONFIG[m]['ref'],True)
        meta=reference_raw['chart']['result'][0]['meta']
        if meta.get('dataGranularity')!='15m' or meta.get('symbol')!=CONFIG[m]['ref']:
            raise FeedError('invalid_reference')
        reference_bars=clean(reference_raw,m,clock)
        if m in COMMODITY_MARKETS:
            # Spot commodities are sourced from Twelve Data. Use the latest actually closed M15
            # candle returned by the official API rather than demanding a wall-clock
            # quarter that may fall inside a provider/session maintenance gap.
            if not reference_bars:
                raise FeedError('twelve_no_complete_15m_bar')
            expected=int(reference_bars[-1][0])
            retrieval=reference_raw.get('_retrieval',{})
            log.info('%s TwelveData reference symbol=%s bars=%s latest=%s volume_available=%s',m,
                     retrieval.get('upstream_symbol'),retrieval.get('bars',len(reference_bars)),
                     expected,retrieval.get('volume_available'))
        elif not reference_bars or reference_bars[-1][0]!=expected:
            raise FeedError('reference_awaiting_closed_bar_or_market_holiday')
        if expected is None:return
        with DB.begin() as s:
            scan=Scan(market=m,boundary=boundary,expected_bar=expected);s.add(scan);s.flush();scan_id=scan.id
        with DB() as s:
            priority=set(s.scalars(select(Plan.symbol).where(Plan.market==m,Plan.state.in_(OPEN))).all())
            stocks=s.scalars(select(Stock).where(Stock.market==m,Stock.last_bar<expected)
                  .order_by(Stock.symbol)).all()
        stocks.sort(key=lambda r:(r.symbol not in priority,r.symbol))
        counts={'total':len(stocks),'ok':0,'errors':0,'new_plans':0,'transitions':0}
        error_counts=Counter()
        error_examples=defaultdict(list)
        strategy_diag={
            'evaluated':0,
            'eligible':0,
            'conditional_plan':0,
            'blocked_symbols':0,
            'blocker_counts':Counter(),
            'blocker_examples':defaultdict(list),
        }

        def remember_error(reason,symbol):
            reason=reason or 'unknown_error'
            error_counts[reason]+=1
            if len(error_examples[reason])<5:
                error_examples[reason].append(symbol)
        async def fetch_stock(stock):
            try:return stock.symbol,await provider.fetch(stock.feed_symbol,not stock.last_bar),None
            except FeedError as error:return stock.symbol,None,str(error)
            except Exception:return stock.symbol,None,'unexpected_feed_error'
        for task in asyncio.as_completed([fetch_stock(r) for r in stocks]):
            symbol,raw,error=await task
            if error:
                with DB.begin() as s:
                    stock=s.get(Stock,symbol);stock.error=error;stock.checked_at=now()
                remember_error(error,symbol)
                counts['errors']+=1;continue
            try:
                # Freeze as-of time at scan start; a long scan cannot include the next quarter's unfinished data.
                created,changed,ok=apply_stock(DB,symbol,raw,expected,reference_bars,settings,clock,strategy_diag)
                counts['new_plans']+=created;counts['transitions']+=changed;counts['ok']+=int(ok);counts['errors']+=int(not ok)
                if not ok:
                    with DB() as s:
                        stock=s.get(Stock,symbol)
                        remember_error(stock.error if stock else 'stock_missing_after_apply',symbol)
            except Exception as exc:
                reason='processing_'+type(exc).__name__
                with DB.begin() as s:
                    stock=s.get(Stock,symbol);stock.error=reason;stock.checked_at=now()
                remember_error(reason,symbol)
                log.exception('%s processing error for %s',m,symbol)
                counts['errors']+=1
            processed=counts['ok']+counts['errors']
            if processed%250==0:log.info('%s progress %s/%s',m,processed,len(stocks))
        with DB.begin() as s:
            scan=s.get(Scan,scan_id)
            for key,value in counts.items():setattr(scan,key,value)
            scan.status='partial' if counts['errors'] else 'complete';scan.finished_at=now()
            summary=dict(counts)
            summary['error_summary']=dict(error_counts)
            summary['strategy_summary']={
                'evaluated':strategy_diag['evaluated'],
                'eligible':strategy_diag['eligible'],
                'conditional_plan':strategy_diag['conditional_plan'],
                'blocked_symbols':strategy_diag['blocked_symbols'],
            }
            summary['blocker_summary']=dict(strategy_diag['blocker_counts'])
            scan.summary_json=json.dumps(summary)
        log.info('%s scan %s',m,counts)
        log.info('%s strategy summary %s',m,{
            'evaluated':strategy_diag['evaluated'],
            'eligible':strategy_diag['eligible'],
            'conditional_plan':strategy_diag['conditional_plan'],
            'blocked_symbols':strategy_diag['blocked_symbols'],
        })
        if strategy_diag['blocker_counts']:
            log.warning('%s blocker summary %s',m,dict(strategy_diag['blocker_counts']))
            for reason,count in strategy_diag['blocker_counts'].most_common():
                log.warning('%s blocker %s: count=%s examples=%s',m,reason,count,','.join(strategy_diag['blocker_examples'][reason]))
        if error_counts:
            log.warning('%s error summary %s',m,dict(error_counts))
            for reason,count in error_counts.most_common():
                log.warning('%s error %s: count=%s examples=%s',m,reason,count,','.join(error_examples[reason]))
    except Exception as error:
        if scan_id is not None:
            with DB.begin() as s:
                scan=s.get(Scan,scan_id);scan.status='waiting_feed';scan.finished_at=now()
                scan.summary_json=json.dumps({'reason':str(error) if isinstance(error,FeedError) else type(error).__name__})
        reason=str(error) if isinstance(error,FeedError) else type(error).__name__
        if m in COMMODITY_MARKETS:
            label=COMMODITY_NAMES.get(m,m)
            if reason.startswith('twelve_cooldown_active;'):
                log.info('%s %s TwelveData retry deferred; %s',m,label,reason)
            else:
                log.warning('%s %s TwelveData reference failed; %s; progress retained',m,label,reason)
        else:
            log.warning('%s reference unavailable or session closed; reason=%s; progress retained',m,reason)

async def run(DB,once=False):
    settings=policy();delay=max(10,int(os.getenv('MONITOR_CLOSE_DELAY_SECONDS','45')))
    concurrency=max(1,min(24,int(os.getenv('MONITOR_CONCURRENCY','12'))))
    ids=os.getenv('MONITOR_ADMIN_ALERT_IDS','')
    if ids and any(not v.strip().lstrip('-').isdigit() for v in ids.split(',')):
        raise RuntimeError('Invalid MONITOR_ADMIN_ALERT_IDS')
    last_attempt=None
    # Development convenience: after every worker restart, run one immediate US scan
    # even when the normal market window is closed. scan_market() still freezes to
    # the latest completed regular-session 15m bar, so no unfinished after-hours
    # candle can create a plan. Set MONITOR_FORCE_STARTUP_SCAN=0 to disable later.
    force_startup=os.getenv('MONITOR_FORCE_STARTUP_SCAN','1').strip().lower() not in {'0','false','no','off'}
    startup_done=False
    egx_last_date=None
    async with YahooProvider(concurrency) as us_provider, TwelveDataCommodityProvider() as commodity_provider:
        while True:
            clock=time.time();boundary=int(clock)//900*900;offset=int(clock)-boundary
            # One forced scan immediately after startup for development/diagnostics.
            if force_startup and not startup_done:
                with exclusive(DB) as locked:
                    if locked:
                        log.info('Startup diagnostic scan forced outside normal market-window rules')
                        for m in ('XA','US'):
                            await scan_market(DB,commodity_provider if m in COMMODITY_MARKETS else us_provider,m,clock,settings)
                        await scan_egx_daily(DB,clock)
                        egx_last_date=datetime.fromtimestamp(clock,ZoneInfo(EGX_TZ)).date().isoformat()
                startup_done=True
                if once:return
                # Mark the current retry key so we do not immediately duplicate this scan.
                attempt=min(3,max(0,(offset-delay)//150))
                last_attempt=(boundary,attempt)
                await asyncio.sleep(10)
                continue
            # Four bounded retries each quarter, fetching only stocks whose watermark is behind.
            attempt=min(3,max(0,(offset-delay)//150))
            key=(boundary,attempt)
            if once or (offset>=delay and key!=last_attempt):
                with exclusive(DB) as locked:
                    if locked:
                        for m in ('XA','US'):
                            if due(m,clock):await scan_market(DB,commodity_provider if m in COMMODITY_MARKETS else us_provider,m,clock,settings)
                last_attempt=key
            egx_today=datetime.fromtimestamp(clock,ZoneInfo(EGX_TZ)).date().isoformat()
            if egx_due(clock) and egx_last_date!=egx_today:
                with exclusive(DB) as locked:
                    if locked:
                        await scan_egx_daily(DB,clock)
                        egx_last_date=egx_today
            if once:return
            await asyncio.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--init-only',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    DB=database()
    with exclusive(DB) as locked:
        if not locked:raise SystemExit('Another monitor is running')
        counts=initialize(DB)
    log.info('Universe initialized: %s; sharia filtering disabled; paper monitoring only',counts)
    if not args.init_only:asyncio.run(run(DB,args.once))

if __name__=='__main__':main()
