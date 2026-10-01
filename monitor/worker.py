"""Railway worker: python -m monitor.worker [--init-only|--once]."""
import argparse, asyncio, contextlib, json, logging, os, time
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import select, delete, func, case, text
from core import database, now
from .models import Stock, Candle, Plan, Scan, OPEN
from .engine import new_plan, advance, policy
from .strategy import clean, local, CONFIG, evaluate
from .provider import YahooProvider, FeedError

log=logging.getLogger('rajih.monitor')
DATA=Path(__file__).parent/'data'
LOCK_KEY=72617368696415

def initialize(DB):
    universe=json.loads((DATA/'universe.json').read_text(encoding='utf-8'))
    counts={m:sum(r['market_key']==m for r in universe) for m in CONFIG}
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
    return {'US':counts['US']}

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
                (meta.get('instrumentType')=='EQUITY','not_equity')]
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

def apply_stock(DB,symbol,raw,expected,reference_bars,settings,clock):
    """One symbol, one transaction: bars, watermark, plans and events commit together."""
    with DB.begin() as s:
        stock=s.get(Stock,symbol);meta=validate(raw,stock)
        if stock.market!='US':return 0,0,False
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
        # Existing WAITING/RETEST/ACTIVE levels stay pinned. A later closed bar may create a fresh plan.
        if not old or latest>old:
            created=int(new_plan(s,stock,result,latest,settings) is not None)
        return created,changes,True

async def scan_market(DB,provider,m,clock,settings):
    boundary=int(clock)//900*900;expected=expected_slot(m,clock)
    if expected is None:return
    with DB.begin() as s:
        scan=Scan(market=m,boundary=boundary,expected_bar=expected);s.add(scan);s.flush();scan_id=scan.id
    try:
        reference_raw=await provider.fetch(CONFIG[m]['ref'],True)
        meta=reference_raw['chart']['result'][0]['meta']
        if meta.get('dataGranularity')!='15m' or meta.get('symbol')!=CONFIG[m]['ref']:
            raise FeedError('invalid_reference')
        reference_bars=clean(reference_raw,m,clock)
        if not reference_bars or reference_bars[-1][0]!=expected:
            raise FeedError('reference_awaiting_closed_bar_or_market_holiday')
        with DB() as s:
            priority=set(s.scalars(select(Plan.symbol).where(Plan.market==m,Plan.state.in_(OPEN))).all())
            stocks=s.scalars(select(Stock).where(Stock.market==m,Stock.last_bar<expected)
                  .order_by(Stock.symbol)).all()
        stocks.sort(key=lambda r:(r.symbol not in priority,r.symbol))
        counts={'total':len(stocks),'ok':0,'errors':0,'new_plans':0,'transitions':0}
        error_counts=Counter()
        error_examples=defaultdict(list)

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
                created,changed,ok=apply_stock(DB,symbol,raw,expected,reference_bars,settings,clock)
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
            scan.summary_json=json.dumps(summary)
        log.info('%s scan %s',m,counts)
        if error_counts:
            log.warning('%s error summary %s',m,dict(error_counts))
            for reason,count in error_counts.most_common():
                log.warning('%s error %s: count=%s examples=%s',m,reason,count,','.join(error_examples[reason]))
    except Exception as error:
        with DB.begin() as s:
            scan=s.get(Scan,scan_id);scan.status='waiting_feed';scan.finished_at=now()
            scan.summary_json=json.dumps({'reason':str(error) if isinstance(error,FeedError) else type(error).__name__})
        log.warning('%s reference unavailable or session closed; progress retained',m)

async def run(DB,once=False):
    settings=policy();delay=max(10,int(os.getenv('MONITOR_CLOSE_DELAY_SECONDS','45')))
    concurrency=max(1,min(24,int(os.getenv('MONITOR_CONCURRENCY','12'))))
    ids=os.getenv('MONITOR_ADMIN_ALERT_IDS','')
    if ids and any(not v.strip().lstrip('-').isdigit() for v in ids.split(',')):
        raise RuntimeError('Invalid MONITOR_ADMIN_ALERT_IDS')
    last_attempt=None
    async with YahooProvider(concurrency) as provider:
        while True:
            clock=time.time();boundary=int(clock)//900*900;offset=int(clock)-boundary
            # Four bounded retries each quarter, fetching only stocks whose watermark is behind.
            attempt=min(3,max(0,(offset-delay)//150))
            key=(boundary,attempt)
            if once or (offset>=delay and key!=last_attempt):
                with exclusive(DB) as locked:
                    if locked:
                        for m in ('US',):
                            if due(m,clock):await scan_market(DB,provider,m,clock,settings)
                last_attempt=key
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
