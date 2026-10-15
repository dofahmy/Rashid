"""US daily Time-Price swing monitor (research / paper signals).

Separate from the legacy 15-minute monitor. Does NOT send signals unless
--publish is requested and a *complete* scan of the configured US universe
finishes for the same NYSE session. Default --once is read-only on trading
recommendations; OHLC cache and staged candidates are updated in PostgreSQL.
"""
import argparse, asyncio, json, logging, math, os, time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import aiohttp
import numpy as np
import pandas as pd
from sqlalchemy import text, select
from core import database, Lead, now
from .models import Stock, Plan, Event
from .customer import eligible, Publication, Recipient, _queue, minimum_score
from .limits import available

LOG=logging.getLogger('rajih.us_tp_daily')
NY=ZoneInfo('America/New_York')
VERSION='US_TP_D1_v1'
RATIOS={'1x2':0.5,'1x1':1.0,'2x1':2.0}
TYPES=('DIFFERENT_ANGLES','SAME_ANGLE_DIFF_ANCHORS','SAME_LINE_TWICE')


def ensure(s):
    s.execute(text('''CREATE TABLE IF NOT EXISTS us_tp_daily_bars (
        symbol TEXT NOT NULL, day TEXT NOT NULL,o DOUBLE PRECISION NOT NULL,
        h DOUBLE PRECISION NOT NULL,l DOUBLE PRECISION NOT NULL,c DOUBLE PRECISION NOT NULL,
        PRIMARY KEY(symbol,day))'''))
    s.execute(text('''CREATE TABLE IF NOT EXISTS us_tp_daily_candidates (
        day TEXT NOT NULL,symbol TEXT NOT NULL,signal_type TEXT NOT NULL,
        anchor_type TEXT NOT NULL,status TEXT NOT NULL,angles TEXT NOT NULL,
        close_price DOUBLE PRECISION NOT NULL,
        PRIMARY KEY(day,symbol,signal_type,anchor_type,status))'''))
    s.execute(text('''CREATE TABLE IF NOT EXISTS us_tp_daily_scan (
        day TEXT PRIMARY KEY,expected_count INT NOT NULL,scanned_count INT NOT NULL DEFAULT 0,
        errors_count INT NOT NULL DEFAULT 0,completed BOOLEAN NOT NULL DEFAULT FALSE,
        published BOOLEAN NOT NULL DEFAULT FALSE,updated_at TIMESTAMPTZ DEFAULT NOW())'''))
    s.execute(text('''CREATE TABLE IF NOT EXISTS us_tp_daily_done (
        day TEXT NOT NULL,symbol TEXT NOT NULL,ok BOOLEAN NOT NULL DEFAULT TRUE,
        PRIMARY KEY(day,symbol))'''))


def fetch_chart(session, symbol, semaphore, price_range="5y"):
    async def go():
        from urllib.parse import quote
        url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')
        params={'interval':'1d','range':price_range,'includePrePost':'false'}
        async with semaphore:
            for attempt in range(2):
                try:
                    async with session.get(url,params=params,timeout=aiohttp.ClientTimeout(total=20,connect=8)) as r:
                        if r.status in (429,500,502,503,504):
                            await asyncio.sleep(2**attempt+2);continue
                        if r.status!=200:raise ValueError('HTTP '+str(r.status))
                        blob=await r.json(content_type=None)
                        obj=(blob.get('chart',{}).get('result') or [None])[0]
                        if not obj:raise ValueError('empty_chart')
                        q=obj['indicators']['quote'][0]
                        dates=[datetime.fromtimestamp(t,timezone.utc).astimezone(NY).strftime('%Y-%m-%d') for t in obj['timestamp']]
                        bars=[]
                        for i,day in enumerate(dates):
                            vals=[q.get(k,[None]*len(dates))[i] for k in ('open','high','low','close')]
                            if all(v is not None and math.isfinite(float(v)) and float(v)>0 for v in vals):
                                o,h,l,c=map(float,vals)
                                if h>=max(o,c,l) and l<=min(o,c,h):bars.append((day,o,h,l,c))
                        return list({r[0]:r for r in bars}.values())
                except (aiohttp.ClientError,asyncio.TimeoutError,ValueError,KeyError,IndexError,TypeError) as e:
                    if attempt==1:raise ValueError(str(e)) from None
                    await asyncio.sleep(2**attempt+1)
        raise ValueError('retries_exhausted')
    return go()


def atr(high, low, close):
    pc=np.roll(close,1);pc[0]=close[0]
    tr=np.maximum(high-low,np.maximum(abs(high-pc),abs(low-pc)))
    return pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy()


def candidates_for_symbol(symbol,bars):
    if len(bars)<60:return []
    bars=sorted(bars)
    dates=np.array([v[0] for v in bars]); high=np.array([v[2] for v in bars],float)
    low=np.array([v[3] for v in bars],float);close=np.array([v[4] for v in bars],float)
    n=len(bars); ar=atr(high,low,close);firsts=defaultdict(list);retouches=[]
    for kind,sign in (('LOW',1),('HIGH',-1)):
        for i in range(14,n-10):
            av=ar[i];
            if not np.isfinite(av) or av<=0:continue
            lr=low[i-10:i+11]; hr=high[i-10:i+11]
            if kind=='LOW':
                if low[i]!=min(lr) or max(hr)-low[i]<5*av or max(hr[11:])-low[i]<2.5*av:continue
                base=low[i]
            else:
                if high[i]!=max(hr) or high[i]-min(lr)<5*av or high[i]-min(lr[11:])<2.5*av:continue
                base=high[i]
            for angle,ratio in RATIOS.items():
                slope=sign*av/20*ratio
                ts=np.arange(i+11,n)
                levels=base+(ts-i)*slope
                touched=(levels>0)&(low[ts]-1e-8<=levels)&(levels<=high[ts]+1e-8)
                offsets=np.flatnonzero(touched)
                if len(offsets)==0:continue
                def row(j,level):
                    prev=level-slope
                    if close[j-1]<prev and close[j]>level:status='CROSS_UP_CLOSE'
                    elif close[j-1]>prev and close[j]<level:status='CROSS_DOWN_CLOSE'
                    elif close[j]>=level:status='TOUCH_ABOVE'
                    else:status='TOUCH_BELOW'
                    return dict(day=str(dates[j]),symbol=symbol,signal_type='',anchor_type=kind,
                                status=status,angles=angle,close_price=float(close[j]),
                                anchor=str(dates[i]))
                fst=int(offsets[0]);j=int(ts[fst]);firsts[j].append(row(j,levels[fst]))
                later=offsets[(offsets>fst+1)&(~touched[offsets-1])]
                if len(later):
                    off=int(later[0]);k=int(ts[off]);v=row(k,levels[off]);v['signal_type']='SAME_LINE_TWICE';retouches.append(v)
    qualified=[]
    for _,items in firsts.items():
        unique={(r['anchor_type'],r['anchor'],r['angles']):r for r in items}
        if len(unique)!=2:continue
        pair=list(unique.values())
        typ='DIFFERENT_ANGLES' if pair[0]['angles']!=pair[1]['angles'] else 'SAME_ANGLE_DIFF_ANCHORS'
        for r in pair:qualified.append({**r,'signal_type':typ})
    qualified.extend(retouches)
    grouped={}
    for r in qualified:
        k=(r['day'],r['symbol'],r['signal_type'],r['anchor_type'],r['status'])
        if k not in grouped:grouped[k]=dict(r)
        else:grouped[k]['angles']='+'.join(sorted(set(grouped[k]['angles'].split('+'))|{r['angles']}))
    return [{k:v for k,v in r.items() if k!='anchor'} for r in grouped.values()]


UPSERT_BARS = text('''INSERT INTO us_tp_daily_bars(symbol,day,o,h,l,c)
 VALUES(:symbol,:day,:o,:h,:l,:c)
 ON CONFLICT(symbol,day) DO UPDATE SET
 o=excluded.o,h=excluded.h,l=excluded.l,c=excluded.c''')
UPSERT_CANDIDATES = text('''INSERT INTO us_tp_daily_candidates
 (day,symbol,signal_type,anchor_type,status,angles,close_price)
 VALUES(:day,:symbol,:signal_type,:anchor_type,:status,:angles,:close_price)
 ON CONFLICT(day,symbol,signal_type,anchor_type,status)
 DO UPDATE SET angles=excluded.angles,close_price=excluded.close_price''')

def cached_bars(DB,symbol):
    with DB() as s:
        return [tuple(r) for r in s.execute(text(
            'SELECT day,o,h,l,c FROM us_tp_daily_bars WHERE symbol=:sym ORDER BY day'),
            {'sym':symbol})]

def save_symbol(DB,day,symbol,bars,old_bars=None):
    old_bars=old_bars or []
    # An initial 5-year fetch is necessary once; later scans fetch only recent bars.
    # Bulk insert/upsert instead of round-tripping for each of ~1,250 daily candles.
    old_by_day={r[0]:r for r in old_bars}
    fresh=[r for r in bars if old_by_day.get(r[0]) != r]
    merged=dict(old_by_day)
    merged.update((r[0],r) for r in bars)
    complete=sorted(merged.values())
    rows=candidates_for_symbol(symbol,complete)
    with DB.begin() as s:
        for i in range(0,len(fresh),300):
            payload=[dict(symbol=symbol,day=d,o=o,h=h,l=l,c=c)
                     for d,o,h,l,c in fresh[i:i+300]]
            if payload:s.execute(UPSERT_BARS,payload)
        s.execute(text('DELETE FROM us_tp_daily_candidates WHERE symbol=:symbol'),{'symbol':symbol})
        for i in range(0,len(rows),300):
            s.execute(UPSERT_CANDIDATES,rows[i:i+300])
        stock=s.get(Stock,symbol)
        if stock is not None:
            stock.last_price=float(complete[-1][4])
            stock.last_bar=int(datetime.strptime(complete[-1][0],'%Y-%m-%d').replace(tzinfo=NY).timestamp())+16*3600
            stock.checked_at=now()
            stock.error=''
        s.execute(text('''INSERT INTO us_tp_daily_done(day,symbol,ok) VALUES(:day,:symbol,TRUE)
           ON CONFLICT(day,symbol) DO UPDATE SET ok=TRUE'''),{'day':day,'symbol':symbol})


def close_plans(s, day):
    """No same-day target from a close entry; start on the next daily bar."""
    plans=s.scalars(select(Plan).where(Plan.market=='US',Plan.strategy_version==VERSION,
                 Plan.state=='ACTIVE')).all()
    for p in plans:
        entries=s.execute(text('''SELECT day,o,h,c FROM us_tp_daily_bars
               WHERE symbol=:sym AND day>:start AND day<=:end ORDER BY day'''),
               {'sym':p.symbol,'start':json.loads(p.context_json)['signal_day'],'end':day}).all()
        for index,(d,o,h,c) in enumerate(entries,1):
            ts=int(datetime.strptime(d,'%Y-%m-%d').replace(tzinfo=NY).timestamp())+16*3600
            # A gap above TP at the open is conservatively credited only at TP.
            if h>=p.target:
                p.state='TARGET';p.exit_price=p.target;p.last_bar=ts; p.updated_at=now()
                ev='TARGET'
            elif index>=20:
                p.state='TIME_EXIT';p.exit_price=float(c);p.last_bar=ts;p.updated_at=now()
                ev='TIME_EXIT'
            else:continue
            ekey=f'{VERSION}:{p.id}:{ev}'
            if not s.scalar(select(Event.id).where(Event.key==ekey)):
                s.add(Event(key=ekey,plan_id=p.id,symbol=p.symbol,kind=ev,bar_ts=ts,
                            details_json=json.dumps({'day':d,'exit':p.exit_price})))
                # Use existing client-report ledger and outbound-delivery policy.
                from .customer import notice
                notice(s,p,ev,ts)
            break


def publish(s,day,clock):
    # Group qualification deliberately based on all symbols, AFTER signal/side/status.
    rows=s.execute(text('''WITH groups AS (
          SELECT day,signal_type,anchor_type,status,COUNT(DISTINCT symbol) AS n
          FROM us_tp_daily_candidates
          WHERE day=:day AND signal_type='SAME_ANGLE_DIFF_ANCHORS'
            AND anchor_type='LOW'
            AND status IN ('CROSS_DOWN_CLOSE','TOUCH_ABOVE')
          GROUP BY 1,2,3,4
          HAVING COUNT(DISTINCT symbol)>=3)
          SELECT c.* FROM us_tp_daily_candidates c JOIN groups g USING(day,signal_type,anchor_type,status)
          WHERE c.day=:day
            AND c.signal_type='SAME_ANGLE_DIFF_ANCHORS'
            AND c.anchor_type='LOW'
            AND c.status IN ('CROSS_DOWN_CLOSE','TOUCH_ABOVE')
          ORDER BY c.symbol,c.signal_type,c.anchor_type,c.status'''),{'day':day}).mappings().all()
    # Defense in depth: preserve only the two explicitly selected patterns.
    allowed_statuses={'CROSS_DOWN_CLOSE','TOUCH_ABOVE'}
    by_symbol={}
    for r in rows:
        if (r['signal_type']=='SAME_ANGLE_DIFF_ANCHORS' and
                r['anchor_type']=='LOW' and r['status'] in allowed_statuses):
            by_symbol.setdefault(r['symbol'],dict(r))
    counts=0
    for sym,r in by_symbol.items():
        entry=float(r['close_price']);ts=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=NY).timestamp())+16*3600
        if s.scalar(select(Plan.id).where(Plan.symbol==sym,Plan.strategy_version==VERSION,Plan.signal_ts==ts)):
            continue
        p=Plan(symbol=sym,market='US',strategy_version=VERSION,state='ACTIVE',signal_ts=ts,last_bar=ts,
               entry=entry,stop=0.,target=round(entry*1.03,6),atr=0.,score=100.,
               trigger_ts=ts,activation_ts=ts,paper_entry=entry,context_json=json.dumps({
                   'signal_day':day,'signal_type':r['signal_type'],'side':r['anchor_type'],
                   'status':r['status'],'angles':r['angles'],'order_type':'MARKET',
                   'strategy':'US_TP_D1_v1','no_stop_loss':True}),
               policy_json=json.dumps({'target_pct':3,'max_sessions':20,'entry_at':'signal_day_close'}))
        s.add(p);s.flush()
        s.add(Event(key=f'{VERSION}:{p.id}:ACTIVE',plan_id=p.id,symbol=sym,kind='ACTIVE',bar_ts=ts,
                    details_json=json.dumps({'signal_day':day})))
        # Live alerts are prospective only. A historical/belated run will not
        # send misleading executable entries to customers.
        if datetime.now(NY).date().isoformat()==day:
            from .customer import score_allowed
            pub=Publication(plan_id=p.id,published_ts=clock,activation_end=clock)
            s.add(pub)
            for lead in s.scalars(select(Lead).where(Lead.status.in_(('trial','subscribed')))).all():
                if not eligible(s,lead) or not available(s,lead.telegram_id,entry,clock):continue
                key=f'usrec:{p.id}:ACTIVE:{lead.telegram_id}'
                s.add(Recipient(plan_id=p.id,telegram_id=lead.telegram_id,entry_key=key))
                note=(f'📈 راجح | Time–Price يومي | {sym}\n'
                      f'إشارة إغلاق {day} — سعر مرجعي {entry:.2f}$\n'
                      f'هدف +3%: {p.target:.2f}$ | خروج بعد 20 جلسة إذا لم يصل\n'
                      f'{r["signal_type"]} | {r["anchor_type"]} | {r["status"]} | {r["angles"]}\n'
                      'تنبيه بعد الإغلاق: سعر الدخول قابل للتغير في الجلسة التالية، ليس تنفيذًا مضمونًا.')
                _queue(s,key,lead.telegram_id,note,p.id,'ACTIVE',clock+86400)
        counts+=1
    return counts


def save_scan_progress(DB,day,expected,allow_publication=False,publish_flag=False):
    with DB.begin() as s:
        n=int(s.scalar(text('SELECT COUNT(*) FROM us_tp_daily_done WHERE day=:day AND ok=TRUE'),{'day':day}) or 0)
        was=s.execute(text('SELECT published FROM us_tp_daily_scan WHERE day=:day'),{'day':day}).scalar()
        s.execute(text('''INSERT INTO us_tp_daily_scan(day,expected_count,scanned_count,errors_count,completed,published,updated_at)
          VALUES(:day,:expected,:n,:err,:complete,:published,NOW())
          ON CONFLICT(day) DO UPDATE SET expected_count=excluded.expected_count,
          scanned_count=excluded.scanned_count,errors_count=excluded.errors_count,
          completed=excluded.completed,updated_at=NOW()'''),
          dict(day=day,expected=expected,n=n,err=max(0,expected-n),complete=n==expected,published=bool(was)))
        if allow_publication and n==expected and expected>0:
            close_plans(s,day)
            if publish_flag and not bool(was):
                total=publish(s,day,int(time.time()))
                s.execute(text('UPDATE us_tp_daily_scan SET published=TRUE WHERE day=:day'),{'day':day})
                LOG.info('Published %s distinct stock signals for %s',total,day)
    return n

async def scan(DB, day, concurrency, batch, publish_flag):
    with DB.begin() as s:ensure(s)
    with DB() as s:
        symbols=s.scalars(select(Stock.symbol).where(Stock.market=='US').order_by(Stock.symbol)).all()
        done=set(s.scalars(text('SELECT symbol FROM us_tp_daily_done WHERE day=:day AND ok=TRUE'),{'day':day}).all())
    todo=[x for x in symbols if x not in done]
    if batch>0:todo=todo[:batch]
    LOG.info('US stocks=%d done=%d this_batch=%d',len(symbols),len(done),len(todo))
    # Progress visible immediately, even if a process is interrupted.
    save_scan_progress(DB,day,len(symbols))
    sem=asyncio.Semaphore(max(1,min(concurrency,8)))
    attempts=0
    async with aiohttp.ClientSession(headers={'User-Agent':'Mozilla/5.0'}) as http:
        async def one(sym):
            try:
                history=await asyncio.to_thread(cached_bars,DB,sym)
                # Reuse existing history when it is long enough to cover the pivots.
                # A 1-month fetch handles recent corrections, holidays and new sessions.
                price_range='1mo' if len(history)>=600 else '5y'
                bars=await fetch_chart(http,sym,sem,price_range)
                last=max(history[-1][0] if history else '',bars[-1][0] if bars else '')
                if last != day or len(history)+len(bars)<60:
                    return sym,'not_ready_or_insufficient_history'
                await asyncio.to_thread(save_symbol,DB,day,sym,bars,history)
                return sym,None
            except Exception as e:
                return sym,str(e)[:130]
        for offset in range(0,len(todo),max(1,concurrency*2)):
            chunk=todo[offset:offset+max(1,concurrency*2)]
            results=await asyncio.gather(*[one(sym) for sym in chunk])
            errors=[(sym,e) for sym,e in results if e]
            attempts+=len(results)
            n=save_scan_progress(DB,day,len(symbols))
            LOG.info('PROGRESS done=%d/%d attempted=%d/%d, recent_failures=%d examples=%s',
                     n,len(symbols),attempts,len(todo),len(errors),errors[:3])
    n=save_scan_progress(DB,day,len(symbols),allow_publication=True,publish_flag=publish_flag)
    if n!=len(symbols):
        LOG.warning('Incomplete universe %d/%d; NO new customer signals.',n,len(symbols))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--once',action='store_true',help='Perform one scan; no publication without --publish')
    p.add_argument('--publish',action='store_true',help='Enable prospect-only Telegram signals after a complete scan')
    p.add_argument('--day',help='Expected completed NYSE trading session YYYY-MM-DD; defaults most recent NY business day')
    p.add_argument('--batch',type=int,default=int(os.getenv('US_TP_BATCH','0')),
                   help='Maximum new symbols to scan in this run; 0 = all outstanding')
    p.add_argument('--concurrency',type=int,default=6)
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    # Calendar date, not an authoritative NYSE holidays calendar. No signals unless
    # all symbol charts have actually closed at the same exact date.
    now_ny=datetime.now(NY)
    day=args.day or ((now_ny-timedelta(days=1)).date().isoformat() if now_ny.hour<18 else now_ny.date().isoformat())
    d=datetime.strptime(day,'%Y-%m-%d').date()
    while d.weekday()>=5:d-=timedelta(days=1)
    day=d.isoformat()
    DB=database()
    asyncio.run(scan(DB,day,max(1,min(args.concurrency,12)),max(0,args.batch),args.publish))

if __name__=='__main__':main()
