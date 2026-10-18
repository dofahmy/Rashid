"""Bounded incremental US refresh for already-known US symbols in market_candles_1d.
Processes one rotating batch per run so the public button is not an unbounded job.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
log=logging.getLogger('us_refresh')
TICKER=re.compile(r'^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$')


def download(sym):
    url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(sym,safe='')+'?range=3mo&interval=1d'
    req=Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'})
    with urlopen(req,timeout=20) as response:
        payload=json.load(response)
    data=(payload.get('chart',{}).get('result') or [None])[0]
    if not data:raise RuntimeError('No Yahoo data')
    quote_data=data['indicators']['quote'][0]
    result=[]
    times=data.get('timestamp') or []
    for i,ts in enumerate(times):
        def value(name):
            seq=quote_data.get(name) or []
            return seq[i] if i<len(seq) else None
        c=value('close')
        if c is None or c<=0:continue
        result.append(dict(symbol=sym,session_date=datetime.fromtimestamp(ts,timezone.utc).date().isoformat(),
                           ts=int(ts),c=c,adj_c=c,o=value('open'),h=value('high'),
                           l=value('low'),v=value('volume'),feed_symbol=sym,
                           source='yahoo_chart_us',retrieved_at=datetime.now(timezone.utc),
                           metadata_json=json.dumps({'market':'US','feed':'yahoo_chart_us','symbol':sym})))
    return result


def main():
    db=create_engine(os.environ['DATABASE_URL'],pool_pre_ping=True)
    max_symbols=max(1,min(int(os.getenv('US_REFRESH_BATCH_SIZE','200')),1000))
    with db.begin() as conn:
        cols={r[0] for r in conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='market_candles_1d'"))}
        required={r[0] for r in conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='market_candles_1d' AND is_nullable='NO' AND column_default IS NULL AND is_generated='NEVER'"))}
        handled={'symbol','session_date','ts','c','adj_c','o','h','l','v','feed_symbol','source','retrieved_at','metadata_json'}
        if missing:=required-handled:raise RuntimeError('Unhandled mandatory DB columns: '+', '.join(sorted(missing)))
        conn.execute(text('CREATE TABLE IF NOT EXISTS us_screening_refresh_cursor (id integer PRIMARY KEY, next_index integer NOT NULL)'))
        conn.execute(text('INSERT INTO us_screening_refresh_cursor(id,next_index) VALUES (1,0) ON CONFLICT (id) DO NOTHING'))
        cursor=conn.execute(text('SELECT next_index FROM us_screening_refresh_cursor WHERE id=1 FOR UPDATE')).scalar_one()
        symbols=sorted(set(str(r[0]).upper() for r in conn.execute(text('SELECT DISTINCT symbol FROM market_candles_1d')) if r[0] and TICKER.fullmatch(str(r[0]).upper())))
        if not symbols:
            log.warning('No US symbols exist in DB. Seed US historical data first.');return
        batch=[symbols[(cursor+i)%len(symbols)] for i in range(min(max_symbols,len(symbols)))]
        conn.execute(text('UPDATE us_screening_refresh_cursor SET next_index=:n WHERE id=1'),{'n':(cursor+len(batch))%len(symbols)})
    ok=failed=0
    log.info('US refresh batch=%s, universe=%s',len(batch),len(symbols))
    for sym in batch:
        try:
            candles=download(sym)
            if not candles:raise RuntimeError('No candles')
            with db.begin() as conn:
                for c in candles:
                    vals={k:v for k,v in c.items() if k in cols}
                    update_fields=[k for k in vals if k not in ('symbol','session_date')]
                    update_sql=', '.join(k+'=:'+k for k in update_fields)
                    r=conn.execute(text('UPDATE market_candles_1d SET '+update_sql+' WHERE symbol=:symbol AND session_date=:session_date'),vals)
                    if r.rowcount==0:
                        conn.execute(text('INSERT INTO market_candles_1d ('+', '.join(vals)+') VALUES ('+', '.join(':'+k for k in vals)+')'),vals)
            ok+=1
        except Exception as exc:
            failed+=1
            log.warning('FAILED %s: %s',sym,exc)
    log.info('DONE updated=%d failed=%d batch=%d universe=%d',ok,failed,len(batch),len(symbols))

if __name__=='__main__': main()
