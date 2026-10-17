"""Saudi daily price loader, optional TradingView universe discovery.
This service does not change Egypt records and writes only *.SR symbols.
"""
import os, json, re, logging
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import quote
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
log=logging.getLogger('saudi_refresh')
FALLBACK=('1010 1020 1030 1050 1060 1080 1111 1120 1140 1150 1180 1211 1301 1302 1303 1304 1320 2001 2010 2020 2030 2050 2060 2070 2080 2090 2100 2110 2120 2130 2140 2150 2160 2170 2180 2190 2200 2210 2220 2222 2223 2230 2240 2250 2270 2280 2290 2300 2310 2320 2330 2340 2350 2360 2370 2380 3001 3002 3003 3004 3010 3020 3030 3040 3050 3060 3080 3090 3091 4001 4002 4003 4004 4005 4006 4007 4008 4009 4010 4011 4012 4013 4014 4020 4030 4031 4040 4050 4051 4061 4070 4080 4090 4100 4110 4130 4140 4141 4142 4143 4144 4150 4160 4161 4162 4163 4164 4165 4170 4180 4190 4191 4192 4200 4210 4220 4230 4240 4250 4260 4261 4262 4263 4270 4280 4290 4291 4292 4300 4310 4320 4321 4322 4323 4324 4325 4330 4331 4332 4333 4334 4335 4336 4337 4338 4339 4340 4342 4344 4345 4346 4347 4348 4349 4350 4351 4352 4353 4354 4355 4356 4357 4358 4359 4360 4361 4362 4363 4364 4365 4366 4367 4368 4369 4370 4371 4372 4373 4374 4375 4376 4377 4378 4379 4380 4381 4382 4383 4384 4385 4386 4387 4388 4389 4402 4420 4430 4440 4450 4502 5110 6001 6010 6012 6013 6014 6015 6016 6017 6018 6019 6020 6030 6040 6050 6060 6070 6080 6090 7010 7020 7030 7040 7200 7201 7202 7203 7204 7211 8010 8012 8020 8030 8040 8050 8060 8070 8100 8120 8150 8160 8170 8180 8190 8200 8210 8230 8240 8250 8260 8270 8280 8300 8310 8311').split()

def universe(conn):
    symbols={x+'.SR' for x in FALLBACK}
    symbols.update(x.strip().upper() for x in os.getenv('SAUDI_SYMBOLS','').replace(',',' ').split() if x.strip())
    for (symbol,) in conn.execute(text("SELECT DISTINCT symbol FROM market_candles_1d WHERE symbol ILIKE '%.SR'")):
        symbols.add(symbol.upper())
    # Optional expanded authoritative universe; use SAUDI_SYMBOLS to supply complete exchange listing.
    return sorted(x for x in symbols if re.fullmatch(r'\d{4}\.SR', x))

def download(symbol):
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?range=10y&interval=1d'
    req=Request(url, headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'})
    with urlopen(req,timeout=30) as response:
        data=json.load(response)['chart']['result'][0]
    timestamps=data.get('timestamp') or []
    quote_rows=data['indicators']['quote'][0]
    out=[]
    for i,ts in enumerate(timestamps):
        close=quote_rows.get('close',[])[i]
        if close is None or close<=0:continue
        o=quote_rows.get('open',[])[i];h=quote_rows.get('high',[])[i];l=quote_rows.get('low',[])[i];v=quote_rows.get('volume',[])[i]
        date=datetime.fromtimestamp(ts,timezone.utc).date().isoformat()
        out.append({'session_date':date,'c':float(close),'adj_c':float(close),'o':o,'h':h,'l':l,'v':v,'ts':ts})
    return out

def main():
    url=os.environ['DATABASE_URL']
    engine=create_engine(url,pool_pre_ping=True)
    with engine.connect() as conn:
        symbols=universe(conn)
        required=conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=current_schema()
              AND table_name='market_candles_1d'
              AND is_nullable='NO'
              AND column_default IS NULL
              AND is_generated='NEVER'
        """)).scalars().all()
        handled={'symbol','session_date','c','adj_c','o','h','l','v','ts',
                 'feed_symbol','source','retrieved_at','metadata_json'}
        missing=set(required)-handled
        if missing:
            raise RuntimeError(f'Unhandled mandatory DB columns: {sorted(missing)}')
    log.info('Required database columns verified: %s', ', '.join(required))
    log.info('Candidate Saudi symbols: %d',len(symbols))
    success=0;failed=0
    for idx,symbol in enumerate(symbols,1):
        try:
            candles=download(symbol)
            if not candles:continue
            with engine.begin() as conn:
                # Write only columns available in market_candles_1d. Keep other-market symbols unchanged.
                existing_cols={r[0] for r in conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='market_candles_1d'"))}
                for c in candles:
                    c['symbol']=symbol
                    c['feed_symbol']=symbol  # Required NOT NULL by the shared market_candles_1d schema.
                    c['source']='yahoo_chart_saudi'
                    c['retrieved_at']=datetime.now(timezone.utc)  # Required NOT NULL timestamp in shared DB.
                    c['metadata_json']=json.dumps({'feed':'yahoo_chart_saudi','market':'SA','symbol':symbol}, ensure_ascii=False)
                    vals={k:v for k,v in c.items() if k in existing_cols}
                    if 'symbol' not in vals or 'session_date' not in vals or 'c' not in vals:raise RuntimeError('Required DB columns absent')
                    upd={k:v for k,v in vals.items() if k not in ('symbol','session_date')}
                    updated=conn.execute(text('UPDATE market_candles_1d SET '+', '.join(f'{k}=:{k}' for k in upd)+' WHERE symbol=:symbol AND session_date=:session_date'),vals)
                    if updated.rowcount==0:
                        conn.execute(text('INSERT INTO market_candles_1d ('+', '.join(vals)+') VALUES ('+', '.join(':'+k for k in vals)+')'),vals)
            success+=1
            log.info('%d/%d %s %d candles',idx,len(symbols),symbol,len(candles))
        except Exception as exc:
            failed+=1
            log.warning('FAILED %s: %s',symbol,exc)
    log.info('DONE updated=%d failed=%d',success,failed)

if __name__=='__main__':main()
