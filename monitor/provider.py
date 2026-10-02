"""Replaceable public research feeds. No realtime/SLA claim."""
import asyncio, time, random, os, math, logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import aiohttp

log=logging.getLogger('rajih.monitor.provider')

class FeedError(Exception): pass

class YahooProvider:
    name='Yahoo public chart (research; latency unknown)'
    def __init__(self,concurrency=12):
        self.limit=asyncio.Semaphore(concurrency)
        self.session=None

    @staticmethod
    def _load_api_key():
        """Load and sanitize the Twelve Data key from Railway env.

        Accepts the canonical TWELVE_DATA_API_KEY plus two compatibility names,
        and tolerates an accidentally pasted `NAME=value` string or surrounding
        quotes.  The secret itself is never logged.
        """
        raw=(os.getenv('TWELVE_DATA_API_KEY') or os.getenv('TWELVEDATA_API_KEY') or os.getenv('TWELVE_API_KEY') or '')
        raw=str(raw).strip()
        for prefix in ('TWELVE_DATA_API_KEY=', 'TWELVEDATA_API_KEY=', 'TWELVE_API_KEY='):
            if raw.startswith(prefix):
                raw=raw[len(prefix):].strip()
                break
        if len(raw)>=2 and raw[0]==raw[-1] and raw[0] in ("'", '"'):
            raw=raw[1:-1].strip()
        return raw

    def _refresh_api_key(self):
        # Railway injects env vars before process start, but refreshing here keeps
        # all commodity requests on the same canonical key and avoids stale values
        # in long-lived provider instances/tests.
        self.api_key=self._load_api_key()
        return self.api_key

    async def __aenter__(self):
        self.session=aiohttp.ClientSession(trust_env=True,timeout=aiohttp.ClientTimeout(total=35),
            headers={'User-Agent':'Mozilla/5.0'},connector=aiohttp.TCPConnector(limit=24))
        return self
    async def __aexit__(self,*args): await self.session.close()
    async def fetch(self,symbol,bootstrap=False):
        url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')
        params={'interval':'15m','includePrePost':'false','range':'60d' if bootstrap else '5d'}
        async with self.limit:
            for attempt in range(3):
                try:
                    async with self.session.get(url,params=params) as resp:
                        if resp.status==422 and bootstrap:
                            params.pop('range',None);params.update(period1=str(int(time.time())-59*86400),period2=str(int(time.time())))
                            continue
                        if resp.status in (429,500,502,503,504):
                            await asyncio.sleep(min(20,2**(attempt+1))+random.random());continue
                        if resp.status!=200: raise FeedError(f'provider_http_{resp.status}')
                        raw=await resp.json(content_type=None)
                        if raw.get('chart',{}).get('error') or not raw.get('chart',{}).get('result'):
                            raise FeedError('provider_no_chart')
                        raw['_retrieval']={'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'provider':self.name,'url':str(resp.url)}
                        return raw
                except (aiohttp.ClientError,asyncio.TimeoutError,ValueError):
                    if attempt==2: raise FeedError('provider_network_or_json') from None
                    await asyncio.sleep(2**attempt)
        raise FeedError('provider_retry_exhausted')


class TwelveDataCommodityProvider:
    """Official Twelve Data adapter for spot commodities on 15-minute candles."""
    name='Twelve Data commodities official API (15m)'
    API_URL='https://api.twelvedata.com/time_series'
    SYMBOLS={
        'XAUUSD':('TWELVE_DATA_XAUUSD_SYMBOL','XAU/USD'),
        'XAGUSD':('TWELVE_DATA_XAGUSD_SYMBOL','XAG/USD'),
        'WTIUSD':('TWELVE_DATA_WTIUSD_SYMBOL','WTI/USD'),
    }

    def __init__(self):
        self.session=None
        self.api_key=self._load_api_key()
        # Per-symbol cooldown prevents repeated paid/plan/auth failures from hammering
        # Twelve Data every worker loop. The first real failure is logged with the
        # upstream HTTP status/code/message; later retries wait for the cooldown.
        self.cooldown_seconds=max(300,int(os.getenv('TWELVE_DATA_ERROR_COOLDOWN_SECONDS','900')))
        self._cooldown_until={}
        self._cooldown_reason={}

    async def __aenter__(self):
        self._refresh_api_key()
        log.info('TwelveData provider ready api_key_loaded=%s',bool(self.api_key))
        self.session=aiohttp.ClientSession(trust_env=True,timeout=aiohttp.ClientTimeout(total=40),
            headers={'User-Agent':'Rajih-Monitor/1.0','Accept':'application/json'})
        return self

    async def __aexit__(self,*args):
        if self.session: await self.session.close()

    @staticmethod
    def _num(value,default=None):
        if value is None:return default
        try:
            value=float(str(value).replace(',','').strip())
            return value if math.isfinite(value) else default
        except (ValueError,TypeError):return default

    @staticmethod
    def _timestamp(value):
        text=str(value or '').strip()
        if not text:return None
        try:dt=datetime.fromisoformat(text.replace('Z','+00:00'))
        except ValueError:return None
        if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    def upstream(self,symbol):
        env,default=self.SYMBOLS.get(symbol,('',symbol))
        return (os.getenv(env) or default).strip() if env else default

    @staticmethod
    def _clean_message(value):
        text=' '.join(str(value or '').replace('\n',' ').replace('\r',' ').split())
        return text[:220] or 'no_message'

    @classmethod
    def _api_error_reason(cls,http_status,payload):
        payload=payload if isinstance(payload,dict) else {}
        code=payload.get('code')
        status=payload.get('status')
        message=cls._clean_message(payload.get('message') or payload.get('detail') or '')
        parts=[f'twelve_http_{http_status}']
        if code is not None: parts.append(f'code={code}')
        if status: parts.append(f'status={status}')
        parts.append(f'message={message}')
        return ';'.join(parts)

    def _set_cooldown(self,symbol,reason):
        self._cooldown_until[symbol]=time.time()+self.cooldown_seconds
        self._cooldown_reason[symbol]=reason

    def _normalize(self,payload,symbol,upstream_symbol):
        if not isinstance(payload,dict):raise FeedError('twelve_invalid_json')
        if payload.get('status')=='error':
            code=payload.get('code');message=str(payload.get('message') or '').lower()
            reason=self._api_error_reason(200,payload)
            if code in (401,403):
                self._set_cooldown(symbol,reason)
                raise FeedError(reason)
            if code==429 or 'limit' in message or 'credits' in message:
                self._set_cooldown(symbol,reason)
                raise FeedError(reason)
            raise FeedError(reason)
        values=payload.get('values')
        if not isinstance(values,list) or not values:raise FeedError('twelve_no_15m_rows')
        rows=[]
        for item in values:
            if not isinstance(item,dict):continue
            ts=self._timestamp(item.get('datetime'));o=self._num(item.get('open'));h=self._num(item.get('high'));l=self._num(item.get('low'));c=self._num(item.get('close'));v=self._num(item.get('volume'),0.0)
            if ts is None or None in (o,h,l,c):continue
            ts=(ts//900)*900
            if min(o,h,l,c)<=0 or not (l<=o<=h and l<=c<=h):continue
            rows.append([ts,o,h,l,c,max(0.0,v or 0.0)])
        rows=sorted({r[0]:r for r in rows}.values())
        if not rows:raise FeedError('twelve_no_valid_15m_rows')
        meta=payload.get('meta') if isinstance(payload.get('meta'),dict) else {};names=['open','high','low','close','volume']
        return {'chart':{'result':[{'meta':{'symbol':symbol,'dataGranularity':'15m','currency':'USD','instrumentType':'CURRENCY','regularMarketPrice':rows[-1][4],'upstreamSymbol':meta.get('symbol') or upstream_symbol,'upstreamInterval':meta.get('interval') or '15min','upstreamType':meta.get('type')},'timestamp':[r[0] for r in rows],'indicators':{'quote':[{k:[r[i+1] for r in rows] for i,k in enumerate(names)}]}}]},'_retrieval':{'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'provider':self.name,'url':self.API_URL,'upstream_symbol':meta.get('symbol') or upstream_symbol,'bars':len(rows),'volume_available':any(r[5]>0 for r in rows)}}

    async def fetch(self,symbol='XAUUSD',bootstrap=False):
        # Re-read the same Railway variable for XA/XS/XO so all commodities use
        # one key path.  Sanitize common paste mistakes before every request.
        if not self._refresh_api_key():raise FeedError('twelve_api_key_missing')
        upstream_symbol=self.upstream(symbol)
        until=self._cooldown_until.get(symbol,0)
        if until>time.time():
            remaining=max(1,int(until-time.time()))
            cached=self._cooldown_reason.get(symbol,'twelve_previous_error')
            raise FeedError(f'twelve_cooldown_active;retry_in={remaining}s;last={cached}')
        params={'symbol':upstream_symbol,'interval':'15min','outputsize':'5000' if bootstrap else '1200','timezone':'UTC','order':'ASC','apikey':self.api_key}
        last_error=None
        for attempt in range(3):
            try:
                async with self.session.get(self.API_URL,params=params) as resp:
                    payload=None
                    try:
                        payload=await resp.json(content_type=None)
                    except Exception:
                        payload=None
                    if resp.status in (429,500,502,503,504):
                        if resp.status==429:
                            reason=self._api_error_reason(resp.status,payload)
                            self._set_cooldown(symbol,reason)
                            raise FeedError(reason)
                        last_error='twelve_http_'+str(resp.status);await asyncio.sleep(min(15,2**(attempt+1))+random.random());continue
                    if resp.status in (401,403):
                        reason=self._api_error_reason(resp.status,payload)
                        self._set_cooldown(symbol,reason)
                        raise FeedError(reason)
                    if resp.status!=200:
                        reason=self._api_error_reason(resp.status,payload)
                        raise FeedError(reason)
                    if payload is None:raise FeedError('twelve_network_or_json')
                    result=self._normalize(payload,symbol,upstream_symbol)
                    # Successful access clears any stale cooldown for this symbol.
                    self._cooldown_until.pop(symbol,None);self._cooldown_reason.pop(symbol,None)
                    return result
            except FeedError:raise
            except (aiohttp.ClientError,asyncio.TimeoutError,ValueError):
                if attempt==2:raise FeedError('twelve_network_or_json') from None
                await asyncio.sleep(2**attempt)
        raise FeedError(last_error or 'twelve_retry_exhausted')

# Backward-compatible name used by older deployments/tests.
TwelveDataGoldProvider=TwelveDataCommodityProvider

class InvestingGoldProvider:
    """Experimental Investing.com XAU/USD 15-minute feed adapter.

    The website exposes XAU/USD and 15-minute historical data, but this is an
    unofficial integration.  The adapter deliberately returns the same chart
    shape used by the rest of the monitor so strategy code stays unchanged.
    """
    name='Investing.com XAU/USD experimental 15m feed'
    SEARCH_URL='https://www.investing.com/search/service/searchTopBar'
    CHART_URL='https://api.investing.com/api/financialdata/{pair_id}/historical/chart'

    def __init__(self):
        self.session=None
        self.pair_id=(os.getenv('INVESTING_XAUUSD_PAIR_ID') or '').strip() or None
        self.last_pair_source='env' if self.pair_id else None

    async def __aenter__(self):
        headers={
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36',
            'Accept':'application/json, text/plain, */*',
            'Accept-Language':'en-US,en;q=0.9',
            'Referer':'https://www.investing.com/currencies/xau-usd-historical-data',
            'Origin':'https://www.investing.com',
            'domain-id':'www',
        }
        self.session=aiohttp.ClientSession(trust_env=True,timeout=aiohttp.ClientTimeout(total=40),headers=headers)
        return self

    async def __aexit__(self,*args):
        if self.session: await self.session.close()

    async def _discover_pair_id(self):
        if self.pair_id:return self.pair_id
        headers={'X-Requested-With':'XMLHttpRequest','Content-Type':'application/x-www-form-urlencoded; charset=UTF-8'}
        candidates=('XAU/USD','XAUUSD','Gold Spot US Dollar')
        for term in candidates:
            try:
                async with self.session.post(self.SEARCH_URL,data={'search_text':term},headers=headers) as resp:
                    if resp.status!=200: continue
                    payload=await resp.json(content_type=None)
                    quotes=payload.get('quotes') if isinstance(payload,dict) else None
                    if not isinstance(quotes,list):continue
                    for item in quotes:
                        url=str(item.get('url') or '').lower()
                        symbol=str(item.get('symbol') or '').upper().replace(' ','')
                        desc=str(item.get('description') or '').lower()
                        if '/currencies/xau-usd' in url or symbol in {'XAU/USD','XAUUSD'} or ('gold spot' in desc and 'us dollar' in desc):
                            value=item.get('pairId') or item.get('pair_id') or item.get('id')
                            if value is not None:
                                self.pair_id=str(value);self.last_pair_source='search';return self.pair_id
            except (aiohttp.ClientError,asyncio.TimeoutError,ValueError):
                continue
        # Widely used Investing.com pair id for XAU/USD spot.  Kept only as a
        # diagnostic fallback and can be overridden by Railway env var.
        self.pair_id='68';self.last_pair_source='fallback';return self.pair_id

    @staticmethod
    def _num(value,default=None):
        if value is None:return default
        if isinstance(value,(int,float)):
            return float(value) if math.isfinite(value) else default
        text=str(value).replace(',','').strip()
        if text in {'','-','None','null'}:return default
        try:
            value=float(text);return value if math.isfinite(value) else default
        except ValueError:return default

    @staticmethod
    def _ts(value):
        if value is None:return None
        if isinstance(value,(int,float)):
            value=float(value)
            if value>1e12:value/=1000.0
            return int(value)
        text=str(value).strip()
        if text.isdigit():
            value=int(text);return int(value/1000) if value>1e12 else value
        text=text.replace('Z','+00:00')
        try:
            dt=datetime.fromisoformat(text)
            if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:return None

    def _extract_rows(self,payload):
        rows=[]
        # TradingView-like response.
        if isinstance(payload,dict) and all(k in payload for k in ('t','o','h','l','c')):
            n=min(len(payload.get(k,[])) for k in ('t','o','h','l','c'))
            vols=payload.get('v') or [0]*n
            for i in range(n):
                rows.append((payload['t'][i],payload['o'][i],payload['h'][i],payload['l'][i],payload['c'][i],vols[i] if i<len(vols) else 0))
        else:
            data=payload
            if isinstance(payload,dict):
                for key in ('data','results','result','rows'):
                    if isinstance(payload.get(key),list):data=payload[key];break
            if isinstance(data,list):
                for item in data:
                    if isinstance(item,(list,tuple)) and len(item)>=5:
                        rows.append(tuple(item[:6]) if len(item)>=6 else tuple(item[:5])+(0,))
                    elif isinstance(item,dict):
                        ts=item.get('date',item.get('time',item.get('timestamp',item.get('rowDateTimestamp'))))
                        o=item.get('price_open',item.get('open',item.get('last_open')))
                        h=item.get('price_high',item.get('high',item.get('last_max')))
                        l=item.get('price_low',item.get('low',item.get('last_min')))
                        c=item.get('price_close',item.get('close',item.get('last_close',item.get('value'))))
                        v=item.get('volume',item.get('volume_raw',item.get('last_volume',0)))
                        rows.append((ts,o,h,l,c,v))
        out=[]
        for ts,o,h,l,c,v in rows:
            ts=self._ts(ts);o=self._num(o);h=self._num(h);l=self._num(l);c=self._num(c);v=self._num(v,0.0)
            if ts is None or None in (o,h,l,c):continue
            # Investing can expose timestamps at second/millisecond precision;
            # keep only exact quarter-hour bars after normalization.
            ts=(int(ts)//900)*900
            if min(o,h,l,c)<=0 or not (l<=o<=h and l<=c<=h):continue
            out.append([ts,o,h,l,c,max(0.0,v or 0.0)])
        return sorted({r[0]:r for r in out}.values())

    async def fetch(self,symbol='XAUUSD',bootstrap=False):
        pair_id=await self._discover_pair_id()
        # Ask for a long enough window for >=200 M15 bars and the 20-session
        # same-time volume baseline.  Investing may cap rows; the strategy will
        # report that explicitly instead of fabricating missing candles.
        params={
            'period':'P3M' if bootstrap else 'P1M',
            'interval':'PT15M',
            'pointscount':'5000' if bootstrap else '1200',
        }
        url=self.CHART_URL.format(pair_id=pair_id)
        last_error=None
        for attempt in range(3):
            try:
                async with self.session.get(url,params=params) as resp:
                    if resp.status in (429,500,502,503,504):
                        last_error=f'investing_http_{resp.status}'
                        await asyncio.sleep(min(15,2**(attempt+1))+random.random());continue
                    if resp.status!=200:raise FeedError(f'investing_http_{resp.status}')
                    payload=await resp.json(content_type=None)
                    rows=self._extract_rows(payload)
                    if not rows:raise FeedError('investing_no_15m_rows')
                    ts=[r[0] for r in rows];o=[r[1] for r in rows];h=[r[2] for r in rows];l=[r[3] for r in rows];c=[r[4] for r in rows];v=[r[5] for r in rows]
                    raw={'chart':{'result':[{
                        'meta':{
                            'symbol':symbol,
                            'currency':'USD',
                            'instrumentType':'CURRENCY',
                            'dataGranularity':'15m',
                            'regularMarketPrice':c[-1],
                            'exchangeName':'Investing.com Real-time FX',
                        },
                        'timestamp':ts,
                        'indicators':{'quote':[{'open':o,'high':h,'low':l,'close':c,'volume':v}]},
                    }], 'error':None}}
                    raw['_retrieval']={
                        'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        'provider':self.name,
                        'url':str(resp.url),
                        'pair_id':pair_id,
                        'pair_id_source':self.last_pair_source,
                        'bars':len(rows),
                    }
                    return raw
            except FeedError:raise
            except (aiohttp.ClientError,asyncio.TimeoutError,ValueError):
                if attempt==2:raise FeedError('investing_network_or_json') from None
                await asyncio.sleep(2**attempt)
        raise FeedError(last_error or 'investing_retry_exhausted')
