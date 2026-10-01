"""Replaceable public research feed. No realtime/SLA claim."""
import asyncio, time, random
from urllib.parse import quote
import aiohttp

class FeedError(Exception): pass
class YahooProvider:
    name='Yahoo public chart (research; latency unknown)'
    def __init__(self,concurrency=12):
        self.limit=asyncio.Semaphore(concurrency)
        self.session=None
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
                        raw['_retrieval']={'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'provider':self.name}
                        return raw
                except (aiohttp.ClientError,asyncio.TimeoutError,ValueError):
                    if attempt==2: raise FeedError('provider_network_or_json') from None
                    await asyncio.sleep(2**attempt)
        raise FeedError('provider_retry_exhausted')
