"""Standalone, fail-closed daily Time-Price worker. Never executes legacy M15 logic.

Set Railway Monitor start command: python -m monitor.us_tp_runner
A single active worker is enforced with the existing monitor's DB advisory lock.
Paper mode by default; publication requires US_TP_LIVE_ALERTS=1.
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from core import database
from .worker import exclusive, LOCK_KEY
from .us_tp_daily import scan

LOG=logging.getLogger('rajih.us_tp.runner')
NY=ZoneInfo('America/New_York')

def last_session_candidate(now):
    day=now.date()
    if now.hour<18:day-=timedelta(days=1)
    while day.weekday()>=5:day-=timedelta(days=1)
    return day.isoformat()

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if os.getenv('RAJIH_STRATEGY','TIME_PRICE_DAILY')!='TIME_PRICE_DAILY':
        raise SystemExit('RAJIH_STRATEGY must be TIME_PRICE_DAILY; legacy M15 is disabled in this worker')
    live=os.getenv('US_TP_LIVE_ALERTS','0')=='1'
    if live and os.getenv('US_TP_ACCEPT_RISK','')!='YES':
        raise SystemExit('US_TP_LIVE_ALERTS requires US_TP_ACCEPT_RISK=YES after verified staging')
    DB=database()
    with exclusive(DB,LOCK_KEY) as acquired:
        if not acquired:raise SystemExit('Another monitor worker is running (legacy or Time–Price)')
        LOG.info('M15 strategy OFF. US daily Time–Price monitor: alerts=%s',live)
        previous=None
        batch=max(1,int(os.getenv('US_TP_BATCH','25')))
        while True:
            try:
                now=datetime.now(NY)
                day=last_session_candidate(now)
                if previous!=day:
                    asyncio.run(scan(DB,day,int(os.getenv('US_TP_CONCURRENCY','4')),batch,live))
                    previous=day
                    LOG.info('Daily scan attempted: %s. Incomplete scans send no new recommendations.',day)
                else:
                    # Retry incomplete scans, e.g. Yahoo rate limiting, without double-publishing.
                    asyncio.run(scan(DB,day,int(os.getenv('US_TP_CONCURRENCY','4')),batch,live))
            except Exception:
                LOG.exception('Time–Price worker scan failed; will retry, M15 remains disabled')
            time.sleep(int(os.getenv('US_TP_POLL_SECONDS','60')))

if __name__=='__main__':main()
