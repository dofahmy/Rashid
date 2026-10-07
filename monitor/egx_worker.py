# monitor/egx_worker.py
import logging,os,time
from datetime import datetime
from zoneinfo import ZoneInfo
from core import database
from .egx_live import refresh
log=logging.getLogger("rajih.egx");CAIRO=ZoneInfo("Africa/Cairo")
def due(now):
    hour=int(os.getenv("EGX_REFRESH_HOUR","16"))
    return now.weekday() in (6,0,1,2,3) and now.hour>=hour
def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    DB=database();last_success_date=None;last_attempt_at=0.0
    while True:
        now=datetime.now(CAIRO)
        if due(now) and last_success_date!=now.date() and time.time()-last_attempt_at>=300:
            last_attempt_at=time.time()
            try:
                log.info("EGX live refresh starting for %s",now.date())
                state=refresh(DB);last_success_date=now.date()
                log.info("EGX live refresh complete date=%s bias=%s low=%s high=%s",state.get("market_date"),state.get("market_bias"),state.get("low_alert_count"),state.get("high_alert_count"))
            except BaseException:
                log.exception("EGX live refresh failed; retry in 5 minutes")
        time.sleep(60)
if __name__=="__main__":main()
