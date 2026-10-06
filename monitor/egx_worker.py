# monitor/egx_worker.py
"""
Dedicated EGX daily worker.

Railway service command:
    python -m monitor.egx_worker

It refreshes after 16:00 Cairo time Sunday-Thursday, once per market date.
Set EGX_REFRESH_HOUR to change the hour.
"""
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from core import database
from .egx_live import refresh

log = logging.getLogger("rajih.egx")

CAIRO = ZoneInfo("Africa/Cairo")


def due(now):
    # Egypt exchange working week: Sunday-Thursday.
    hour = int(os.getenv("EGX_REFRESH_HOUR", "16"))
    return now.weekday() in (6, 0, 1, 2, 3) and now.hour >= hour


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DB = database()
    last_date = None

    while True:
        now = datetime.now(CAIRO)
        if due(now) and last_date != now.date():
            try:
                log.info("EGX live refresh starting for %s", now.date())
                state = refresh(DB)
                last_date = now.date()
                log.info(
                    "EGX live refresh complete date=%s bias=%s low=%s high=%s",
                    state.get("market_date"),
                    state.get("market_bias"),
                    state.get("low_alert_count"),
                    state.get("high_alert_count"),
                )
            except Exception:
                log.exception("EGX live refresh failed")
        time.sleep(60)


if __name__ == "__main__":
    main()
