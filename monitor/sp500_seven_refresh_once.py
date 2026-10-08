# monitor/sp500_seven_refresh_once.py
from datetime import datetime, timezone
import traceback
from monitor.sp500_seven_data import refresh_sp500_data, set_status

def main():
    try:
        refresh_sp500_data(force_full=False)
    except Exception as exc:
        set_status({
            "status":"failed",
            "finished_at_utc":datetime.now(timezone.utc).isoformat(),
            "message":f"فشل تحديث S&P 500: {type(exc).__name__}: {exc}"
        })
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
