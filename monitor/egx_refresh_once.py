# monitor/egx_refresh_once.py
from datetime import datetime,timezone
import json,traceback
from core import database
from .egx_data_refresh import refresh_daily_data
from .egx_live import refresh,set_refresh_status

def main():
    DB=database();started=datetime.now(timezone.utc).isoformat()
    set_refresh_status(DB,{"status":"running","started_at_utc":started,"message":"جارٍ تحديث بيانات السوق المصري ثم إعادة الحساب..."})
    try:
        data_result=refresh_daily_data()
        state=refresh(DB)
        set_refresh_status(DB,{"status":"complete","started_at_utc":started,"finished_at_utc":datetime.now(timezone.utc).isoformat(),"message":"اكتمل تحديث البيانات وإعادة حساب Market Turn.","market_date":state.get("market_date"),"data_result":data_result})
        print(json.dumps({"ok":True,"market_date":state.get("market_date"),"bias":state.get("market_bias"),"low_alerts":state.get("low_alert_count"),"high_alerts":state.get("high_alert_count"),"data":data_result},ensure_ascii=False,indent=2))
    except Exception as exc:
        set_refresh_status(DB,{"status":"failed","started_at_utc":started,"finished_at_utc":datetime.now(timezone.utc).isoformat(),"message":f"فشل التحديث: {type(exc).__name__}: {exc}"})
        traceback.print_exc();raise

if __name__=="__main__":main()
