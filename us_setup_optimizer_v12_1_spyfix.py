#!/usr/bin/env python3
import sys, subprocess
from datetime import datetime, timezone

def ensure(pkg, import_name=None):
    import_name = import_name or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

ensure("requests")
ensure("pandas")

import requests
import pandas as pd
import us_setup_optimizer_v12 as v12

_original = v12.load_spy_from_db

def yahoo_spy():
    print("[SPY] not in DB - downloading from Yahoo Finance...", flush=True)
    start = int(datetime(2022,1,1,tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(timezone.utc).timestamp()) + 86400
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    params = {
        "period1": start,
        "period2": end,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    r = requests.get(url, params=params, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    obj = r.json()
    result = (obj.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError("Yahoo SPY download failed")
    item = result[0]
    ts = item.get("timestamp") or []
    ind = item.get("indicators") or {}
    adj_items = ind.get("adjclose") or []
    q_items = ind.get("quote") or []
    adj = adj_items[0].get("adjclose") if adj_items else None
    close = adj if adj and len(adj)==len(ts) else (q_items[0].get("close") if q_items else None)
    if not ts or not close:
        raise RuntimeError("Yahoo returned no usable SPY data")
    spy = pd.DataFrame({
        "d": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "c": close,
    })
    spy["c"] = pd.to_numeric(spy["c"], errors="coerce")
    spy = spy.dropna().drop_duplicates("d", keep="last").sort_values("d").reset_index(drop=True)
    print(f"[SPY] Yahoo loaded {len(spy):,} rows {spy.d.min().date()} -> {spy.d.max().date()}", flush=True)
    return spy

def patched():
    try:
        return _original()
    except Exception as e:
        print(f"[SPY] DB source unavailable: {e}", flush=True)
        return yahoo_spy()

v12.load_spy_from_db = patched

if __name__ == "__main__":
    v12.main()
