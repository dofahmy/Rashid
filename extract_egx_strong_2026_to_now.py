#!/usr/bin/env python3
"""
Extract all EGX STRONG signals from 2026 and calculate each signal's performance
from signal close up to the latest available trading day in the stored result file.

Input:
  /data/egx_accum_strong_1y2y_signals.csv

Output:
  /data/egx_strong_2026_signals_to_now.csv
  /data/egx_strong_2026_signals_to_now.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
import time

DATA = Path("/data")
INFILE = DATA / "egx_accum_strong_1y2y_signals.csv"
OUTCSV = DATA / "egx_strong_2026_signals_to_now.csv"
OUTTXT = DATA / "egx_strong_2026_signals_to_now.txt"

df = pd.read_csv(INFILE, dtype={"symbol": str})
df["signal_date"] = pd.to_datetime(df["signal_date"])
sig = df[df["signal_date"].dt.year == 2026].copy().sort_values(["signal_date","symbol"])

if sig.empty:
    raise SystemExit("No 2026 signals found.")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})

rows = []

def download(symbol):
    yahoo = f"{symbol}.CA"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo}"
    params = {
        "period1": int(pd.Timestamp("2025-12-01", tz="UTC").timestamp()),
        "period2": int(pd.Timestamp.utcnow().timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    obj = r.json()
    result = ((obj.get("chart") or {}).get("result") or [])
    if not result:
        return None, str((obj.get("chart") or {}).get("error") or "no result")
    z = result[0]
    ts = z.get("timestamp") or []
    q = (((z.get("indicators") or {}).get("quote") or [{}])[0])
    adj = ((z.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not ts:
        return None, "no timestamps"
    x = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "o": q.get("open"),
        "h": q.get("high"),
        "l": q.get("low"),
        "c": q.get("close"),
        "v": q.get("volume"),
    })
    x["adj_c"] = adj if adj and len(adj) == len(x) else x["c"]
    for c in ["o","h","l","c","v","adj_c"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["c","adj_c"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return x, None

for n, r in enumerate(sig.itertuples(index=False), 1):
    symbol = str(r.symbol).strip()
    hist, err = download(symbol)

    if hist is None or hist.empty:
        rows.append({
            "signal_date": r.signal_date.date().isoformat(),
            "symbol": symbol,
            "name": getattr(r, "name", ""),
            "signal_price": getattr(r, "raw_signal_close_egp", np.nan),
            "latest_date": None,
            "latest_close": None,
            "return_to_now_pct": None,
            "max_gain_to_now_pct": None,
            "max_drawdown_to_now_pct": None,
            "trading_sessions_since_signal": None,
            "status": f"DATA ERROR: {err}",
        })
        continue

    after = hist[hist["date"] >= r.signal_date].copy()
    if after.empty:
        continue

    # Use the original recorded signal price for entry, to stay consistent with the backtest.
    entry = float(getattr(r, "raw_signal_close_egp"))
    latest = after.iloc[-1]
    latest_close = float(latest["c"])

    future = after[after["date"] > r.signal_date].copy()

    if len(future):
        max_high = float(future["h"].max())
        min_low = float(future["l"].min())
        max_gain = 100 * (max_high / entry - 1)
        max_dd = 100 * (min_low / entry - 1)
    else:
        max_gain = np.nan
        max_dd = np.nan

    ret_now = 100 * (latest_close / entry - 1)

    rows.append({
        "signal_date": r.signal_date.date().isoformat(),
        "symbol": symbol,
        "name": getattr(r, "name", ""),
        "signal_price": entry,
        "latest_date": latest["date"].date().isoformat(),
        "latest_close": latest_close,
        "return_to_now_pct": ret_now,
        "max_gain_to_now_pct": max_gain,
        "max_drawdown_to_now_pct": max_dd,
        "trading_sessions_since_signal": int(len(future)),
        "status": "OK",
    })

    print(
        f"[{n}/{len(sig)}] {symbol} | "
        f"{r.signal_date.date()} -> {latest['date'].date()} | "
        f"now {ret_now:+.2f}% | max {max_gain:+.2f}% | dd {max_dd:+.2f}%"
    )
    time.sleep(0.05)

out = pd.DataFrame(rows)
out = out.sort_values(["signal_date","symbol"]).reset_index(drop=True)
out.to_csv(OUTCSV, index=False, encoding="utf-8-sig")

ok = out[out["status"] == "OK"].copy()

print("\n=== EGX STRONG SIGNALS 2026 — PERFORMANCE TO LATEST DATE ===")
print("Signals:", len(out))
print("With current data:", len(ok))
if len(ok):
    print("Positive now:", int((ok["return_to_now_pct"] > 0).sum()),
          f"({100*(ok['return_to_now_pct'] > 0).mean():.2f}%)")
    print("Average return now:", f"{ok['return_to_now_pct'].mean():.2f}%")
    print("Median return now:", f"{ok['return_to_now_pct'].median():.2f}%")
    print("Average max gain:", f"{ok['max_gain_to_now_pct'].mean():.2f}%")
    print("Median max gain:", f"{ok['max_gain_to_now_pct'].median():.2f}%")
    print("Worst current return:", f"{ok['return_to_now_pct'].min():.2f}%")

cols = [
    "signal_date","symbol","signal_price","latest_date","latest_close",
    "return_to_now_pct","max_gain_to_now_pct","max_drawdown_to_now_pct",
    "trading_sessions_since_signal"
]

print("\n=== ALL 2026 SIGNALS ===")
print(ok[cols].to_string(index=False))

lines = []
lines.append("EGX STRONG SIGNALS 2026 — PERFORMANCE TO LATEST AVAILABLE DATE")
lines.append("="*110)
lines.append(f"Signals: {len(out)}")
lines.append(f"With current data: {len(ok)}")
if len(ok):
    lines.append(f"Positive now: {(ok['return_to_now_pct'] > 0).sum()} / {len(ok)} = {100*(ok['return_to_now_pct'] > 0).mean():.2f}%")
    lines.append(f"Average return now: {ok['return_to_now_pct'].mean():.2f}%")
    lines.append(f"Median return now: {ok['return_to_now_pct'].median():.2f}%")
    lines.append("")
    lines.append(ok[cols].to_string(index=False))

OUTTXT.write_text("\n".join(lines), encoding="utf-8")

print("\nCreated:")
print(OUTCSV)
print(OUTTXT)
