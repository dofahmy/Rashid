#!/usr/bin/env python3
"""
Rajih — Accumulation STRONG backtest on EGX daily stocks

Goal
----
Apply the SAME supplied Pine Script logic to Egyptian Exchange stocks on DAILY
bars, then evaluate independent STRONG signals over:
  - 3 months = 63 trading sessions
  - 6 months = 126 trading sessions

Data
----
1) Discover current Egypt-listed equities from TradingView scanner.
2) Download daily OHLCV history from Yahoo Finance using .CA tickers.

IMPORTANT:
Yahoo coverage for EGX is incomplete. The script prints:
  - symbols discovered
  - symbols with usable Yahoo history
  - failed symbols
So we can judge coverage before trusting the result.

Pine logic mirrored
-------------------
MIN_BARS = 80
range = 22%..80%
avgVolNow(80) < avgVolHist(200)*0.70
green volume ratio over 80 bars >= 0.55
close in top 30% of 80-bar range
STRONG = score >=90
With supplied weights, STRONG effectively means all four conditions are true.

Signal:
  new STRONG only = isStrong and not isStrong[1]

Independence:
  after a kept STRONG signal, ignore new STRONG signals in the same stock
  for 126 actual trading sessions.

EGX basic cleaning
------------------
- actual/raw close >= 1 EGP
- prior 20-day average traded value >= 1,000,000 EGP
- prior 20-day average volume >= 10,000 shares
- no absurd one-day adjusted/raw discontinuity > 100%
These are intentionally lighter than the US filters.

Outputs
-------
/data/egx_accum_strong_signals.csv
/data/egx_accum_strong_yearly.csv
/data/egx_accum_strong_coverage.csv
/data/egx_accum_strong_report.txt

Run
---
cd /app
python backtest_accumulation_strong_egx.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path("/data")

OUT_SIGNALS = DATA / "egx_accum_strong_1y2y_signals.csv"
OUT_YEARLY = DATA / "egx_accum_strong_1y2y_yearly.csv"
OUT_COVERAGE = DATA / "egx_accum_strong_1y2y_coverage.csv"
OUT_REPORT = DATA / "egx_accum_strong_1y2y_report.txt"

# Pine defaults
MIN_BARS = 80
MIN_RANGE = 22.0
MAX_RANGE = 80.0
VOL_DRY_RATIO = 0.70
VOL_HIST_BARS = 200
GREEN_VOL_MIN = 0.55

# EGX light cleaning
MIN_PRICE_EGP = 1.0
MIN_ADV20_EGP = 1_000_000.0
MIN_AVG_VOL20 = 10_000.0

START_TS = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
END_TS = int(pd.Timestamp.utcnow().timestamp())

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})


def discover_egx_symbols():
    url = "https://scanner.tradingview.com/egypt/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"}
        ],
        "options": {"lang": "en"},
        "markets": ["egypt"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "description", "exchange"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 1000]
    }

    r = SESSION.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])

    out = []
    for item in data:
        s = item.get("s", "")
        if ":" in s:
            exch, ticker = s.split(":", 1)
        else:
            exch, ticker = "", s
        vals = item.get("d") or []
        name = vals[0] if len(vals) > 0 else ticker
        desc = vals[1] if len(vals) > 1 else ""
        exchange = vals[2] if len(vals) > 2 else exch

        if ticker:
            out.append({
                "tv_symbol": s,
                "ticker": ticker.strip(),
                "name": name,
                "description": desc,
                "exchange": exchange,
            })

    # unique ticker
    seen = set()
    clean = []
    for x in out:
        if x["ticker"] not in seen:
            seen.add(x["ticker"])
            clean.append(x)
    return clean


def yahoo_download(ticker):
    yahoo_symbol = f"{ticker}.CA"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    params = {
        "period1": START_TS,
        "period2": END_TS,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }

    r = SESSION.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"

    obj = r.json()
    result = ((obj.get("chart") or {}).get("result") or [])
    if not result:
        err = (obj.get("chart") or {}).get("error")
        return None, str(err or "no result")

    z = result[0]
    ts = z.get("timestamp") or []
    q = (((z.get("indicators") or {}).get("quote") or [{}])[0])
    adj_list = ((z.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")

    if not ts or not q:
        return None, "no OHLCV"

    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "o": q.get("open"),
        "h": q.get("high"),
        "l": q.get("low"),
        "c": q.get("close"),
        "v": q.get("volume"),
    })

    if adj_list and len(adj_list) == len(df):
        df["adj_c"] = adj_list
    else:
        df["adj_c"] = df["c"]

    for c in ["o", "h", "l", "c", "v", "adj_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["o", "h", "l", "c", "adj_c"])
    df = df[(df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) & (df["c"] > 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if len(df) < VOL_HIST_BARS:
        return None, f"only {len(df)} bars"

    return df, None


def add_features(df):
    df = df.copy()

    # Adjusted OHLC for split/dividend continuity.
    factor = df["adj_c"] / df["c"]
    df["ao"] = df["o"] * factor
    df["ah"] = df["h"] * factor
    df["al"] = df["l"] * factor
    df["ac"] = df["adj_c"]

    # basic data-quality discontinuity guard
    df["adj_ret1"] = 100 * (df["ac"] / df["ac"].shift(1) - 1)

    df["raw_value"] = df["c"] * df["v"].fillna(0)
    df["adv20"] = df["raw_value"].rolling(20).mean()
    df["avgvol20"] = df["v"].rolling(20).mean()

    # exact Pine logic
    df["patHigh"] = df["h"].rolling(MIN_BARS).max()
    df["patLow"] = df["l"].rolling(MIN_BARS).min()
    df["patRange"] = 100 * (df["patHigh"] - df["patLow"]) / df["patLow"]

    df["avgVolNow"] = df["v"].rolling(MIN_BARS).mean()
    df["avgVolHist"] = df["v"].rolling(VOL_HIST_BARS).mean()

    green_vol = df["v"].where(df["c"] >= df["o"], 0.0)
    total_vol = df["v"].fillna(0.0)

    df["greenVolRatio"] = (
        green_vol.rolling(MIN_BARS).sum() /
        total_vol.rolling(MIN_BARS).sum().replace(0, np.nan)
    )

    rng = (df["patHigh"] - df["patLow"]).replace(0, np.nan)
    df["nearTop"] = (df["c"] - df["patLow"]) / rng

    c1 = (df["patRange"] >= MIN_RANGE) & (df["patRange"] <= MAX_RANGE)
    c2 = df["avgVolNow"] < df["avgVolHist"] * VOL_DRY_RATIO
    c3 = df["greenVolRatio"] >= GREEN_VOL_MIN
    c4 = df["nearTop"] >= 0.70

    df["score"] = (
        c1.astype(int) * 25 +
        c2.astype(int) * 25 +
        c3.astype(int) * 30 +
        c4.astype(int) * 20
    )
    df["isStrong"] = df["score"] >= 90
    df["newStrong"] = df["isStrong"] & (~df["isStrong"].shift(1).fillna(False))

    return df


def clean_signal(df, i):
    if i < VOL_HIST_BARS - 1:
        return False, "INSUFFICIENT_HISTORY"

    if df.at[i, "c"] < MIN_PRICE_EGP:
        return False, "PRICE"

    if pd.isna(df.at[i-1, "adv20"]) or df.at[i-1, "adv20"] < MIN_ADV20_EGP:
        return False, "LIQUIDITY_VALUE"

    if pd.isna(df.at[i-1, "avgvol20"]) or df.at[i-1, "avgvol20"] < MIN_AVG_VOL20:
        return False, "LIQUIDITY_VOLUME"

    if (df.loc[max(1, i-20):i-1, "adj_ret1"].abs() > 100).any():
        return False, "DATA_DISCONTINUITY"

    return True, "ELIGIBLE"


def horizon(df, i, n):
    if i + n >= len(df):
        return None

    entry = float(df.at[i, "ac"])
    f = df.iloc[i+1:i+n+1]

    high_gain = 100 * (f["ah"].to_numpy(float) / entry - 1)
    low_dd = 100 * (f["al"].to_numpy(float) / entry - 1)
    close_ret = 100 * (float(df.at[i+n, "ac"]) / entry - 1)

    return {
        "close_ret_pct": float(close_ret),
        "max_gain_pct": float(np.nanmax(high_gain)),
        "max_drawdown_pct": float(np.nanmin(low_dd)),
        "hit20": int(np.any(high_gain >= 20)),
        "hit30": int(np.any(high_gain >= 30)),
        "hit50": int(np.any(high_gain >= 50)),
        "hit100": int(np.any(high_gain >= 100)),
    }


symbols = discover_egx_symbols()
print("\nRAJIH — EGX ACCUMULATION STRONG BACKTEST")
print("TradingView EGX equities discovered:", len(symbols))

coverage_rows = []
signal_rows = []
reject_counts = {}

for n, info in enumerate(symbols, 1):
    ticker = info["ticker"]

    try:
        df, err = yahoo_download(ticker)
        if df is None:
            coverage_rows.append({
                **info,
                "yahoo_symbol": f"{ticker}.CA",
                "usable": 0,
                "bars": 0,
                "reason": err,
            })
            if n % 25 == 0:
                print(f"[{n}/{len(symbols)}] usable so far: {sum(x['usable'] for x in coverage_rows)}")
            continue

        coverage_rows.append({
            **info,
            "yahoo_symbol": f"{ticker}.CA",
            "usable": 1,
            "bars": len(df),
            "reason": "",
        })

        df = add_features(df)

        last_kept_i = -10**9
        found = 0

        for i in df.index[df["newStrong"]]:
            ok, reason = clean_signal(df, i)
            if not ok:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue

            # exact 126 trading-session independence
            if i - last_kept_i < 126:
                reject_counts["COOLDOWN_126"] = reject_counts.get("COOLDOWN_126", 0) + 1
                continue

            last_kept_i = i
            found += 1

            m3 = horizon(df, i, 63)
            m6 = horizon(df, i, 126)
            m12 = horizon(df, i, 252)
            m24 = horizon(df, i, 504)

            row = {
                "signal_date": df.at[i, "date"].date().isoformat(),
                "year": int(df.at[i, "date"].year),
                "symbol": ticker,
                "name": info["name"],
                "raw_signal_close_egp": float(df.at[i, "c"]),
                "score": int(df.at[i, "score"]),
                "patRange_pct": float(df.at[i, "patRange"]),
                "dry_vol_ratio": float(df.at[i, "avgVolNow"] / df.at[i, "avgVolHist"]),
                "greenVolRatio": float(df.at[i, "greenVolRatio"]),
                "nearTop": float(df.at[i, "nearTop"]),
                "adv20_egp": float(df.at[i-1, "adv20"]),
                "mature_3m": int(m3 is not None),
                "mature_6m": int(m6 is not None),
                "mature_1y": int(m12 is not None),
                "mature_2y": int(m24 is not None),
            }

            for prefix, m in [("m3", m3), ("m6", m6), ("m12", m12), ("m24", m24)]:
                for key in [
                    "close_ret_pct", "max_gain_pct", "max_drawdown_pct",
                    "hit20", "hit30", "hit50", "hit100"
                ]:
                    row[f"{prefix}_{key}"] = m[key] if m else None

            signal_rows.append(row)

        if found:
            print(f"[{n}/{len(symbols)}] {ticker}: independent STRONG={found}")
        elif n % 25 == 0:
            print(f"[{n}/{len(symbols)}] usable so far: {sum(x['usable'] for x in coverage_rows)}")

        time.sleep(0.05)

    except Exception as e:
        coverage_rows.append({
            **info,
            "yahoo_symbol": f"{ticker}.CA",
            "usable": 0,
            "bars": 0,
            "reason": f"ERROR: {e}",
        })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(OUT_COVERAGE, index=False, encoding="utf-8-sig")

if not signal_rows:
    print("\nNo clean independent STRONG signals found.")
    print("Yahoo usable symbols:", int(coverage["usable"].sum()))
    raise SystemExit(0)

sig = pd.DataFrame(signal_rows).sort_values(["signal_date","symbol"]).reset_index(drop=True)
sig.to_csv(OUT_SIGNALS, index=False, encoding="utf-8-sig")


def summarize(x, prefix, mature_col):
    y = x[x[mature_col] == 1].copy()
    if len(y) == 0:
        return {}
    return {
        "n": len(y),
        "positive_close_n": int((y[f"{prefix}_close_ret_pct"] > 0).sum()),
        "positive_close_pct": 100 * (y[f"{prefix}_close_ret_pct"] > 0).mean(),
        "avg_close_ret_pct": y[f"{prefix}_close_ret_pct"].mean(),
        "median_close_ret_pct": y[f"{prefix}_close_ret_pct"].median(),
        "avg_max_gain_pct": y[f"{prefix}_max_gain_pct"].mean(),
        "median_max_gain_pct": y[f"{prefix}_max_gain_pct"].median(),
        "median_max_drawdown_pct": y[f"{prefix}_max_drawdown_pct"].median(),
        "worst_drawdown_pct": y[f"{prefix}_max_drawdown_pct"].min(),
        "hit20_n": int(y[f"{prefix}_hit20"].sum()),
        "hit20_pct": 100 * y[f"{prefix}_hit20"].mean(),
        "hit30_n": int(y[f"{prefix}_hit30"].sum()),
        "hit30_pct": 100 * y[f"{prefix}_hit30"].mean(),
        "hit50_n": int(y[f"{prefix}_hit50"].sum()),
        "hit50_pct": 100 * y[f"{prefix}_hit50"].mean(),
        "hit100_n": int(y[f"{prefix}_hit100"].sum()),
        "hit100_pct": 100 * y[f"{prefix}_hit100"].mean(),
    }


s3 = summarize(sig, "m3", "mature_3m")
s6 = summarize(sig, "m6", "mature_6m")
s12 = summarize(sig, "m12", "mature_1y")
s24 = summarize(sig, "m24", "mature_2y")

year_rows = []
for year in sorted(sig["year"].unique()):
    y = sig[sig["year"] == year]
    row = {"year": int(year), "signals": len(y)}
    for prefix, mature in [
        ("m3","mature_3m"),
        ("m6","mature_6m"),
        ("m12","mature_1y"),
        ("m24","mature_2y"),
    ]:
        sm = summarize(y, prefix, mature)
        for k, v in sm.items():
            row[f"{prefix}_{k}"] = v
    year_rows.append(row)

yearly = pd.DataFrame(year_rows)
yearly.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")

usable = int(coverage["usable"].sum())

print("\n=== EGX DATA COVERAGE ===")
print("TradingView symbols:", len(symbols))
print("Yahoo usable symbols:", usable)
print("Coverage %:", f"{100*usable/len(symbols):.2f}%" if symbols else "0%")

print("\n=== EGX INDEPENDENT STRONG SIGNALS ===")
print("Signals:", len(sig))
print("Distinct symbols:", sig["symbol"].nunique())
print("Distinct dates:", sig["signal_date"].nunique())

print("\n=== EGX 3 MONTHS / 63 SESSIONS ===")
for k, v in s3.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== EGX 6 MONTHS / 126 SESSIONS ===")
for k, v in s6.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== EGX 1 YEAR / 252 SESSIONS ===")
for k, v in s12.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== EGX 2 YEARS / 504 SESSIONS ===")
for k, v in s24.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== EGX YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== EGX TOP 20 BY 2Y MAX GAIN ===")
top = sig[sig["mature_2y"] == 1].sort_values("m24_max_gain_pct", ascending=False).head(20)
print(top[[
    "year","signal_date","symbol","raw_signal_close_egp",
    "m24_close_ret_pct","m24_max_gain_pct","m24_max_drawdown_pct"
]].to_string(index=False))

print("\n=== EGX WORST 20 BY 2Y CLOSE RETURN ===")
worst = sig[sig["mature_2y"] == 1].sort_values("m24_close_ret_pct").head(20)
print(worst[[
    "year","signal_date","symbol","raw_signal_close_egp",
    "m24_close_ret_pct","m24_max_gain_pct","m24_max_drawdown_pct"
]].to_string(index=False))

print("\n=== EGX CLEANING REJECTIONS ===")
for k, v in sorted(reject_counts.items(), key=lambda x: (-x[1], x[0])):
    print(f"{k}: {v}")

report = []
report.append("RAJIH — EGX ACCUMULATION STRONG BACKTEST")
report.append("="*100)
report.append(f"TradingView symbols: {len(symbols)}")
report.append(f"Yahoo usable symbols: {usable}")
report.append(f"Coverage pct: {100*usable/len(symbols) if symbols else 0:.2f}")
report.append(f"Independent STRONG signals: {len(sig)}")
report.append(f"Distinct symbols: {sig['symbol'].nunique()}")
report.append("")
report.append("3 MONTHS / 63 SESSIONS")
for k,v in s3.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("6 MONTHS / 126 SESSIONS")
for k,v in s6.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("1 YEAR / 252 SESSIONS")
for k,v in s12.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("2 YEARS / 504 SESSIONS")
for k,v in s24.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("TOP 20 BY 2Y MAX GAIN")
report.append(top[[
    "year","signal_date","symbol","raw_signal_close_egp",
    "m24_close_ret_pct","m24_max_gain_pct","m24_max_drawdown_pct"
]].to_string(index=False))
report.append("")
report.append("WORST 20 BY 2Y CLOSE RETURN")
report.append(worst[[
    "year","signal_date","symbol","raw_signal_close_egp",
    "m24_close_ret_pct","m24_max_gain_pct","m24_max_drawdown_pct"
]].to_string(index=False))
report.append("")
report.append("CLEANING REJECTIONS")
for k,v in sorted(reject_counts.items(), key=lambda x: (-x[1], x[0])):
    report.append(f"{k}: {v}")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_SIGNALS, OUT_YEARLY, OUT_COVERAGE, OUT_REPORT]:
    print(" ", p)
