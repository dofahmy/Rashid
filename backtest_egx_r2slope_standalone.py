#!/usr/bin/env python3
"""
EGX standalone R2 + Slope signal backtest (NO Claude STRONG)

Signal definition
-----------------
On every daily bar, using ONLY the previous 126 trading sessions:
    pre_trend_r2 >= 0.791694
    AND
    pre_trend_slope_pct >= 67.5062

This is a standalone signal. It does NOT require the Claude/Pine STRONG signal.

Independence
------------
To avoid firing every day while the same trend remains valid:
- first day the standalone rule becomes true = signal
- then ignore the same stock for 126 actual trading sessions

Entry / Exit
------------
Entry: signal-day close
TP: first touch of +50% using adjusted daily high
If +50% not reached, position remains OPEN in current-status output.

Universe
--------
Current EGX equities discovered from TradingView.
Daily price history from Yahoo .CA tickers.
The script prints coverage so we know how much of EGX is actually testable.

Light filters at signal date
----------------------------
- raw close >= 1 EGP
- prior-20d average traded value >= 1,000,000 EGP
- prior-20d average volume >= 10,000 shares

Outputs
-------
/data/egx_r2slope_standalone_all.csv
/data/egx_r2slope_standalone_yearly.csv
/data/egx_r2slope_standalone_2026.csv
/data/egx_r2slope_standalone_coverage.csv
/data/egx_r2slope_standalone_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time

DATA = Path("/data")

OUT_ALL = DATA / "egx_r2slope_standalone_all.csv"
OUT_YEAR = DATA / "egx_r2slope_standalone_yearly.csv"
OUT_2026 = DATA / "egx_r2slope_standalone_2026.csv"
OUT_COVER = DATA / "egx_r2slope_standalone_coverage.csv"
OUT_REPORT = DATA / "egx_r2slope_standalone_report.txt"

R2_MIN = 0.791694
SLOPE_MIN = 67.5062
LOOKBACK = 126
COOLDOWN = 126
TP = 50.0

MIN_PRICE = 1.0
MIN_ADV20 = 1_000_000.0
MIN_AVGVOL20 = 10_000.0

START = "2019-01-01"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})


def discover_egx():
    url = "https://scanner.tradingview.com/egypt/scan"
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "options": {"lang": "en"},
        "markets": ["egypt"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "description", "exchange"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 1000],
    }
    r = session.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    out = []
    seen = set()
    for item in data:
        s = item.get("s", "")
        ticker = s.split(":",1)[1] if ":" in s else s
        vals = item.get("d") or []
        name = vals[0] if len(vals) > 0 else ticker
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append({"ticker":ticker, "name":name, "tv_symbol":s})
    return out


def download(symbol):
    y = f"{symbol}.CA"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y}"
    params = {
        "period1": int(pd.Timestamp(START, tz="UTC").timestamp()),
        "period2": int(pd.Timestamp.now("UTC").timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"

    obj = r.json()
    res = ((obj.get("chart") or {}).get("result") or [])
    if not res:
        return None, "no result"

    z = res[0]
    ts = z.get("timestamp") or []
    q = (((z.get("indicators") or {}).get("quote") or [{}])[0])
    adj = ((z.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")

    if not ts:
        return None, "no timestamps"

    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "o": q.get("open"), "h": q.get("high"), "l": q.get("low"),
        "c": q.get("close"), "v": q.get("volume"),
    })
    df["adj_c"] = adj if adj and len(adj) == len(df) else df["c"]

    for c in ["o","h","l","c","v","adj_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["o","h","l","c","adj_c"])
    df = df[(df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) & (df["c"] > 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if len(df) < LOOKBACK + 21:
        return None, f"only {len(df)} bars"

    factor = df["adj_c"] / df["c"]
    df["ah"] = df["h"] * factor
    df["al"] = df["l"] * factor
    df["ac"] = df["adj_c"]

    df["raw_value"] = df["c"] * df["v"].fillna(0)
    df["adv20"] = df["raw_value"].rolling(20).mean()
    df["avgvol20"] = df["v"].rolling(20).mean()

    return df, ""


def rolling_linreg_metrics(values, window):
    arr = np.asarray(values, dtype=float)
    slopes = np.full(len(arr), np.nan)
    r2s = np.full(len(arr), np.nan)

    x = np.arange(window, dtype=float)
    xm = x.mean()
    xdev = x - xm
    xss = np.sum(xdev*xdev)

    for i in range(window, len(arr)):
        # PREVIOUS window only; current bar not included
        y = arr[i-window:i]
        if np.any(~np.isfinite(y)):
            continue
        ym = y.mean()
        ydev = y - ym
        slope = np.sum(xdev*ydev) / xss
        pred = ym + slope*xdev
        ss_res = np.sum((y-pred)**2)
        ss_tot = np.sum(ydev*ydev)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        slope_pct = 100 * (slope*(window-1)) / ym if ym != 0 else np.nan
        slopes[i] = slope_pct
        r2s[i] = r2

    return slopes, r2s


symbols = discover_egx()
print("\nEGX STANDALONE R2 + SLOPE")
print("TradingView EGX equities:", len(symbols))
print(f"Rule: R2 >= {R2_MIN} AND slope >= {SLOPE_MIN}")
print("NO Claude STRONG required.")

coverage = []
rows = []
rejects = {}

for n, info in enumerate(symbols, 1):
    symbol = info["ticker"]
    try:
        df, err = download(symbol)
        if df is None:
            coverage.append({**info, "usable":0, "bars":0, "reason":err})
            continue

        coverage.append({**info, "usable":1, "bars":len(df), "reason":""})

        slope, r2 = rolling_linreg_metrics(df["ac"].to_numpy(float), LOOKBACK)
        df["pre_trend_slope_pct"] = slope
        df["pre_trend_r2"] = r2

        rule = (
            (df["pre_trend_r2"] >= R2_MIN) &
            (df["pre_trend_slope_pct"] >= SLOPE_MIN)
        )

        # first TRUE after FALSE = standalone signal episode start
        new_rule = rule & (~rule.shift(1).fillna(False))

        last_kept = -10**9
        for i in df.index[new_rule]:
            if i - last_kept < COOLDOWN:
                rejects["COOLDOWN_126"] = rejects.get("COOLDOWN_126",0)+1
                continue

            # basic Egypt filters using prior day rolling values
            if df.at[i,"c"] < MIN_PRICE:
                rejects["PRICE"] = rejects.get("PRICE",0)+1
                continue
            if i < 1 or pd.isna(df.at[i-1,"adv20"]) or df.at[i-1,"adv20"] < MIN_ADV20:
                rejects["LIQUIDITY_VALUE"] = rejects.get("LIQUIDITY_VALUE",0)+1
                continue
            if pd.isna(df.at[i-1,"avgvol20"]) or df.at[i-1,"avgvol20"] < MIN_AVGVOL20:
                rejects["LIQUIDITY_VOLUME"] = rejects.get("LIQUIDITY_VOLUME",0)+1
                continue

            last_kept = i

            entry = float(df.at[i,"c"])
            future = df.iloc[i+1:].copy()
            if future.empty:
                continue

            tp_price = entry*1.50
            hits = future.index[future["ah"] >= tp_price].tolist()
            tp_hit = len(hits) > 0

            latest = df.iloc[-1]
            latest_ret = 100*(float(latest["ac"])/entry - 1)
            max_gain = 100*(float(future["ah"].max())/entry - 1)
            max_dd = 100*(float(future["al"].min())/entry - 1)

            if tp_hit:
                j = hits[0]
                sessions_to_tp = int(j-i)
                tp_date = df.at[j,"date"]
                mae_to_tp = 100*(float(df.loc[i+1:j,"al"].min())/entry - 1)
                status = "CLOSED_TP50"
            else:
                sessions_to_tp = np.nan
                tp_date = pd.NaT
                mae_to_tp = max_dd
                status = "OPEN"

            row = {
                "signal_date": df.at[i,"date"].date().isoformat(),
                "year": int(df.at[i,"date"].year),
                "symbol": symbol,
                "name": info["name"],
                "signal_price": entry,
                "pre_trend_r2": float(df.at[i,"pre_trend_r2"]),
                "pre_trend_slope_pct": float(df.at[i,"pre_trend_slope_pct"]),
                "tp50_hit": int(tp_hit),
                "tp50_date": tp_date.date().isoformat() if tp_hit else "",
                "sessions_to_tp50": sessions_to_tp,
                "status": status,
                "latest_date": latest["date"].date().isoformat(),
                "latest_close": float(latest["c"]),
                "latest_return_pct": latest_ret,
                "max_gain_to_now_pct": max_gain,
                "mae_before_tp_or_now_pct": mae_to_tp,
            }

            for label,h in [("3m",63),("6m",126),("1y",252),("2y",504)]:
                if i+h < len(df):
                    row[f"{label}_mature"] = 1
                    row[f"{label}_close_ret_pct"] = 100*(float(df.at[i+h,"ac"])/entry - 1)
                    row[f"{label}_tp50_hit"] = int(tp_hit and sessions_to_tp <= h)
                else:
                    row[f"{label}_mature"] = 0
                    row[f"{label}_close_ret_pct"] = np.nan
                    row[f"{label}_tp50_hit"] = np.nan

            rows.append(row)

        if n % 25 == 0 or n == len(symbols):
            print(f"[{n}/{len(symbols)}] usable={sum(x['usable'] for x in coverage)} signals={len(rows)}")
        time.sleep(0.03)

    except Exception as e:
        coverage.append({**info, "usable":0, "bars":0, "reason":f"ERROR: {e}"})

cover = pd.DataFrame(coverage)
cover.to_csv(OUT_COVER, index=False, encoding="utf-8-sig")

out = pd.DataFrame(rows)
if out.empty:
    raise SystemExit("No standalone R2+Slope signals found.")

out = out.sort_values(["signal_date","symbol"]).reset_index(drop=True)
out.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

year_rows = []
for year,g in out.groupby("year"):
    m6 = g[g["6m_mature"]==1]
    year_rows.append({
        "year": int(year),
        "signals": len(g),
        "closed_tp50_n": int((g["status"]=="CLOSED_TP50").sum()),
        "open_now_n": int((g["status"]=="OPEN").sum()),
        "tp50_ever_pct": 100*g["tp50_hit"].mean(),
        "median_sessions_to_tp50": g.loc[g["tp50_hit"]==1,"sessions_to_tp50"].median(),
        "mature6_n": len(m6),
        "mature6_tp50_pct": 100*m6["6m_tp50_hit"].mean() if len(m6) else np.nan,
    })

yearly = pd.DataFrame(year_rows).sort_values("year")
yearly.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

y2026 = out[out["year"]==2026].copy()
y2026.to_csv(OUT_2026, index=False, encoding="utf-8-sig")

usable = int(cover["usable"].sum())

print("\n=== DATA COVERAGE ===")
print("TradingView symbols:", len(symbols))
print("Yahoo usable:", usable)
print("Coverage %:", f"{100*usable/len(symbols):.2f}%")

print("\n=== STANDALONE R2+SLOPE SAMPLE ===")
print("Signals:", len(out))
print("Distinct symbols:", out["symbol"].nunique())
print("TP50 ever:", int(out["tp50_hit"].sum()), f"({100*out['tp50_hit'].mean():.2f}%)")
print("Open now:", int((out["status"]=="OPEN").sum()))
if out["tp50_hit"].sum():
    print("Median sessions to TP50:", out.loc[out["tp50_hit"]==1,"sessions_to_tp50"].median())
    print("Median MAE before TP50:", f"{out.loc[out['tp50_hit']==1,'mae_before_tp_or_now_pct'].median():.2f}%")

print("\n=== YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== 2026 STATUS ===")
print("2026 signals:", len(y2026))
print("2026 closed at +50%:", int((y2026["status"]=="CLOSED_TP50").sum()))
print("2026 still open:", int((y2026["status"]=="OPEN").sum()))

cols = [
    "signal_date","symbol","signal_price",
    "pre_trend_r2","pre_trend_slope_pct",
    "status","tp50_date","sessions_to_tp50",
    "latest_return_pct","max_gain_to_now_pct","mae_before_tp_or_now_pct"
]
print(y2026[cols].to_string(index=False))

print("\n=== CLEANING / COOLDOWN REJECTIONS ===")
for k,v in sorted(rejects.items(), key=lambda x:(-x[1],x[0])):
    print(f"{k}: {v}")

report = []
report.append("EGX STANDALONE R2+SLOPE SIGNAL — NO CLAUDE STRONG")
report.append("="*120)
report.append(f"Rule: R2 >= {R2_MIN} AND slope >= {SLOPE_MIN}")
report.append(f"TradingView symbols: {len(symbols)}")
report.append(f"Yahoo usable: {usable} ({100*usable/len(symbols):.2f}%)")
report.append(f"Signals: {len(out)}")
report.append(f"Distinct symbols: {out['symbol'].nunique()}")
report.append(f"TP50 ever: {int(out['tp50_hit'].sum())}/{len(out)} = {100*out['tp50_hit'].mean():.2f}%")
report.append(f"Open now: {int((out['status']=='OPEN').sum())}")
if out["tp50_hit"].sum():
    report.append(f"Median sessions to TP50: {out.loc[out['tp50_hit']==1,'sessions_to_tp50'].median()}")
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("2026 STATUS")
report.append(y2026[cols].to_string(index=False))
report.append("")
report.append("REJECTIONS")
for k,v in sorted(rejects.items(), key=lambda x:(-x[1],x[0])):
    report.append(f"{k}: {v}")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_ALL,OUT_YEAR,OUT_2026,OUT_COVER,OUT_REPORT]:
    print(" ",p)
