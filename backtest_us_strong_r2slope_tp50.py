#!/usr/bin/env python3
"""
US STRONG — apply the SAME Egypt-derived pre-signal rule + TP50

Rule learned from EGX:
    pre_trend_r2 >= 0.791694
    pre_trend_slope_pct >= 67.5062

Entry:
    STRONG signal close

Exit:
    Take profit at +50% when daily adjusted HIGH first touches +50%.
    Otherwise position remains open in the live-status view.

Input:
    /data/accum_strong_clean_signals.csv

Outputs:
    /data/us_strong_r2slope_tp50_all.csv
    /data/us_strong_r2slope_tp50_yearly.csv
    /data/us_strong_r2slope_tp50_2026.csv
    /data/us_strong_r2slope_tp50_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time

DATA = Path("/data")
INFILE = DATA / "accum_strong_clean_signals.csv"

OUT_ALL = DATA / "us_strong_r2slope_tp50_all.csv"
OUT_YEAR = DATA / "us_strong_r2slope_tp50_yearly.csv"
OUT_2026 = DATA / "us_strong_r2slope_tp50_2026.csv"
OUT_REPORT = DATA / "us_strong_r2slope_tp50_report.txt"

R2_MIN = 0.791694
SLOPE_MIN = 67.5062
LOOKBACK = 126
TP_PCT = 50.0

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})

def download(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": int(pd.Timestamp("2023-01-01", tz="UTC").timestamp()),
        "period2": int(pd.Timestamp.now("UTC").timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    obj = r.json()
    res = ((obj.get("chart") or {}).get("result") or [])
    if not res:
        return None
    z = res[0]
    ts = z.get("timestamp") or []
    q = (((z.get("indicators") or {}).get("quote") or [{}])[0])
    adj = ((z.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not ts:
        return None

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

    factor = df["adj_c"] / df["c"]
    df["ah"] = df["h"] * factor
    df["al"] = df["l"] * factor
    df["ac"] = df["adj_c"]
    return df

def linreg_stats(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 5 or np.any(~np.isfinite(y)):
        return np.nan, np.nan
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    pred = coef[0] * x + coef[1]
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    slope_pct = 100.0 * (coef[0] * (len(y)-1)) / np.mean(y)
    return float(slope_pct), float(r2)

src = pd.read_csv(INFILE, dtype={"symbol": str})
src["signal_date"] = pd.to_datetime(src["signal_date"])
src = src.sort_values(["symbol","signal_date"]).reset_index(drop=True)

symbols = sorted(src["symbol"].dropna().astype(str).unique())
histories = {}
rows = []

print("\nUS STRONG — EGX R2/SLOPE RULE + TP50")
print("Source STRONG signals:", len(src))
print("Distinct symbols:", len(symbols))
print("Rule: R2 >= %.6f AND slope >= %.4f" % (R2_MIN, SLOPE_MIN))

for n, symbol in enumerate(symbols, 1):
    histories[symbol] = download(symbol)
    if n % 20 == 0 or n == len(symbols):
        usable = sum(v is not None and len(v) >= LOOKBACK+1 for v in histories.values())
        print(f"[{n}/{len(symbols)}] usable={usable}")
    time.sleep(0.03)

matched = 0

for r in src.itertuples(index=False):
    symbol = str(r.symbol)
    hist = histories.get(symbol)
    if hist is None or hist.empty:
        continue

    d = pd.Timestamp(r.signal_date).normalize()

    prev = hist[hist["date"] < d].copy()
    if len(prev) < LOOKBACK:
        continue

    pre = prev.tail(LOOKBACK)
    slope, r2 = linreg_stats(pre["ac"].to_numpy(float))

    if not (np.isfinite(r2) and np.isfinite(slope)):
        continue
    if r2 < R2_MIN or slope < SLOPE_MIN:
        continue

    idxs = hist.index[hist["date"] == d].tolist()
    if not idxs:
        prev_idx = hist.index[hist["date"] <= d].tolist()
        if not prev_idx:
            continue
        i = prev_idx[-1]
    else:
        i = idxs[-1]

    if hasattr(r, "raw_signal_close"):
        entry = float(r.raw_signal_close)
    elif hasattr(r, "signal_price"):
        entry = float(r.signal_price)
    else:
        entry = float(hist.at[i, "c"])

    future = hist.iloc[i+1:].copy()
    if future.empty:
        continue

    tp_price = entry * 1.50
    tp_hits = future.index[future["ah"] >= tp_price].tolist()
    tp_hit = len(tp_hits) > 0

    latest = hist.iloc[-1]
    latest_ret = 100 * (float(latest["ac"]) / entry - 1)

    if tp_hit:
        j = tp_hits[0]
        sessions_to_tp = int(j-i)
        tp_date = hist.at[j, "date"]
        mae = 100 * (float(hist.loc[i+1:j, "al"].min()) / entry - 1)
        status = "CLOSED_TP50"
    else:
        sessions_to_tp = np.nan
        tp_date = pd.NaT
        mae = 100 * (float(future["al"].min()) / entry - 1)
        status = "OPEN"

    max_gain_now = 100 * (float(future["ah"].max()) / entry - 1)

    row = {
        "signal_date": d.date().isoformat(),
        "year": int(d.year),
        "symbol": symbol,
        "signal_price": entry,
        "pre_trend_r2": r2,
        "pre_trend_slope_pct": slope,
        "tp50_hit": int(tp_hit),
        "tp50_date": tp_date.date().isoformat() if tp_hit else "",
        "sessions_to_tp50": sessions_to_tp,
        "status": status,
        "latest_date": latest["date"].date().isoformat(),
        "latest_close": float(latest["c"]),
        "latest_return_pct": latest_ret,
        "max_gain_to_now_pct": max_gain_now,
        "mae_before_tp_or_now_pct": mae,
    }

    for label, h in [("3m",63),("6m",126),("1y",252)]:
        if i+h < len(hist):
            row[f"{label}_mature"] = 1
            row[f"{label}_close_ret_pct"] = 100*(float(hist.at[i+h,"ac"])/entry - 1)
            row[f"{label}_tp50_hit"] = int(tp_hit and sessions_to_tp <= h)
        else:
            row[f"{label}_mature"] = 0
            row[f"{label}_close_ret_pct"] = np.nan
            row[f"{label}_tp50_hit"] = np.nan

    rows.append(row)
    matched += 1

out = pd.DataFrame(rows)
if out.empty:
    print("\nNo US signals matched the rule.")
    raise SystemExit(0)

out = out.sort_values(["signal_date","symbol"]).reset_index(drop=True)
out.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

year_rows = []
for year, g in out.groupby("year"):
    mature6 = g[g["6m_mature"] == 1]
    year_rows.append({
        "year": int(year),
        "signals": len(g),
        "tp50_ever_n": int(g["tp50_hit"].sum()),
        "tp50_ever_pct": 100*g["tp50_hit"].mean(),
        "open_now_n": int((g["status"]=="OPEN").sum()),
        "median_sessions_to_tp50": g.loc[g["tp50_hit"]==1,"sessions_to_tp50"].median(),
        "mature6_n": len(mature6),
        "mature6_tp50_pct": 100*mature6["6m_tp50_hit"].mean() if len(mature6) else np.nan,
    })

yearly = pd.DataFrame(year_rows).sort_values("year")
yearly.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

y2026 = out[out["year"] == 2026].copy()
y2026.to_csv(OUT_2026, index=False, encoding="utf-8-sig")

print("\n=== US RULE SAMPLE ===")
print("Matched signals:", len(out))
print("Distinct symbols:", out["symbol"].nunique())
print("TP50 ever:", int(out["tp50_hit"].sum()), f"({100*out['tp50_hit'].mean():.2f}%)")
print("Open now:", int((out["status"]=="OPEN").sum()))
if out["tp50_hit"].sum():
    print("Median sessions to TP50:", out.loc[out["tp50_hit"]==1,"sessions_to_tp50"].median())
    print("Median MAE before TP50:", f"{out.loc[out['tp50_hit']==1,'mae_before_tp_or_now_pct'].median():.2f}%")

print("\n=== US YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== US 2026 STATUS ===")
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

print("\n=== ALL MATCHED US SIGNALS ===")
print(out[cols].to_string(index=False))

report = []
report.append("US STRONG — EGX R2/SLOPE RULE + TP50")
report.append("="*110)
report.append(f"Rule: pre_trend_r2 >= {R2_MIN} AND pre_trend_slope_pct >= {SLOPE_MIN}")
report.append(f"Matched signals: {len(out)}")
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
report.append("ALL MATCHED SIGNALS")
report.append(out[cols].to_string(index=False))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_ALL, OUT_YEAR, OUT_2026, OUT_REPORT]:
    print(" ", p)
