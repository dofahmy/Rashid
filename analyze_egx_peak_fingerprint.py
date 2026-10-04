#!/usr/bin/env python3
"""
EGX Peak Fingerprint Research
=============================

Purpose
-------
Extract the "fingerprint of the top" for standalone R2+Slope signals in Egypt.

Input signals:
    /data/egx_r2slope_standalone_all.csv

For every signal:
1) Download Yahoo daily history for SYMBOL.CA.
2) Entry = signal-day raw close.
3) Look forward up to 252 trading sessions (about 1 year) or to latest available data.
4) Identify the post-entry PEAK as the day with the highest adjusted HIGH.
5) Recalculate daily trend metrics using ONLY the previous 126 sessions:
       trend_slope_pct
       trend_r2
6) Measure the behavior at:
       peak day
       -1, -3, -5, -10, -20 sessions before peak
7) Measure "rollover / divergence" features:
       - slope change vs 5 / 10 / 20 sessions earlier
       - R2 change vs 5 / 10 / 20 sessions earlier
       - whether price makes a new high while slope fails to make a new high
       - whether price makes a new high while R2 falls
       - slope drawdown from its rolling post-entry peak
       - R2 drawdown from its rolling post-entry peak
8) Summarize all signals and also a "meaningful winners" subset:
       peak gain >= +50%

Outputs
-------
/data/egx_peak_fingerprint_all.csv
/data/egx_peak_fingerprint_winners50.csv
/data/egx_peak_fingerprint_summary.csv
/data/egx_peak_fingerprint_2026.csv
/data/egx_peak_fingerprint_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time

DATA = Path("/data")
SIGNALS = DATA / "egx_r2slope_standalone_all.csv"

OUT_ALL = DATA / "egx_peak_fingerprint_all.csv"
OUT_WIN = DATA / "egx_peak_fingerprint_winners50.csv"
OUT_SUM = DATA / "egx_peak_fingerprint_summary.csv"
OUT_2026 = DATA / "egx_peak_fingerprint_2026.csv"
OUT_REPORT = DATA / "egx_peak_fingerprint_report.txt"

LOOKBACK = 126
HORIZON = 252
OFFSETS = [0, 1, 3, 5, 10, 20]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})

def download(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.CA"
    params = {
        "period1": int(pd.Timestamp("2019-01-01", tz="UTC").timestamp()),
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
    df = df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    factor = df["adj_c"] / df["c"]
    df["ah"] = df["h"] * factor
    df["al"] = df["l"] * factor
    df["ac"] = df["adj_c"]
    return df

def add_trend_metrics(df):
    n = len(df)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    x = np.arange(LOOKBACK, dtype=float)
    xm = x.mean()
    xd = x - xm
    xss = np.sum(xd*xd)
    ac = df["ac"].to_numpy(float)

    for i in range(LOOKBACK, n):
        y = ac[i-LOOKBACK:i]  # strictly prior 126 sessions
        if np.any(~np.isfinite(y)):
            continue
        ym = y.mean()
        yd = y - ym
        b = np.sum(xd*yd)/xss
        pred = ym + b*xd
        ss_res = np.sum((y-pred)**2)
        ss_tot = np.sum(yd*yd)
        r2[i] = 1-ss_res/ss_tot if ss_tot > 0 else 0.0
        slope[i] = 100*(b*(LOOKBACK-1))/ym if ym else np.nan

    out = df.copy()
    out["trend_slope"] = slope
    out["trend_r2"] = r2
    return out

signals = pd.read_csv(SIGNALS, dtype={"symbol": str})
signals["signal_date"] = pd.to_datetime(signals["signal_date"])

symbols = sorted(signals["symbol"].unique())
histories = {}

print("\nEGX PEAK FINGERPRINT RESEARCH")
print("Signals:", len(signals), "Symbols:", len(symbols))

for n, s in enumerate(symbols, 1):
    df = download(s)
    if df is not None and len(df) > LOOKBACK:
        histories[s] = add_trend_metrics(df)
    if n % 25 == 0 or n == len(symbols):
        print(f"[{n}/{len(symbols)}] usable={len(histories)}")
    time.sleep(0.02)

rows = []

for k, sig in enumerate(signals.itertuples(index=False), 1):
    symbol = sig.symbol
    df = histories.get(symbol)
    if df is None:
        continue

    d = pd.Timestamp(sig.signal_date).normalize()
    ids = df.index[df["date"] == d].tolist()
    if not ids:
        continue
    i = ids[-1]
    if i+1 >= len(df):
        continue

    entry = float(sig.signal_price)
    end = min(len(df)-1, i+HORIZON)
    seg = df.loc[i+1:end].copy()
    if seg.empty:
        continue

    peak_idx = int(seg["ah"].idxmax())
    peak_date = df.at[peak_idx, "date"]
    peak_price = float(df.at[peak_idx, "ah"])
    peak_gain = 100*(peak_price/entry - 1)
    sessions_to_peak = peak_idx - i

    # post-entry rolling peaks in slope/r2 up to the price peak
    prepeak = df.loc[i+1:peak_idx].copy()
    peak_slope_seen = prepeak["trend_slope"].max()
    peak_r2_seen = prepeak["trend_r2"].max()

    row = {
        "signal_date": d.date().isoformat(),
        "year": int(d.year),
        "symbol": symbol,
        "signal_price": entry,
        "peak_date": peak_date.date().isoformat(),
        "sessions_to_peak": int(sessions_to_peak),
        "peak_gain_pct": peak_gain,
        "peak_price": peak_price,
        "postentry_max_slope_before_peak": float(peak_slope_seen) if np.isfinite(peak_slope_seen) else np.nan,
        "postentry_max_r2_before_peak": float(peak_r2_seen) if np.isfinite(peak_r2_seen) else np.nan,
    }

    # exact snapshots before peak
    for off in OFFSETS:
        j = peak_idx - off
        key = "peak" if off == 0 else f"m{off}"
        if j >= 0:
            row[f"{key}_slope"] = float(df.at[j,"trend_slope"]) if np.isfinite(df.at[j,"trend_slope"]) else np.nan
            row[f"{key}_r2"] = float(df.at[j,"trend_r2"]) if np.isfinite(df.at[j,"trend_r2"]) else np.nan
            row[f"{key}_close"] = float(df.at[j,"ac"])
        else:
            row[f"{key}_slope"] = np.nan
            row[f"{key}_r2"] = np.nan
            row[f"{key}_close"] = np.nan

    # changes into the peak
    for off in [5,10,20]:
        row[f"slope_change_{off}_to_peak"] = row["peak_slope"] - row[f"m{off}_slope"] if np.isfinite(row["peak_slope"]) and np.isfinite(row[f"m{off}_slope"]) else np.nan
        row[f"r2_change_{off}_to_peak"] = row["peak_r2"] - row[f"m{off}_r2"] if np.isfinite(row["peak_r2"]) and np.isfinite(row[f"m{off}_r2"]) else np.nan
        row[f"price_change_{off}_to_peak_pct"] = 100*(row["peak_close"]/row[f"m{off}_close"]-1) if np.isfinite(row["peak_close"]) and np.isfinite(row[f"m{off}_close"]) and row[f"m{off}_close"] != 0 else np.nan

    # drawdown from metric peaks at price peak
    row["slope_drawdown_from_metric_peak_pct"] = (
        100*(row["peak_slope"]/peak_slope_seen - 1)
        if np.isfinite(row["peak_slope"]) and np.isfinite(peak_slope_seen) and peak_slope_seen != 0
        else np.nan
    )
    row["r2_drop_from_metric_peak"] = (
        float(peak_r2_seen - row["peak_r2"])
        if np.isfinite(row["peak_r2"]) and np.isfinite(peak_r2_seen)
        else np.nan
    )

    # divergence flags: price rises into top while metric weakens
    for off in [5,10,20]:
        price_up = row[f"price_change_{off}_to_peak_pct"] > 0 if np.isfinite(row[f"price_change_{off}_to_peak_pct"]) else False
        slope_down = row[f"slope_change_{off}_to_peak"] < 0 if np.isfinite(row[f"slope_change_{off}_to_peak"]) else False
        r2_down = row[f"r2_change_{off}_to_peak"] < 0 if np.isfinite(row[f"r2_change_{off}_to_peak"]) else False
        row[f"price_up_slope_down_{off}"] = int(price_up and slope_down)
        row[f"price_up_r2_down_{off}"] = int(price_up and r2_down)
        row[f"price_up_both_down_{off}"] = int(price_up and slope_down and r2_down)

    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

win = out[out["peak_gain_pct"] >= 50].copy()
win.to_csv(OUT_WIN, index=False, encoding="utf-8-sig")

y2026 = out[out["year"] == 2026].copy()
y2026.to_csv(OUT_2026, index=False, encoding="utf-8-sig")

# summaries for all and winners50
summary_rows = []

def add_summary(label, g):
    if g.empty:
        return
    r = {
        "group": label,
        "n": len(g),
        "median_peak_gain_pct": g["peak_gain_pct"].median(),
        "median_sessions_to_peak": g["sessions_to_peak"].median(),
        "median_peak_slope": g["peak_slope"].median(),
        "median_peak_r2": g["peak_r2"].median(),
        "median_slope_dd_from_metric_peak_pct": g["slope_drawdown_from_metric_peak_pct"].median(),
        "median_r2_drop_from_metric_peak": g["r2_drop_from_metric_peak"].median(),
    }
    for off in [5,10,20]:
        r[f"pct_price_up_slope_down_{off}"] = 100*g[f"price_up_slope_down_{off}"].mean()
        r[f"pct_price_up_r2_down_{off}"] = 100*g[f"price_up_r2_down_{off}"].mean()
        r[f"pct_price_up_both_down_{off}"] = 100*g[f"price_up_both_down_{off}"].mean()
        r[f"median_slope_change_{off}_to_peak"] = g[f"slope_change_{off}_to_peak"].median()
        r[f"median_r2_change_{off}_to_peak"] = g[f"r2_change_{off}_to_peak"].median()
    summary_rows.append(r)

add_summary("ALL", out)
add_summary("PEAK_GAIN_GE_50", win)
add_summary("2026_ALL", y2026)
add_summary("2026_PEAK_GAIN_GE_50", y2026[y2026["peak_gain_pct"] >= 50])

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

print("\n=== PEAK FINGERPRINT SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== WINNERS >= +50% : MOST COMMON DIVERGENCES ===")
if not win.empty:
    for off in [5,10,20]:
        print(
            f"{off} sessions before peak -> "
            f"price up / slope down: {100*win[f'price_up_slope_down_{off}'].mean():.2f}% | "
            f"price up / R2 down: {100*win[f'price_up_r2_down_{off}'].mean():.2f}% | "
            f"price up / BOTH down: {100*win[f'price_up_both_down_{off}'].mean():.2f}%"
        )

print("\n=== 2026 TOPS ===")
cols = [
    "signal_date","symbol","peak_date","sessions_to_peak","peak_gain_pct",
    "peak_slope","peak_r2",
    "slope_drawdown_from_metric_peak_pct","r2_drop_from_metric_peak",
    "price_up_both_down_5","price_up_both_down_10","price_up_both_down_20"
]
print(y2026[cols].sort_values("peak_gain_pct",ascending=False).to_string(index=False))

report = []
report.append("EGX PEAK FINGERPRINT RESEARCH")
report.append("="*120)
report.append(f"Signals analyzed: {len(out)}")
report.append(f"Winners with peak gain >=50%: {len(win)}")
report.append("")
report.append("SUMMARY")
report.append(summary.to_string(index=False))
report.append("")
report.append("2026 TOPS")
report.append(y2026[cols].sort_values("peak_gain_pct",ascending=False).to_string(index=False))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_ALL,OUT_WIN,OUT_SUM,OUT_2026,OUT_REPORT]:
    print(" ",p)
