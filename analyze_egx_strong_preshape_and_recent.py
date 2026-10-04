#!/usr/bin/env python3
"""
EGX STRONG — 6-month pre-signal shape study + recent signals

Purpose
-------
Study what the stock looked like during the 126 trading sessions (~6 months)
BEFORE each STRONG signal, then relate that pre-signal shape to the signal's
subsequent performance.

Primary success definition:
  SUCCESS50 = touched +50% within 126 sessions after STRONG.

Also reports:
  hit +20%, +30%, +50%, +100%
  positive close after 6 months
  6M close return
  6M max gain

Pre-signal features include:
  - 6M return
  - 3M return
  - 1M return
  - 6M range %
  - realized daily volatility
  - ATR14 %
  - distance from 6M high
  - distance from 6M low
  - slope / linear trend quality (R²)
  - fraction of closes inside middle 50% of 6M range
  - 20d vs 126d volume ratio
  - 20d average traded value
  - 20d return volatility
  - drawdown from 6M peak

Regime labels:
  LOW_VOL_SIDEWAYS
  HIGH_VOL_SIDEWAYS
  QUIET_UPTREND
  STRONG_UPTREND
  DOWNTREND
  RECOVERY_FROM_LOW
  NEAR_6M_HIGH
  MIXED

The regime thresholds are DATA-RELATIVE, based on quartiles/medians across
historical mature signals, to avoid hard-coding US-style volatility levels.

Recent signals:
  Signals in the latest 126 trading-session-equivalent period available in the
  result set are listed with their pre-signal regime and historical SUCCESS50
  rate for that regime.

Inputs
------
/data/egx_accum_strong_1y2y_signals.csv

Outputs
-------
/data/egx_strong_preshape_all.csv
/data/egx_strong_preshape_feature_compare.csv
/data/egx_strong_preshape_regimes.csv
/data/egx_strong_recent_signals_ranked.csv
/data/egx_strong_preshape_report.txt

Run
---
cd /app
python analyze_egx_strong_preshape_and_recent.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path("/data")
INFILE = DATA / "egx_accum_strong_1y2y_signals.csv"

OUT_ALL = DATA / "egx_strong_preshape_all.csv"
OUT_FEATURES = DATA / "egx_strong_preshape_feature_compare.csv"
OUT_REGIMES = DATA / "egx_strong_preshape_regimes.csv"
OUT_RECENT = DATA / "egx_strong_recent_signals_ranked.csv"
OUT_REPORT = DATA / "egx_strong_preshape_report.txt"

LOOKBACK = 126
MIN_PRE_BARS = 110
RECENT_CALENDAR_DAYS = 220  # roughly >= 126 trading sessions, safe display window

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})


def yahoo_download(symbol, start="2019-01-01"):
    y = f"{symbol}.CA"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y}"
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
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
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def safe_ret(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return np.nan
    return 100.0 * (a / b - 1.0)


def linreg_stats(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 5 or np.any(~np.isfinite(y)):
        return np.nan, np.nan
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    pred = coef[0] * x + coef[1]
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # normalized slope over the full window, %
    slope_pct = 100.0 * (coef[0] * (len(y)-1)) / np.mean(y)
    return float(slope_pct), float(r2)


def pre_features(hist, signal_date):
    before = hist[hist["date"] < signal_date].copy()
    if len(before) < MIN_PRE_BARS:
        return None

    pre = before.tail(LOOKBACK).copy()
    if len(pre) < MIN_PRE_BARS:
        return None

    # adjusted OHLC
    factor = pre["adj_c"] / pre["c"]
    pre["ah"] = pre["h"] * factor
    pre["al"] = pre["l"] * factor
    pre["ac"] = pre["adj_c"]

    ac = pre["ac"].to_numpy(float)
    ah = pre["ah"].to_numpy(float)
    al = pre["al"].to_numpy(float)
    vol = pre["v"].fillna(0).to_numpy(float)

    ret1 = pd.Series(ac).pct_change().dropna().to_numpy(float)
    realized_vol = float(np.std(ret1, ddof=0) * 100) if len(ret1) else np.nan

    high126 = float(np.max(ah))
    low126 = float(np.min(al))
    last = float(ac[-1])

    ret126 = safe_ret(last, ac[0])
    ret63 = safe_ret(last, ac[-64]) if len(ac) >= 64 else np.nan
    ret21 = safe_ret(last, ac[-22]) if len(ac) >= 22 else np.nan

    range126 = safe_ret(high126, low126)
    dist_high = safe_ret(last, high126)
    dist_low = safe_ret(last, low126)

    # max drawdown from running peak
    running_peak = np.maximum.accumulate(ac)
    dd = 100.0 * (ac / running_peak - 1.0)
    max_dd = float(np.min(dd))

    slope_pct, trend_r2 = linreg_stats(ac)

    # ATR14 on adjusted OHLC, last value
    prev = pd.Series(ac).shift(1).to_numpy()
    tr = np.nanmax(np.vstack([
        ah - al,
        np.abs(ah - prev),
        np.abs(al - prev),
    ]), axis=0)
    atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]
    atr14_pct = float(100.0 * atr14 / last)

    mid_low = low126 + 0.25*(high126-low126)
    mid_high = low126 + 0.75*(high126-low126)
    middle50_frac = float(np.mean((ac >= mid_low) & (ac <= mid_high)))

    avgvol126 = float(np.mean(vol))
    avgvol20 = float(np.mean(vol[-20:]))
    vol_ratio20_126 = avgvol20 / avgvol126 if avgvol126 > 0 else np.nan

    raw_value = pre["c"].to_numpy(float) * vol
    adv20 = float(np.mean(raw_value[-20:]))

    ret20std = float(pd.Series(ac).pct_change().tail(20).std(ddof=0) * 100)

    return {
        "pre_6m_ret_pct": ret126,
        "pre_3m_ret_pct": ret63,
        "pre_1m_ret_pct": ret21,
        "pre_6m_range_pct": range126,
        "pre_realized_vol_pct": realized_vol,
        "pre_atr14_pct": atr14_pct,
        "pre_dist_6m_high_pct": dist_high,
        "pre_dist_6m_low_pct": dist_low,
        "pre_max_drawdown_pct": max_dd,
        "pre_trend_slope_pct": slope_pct,
        "pre_trend_r2": trend_r2,
        "pre_middle50_frac": middle50_frac,
        "pre_vol_ratio20_126": vol_ratio20_126,
        "pre_adv20": adv20,
        "pre_ret20_std_pct": ret20std,
    }


signals = pd.read_csv(INFILE, dtype={"symbol": str})
signals["signal_date"] = pd.to_datetime(signals["signal_date"])
signals = signals.sort_values(["symbol","signal_date"]).reset_index(drop=True)

symbols = sorted(signals["symbol"].dropna().astype(str).unique())
histories = {}

print("\nEGX STRONG — PRE-SIGNAL SHAPE STUDY")
print("Signals:", len(signals))
print("Symbols:", len(symbols))
print("Downloading histories...")

for n, symbol in enumerate(symbols, 1):
    hist = yahoo_download(symbol)
    histories[symbol] = hist
    if n % 20 == 0 or n == len(symbols):
        usable = sum(v is not None and len(v) >= MIN_PRE_BARS for v in histories.values())
        print(f"[{n}/{len(symbols)}] usable histories={usable}")
    time.sleep(0.03)

rows = []
for r in signals.itertuples(index=False):
    symbol = str(r.symbol)
    hist = histories.get(symbol)
    if hist is None:
        continue

    f = pre_features(hist, r.signal_date)
    if f is None:
        continue

    row = {
        "signal_date": r.signal_date.date().isoformat(),
        "year": int(r.year),
        "symbol": symbol,
        "name": getattr(r, "name", ""),
        "signal_price": getattr(r, "raw_signal_close_egp", np.nan),

        "mature_6m": int(getattr(r, "mature_6m", 0)),
        "m6_close_ret_pct": getattr(r, "m6_close_ret_pct", np.nan),
        "m6_max_gain_pct": getattr(r, "m6_max_gain_pct", np.nan),
        "m6_max_drawdown_pct": getattr(r, "m6_max_drawdown_pct", np.nan),
        "m6_hit20": getattr(r, "m6_hit20", np.nan),
        "m6_hit30": getattr(r, "m6_hit30", np.nan),
        "m6_hit50": getattr(r, "m6_hit50", np.nan),
        "m6_hit100": getattr(r, "m6_hit100", np.nan),
    }
    row.update(f)
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No signals had enough pre-signal history.")

# Historical training population = mature 6M signals.
train = df[df["mature_6m"] == 1].copy()
if train.empty:
    raise SystemExit("No mature 6M signals.")

for col in ["m6_hit20","m6_hit30","m6_hit50","m6_hit100"]:
    train[col] = pd.to_numeric(train[col], errors="coerce")

# Data-relative thresholds from mature historical signals.
q_vol25 = train["pre_realized_vol_pct"].quantile(.25)
q_vol75 = train["pre_realized_vol_pct"].quantile(.75)
q_range25 = train["pre_6m_range_pct"].quantile(.25)
q_range75 = train["pre_6m_range_pct"].quantile(.75)
q_r2_60 = train["pre_trend_r2"].quantile(.60)
q_distlow25 = train["pre_dist_6m_low_pct"].quantile(.25)
q_disthigh75 = train["pre_dist_6m_high_pct"].quantile(.75)


def regime(row):
    ret6 = row["pre_6m_ret_pct"]
    vol = row["pre_realized_vol_pct"]
    rng = row["pre_6m_range_pct"]
    r2 = row["pre_trend_r2"]
    dh = row["pre_dist_6m_high_pct"]
    dl = row["pre_dist_6m_low_pct"]
    slope = row["pre_trend_slope_pct"]

    if (
        vol <= q_vol25 and rng <= q_range25
        and abs(ret6) <= 15 and r2 < q_r2_60
    ):
        return "LOW_VOL_SIDEWAYS"

    if (
        vol >= q_vol75 and rng >= q_range75
        and abs(ret6) <= 25
    ):
        return "HIGH_VOL_SIDEWAYS"

    if slope > 15 and r2 >= q_r2_60 and ret6 > 15:
        if ret6 >= 40:
            return "STRONG_UPTREND"
        return "QUIET_UPTREND"

    if slope < -15 and r2 >= q_r2_60 and ret6 < -15:
        return "DOWNTREND"

    if dl <= train["pre_dist_6m_low_pct"].median() and ret21 > 0:
        return "RECOVERY_FROM_LOW"

    if dh >= q_disthigh75:
        return "NEAR_6M_HIGH"

    return "MIXED"


# fix local ref for function
ret21 = None

def assign_regime(row):
    ret6 = row["pre_6m_ret_pct"]
    vol = row["pre_realized_vol_pct"]
    rng = row["pre_6m_range_pct"]
    r2 = row["pre_trend_r2"]
    dh = row["pre_dist_6m_high_pct"]
    dl = row["pre_dist_6m_low_pct"]
    slope = row["pre_trend_slope_pct"]
    ret1m = row["pre_1m_ret_pct"]

    if (
        vol <= q_vol25 and rng <= q_range25
        and abs(ret6) <= 15 and r2 < q_r2_60
    ):
        return "LOW_VOL_SIDEWAYS"
    if (
        vol >= q_vol75 and rng >= q_range75
        and abs(ret6) <= 25
    ):
        return "HIGH_VOL_SIDEWAYS"
    if slope > 15 and r2 >= q_r2_60 and ret6 > 15:
        if ret6 >= 40:
            return "STRONG_UPTREND"
        return "QUIET_UPTREND"
    if slope < -15 and r2 >= q_r2_60 and ret6 < -15:
        return "DOWNTREND"
    if dl <= train["pre_dist_6m_low_pct"].median() and ret1m > 0:
        return "RECOVERY_FROM_LOW"
    if dh >= q_disthigh75:
        return "NEAR_6M_HIGH"
    return "MIXED"


df["pre_regime"] = df.apply(assign_regime, axis=1)
train = df[df["mature_6m"] == 1].copy()

# Feature comparison: SUCCESS50 vs non-success.
feature_cols = [
    "pre_6m_ret_pct","pre_3m_ret_pct","pre_1m_ret_pct",
    "pre_6m_range_pct","pre_realized_vol_pct","pre_atr14_pct",
    "pre_dist_6m_high_pct","pre_dist_6m_low_pct",
    "pre_max_drawdown_pct","pre_trend_slope_pct","pre_trend_r2",
    "pre_middle50_frac","pre_vol_ratio20_126","pre_adv20","pre_ret20_std_pct"
]

w = train[train["m6_hit50"] == 1]
l = train[train["m6_hit50"] == 0]

feat_rows = []
for col in feature_cols:
    wm = w[col].median()
    lm = l[col].median()
    # rank-biserial/AUC-style separation via pairwise probability approximation
    vals_w = w[col].dropna().to_numpy(float)
    vals_l = l[col].dropna().to_numpy(float)
    sep = np.nan
    direction = ""
    if len(vals_w) and len(vals_l):
        # efficient rank-based AUC
        comb = pd.Series(np.concatenate([vals_w, vals_l]))
        ranks = comb.rank(method="average").to_numpy()
        rw = ranks[:len(vals_w)].sum()
        U = rw - len(vals_w)*(len(vals_w)+1)/2
        auc = U / (len(vals_w)*len(vals_l))
        if auc >= .5:
            sep = auc
            direction = "HIGHER"
        else:
            sep = 1-auc
            direction = "LOWER"

    feat_rows.append({
        "feature": col,
        "success50_median": wm,
        "non_success50_median": lm,
        "direction_for_success": direction,
        "separation": sep,
    })

feat = pd.DataFrame(feat_rows).sort_values("separation", ascending=False)
feat.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

# Regime performance.
reg_rows = []
for rg, g in train.groupby("pre_regime"):
    reg_rows.append({
        "regime": rg,
        "n": len(g),
        "positive_close_6m_pct": 100*(g["m6_close_ret_pct"] > 0).mean(),
        "avg_close_6m_pct": g["m6_close_ret_pct"].mean(),
        "median_close_6m_pct": g["m6_close_ret_pct"].median(),
        "hit20_pct": 100*g["m6_hit20"].mean(),
        "hit30_pct": 100*g["m6_hit30"].mean(),
        "hit50_pct": 100*g["m6_hit50"].mean(),
        "hit100_pct": 100*g["m6_hit100"].mean(),
        "median_max_gain_6m_pct": g["m6_max_gain_pct"].median(),
        "median_max_dd_6m_pct": g["m6_max_drawdown_pct"].median(),
    })

reg = pd.DataFrame(reg_rows).sort_values(["hit50_pct","n"], ascending=[False,False])
reg.to_csv(OUT_REGIMES, index=False, encoding="utf-8-sig")

# Map historical regime stats back to all signals.
reg_map = reg.set_index("regime")
df["regime_hist_n"] = df["pre_regime"].map(reg_map["n"])
df["regime_hist_hit20_pct"] = df["pre_regime"].map(reg_map["hit20_pct"])
df["regime_hist_hit50_pct"] = df["pre_regime"].map(reg_map["hit50_pct"])
df["regime_hist_hit100_pct"] = df["pre_regime"].map(reg_map["hit100_pct"])

# Simple historical-quality score for ranking recent signals.
# Uses regime's hit50 rate + favorable z-like ranks on top 3 discovered features.
top3 = feat.head(3)["feature"].tolist()

for col in top3:
    med = train[col].median()
    scale = train[col].mad() if hasattr(train[col], "mad") else None
    # pandas 3 removed Series.mad, use median absolute deviation:
    mad = float((train[col] - med).abs().median())
    if not np.isfinite(mad) or mad == 0:
        mad = float(train[col].std(ddof=0)) or 1.0
    direction = feat.loc[feat["feature"] == col, "direction_for_success"].iloc[0]
    z = (df[col] - med) / mad
    if direction == "LOWER":
        z = -z
    df[f"score_{col}"] = z.clip(-3,3)

base = df["regime_hist_hit50_pct"].fillna(train["m6_hit50"].mean()*100) / 100.0
shape = sum(df[f"score_{c}"] for c in top3) / max(len(top3), 1)
df["historical_shape_score"] = 100 * (0.65*base + 0.35*(1/(1+np.exp(-shape))))

df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

# Recent signals = latest ~220 calendar days among signals.
max_date = df["signal_date"].max()
cutoff = (pd.Timestamp(max_date) - pd.Timedelta(days=RECENT_CALENDAR_DAYS)).date().isoformat()
recent = df[df["signal_date"] >= cutoff].copy()
recent = recent.sort_values(["historical_shape_score","signal_date"], ascending=[False,False])

recent_cols = [
    "signal_date","symbol","name","signal_price","pre_regime",
    "historical_shape_score","regime_hist_n",
    "regime_hist_hit20_pct","regime_hist_hit50_pct","regime_hist_hit100_pct",
    "pre_6m_ret_pct","pre_3m_ret_pct","pre_1m_ret_pct",
    "pre_6m_range_pct","pre_realized_vol_pct","pre_atr14_pct",
    "pre_dist_6m_high_pct","pre_dist_6m_low_pct",
    "pre_trend_slope_pct","pre_trend_r2","pre_vol_ratio20_126",
]
recent[recent_cols].to_csv(OUT_RECENT, index=False, encoding="utf-8-sig")

print("\n=== HISTORICAL MATURE 6M SAMPLE ===")
print("Signals:", len(train))
print("Hit +20%:", f"{100*train['m6_hit20'].mean():.2f}%")
print("Hit +50%:", f"{100*train['m6_hit50'].mean():.2f}%")
print("Hit +100%:", f"{100*train['m6_hit100'].mean():.2f}%")

print("\n=== PRE-SIGNAL SHAPE REGIMES ===")
print(reg.to_string(index=False))

print("\n=== TOP PRE-SIGNAL FEATURES: +50% SUCCESS VS OTHERS ===")
print(feat.head(10).to_string(index=False))

print("\n=== RECENT STRONG SIGNALS RANKED BY HISTORICAL PRE-SHAPE ===")
print(recent[recent_cols].head(60).to_string(index=False))

report = []
report.append("EGX STRONG — 6M PRE-SIGNAL SHAPE STUDY")
report.append("="*120)
report.append(f"Mature 6M historical signals: {len(train)}")
report.append(f"Historical +20 hit rate: {100*train['m6_hit20'].mean():.2f}%")
report.append(f"Historical +50 hit rate: {100*train['m6_hit50'].mean():.2f}%")
report.append(f"Historical +100 hit rate: {100*train['m6_hit100'].mean():.2f}%")
report.append("")
report.append("PRE-SIGNAL REGIMES")
report.append(reg.to_string(index=False))
report.append("")
report.append("TOP FEATURES FOR +50% SUCCESS")
report.append(feat.head(15).to_string(index=False))
report.append("")
report.append("RECENT SIGNALS")
report.append(recent[recent_cols].to_string(index=False))
OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_ALL, OUT_FEATURES, OUT_REGIMES, OUT_RECENT, OUT_REPORT]:
    print(" ", p)
