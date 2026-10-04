#!/usr/bin/env python3
"""
Rajih — Accumulation Pattern Scanner backtest on CLEAN US daily stocks

This script reproduces the supplied Pine Script logic as closely as possible
on daily bars, then evaluates every NEW STRONG signal over ~3 months (63
trading sessions) and ~6 months (126 trading sessions).

IMPORTANT PINE DETAILS MIRRORED EXACTLY
---------------------------------------
Defaults:
  i_minBars      = 80
  i_minRange     = 22%
  i_maxRange     = 80%
  i_volDryRatio  = 0.70
  i_volHistBars  = 200
  i_greenVolMin  = 0.55

Pine conditions:
  C1: 80-bar price range is 22%..80%
  C2: SMA(volume,80) < SMA(volume,200) * 0.70
  C3: green-candle volume / total volume over 80 bars >= 0.55
  C4: close is in the top 30% of the 80-bar range

Score:
  C1 25 + C2 25 + C3 30 + C4 20

Because of those exact weights, score >= 90 can ONLY happen when ALL FOUR
conditions are true, which actually gives score = 100.

STRONG alert:
  isStrongPattern and not isStrongPattern[1]
So the signal is the FIRST daily bar of a new STRONG episode.

NOTE:
The Pine comment says historical volume is "outside the current window", but
the actual code uses ta.sma(volume, 200), which includes the current bars.
This backtest mirrors the ACTUAL code, not the comment.

CLEAN US FILTERS (V3-style, applied at signal date)
---------------------------------------------------
- simple alphabetic ticker 1..5 chars, exclude obvious W/WS/U/R suffixes
- RAW close >= $5
- prior-20 avg RAW dollar volume >= $5M
- prior-20 avg RAW share volume >= 100k/day
- no RAW close below $1 during prior 126 sessions
- no split/reverse-split-like flag in prior up-to-252 sessions
- no suspicious raw/adjusted discontinuity in prior up-to-252 sessions
- at least 60 prior sessions for cleaning history

PERFORMANCE
-----------
For each clean NEW STRONG signal:
3 months  = 63 trading sessions
6 months  = 126 trading sessions

Reports:
- close return at 63 / 126 sessions
- max gain (using adjusted HIGH) within each horizon
- max drawdown (using adjusted LOW) within each horizon
- hit +20 / +50 / +100 / +200 within each horizon
- yearly summaries
- top and worst signals

Outputs
-------
/data/accum_strong_clean_signals.csv
/data/accum_strong_clean_3m_summary.csv
/data/accum_strong_clean_6m_summary.csv
/data/accum_strong_clean_yearly.csv
/data/accum_strong_clean_report.txt

Run
---
cd /app
python backtest_accumulation_strong_clean_us.py
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select, func

from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

OUT_SIGNALS = DATA_DIR / "accum_strong_clean_signals.csv"
OUT_3M = DATA_DIR / "accum_strong_clean_3m_summary.csv"
OUT_6M = DATA_DIR / "accum_strong_clean_6m_summary.csv"
OUT_YEARLY = DATA_DIR / "accum_strong_clean_yearly.csv"
OUT_REPORT = DATA_DIR / "accum_strong_clean_report.txt"

YEARS = (2024, 2025, 2026)

# Pine defaults
MIN_BARS = 80
MIN_RANGE = 22.0
MAX_RANGE = 80.0
VOL_DRY_RATIO = 0.70
VOL_HIST_BARS = 200
GREEN_VOL_MIN = 0.55

# Clean V3 universe
MIN_RAW_PRICE = 5.0
MIN_ADV20 = 5_000_000.0
MIN_AVG_SHARE_VOL20 = 100_000.0
LOW_PRICE_FLOOR = 1.0
LOW_PRICE_LOOKBACK = 126

SPLIT_LOOKBACK_MAX = 252
MIN_CLEAN_LOOKBACK = 60
SPLIT_FACTOR_THRESHOLD = 1.50

RAW_JUMP_THRESHOLD_PCT = 80.0
ADJ_SMALL_MOVE_THRESHOLD_PCT = 30.0
EXTREME_RAW_RATIO = 2.0
MAX_ABS_ADJ_CLOSE_JUMP_PCT = 100.0


def looks_like_common_stock_symbol(symbol: str) -> bool:
    s = (symbol or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}", s):
        return False
    if s.endswith(("WS", "W", "U", "R")):
        return False
    return True


def pct(new, old):
    if old is None or not np.isfinite(old) or old <= 0:
        return np.nan
    return 100.0 * (new / old - 1.0)


def add_features(raw_rows):
    df = pd.DataFrame(
        raw_rows,
        columns=["session_date", "o", "h", "l", "c", "v", "adj_c"],
    )
    if df.empty:
        return df

    for col in ["o", "h", "l", "c", "v", "adj_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["o", "h", "l", "c", "adj_c"])
    df = df[
        (df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) &
        (df["c"] > 0) & (df["adj_c"] > 0)
    ].copy()

    if df.empty:
        return df

    df = df.sort_values("session_date").reset_index(drop=True)

    # Adjusted OHLC for return continuity
    df["adj_factor"] = df["adj_c"] / df["c"]
    df["ao"] = df["o"] * df["adj_factor"]
    df["ah"] = df["h"] * df["adj_factor"]
    df["al"] = df["l"] * df["adj_factor"]
    df["ac"] = df["adj_c"]

    # cleaning diagnostics
    df["raw_close_ret_pct"] = 100.0 * (df["c"] / df["c"].shift(1) - 1.0)
    df["adj_close_ret_pct"] = 100.0 * (df["ac"] / df["ac"].shift(1) - 1.0)

    factor_ratio = df["adj_factor"] / df["adj_factor"].shift(1)
    df["split_factor_flag"] = (
        (factor_ratio >= SPLIT_FACTOR_THRESHOLD) |
        (factor_ratio <= 1.0 / SPLIT_FACTOR_THRESHOLD)
    ).astype(int)

    raw_ratio = df["c"] / df["c"].shift(1)

    df["raw_adj_mismatch_flag"] = (
        (df["raw_close_ret_pct"].abs() >= RAW_JUMP_THRESHOLD_PCT) &
        (df["adj_close_ret_pct"].abs() <= ADJ_SMALL_MOVE_THRESHOLD_PCT)
    ).astype(int)

    df["extreme_raw_ratio_flag"] = (
        ((raw_ratio >= EXTREME_RAW_RATIO) | (raw_ratio <= 1.0 / EXTREME_RAW_RATIO)) &
        (df["adj_close_ret_pct"].abs() < df["raw_close_ret_pct"].abs() * 0.60)
    ).astype(int)

    # Cleaning liquidity
    df["raw_dollar_volume"] = df["c"] * df["v"].fillna(0.0)
    df["avg_raw_dollar_volume20"] = df["raw_dollar_volume"].rolling(20).mean()
    df["avg_raw_share_volume20"] = df["v"].rolling(20).mean()

    # -------- Exact Pine logic --------
    df["patHigh"] = df["h"].rolling(MIN_BARS).max()
    df["patLow"] = df["l"].rolling(MIN_BARS).min()
    df["patRange"] = 100.0 * (df["patHigh"] - df["patLow"]) / df["patLow"]

    df["avgVolNow"] = df["v"].rolling(MIN_BARS).mean()
    df["avgVolHist"] = df["v"].rolling(VOL_HIST_BARS).mean()

    green_vol = df["v"].where(df["c"] >= df["o"], 0.0)
    total_vol = df["v"].fillna(0.0)

    df["greenVolSum"] = green_vol.rolling(MIN_BARS).sum()
    df["totalVolSum"] = total_vol.rolling(MIN_BARS).sum()
    df["greenVolRatio"] = (
        df["greenVolSum"] /
        df["totalVolSum"].replace(0, np.nan)
    )

    denom = (df["patHigh"] - df["patLow"]).replace(0, np.nan)
    df["nearTop"] = (df["c"] - df["patLow"]) / denom

    df["c1_range"] = (
        (df["patRange"] >= MIN_RANGE) &
        (df["patRange"] <= MAX_RANGE)
    )
    df["c2_dryVol"] = df["avgVolNow"] < (df["avgVolHist"] * VOL_DRY_RATIO)
    df["c3_greenVol"] = df["greenVolRatio"] >= GREEN_VOL_MIN
    df["c4_nearTop"] = df["nearTop"] >= 0.70

    df["score"] = (
        df["c1_range"].astype(int) * 25 +
        df["c2_dryVol"].astype(int) * 25 +
        df["c3_greenVol"].astype(int) * 30 +
        df["c4_nearTop"].astype(int) * 20
    )

    df["isStrong"] = df["score"] >= 90
    df["newStrong"] = df["isStrong"] & (~df["isStrong"].shift(1).fillna(False))

    return df


def clean_at_signal(df, i):
    if i < MIN_CLEAN_LOOKBACK:
        return False, "INSUFFICIENT_LOOKBACK"

    t1 = i - 1

    vals = [
        df.at[i, "c"],
        df.at[t1, "avg_raw_dollar_volume20"],
        df.at[t1, "avg_raw_share_volume20"],
    ]
    if any(pd.isna(x) or not np.isfinite(x) for x in vals):
        return False, "INSUFFICIENT_LOOKBACK"

    if df.at[i, "c"] < MIN_RAW_PRICE:
        return False, "PRICE"

    if (
        df.at[t1, "avg_raw_dollar_volume20"] < MIN_ADV20 or
        df.at[t1, "avg_raw_share_volume20"] < MIN_AVG_SHARE_VOL20
    ):
        return False, "LIQUIDITY"

    lb_low = max(0, i - min(LOW_PRICE_LOOKBACK, i))
    if (df.loc[lb_low:i-1, "c"] < LOW_PRICE_FLOOR).any():
        return False, "LOW_PRICE_HISTORY"

    lb_split = max(0, i - min(SPLIT_LOOKBACK_MAX, i))

    if df.loc[lb_split:i-1, "split_factor_flag"].any():
        return False, "RECENT_SPLIT"

    if df.loc[
        lb_split:i-1,
        ["raw_adj_mismatch_flag", "extreme_raw_ratio_flag"]
    ].any().any():
        return False, "SUSPICIOUS_DISCONTINUITY"

    if (
        df.loc[max(1, i-20):i-1, "adj_close_ret_pct"].abs()
        > MAX_ABS_ADJ_CLOSE_JUMP_PCT
    ).any():
        return False, "SUSPICIOUS_DISCONTINUITY"

    return True, "ELIGIBLE"


def horizon_metrics(df, i, sessions):
    if i + sessions >= len(df):
        return None

    entry = float(df.at[i, "ac"])
    f = df.iloc[i+1:i+sessions+1]

    highs = f["ah"].to_numpy(float)
    lows = f["al"].to_numpy(float)

    high_gains = 100.0 * (highs / entry - 1.0)
    low_dd = 100.0 * (lows / entry - 1.0)

    close_ret = pct(float(df.at[i+sessions, "ac"]), entry)

    return {
        "close_ret_pct": float(close_ret),
        "max_gain_pct": float(np.nanmax(high_gains)),
        "max_drawdown_pct": float(np.nanmin(low_dd)),
        "hit20": int(np.any(high_gains >= 20)),
        "hit50": int(np.any(high_gains >= 50)),
        "hit100": int(np.any(high_gains >= 100)),
        "hit200": int(np.any(high_gains >= 200)),
    }


DB = database()

with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

    all_symbols = [
        r[0]
        for r in s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count() >= VOL_HIST_BARS)
            .order_by(daily.c.symbol)
        ).all()
    ]

symbols = [s for s in all_symbols if looks_like_common_stock_symbol(s)]

print("\nRAJIH — ACCUMULATION STRONG CLEAN-US BACKTEST")
print(f"Symbols with >= {VOL_HIST_BARS} daily bars: {len(all_symbols)}")
print(f"Ticker-clean symbols retained: {len(symbols)}")
print("Mirroring supplied Pine logic exactly on DAILY bars.")
print("STRONG = first bar where score >=90 (in practice all 4 conditions true).")

rows = []
rejections = {}

for n, symbol in enumerate(symbols, 1):
    try:
        with DB() as s:
            raw = s.execute(
                select(
                    daily.c.session_date,
                    daily.c.o,
                    daily.c.h,
                    daily.c.l,
                    daily.c.c,
                    daily.c.v,
                    daily.c.adj_c,
                )
                .where(daily.c.symbol == symbol)
                .order_by(daily.c.session_date)
            ).all()

        df = add_features(raw)
        if len(df) < VOL_HIST_BARS:
            continue

        strong_count = 0
        clean_count = 0

        for i in df.index[df["newStrong"]]:
            dt = pd.Timestamp(df.at[i, "session_date"])
            if dt.year not in YEARS:
                continue

            strong_count += 1
            ok, reason = clean_at_signal(df, i)

            if not ok:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            clean_count += 1

            m3 = horizon_metrics(df, i, 63)
            m6 = horizon_metrics(df, i, 126)

            row = {
                "year": int(dt.year),
                "signal_date": dt.date().isoformat(),
                "symbol": symbol,
                "raw_signal_close": float(df.at[i, "c"]),
                "adjusted_signal_close": float(df.at[i, "ac"]),
                "score": int(df.at[i, "score"]),
                "patRange_pct": float(df.at[i, "patRange"]),
                "avgVolNow": float(df.at[i, "avgVolNow"]),
                "avgVolHist": float(df.at[i, "avgVolHist"]),
                "dryVolRatio_actual": float(df.at[i, "avgVolNow"] / df.at[i, "avgVolHist"]),
                "greenVolRatio": float(df.at[i, "greenVolRatio"]),
                "nearTop": float(df.at[i, "nearTop"]),
                "adv20_raw": float(df.at[i-1, "avg_raw_dollar_volume20"]),
                "avg_share_vol20": float(df.at[i-1, "avg_raw_share_volume20"]),
                "mature_3m": int(m3 is not None),
                "mature_6m": int(m6 is not None),
            }

            for prefix, m in [("m3", m3), ("m6", m6)]:
                for k in [
                    "close_ret_pct","max_gain_pct","max_drawdown_pct",
                    "hit20","hit50","hit100","hit200"
                ]:
                    row[f"{prefix}_{k}"] = m[k] if m else None

            rows.append(row)

        if clean_count:
            print(
                f"[{n}/{len(symbols)}] {symbol}: "
                f"STRONG={strong_count} clean={clean_count}"
            )
        elif n % 250 == 0:
            print(
                f"[{n}/{len(symbols)}] progress | "
                f"clean STRONG signals={len(rows)}"
            )

    except Exception as e:
        print(f"[{n}/{len(symbols)}] {symbol} ERROR: {e}")

if not rows:
    raise SystemExit("No clean STRONG signals found.")

sig = pd.DataFrame(rows).sort_values(["signal_date","symbol"]).reset_index(drop=True)
sig.to_csv(OUT_SIGNALS, index=False, encoding="utf-8-sig")


def summarize_horizon(df, prefix):
    x = df[df[f"mature_{'3m' if prefix == 'm3' else '6m'}"] == 1].copy()
    if not len(x):
        return {}

    return {
        "n": int(len(x)),
        "positive_close_n": int((x[f"{prefix}_close_ret_pct"] > 0).sum()),
        "positive_close_pct": float(100 * (x[f"{prefix}_close_ret_pct"] > 0).mean()),
        "avg_close_ret_pct": float(x[f"{prefix}_close_ret_pct"].mean()),
        "median_close_ret_pct": float(x[f"{prefix}_close_ret_pct"].median()),
        "avg_max_gain_pct": float(x[f"{prefix}_max_gain_pct"].mean()),
        "median_max_gain_pct": float(x[f"{prefix}_max_gain_pct"].median()),
        "median_max_drawdown_pct": float(x[f"{prefix}_max_drawdown_pct"].median()),
        "hit20_n": int(x[f"{prefix}_hit20"].sum()),
        "hit20_pct": float(100 * x[f"{prefix}_hit20"].mean()),
        "hit50_n": int(x[f"{prefix}_hit50"].sum()),
        "hit50_pct": float(100 * x[f"{prefix}_hit50"].mean()),
        "hit100_n": int(x[f"{prefix}_hit100"].sum()),
        "hit100_pct": float(100 * x[f"{prefix}_hit100"].mean()),
        "hit200_n": int(x[f"{prefix}_hit200"].sum()),
        "hit200_pct": float(100 * x[f"{prefix}_hit200"].mean()),
    }


s3 = summarize_horizon(sig, "m3")
s6 = summarize_horizon(sig, "m6")

pd.DataFrame([s3]).to_csv(OUT_3M, index=False, encoding="utf-8-sig")
pd.DataFrame([s6]).to_csv(OUT_6M, index=False, encoding="utf-8-sig")

year_rows = []
for y in YEARS:
    sy = sig[sig["year"] == y]
    r = {"year": y, "signals": int(len(sy))}
    for prefix in ("m3", "m6"):
        ss = summarize_horizon(sy, prefix)
        for k, v in ss.items():
            r[f"{prefix}_{k}"] = v
    year_rows.append(r)

yearly = pd.DataFrame(year_rows)
yearly.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")

print("\n=== STRONG SIGNALS OVERALL ===")
print(f"Clean STRONG signals: {len(sig)}")
print(f"Distinct symbols: {sig['symbol'].nunique()}")
print(f"Distinct dates: {sig['signal_date'].nunique()}")

print("\n=== 3 MONTHS / 63 SESSIONS ===")
for k, v in s3.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

print("\n=== 6 MONTHS / 126 SESSIONS ===")
for k, v in s6.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

print("\n=== YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== TOP 20 BY 3M MAX GAIN ===")
m3_top = sig[sig["mature_3m"] == 1].sort_values("m3_max_gain_pct", ascending=False).head(20)
print(
    m3_top[
        ["year","signal_date","symbol","raw_signal_close",
         "m3_close_ret_pct","m3_max_gain_pct","m3_max_drawdown_pct"]
    ].to_string(index=False)
)

print("\n=== WORST 20 BY 3M CLOSE RETURN ===")
m3_worst = sig[sig["mature_3m"] == 1].sort_values("m3_close_ret_pct").head(20)
print(
    m3_worst[
        ["year","signal_date","symbol","raw_signal_close",
         "m3_close_ret_pct","m3_max_gain_pct","m3_max_drawdown_pct"]
    ].to_string(index=False)
)

print("\n=== CLEANING REJECTIONS OF STRONG SIGNALS ===")
if rejections:
    for k, v in sorted(rejections.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")
else:
    print("None")

report = []
report.append("RAJIH — ACCUMULATION STRONG CLEAN-US BACKTEST")
report.append("=" * 105)
report.append(f"Clean STRONG signals: {len(sig)}")
report.append(f"Distinct symbols: {sig['symbol'].nunique()}")
report.append(f"Distinct dates: {sig['signal_date'].nunique()}")
report.append("")
report.append("3 MONTHS / 63 SESSIONS")
for k, v in s3.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("6 MONTHS / 126 SESSIONS")
for k, v in s6.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("TOP 20 BY 3M MAX GAIN")
report.append(
    m3_top[
        ["year","signal_date","symbol","raw_signal_close",
         "m3_close_ret_pct","m3_max_gain_pct","m3_max_drawdown_pct"]
    ].to_string(index=False)
)
report.append("")
report.append("WORST 20 BY 3M CLOSE RETURN")
report.append(
    m3_worst[
        ["year","signal_date","symbol","raw_signal_close",
         "m3_close_ret_pct","m3_max_gain_pct","m3_max_drawdown_pct"]
    ].to_string(index=False)
)
report.append("")
report.append("CLEANING REJECTIONS OF STRONG SIGNALS")
for k, v in sorted(rejections.items(), key=lambda x: (-x[1], x[0])):
    report.append(f"{k}: {v}")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_SIGNALS, OUT_3M, OUT_6M, OUT_YEARLY, OUT_REPORT]:
    print(" ", p)
