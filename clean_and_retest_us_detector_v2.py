#!/usr/bin/env python3
"""
Rajih — CLEAN US DATA + RE-TEST DETECTOR ON FULL MARKET (2024-2026)

Goal
----
Clean the historical universe FIRST, then re-run the fixed detector from scratch
on every eligible US stock/date.

CLEANING RULES
--------------
Ticker:
  - simple alphabetic ticker, 1..5 chars
  - exclude obvious W / WS / U / R suffixes

At signal date:
  - RAW close >= $3.00
  - prior-20 average RAW dollar volume >= $5,000,000
  - prior-20 average RAW share volume >= 100,000 shares/day
  - no split-like adjustment-factor change in prior 252 sessions
  - no >100% adjusted-close jump in prior 20 sessions

Why RAW price?
--------------
Eligibility must use the actual traded price at the time, not a later
split-adjusted historical price.

Split detection
---------------
factor = adj_close / raw_close
A large day-to-day change in factor usually indicates a split/reverse split.
We reject a signal if factor changes by >= 1.5x or <= 1/1.5x during the prior
252 sessions.

FIXED DETECTOR
--------------
T-1 ATR14% >= 10.96
T-1 5-day return <= -9.22%
T-1 Bollinger Width% > 100
T-1 Dollar-volume ratio 5/20 > 1.5
T-20 10-day return <= -15.3917%

INDEPENDENCE
------------
One signal per symbol every 63 trading sessions.

SYSTEM TEST
-----------
Primary rule:
  - Entry = signal close
  - TP1 = +20% within first 5 trading sessions
  - If TP1 not hit: exit at Day-5 close

Also report:
  - +50%, +100%, +200% within 63 sessions from signal
  - max drawdown before TP1 for 5-day winners
  - yearly results

Outputs
-------
/data/us_clean_v2_signals.csv
/data/us_clean_v2_yearly.csv
/data/us_clean_v2_tp20_winners.csv
/data/us_clean_v2_summary.json
/data/us_clean_v2_report.txt

Run
---
cd /app
python clean_and_retest_us_detector_v2.py
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select, func

from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

OUT_SIGNALS = DATA_DIR / "us_clean_v2_signals.csv"
OUT_YEARLY = DATA_DIR / "us_clean_v2_yearly.csv"
OUT_WINNERS = DATA_DIR / "us_clean_v2_tp20_winners.csv"
OUT_JSON = DATA_DIR / "us_clean_v2_summary.json"
OUT_REPORT = DATA_DIR / "us_clean_v2_report.txt"

YEARS = (2024, 2025, 2026)

# Cleaning.
MIN_RAW_PRICE = 3.00
MIN_AVG_DOLLAR_VOL20 = 5_000_000.0
MIN_AVG_SHARE_VOL20 = 100_000.0
SPLIT_LOOKBACK = 252
SPLIT_FACTOR_THRESHOLD = 1.50
MAX_ABS_ADJ_CLOSE_JUMP_PCT = 100.0

# Detector.
ATR14_MIN = 10.96
RET5_MAX = -9.22
BB_WIDTH_MIN = 100.0
DOLLARVOL_RATIO_MIN = 1.5
T20_RET10_MAX = -15.3917

COOLDOWN_SESSIONS = 63
MIN_SIGNAL_INDEX = 270  # enough for 252-session split history

TP1 = 20.0
TP1_DAYS = 5


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

    # Split-adjusted OHLC for continuous return calculations.
    df["adj_factor"] = df["adj_c"] / df["c"]
    df["ao"] = df["o"] * df["adj_factor"]
    df["ah"] = df["h"] * df["adj_factor"]
    df["al"] = df["l"] * df["adj_factor"]
    df["ac"] = df["adj_c"]

    # Detect changes in adjustment factor (split/reverse split proxy).
    factor_ratio = df["adj_factor"] / df["adj_factor"].shift(1)
    df["split_like"] = (
        (factor_ratio >= SPLIT_FACTOR_THRESHOLD) |
        (factor_ratio <= 1.0 / SPLIT_FACTOR_THRESHOLD)
    ).astype(int)

    prev = df["ac"].shift(1)
    tr = pd.concat(
        [
            df["ah"] - df["al"],
            (df["ah"] - prev).abs(),
            (df["al"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["atr14_pct"] = 100.0 * atr14 / df["ac"]

    df["ret5_pct"] = 100.0 * (df["ac"] / df["ac"].shift(5) - 1.0)
    df["ret10_pct"] = 100.0 * (df["ac"] / df["ac"].shift(10) - 1.0)

    sma20 = df["ac"].rolling(20).mean()
    std20 = df["ac"].rolling(20).std(ddof=0)
    df["bb_width_pct"] = 100.0 * (4.0 * std20) / sma20

    # Eligibility is based on actual/raw traded values.
    df["raw_dollar_volume"] = df["c"] * df["v"].fillna(0.0)
    df["avg_raw_dollar_volume20"] = df["raw_dollar_volume"].rolling(20).mean()
    df["avg_raw_share_volume20"] = df["v"].rolling(20).mean()

    # Detector's relative volume feature.
    df["dollarvol_ratio_5_20"] = (
        df["raw_dollar_volume"].rolling(5).mean() /
        df["avg_raw_dollar_volume20"].replace(0, np.nan)
    )

    df["adj_close_jump_pct"] = 100.0 * (df["ac"] / df["ac"].shift(1) - 1.0)

    return df


def passes_cleaning(df, i):
    t1 = i - 1

    vals = [
        df.at[i, "c"],
        df.at[t1, "avg_raw_dollar_volume20"],
        df.at[t1, "avg_raw_share_volume20"],
    ]
    if any(pd.isna(x) or not np.isfinite(x) for x in vals):
        return False

    if df.at[i, "c"] < MIN_RAW_PRICE:
        return False
    if df.at[t1, "avg_raw_dollar_volume20"] < MIN_AVG_DOLLAR_VOL20:
        return False
    if df.at[t1, "avg_raw_share_volume20"] < MIN_AVG_SHARE_VOL20:
        return False

    # No split / reverse split during the prior trading year.
    lb = max(1, i - SPLIT_LOOKBACK)
    if df.loc[lb:i-1, "split_like"].any():
        return False

    # Extra anti-artifact guard close to signal.
    if (df.loc[max(1, i-20):i-1, "adj_close_jump_pct"].abs() > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any():
        return False

    return True


def detector_passes(df, i):
    t1 = i - 1
    t20 = i - 20

    if i < MIN_SIGNAL_INDEX or t20 < 10:
        return False

    if not passes_cleaning(df, i):
        return False

    vals = [
        df.at[t1, "atr14_pct"],
        df.at[t1, "ret5_pct"],
        df.at[t1, "bb_width_pct"],
        df.at[t1, "dollarvol_ratio_5_20"],
        df.at[t20, "ret10_pct"],
    ]
    if any(pd.isna(x) or not np.isfinite(x) for x in vals):
        return False

    return (
        df.at[t1, "atr14_pct"] >= ATR14_MIN and
        df.at[t1, "ret5_pct"] <= RET5_MAX and
        df.at[t1, "bb_width_pct"] > BB_WIDTH_MIN and
        df.at[t1, "dollarvol_ratio_5_20"] > DOLLARVOL_RATIO_MIN and
        df.at[t20, "ret10_pct"] <= T20_RET10_MAX
    )


def analyze_symbol(symbol, raw_rows):
    df = add_features(raw_rows)
    if len(df) < MIN_SIGNAL_INDEX + 6:
        return []

    rows = []
    last_kept_i = -10**9

    for i in range(MIN_SIGNAL_INDEX, len(df)-5):
        dt = pd.Timestamp(df.at[i, "session_date"])
        if dt.year not in YEARS:
            continue

        if not detector_passes(df, i):
            continue

        if i - last_kept_i < COOLDOWN_SESSIONS:
            continue
        last_kept_i = i

        sig_adj = float(df.at[i, "ac"])
        sig_raw = float(df.at[i, "c"])

        # First 5 sessions.
        f5 = df.iloc[i+1:i+6]
        gains5 = 100.0 * (f5["ah"].to_numpy(float) / sig_adj - 1.0)
        hit_idx = np.flatnonzero(gains5 >= TP1)
        hit20_5d = int(len(hit_idx) > 0)
        first_tp20_day = int(hit_idx[0] + 1) if len(hit_idx) else None

        day5_close_ret = pct(float(df.at[i+5, "ac"]), sig_adj)
        system_ret = TP1 if hit20_5d else day5_close_ret

        dd_before_tp20 = None
        if hit20_5d:
            d = first_tp20_day
            low = float(df.iloc[i+1:i+d+1]["al"].min())
            dd_before_tp20 = pct(low, sig_adj)

        # 63-session tail metrics where mature.
        mature63 = i + 63 < len(df)
        hit50 = hit100 = hit200 = None
        max63 = None
        if mature63:
            f63 = df.iloc[i+1:i+64]
            g63 = 100.0 * (f63["ah"].to_numpy(float) / sig_adj - 1.0)
            hit50 = int(np.any(g63 >= 50))
            hit100 = int(np.any(g63 >= 100))
            hit200 = int(np.any(g63 >= 200))
            max63 = float(np.nanmax(g63))

        rows.append({
            "year": int(dt.year),
            "signal_date": dt.date().isoformat(),
            "symbol": symbol,
            "raw_signal_close": sig_raw,
            "adjusted_signal_close": sig_adj,
            "avg_raw_dollar_volume20": float(df.at[i-1, "avg_raw_dollar_volume20"]),
            "avg_raw_share_volume20": float(df.at[i-1, "avg_raw_share_volume20"]),
            "t1_atr14_pct": float(df.at[i-1, "atr14_pct"]),
            "t1_ret5_pct": float(df.at[i-1, "ret5_pct"]),
            "t1_bb_width_pct": float(df.at[i-1, "bb_width_pct"]),
            "t1_dollarvol_ratio_5_20": float(df.at[i-1, "dollarvol_ratio_5_20"]),
            "t20_ret10_pct": float(df.at[i-20, "ret10_pct"]),
            "tp20_within_5d": hit20_5d,
            "first_tp20_day": first_tp20_day,
            "drawdown_before_tp20_pct": dd_before_tp20,
            "day5_close_return_pct": float(day5_close_ret),
            "system_return_pct": float(system_ret),
            "mature63": int(mature63),
            "hit50_63d": hit50,
            "hit100_63d": hit100,
            "hit200_63d": hit200,
            "max_gain63_pct": max63,
        })

    return rows


DB = database()

with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

    all_symbols = [
        r[0]
        for r in s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count() >= MIN_SIGNAL_INDEX + 6)
            .order_by(daily.c.symbol)
        ).all()
    ]

symbols = [s for s in all_symbols if looks_like_common_stock_symbol(s)]

print("\nRAJIH — CLEAN DATA V2 + FULL MARKET RETEST")
print(f"Symbols with enough history: {len(all_symbols)}")
print(f"Ticker-clean symbols retained: {len(symbols)}")
print("Cleaning: raw price >= $3, $5M ADV20, 100k shares/day, no prior-252 split-like event")
print("Scanning full 2024-2026 universe ...")

rows = []

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

        rr = analyze_symbol(symbol, raw)
        if rr:
            rows.extend(rr)
            print(f"[{n}/{len(symbols)}] {symbol}: signals={len(rr)} tp20_5d={sum(x['tp20_within_5d'] for x in rr)}")
        elif n % 250 == 0:
            print(f"[{n}/{len(symbols)}] progress | clean signals={len(rows)}")

    except Exception as e:
        print(f"[{n}/{len(symbols)}] {symbol} ERROR: {e}")

if not rows:
    raise SystemExit("No clean signals found.")

df = pd.DataFrame(rows).sort_values(["signal_date","symbol"]).reset_index(drop=True)
df.to_csv(OUT_SIGNALS, index=False, encoding="utf-8-sig")

wins = df[df["tp20_within_5d"] == 1].copy()
wins.to_csv(OUT_WINNERS, index=False, encoding="utf-8-sig")


def summarize(x):
    n = len(x)
    mature = x[x["mature63"] == 1]
    return {
        "signals": n,
        "tp20_5d_n": int(x["tp20_within_5d"].sum()),
        "tp20_5d_pct": 100.0*x["tp20_within_5d"].mean() if n else None,
        "positive_system_n": int((x["system_return_pct"] > 0).sum()),
        "positive_system_pct": 100.0*(x["system_return_pct"] > 0).mean() if n else None,
        "avg_system_return_pct": float(x["system_return_pct"].mean()) if n else None,
        "median_system_return_pct": float(x["system_return_pct"].median()) if n else None,
        "worst_system_return_pct": float(x["system_return_pct"].min()) if n else None,
        "mature63": len(mature),
        "hit50_63d_pct": 100.0*mature["hit50_63d"].mean() if len(mature) else None,
        "hit100_63d_pct": 100.0*mature["hit100_63d"].mean() if len(mature) else None,
        "hit200_63d_pct": 100.0*mature["hit200_63d"].mean() if len(mature) else None,
    }


overall = summarize(df)

year_rows = []
for y in YEARS:
    s = summarize(df[df["year"] == y])
    s["year"] = y
    year_rows.append(s)

yearly = pd.DataFrame(year_rows)
yearly = yearly[["year"] + [c for c in yearly.columns if c != "year"]]
yearly.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")

winner_dd = {
    "winner_count": int(len(wins)),
    "median_dd_before_tp20_pct": float(wins["drawdown_before_tp20_pct"].median()) if len(wins) else None,
    "p10_dd_before_tp20_pct": float(wins["drawdown_before_tp20_pct"].quantile(.10)) if len(wins) else None,
    "p05_dd_before_tp20_pct": float(wins["drawdown_before_tp20_pct"].quantile(.05)) if len(wins) else None,
    "worst_dd_before_tp20_pct": float(wins["drawdown_before_tp20_pct"].min()) if len(wins) else None,
}

summary = {
    "cleaning": {
        "min_raw_price": MIN_RAW_PRICE,
        "min_avg_raw_dollar_volume20": MIN_AVG_DOLLAR_VOL20,
        "min_avg_raw_share_volume20": MIN_AVG_SHARE_VOL20,
        "split_lookback_sessions": SPLIT_LOOKBACK,
        "split_factor_threshold": SPLIT_FACTOR_THRESHOLD,
        "cooldown_sessions": COOLDOWN_SESSIONS,
    },
    "overall": overall,
    "winner_drawdown": winner_dd,
    "yearly": year_rows,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== CLEAN V2 OVERALL ===")
for k, v in overall.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

print("\n=== CLEAN V2 YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== TP20 <=5D WINNER DRAWDOWN ===")
for k, v in winner_dd.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

print("\n=== WORST 20 SYSTEM TRADES ===")
print(
    df.sort_values("system_return_pct")
      .head(20)[
          ["year","signal_date","symbol","raw_signal_close",
           "avg_raw_dollar_volume20","tp20_within_5d",
           "day5_close_return_pct","system_return_pct"]
      ]
      .to_string(index=False)
)

print("\n=== TP20 WINNERS ===")
print(
    wins[[
        "year","signal_date","symbol","raw_signal_close",
        "first_tp20_day","drawdown_before_tp20_pct"
    ]].to_string(index=False)
)

report = []
report.append("RAJIH — CLEAN DATA V2 + FULL MARKET RETEST")
report.append("="*100)
report.append("CLEANING")
report.append(
    f"raw price >= ${MIN_RAW_PRICE}; ADV20 >= ${MIN_AVG_DOLLAR_VOL20:,.0f}; "
    f"avg shares20 >= {MIN_AVG_SHARE_VOL20:,.0f}; "
    f"no prior-{SPLIT_LOOKBACK} split-like factor change; cooldown={COOLDOWN_SESSIONS}"
)
report.append("")
report.append("OVERALL")
for k, v in overall.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("TP20 <=5D WINNER DRAWDOWN")
for k, v in winner_dd.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("WORST 20 SYSTEM TRADES")
report.append(
    df.sort_values("system_return_pct")
      .head(20)[
          ["year","signal_date","symbol","raw_signal_close",
           "avg_raw_dollar_volume20","tp20_within_5d",
           "day5_close_return_pct","system_return_pct"]
      ].to_string(index=False)
)

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_SIGNALS, OUT_YEARLY, OUT_WINNERS, OUT_JSON, OUT_REPORT]:
    print(" ", p)
