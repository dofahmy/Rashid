#!/usr/bin/env python3
"""
Rajih — Find stop-loss from maximum drawdown BEFORE first +20% target
for the CLEAN full-US detector signals (2024-2026).

Uses the SAME clean detector logic as:
  scan_us_clean_detector_cooldown_entries.py

Signal definition:
  - signal price >= $1
  - prior-20 avg dollar volume >= $1M
  - T-1 ATR14% >= 10.96
  - T-1 Ret5 <= -9.22%
  - T-1 BB Width > 100
  - T-1 Dollar Volume Ratio 5/20 > 1.5
  - T-20 Ret10 <= -15.3917%
  - no prior adjusted-close jump >100%
  - 63-session cooldown per symbol

Goal:
For every MATURE signal that eventually reaches +20% within 63 sessions:
  1) find the FIRST day whose adjusted HIGH reaches +20%
  2) find the lowest adjusted LOW from Day+1 through that first-hit day
  3) measure max drawdown from signal close before +20%

Because daily candles do not reveal intraday order:
If the target day itself has both:
    low <= proposed stop AND high >= +20%
then the stop-vs-target order is AMBIGUOUS.
The report therefore shows:
  - conservative survival: ambiguous counts as stopped
  - optimistic survival: ambiguous counts as survived

Outputs:
  /data/us_stop_before_tp20_cases.csv
  /data/us_stop_before_tp20_distribution.csv
  /data/us_stop_before_tp20_stop_grid.csv
  /data/us_stop_before_tp20_yearly.csv
  /data/us_stop_before_tp20_report.txt

Run:
  cd /app
  python analyze_stop_before_tp20.py
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

OUT_CASES = DATA_DIR / "us_stop_before_tp20_cases.csv"
OUT_DIST = DATA_DIR / "us_stop_before_tp20_distribution.csv"
OUT_GRID = DATA_DIR / "us_stop_before_tp20_stop_grid.csv"
OUT_YEAR = DATA_DIR / "us_stop_before_tp20_yearly.csv"
OUT_REPORT = DATA_DIR / "us_stop_before_tp20_report.txt"

YEARS = (2024, 2025, 2026)

MIN_PRICE = 1.00
MIN_AVG_DOLLAR_VOL20 = 1_000_000.0

ATR14_MIN = 10.96
RET5_MAX = -9.22
BB_WIDTH_MIN = 100.0
DOLLARVOL_RATIO_MIN = 1.5
T20_RET10_MAX = -15.3917

COOLDOWN_SESSIONS = 63
MAX_ABS_ADJ_CLOSE_JUMP_PCT = 100.0
MIN_SIGNAL_INDEX = 40
TP20 = 20.0

# Stop grid to inspect.
STOP_LEVELS = [-5, -7.5, -10, -12.5, -15, -17.5, -20, -22.5, -25, -30, -35, -40, -50]


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

    for c in ["o", "h", "l", "c", "v", "adj_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["o", "h", "l", "c", "adj_c"])
    df = df[
        (df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) &
        (df["c"] > 0) & (df["adj_c"] > 0)
    ].copy()

    if df.empty:
        return df

    df = df.sort_values("session_date").reset_index(drop=True)

    fac = df["adj_c"] / df["c"]
    df["ao"] = df["o"] * fac
    df["ah"] = df["h"] * fac
    df["al"] = df["l"] * fac
    df["ac"] = df["adj_c"]

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

    df["dollar_volume"] = df["c"] * df["v"].fillna(0.0)
    df["avg_dollar_volume20"] = df["dollar_volume"].rolling(20).mean()
    df["dollarvol_ratio_5_20"] = (
        df["dollar_volume"].rolling(5).mean() /
        df["avg_dollar_volume20"].replace(0, np.nan)
    )

    df["adj_close_jump_pct"] = 100.0 * (df["ac"] / df["ac"].shift(1) - 1.0)

    return df


def detector_passes(df, i):
    t1 = i - 1
    t20 = i - 20

    if i < MIN_SIGNAL_INDEX or t20 < 10:
        return False

    vals = [
        df.at[t1, "atr14_pct"],
        df.at[t1, "ret5_pct"],
        df.at[t1, "bb_width_pct"],
        df.at[t1, "dollarvol_ratio_5_20"],
        df.at[t20, "ret10_pct"],
        df.at[t1, "avg_dollar_volume20"],
        df.at[i, "ac"],
    ]
    if any(pd.isna(x) or not np.isfinite(x) for x in vals):
        return False

    if df.at[i, "ac"] < MIN_PRICE:
        return False
    if df.at[t1, "avg_dollar_volume20"] < MIN_AVG_DOLLAR_VOL20:
        return False

    if df.at[t1, "atr14_pct"] < ATR14_MIN:
        return False
    if df.at[t1, "ret5_pct"] > RET5_MAX:
        return False
    if df.at[t1, "bb_width_pct"] <= BB_WIDTH_MIN:
        return False
    if df.at[t1, "dollarvol_ratio_5_20"] <= DOLLARVOL_RATIO_MIN:
        return False
    if df.at[t20, "ret10_pct"] > T20_RET10_MAX:
        return False

    prior_jump = df.loc[max(1, i-20):i-1, "adj_close_jump_pct"].abs()
    if (prior_jump > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any():
        return False

    return True


def analyze_symbol(symbol, raw_rows):
    df = add_features(raw_rows)
    if len(df) < 70:
        return []

    out = []
    last_kept_i = -10**9

    for i in range(MIN_SIGNAL_INDEX, len(df)):
        dt = pd.Timestamp(df.at[i, "session_date"])
        if dt.year not in YEARS:
            continue

        if not detector_passes(df, i):
            continue

        if i - last_kept_i < COOLDOWN_SESSIONS:
            continue
        last_kept_i = i

        # Only mature 63-session signals are usable for this analysis.
        if i + 63 >= len(df):
            continue

        # Reject suspicious forward corporate-action-like jumps.
        forward_jump = df.loc[i+1:i+63, "adj_close_jump_pct"].abs()
        if (forward_jump > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any():
            continue

        base = float(df.at[i, "ac"])
        f = df.iloc[i+1:i+64].copy()

        high_gain = 100.0 * (f["ah"].to_numpy(dtype=float) / base - 1.0)
        hit_idx = np.flatnonzero(high_gain >= TP20)

        if not len(hit_idx):
            out.append({
                "symbol": symbol,
                "signal_date": dt.date().isoformat(),
                "year": int(dt.year),
                "signal_close": base,
                "hit20": 0,
                "first_tp20_day": None,
                "max_drawdown_before_tp20_pct": None,
                "lowest_low_before_tp20": None,
                "tp20_day_low_drawdown_pct": None,
            })
            continue

        # +1 because future window starts at Day+1.
        first_day = int(hit_idx[0] + 1)
        through_target = df.iloc[i+1:i+first_day+1]
        lowest_low = float(through_target["al"].min())
        max_dd = pct(lowest_low, base)

        target_day_low = float(df.at[i+first_day, "al"])
        target_day_low_dd = pct(target_day_low, base)

        out.append({
            "symbol": symbol,
            "signal_date": dt.date().isoformat(),
            "year": int(dt.year),
            "signal_close": base,
            "hit20": 1,
            "first_tp20_day": first_day,
            "max_drawdown_before_tp20_pct": float(max_dd),
            "lowest_low_before_tp20": lowest_low,
            "tp20_day_low_drawdown_pct": float(target_day_low_dd),
        })

    return out


DB = database()

with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())
    all_symbols = [
        r[0]
        for r in s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count() >= 80)
            .order_by(daily.c.symbol)
        ).all()
    ]

symbols = [s for s in all_symbols if looks_like_common_stock_symbol(s)]

print("\nRAJIH — STOP-LOSS BEFORE FIRST +20%")
print(f"Scanning {len(symbols)} cleaned symbols ...")

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

        r = analyze_symbol(symbol, raw)
        if r:
            rows.extend(r)

        if n % 250 == 0:
            print(f"[{n}/{len(symbols)}] mature signals collected={len(rows)}")

    except Exception as e:
        print(f"{symbol} ERROR: {e}")

if not rows:
    raise SystemExit("No mature signals found.")

df = pd.DataFrame(rows).sort_values(["signal_date","symbol"]).reset_index(drop=True)
df.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

wins = df[df["hit20"] == 1].copy()
losses = df[df["hit20"] == 0].copy()

# Distribution of drawdown before first TP20.
quantiles = [0, .01, .05, .10, .20, .25, .50, .75, .90, .95, .99, 1.0]
dist_rows = []
for q in quantiles:
    dist_rows.append({
        "quantile": q,
        "max_drawdown_before_tp20_pct": float(
            wins["max_drawdown_before_tp20_pct"].quantile(q)
        ) if len(wins) else None
    })
dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")

# Stop grid.
grid_rows = []
for stop in STOP_LEVELS:
    survived_opt = 0
    survived_cons = 0
    ambiguous = 0
    definite_stop = 0

    for _, r in wins.iterrows():
        dd = float(r["max_drawdown_before_tp20_pct"])
        target_day_dd = float(r["tp20_day_low_drawdown_pct"])

        # If the worst low never touched stop, definitely survives.
        if dd > stop:
            survived_opt += 1
            survived_cons += 1
            continue

        # The path touched stop somewhere before/on target day.
        # If target-day low also touches stop, target/stop order could be ambiguous.
        # If worst stop touch occurred before target day, definitely stopped.
        if target_day_dd <= stop:
            ambiguous += 1
            survived_opt += 1      # optimistic assumes +20 hit first
            # conservative assumes stop hit first -> no increment
        else:
            definite_stop += 1

    n = len(wins)
    grid_rows.append({
        "stop_loss_pct": stop,
        "tp20_winners": n,
        "definite_stopped_before_tp20": definite_stop,
        "ambiguous_same_target_day": ambiguous,
        "optimistic_survivors": survived_opt,
        "optimistic_survival_pct": 100.0 * survived_opt / n if n else None,
        "conservative_survivors": survived_cons,
        "conservative_survival_pct": 100.0 * survived_cons / n if n else None,
    })

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")

# Yearly.
year_rows = []
for y in YEARS:
    ydf = df[df["year"] == y]
    yw = ydf[ydf["hit20"] == 1]
    year_rows.append({
        "year": y,
        "mature_signals": int(len(ydf)),
        "hit20_n": int(len(yw)),
        "hit20_pct": 100.0 * len(yw) / len(ydf) if len(ydf) else None,
        "median_first_tp20_day": float(yw["first_tp20_day"].median()) if len(yw) else None,
        "median_dd_before_tp20_pct": float(yw["max_drawdown_before_tp20_pct"].median()) if len(yw) else None,
        "worst_dd_before_tp20_pct": float(yw["max_drawdown_before_tp20_pct"].min()) if len(yw) else None,
    })

year_df = pd.DataFrame(year_rows)
year_df.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

print("\n=== OVERALL ===")
print(f"Mature detector signals: {len(df)}")
print(f"Reached +20%: {len(wins)} ({100*len(wins)/len(df):.2f}%)")
print(f"Did not reach +20%: {len(losses)} ({100*len(losses)/len(df):.2f}%)")

if len(wins):
    print(f"Median first +20 day: {wins['first_tp20_day'].median():.1f}")
    print(f"Median drawdown before +20: {wins['max_drawdown_before_tp20_pct'].median():.2f}%")
    print(f"Worst drawdown before +20: {wins['max_drawdown_before_tp20_pct'].min():.2f}%")

print("\n=== DRAWDOWN DISTRIBUTION BEFORE +20 ===")
print(dist_df.to_string(index=False))

print("\n=== STOP GRID ===")
print(grid_df.to_string(index=False))

print("\n=== YEARLY ===")
print(year_df.to_string(index=False))

print("\n=== 20 WORST WINNERS BEFORE +20 ===")
worst_cols = [
    "year","signal_date","symbol","signal_close","first_tp20_day",
    "max_drawdown_before_tp20_pct","tp20_day_low_drawdown_pct"
]
print(
    wins.sort_values("max_drawdown_before_tp20_pct")
        .head(20)[worst_cols]
        .to_string(index=False)
)

report = []
report.append("RAJIH — STOP-LOSS BEFORE FIRST +20%")
report.append("="*100)
report.append(f"Mature detector signals: {len(df)}")
report.append(f"Reached +20%: {len(wins)} ({100*len(wins)/len(df):.2f}%)")
report.append(f"Did not reach +20%: {len(losses)} ({100*len(losses)/len(df):.2f}%)")
if len(wins):
    report.append(f"Median first +20 day: {wins['first_tp20_day'].median():.1f}")
    report.append(f"Median DD before +20: {wins['max_drawdown_before_tp20_pct'].median():.2f}%")
    report.append(f"Worst DD before +20: {wins['max_drawdown_before_tp20_pct'].min():.2f}%")
report.append("")
report.append("DRAWDOWN DISTRIBUTION")
report.append(dist_df.to_string(index=False))
report.append("")
report.append("STOP GRID")
report.append(grid_df.to_string(index=False))
report.append("")
report.append("YEARLY")
report.append(year_df.to_string(index=False))
report.append("")
report.append("20 WORST +20 WINNERS")
report.append(
    wins.sort_values("max_drawdown_before_tp20_pct")
        .head(20)[worst_cols]
        .to_string(index=False)
)
report.append("")
report.append(
    "CAUTION: daily bars cannot determine whether stop or +20 target occurred first "
    "when both are touched in the same candle. Use conservative survival when choosing "
    "a production stop."
)

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_CASES, OUT_DIST, OUT_GRID, OUT_YEAR, OUT_REPORT]:
    print(" ", p)
