#!/usr/bin/env python3
"""
Rajih — CLEAN FULL-US DETECTOR RE-TEST (2024-2026)

Purpose
-------
Re-run the detector from the full US daily database, but clean the population:

1) Start from ALL symbols/dates in market_candles_1d.
2) Keep ordinary-looking stock tickers only (conservative local heuristic).
3) Signal-day adjusted close >= $1.
4) Prior-20-session average dollar volume >= $1,000,000.
5) Reject prior corporate-action-like adjusted-close jumps >100%.
6) Apply the FIXED detector thresholds.
7) Enforce 63-session cooldown per symbol (one independent signal per stock per 63 sessions).
8) Apply the fixed Day-5 confirmation.
9) Report outcomes from:
      - signal close
      - Day-1 close
      - Day-2 close
      - Day-3 close
      - Day-5 close
   through signal Day+63.

IMPORTANT
---------
Day-1/2/3 entry results among Day-5-confirmed stocks are DIAGNOSTIC / HINDSIGHT:
the Day-5 confirmation was not yet knowable at those earlier entries.
Day-5-close results are the directly executable version of the fixed confirmation.

FIXED PRE-SIGNAL DETECTOR
-------------------------
T-1 ATR14% >= 10.96
T-1 5-day return <= -9.22%
T-1 Bollinger Width% > 100
T-1 Dollar-volume ratio 5/20 > 1.5
T-20 10-day return <= -15.3917%

FIXED DAY-5 CONFIRMATION
------------------------
Day-5 close > signal-day high
First-5-day max drawdown from signal close >= -16.2879%

OUTPUTS
-------
/data/us_clean_detector_presignals.csv
/data/us_clean_detector_confirmed.csv
/data/us_clean_detector_yearly.csv
/data/us_clean_detector_entry_comparison.csv
/data/us_clean_detector_summary.json
/data/us_clean_detector_report.txt

Run
---
cd /app
python scan_us_clean_detector_cooldown_entries.py
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

OUT_PRE = DATA_DIR / "us_clean_detector_presignals.csv"
OUT_CONF = DATA_DIR / "us_clean_detector_confirmed.csv"
OUT_YEAR = DATA_DIR / "us_clean_detector_yearly.csv"
OUT_ENTRY = DATA_DIR / "us_clean_detector_entry_comparison.csv"
OUT_JSON = DATA_DIR / "us_clean_detector_summary.json"
OUT_REPORT = DATA_DIR / "us_clean_detector_report.txt"

YEARS = (2024, 2025, 2026)

# Original clean-US universe requirements.
MIN_PRICE = 1.00
MIN_AVG_DOLLAR_VOL20 = 1_000_000.0

# Fixed detector thresholds.
ATR14_MIN = 10.96
RET5_MAX = -9.22
BB_WIDTH_MIN = 100.0
DOLLARVOL_RATIO_MIN = 1.5
T20_RET10_MAX = -15.3917

# Fixed confirmation.
D5_DD_MIN = -16.2879

# Independence / artifact controls.
COOLDOWN_SESSIONS = 63
MAX_ABS_ADJ_CLOSE_JUMP_PCT = 100.0
MIN_SIGNAL_INDEX = 40


def looks_like_common_stock_symbol(symbol: str) -> bool:
    """
    Conservative ticker heuristic when no local security-type metadata is available.

    Keeps simple alphabetic NASDAQ/NYSE-like symbols.
    Excludes common warrant/unit/right/preferred patterns:
      .  -  /  ^  =
      suffix W / WS / U / R
      very long/odd strings

    This cannot perfectly distinguish ETFs from common stocks without a security
    master. The report states this limitation explicitly.
    """
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

    prev_close = df["ac"].shift(1)
    tr = pd.concat(
        [
            df["ah"] - df["al"],
            (df["ah"] - prev_close).abs(),
            (df["al"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["atr14_pct"] = 100.0 * atr14 / df["ac"]

    df["ret5_pct"] = 100.0 * (df["ac"] / df["ac"].shift(5) - 1.0)
    df["ret10_pct"] = 100.0 * (df["ac"] / df["ac"].shift(10) - 1.0)

    sma20 = df["ac"].rolling(20).mean()
    std20 = df["ac"].rolling(20).std(ddof=0)
    df["bb_width_pct"] = 100.0 * ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20

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

    # Clean tradability filters.
    if df.at[i, "ac"] < MIN_PRICE:
        return False
    if df.at[t1, "avg_dollar_volume20"] < MIN_AVG_DOLLAR_VOL20:
        return False

    # Fixed detector.
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

    # Prior-only corporate-action guard.
    prior_jump = df.loc[max(1, i-20):i-1, "adj_close_jump_pct"].abs()
    if (prior_jump > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any():
        return False

    return True


def horizon_outcomes(df, signal_i, entry_i):
    """
    Entry at close of entry_i.
    Outcome window ends at signal_i+63.
    Requires signal_i+63 to exist.
    """
    if signal_i + 63 >= len(df):
        return None

    # reject suspicious future adjusted-close discontinuities
    future_jump = df.loc[signal_i+1:signal_i+63, "adj_close_jump_pct"].abs()
    if (future_jump > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any():
        return None

    entry = float(df.at[entry_i, "ac"])
    if not np.isfinite(entry) or entry <= 0:
        return None

    start = entry_i + 1
    end = signal_i + 63
    if start > end:
        return None

    f = df.iloc[start:end+1]
    highs = f["ah"].to_numpy(dtype=float)
    lows = f["al"].to_numpy(dtype=float)

    gains = 100.0 * (highs / entry - 1.0)
    dds = 100.0 * (lows / entry - 1.0)

    return {
        "hit20": int(np.any(gains >= 20)),
        "hit50": int(np.any(gains >= 50)),
        "hit100": int(np.any(gains >= 100)),
        "hit200": int(np.any(gains >= 200)),
        "max_gain_pct": float(np.nanmax(gains)) if len(gains) else None,
        "max_drawdown_pct": float(np.nanmin(dds)) if len(dds) else None,
        "close63_ret_pct": pct(float(df.at[signal_i+63, "ac"]), entry),
    }


def analyze_symbol(symbol, raw_rows):
    df = add_features(raw_rows)
    if len(df) < 70:
        return []

    out = []
    last_kept_i = -10**9

    for i in range(MIN_SIGNAL_INDEX, len(df)-5):
        dt = pd.Timestamp(df.at[i, "session_date"])
        if dt.year not in YEARS:
            continue

        if not detector_passes(df, i):
            continue

        # One independent signal per stock per 63 trading sessions.
        if i - last_kept_i < COOLDOWN_SESSIONS:
            continue

        last_kept_i = i

        sig_close = float(df.at[i, "ac"])
        sig_high = float(df.at[i, "ah"])

        first5 = df.iloc[i+1:i+6]
        d5_close = float(df.at[i+5, "ac"])
        dd5 = pct(float(first5["al"].min()), sig_close)

        d5_above = d5_close > sig_high
        confirmed = d5_above and dd5 >= D5_DD_MIN

        row = {
            "symbol": symbol,
            "signal_date": dt.date().isoformat(),
            "year": int(dt.year),
            "signal_close": sig_close,
            "avg_dollar_volume20": float(df.at[i-1, "avg_dollar_volume20"]),
            "t1_atr14_pct": float(df.at[i-1, "atr14_pct"]),
            "t1_ret5_pct": float(df.at[i-1, "ret5_pct"]),
            "t1_bb_width_pct": float(df.at[i-1, "bb_width_pct"]),
            "t1_dollarvol_ratio_5_20": float(df.at[i-1, "dollarvol_ratio_5_20"]),
            "t20_ret10_pct": float(df.at[i-20, "ret10_pct"]),
            "d5_close_above_signal_high": int(d5_above),
            "first5_max_drawdown_pct": float(dd5),
            "confirmed": int(confirmed),
        }

        # Signal-close outcome.
        sig_out = horizon_outcomes(df, i, i)
        mature = sig_out is not None
        row["mature_63d"] = int(mature)

        if sig_out:
            for k, v in sig_out.items():
                row[f"signal_{k}"] = v
        else:
            for k in [
                "hit20","hit50","hit100","hit200",
                "max_gain_pct","max_drawdown_pct","close63_ret_pct"
            ]:
                row[f"signal_{k}"] = None

        # Diagnostic/executable entry comparisons.
        for day in (1, 2, 3, 5):
            ent_i = i + day
            row[f"d{day}_entry_close"] = float(df.at[ent_i, "ac"])
            res = horizon_outcomes(df, i, ent_i) if mature else None
            for k in [
                "hit20","hit50","hit100","hit200",
                "max_gain_pct","max_drawdown_pct","close63_ret_pct"
            ]:
                row[f"d{day}_{k}"] = res[k] if res else None

        out.append(row)

    return out


def summarize(df, prefix):
    mature = df[df["mature_63d"] == 1].copy()
    out = {
        "n": int(len(df)),
        "mature": int(len(mature)),
        "unmatured": int(len(df)-len(mature)),
        "symbols": int(df["symbol"].nunique()) if len(df) else 0,
        "dates": int(df["signal_date"].nunique()) if len(df) else 0,
    }
    for th in (20,50,100,200):
        col = f"{prefix}_hit{th}"
        if len(mature):
            n = int(mature[col].fillna(0).sum())
            out[f"hit{th}_n"] = n
            out[f"hit{th}_pct"] = 100.0*n/len(mature)
        else:
            out[f"hit{th}_n"] = 0
            out[f"hit{th}_pct"] = None
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

print("\nRAJIH — CLEAN FULL-US DETECTOR RE-TEST")
print(f"All symbols with >=80 bars: {len(all_symbols)}")
print(f"Ordinary-looking stock symbols retained: {len(symbols)}")
print("NOTE: local DB has no confirmed security-type master; ETF exclusion is not perfect.")
print("Scanning 2024-2026 with price/liquidity filters + 63-session cooldown ...")

all_rows = []

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

        rows = analyze_symbol(symbol, raw)
        if rows:
            all_rows.extend(rows)
            print(
                f"[{n}/{len(symbols)}] {symbol}: "
                f"independent pre={len(rows)} confirmed={sum(r['confirmed'] for r in rows)}"
            )
        elif n % 250 == 0:
            print(f"[{n}/{len(symbols)}] progress | pre={len(all_rows)}")

    except Exception as e:
        print(f"[{n}/{len(symbols)}] {symbol} ERROR: {e}")

if not all_rows:
    raise SystemExit("No clean detector signals found.")

df = pd.DataFrame(all_rows).sort_values(["signal_date","symbol"]).reset_index(drop=True)
conf = df[df["confirmed"] == 1].copy()

df.to_csv(OUT_PRE, index=False, encoding="utf-8-sig")
conf.to_csv(OUT_CONF, index=False, encoding="utf-8-sig")

# Overall entry comparison among confirmed cases.
entry_rows = []
for prefix, label, executable in [
    ("signal", "Signal close", False),
    ("d1", "Day 1 close", False),
    ("d2", "Day 2 close", False),
    ("d3", "Day 3 close", False),
    ("d5", "Day 5 close", True),
]:
    s = summarize(conf, prefix)
    entry_rows.append({
        "entry": label,
        "executable_with_fixed_D5_confirmation": int(executable),
        **s,
    })

entry_df = pd.DataFrame(entry_rows)
entry_df.to_csv(OUT_ENTRY, index=False, encoding="utf-8-sig")

year_rows = []
for y in YEARS:
    yp = df[df["year"] == y]
    yc = conf[conf["year"] == y]

    pre_s = summarize(yp, "signal")
    row = {
        "year": y,
        "presignals": pre_s["n"],
        "presignal_mature": pre_s["mature"],
        "presignal_hit200_n": pre_s["hit200_n"],
        "presignal_hit200_pct": pre_s["hit200_pct"],
        "confirmed": int(len(yc)),
        "confirmed_mature": int((yc["mature_63d"] == 1).sum()),
    }

    for prefix in ("signal","d1","d2","d3","d5"):
        s = summarize(yc, prefix)
        for th in (20,50,100,200):
            row[f"{prefix}_hit{th}_n"] = s[f"hit{th}_n"]
            row[f"{prefix}_hit{th}_pct"] = s[f"hit{th}_pct"]

    year_rows.append(row)

year_df = pd.DataFrame(year_rows)
year_df.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

summary = {
    "population": {
        "all_symbols_with_80_bars": len(all_symbols),
        "ordinary_symbol_heuristic_retained": len(symbols),
        "security_type_limitation": (
            "No local security master was used; ticker heuristic removes obvious "
            "warrants/units/rights, but ETF exclusion is not guaranteed."
        ),
    },
    "filters": {
        "min_signal_price": MIN_PRICE,
        "min_prior20_avg_dollar_volume": MIN_AVG_DOLLAR_VOL20,
        "cooldown_sessions_per_symbol": COOLDOWN_SESSIONS,
        "prior_adj_close_jump_abs_max_pct": MAX_ABS_ADJ_CLOSE_JUMP_PCT,
    },
    "detector": {
        "t1_atr14_pct_gte": ATR14_MIN,
        "t1_ret5_pct_lte": RET5_MAX,
        "t1_bb_width_pct_gt": BB_WIDTH_MIN,
        "t1_dollarvol_ratio_5_20_gt": DOLLARVOL_RATIO_MIN,
        "t20_ret10_pct_lte": T20_RET10_MAX,
    },
    "confirmation": {
        "d5_close_gt_signal_high": True,
        "first5_max_drawdown_pct_gte": D5_DD_MIN,
    },
    "entry_comparison": entry_rows,
    "yearly": year_rows,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== CLEAN FULL MARKET OVERALL ===")
print(f"Independent pre-signals: {len(df)}")
print(f"Confirmed: {len(conf)}")
print(entry_df.to_string(index=False))

print("\n=== CLEAN FULL MARKET YEARLY ===")
print(year_df.to_string(index=False))

print("\n=== CLEAN CONFIRMED CASES ===")
cols = [
    "year","signal_date","symbol","signal_close","avg_dollar_volume20",
    "first5_max_drawdown_pct",
    "signal_hit50","signal_hit100","signal_hit200",
    "d1_hit50","d1_hit100","d1_hit200",
    "d2_hit50","d2_hit100","d2_hit200",
    "d3_hit50","d3_hit100","d3_hit200",
    "d5_hit50","d5_hit100","d5_hit200",
]
print(conf[cols].to_string(index=False) if len(conf) else "None")

report = []
report.append("RAJIH — CLEAN FULL-US DETECTOR RE-TEST")
report.append("="*110)
report.append(f"All >=80-bar symbols: {len(all_symbols)}")
report.append(f"Ordinary-looking ticker heuristic retained: {len(symbols)}")
report.append(f"Independent pre-signals after 63-session cooldown: {len(df)}")
report.append(f"Confirmed: {len(conf)}")
report.append("")
report.append("ENTRY COMPARISON AMONG CONFIRMED CASES")
report.append(entry_df.to_string(index=False))
report.append("")
report.append("YEARLY")
report.append(year_df.to_string(index=False))
report.append("")
report.append(
    "NOTE: Day1/Day2/Day3 rows are hindsight diagnostics because Day5 confirmation "
    "was not knowable yet. Day5 close is the executable fixed-confirmation entry."
)
OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_PRE, OUT_CONF, OUT_YEAR, OUT_ENTRY, OUT_JSON, OUT_REPORT]:
    print(" ", p)
