#!/usr/bin/env python3
"""
Rajih — FULL US MARKET DETECTOR FROM SCRATCH (2024-2026)

This deliberately DOES NOT use:
- the historical +200% winner list
- the 4,066 precomputed first-stage signals
- Top-1/day files
- Candidate-A match files

It starts from every US symbol/date available in market_candles_1d and applies
the fixed detector forward, exactly as a scanner would have done in real time.

FIXED PRE-SIGNAL DETECTOR
-------------------------
At signal date S, using data available through S-1 / T-20 only:

1) T-1 ATR14% >= 10.96
2) T-1 5-session return <= -9.22%
3) T-1 Bollinger Width% > 100
4) T-1 Dollar-volume ratio 5/20 > 1.5
5) T-20 10-session return <= -15.3917%

EARLY CONFIRMATION
------------------
After the signal appears, wait through Day +5:

6) Day-5 close > signal-day high
7) Lowest adjusted low during Days +1..+5, measured from signal close,
   is >= -16.2879%

OUTCOMES
--------
From SIGNAL-DAY adjusted close, using adjusted HIGH over next 63 sessions:
+20%, +50%, +100%, +200%.

Also reports a second, more trade-realistic outcome from DAY-5 CLOSE
(the moment confirmation becomes knowable):
+20%, +50%, +100%, +200% over the remaining horizon through signal Day +63.

YEARS
-----
Signal dates in 2024, 2025, 2026.

DATA
----
PostgreSQL table: market_candles_1d

OUTPUTS
-------
/data/us_full_market_detector_presignals.csv
/data/us_full_market_detector_confirmed.csv
/data/us_full_market_detector_yearly.csv
/data/us_full_market_detector_summary.json
/data/us_full_market_detector_report.txt

Run:
  cd /app
  python scan_us_full_market_detector_from_scratch.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select, func

from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

OUT_PRE = DATA_DIR / "us_full_market_detector_presignals.csv"
OUT_CONF = DATA_DIR / "us_full_market_detector_confirmed.csv"
OUT_YEAR = DATA_DIR / "us_full_market_detector_yearly.csv"
OUT_JSON = DATA_DIR / "us_full_market_detector_summary.json"
OUT_REPORT = DATA_DIR / "us_full_market_detector_report.txt"

YEARS = (2024, 2025, 2026)

# Fixed thresholds — NO retuning.
ATR14_MIN = 10.96
RET5_MAX = -9.22
BB_WIDTH_MIN = 100.0
DOLLARVOL_RATIO_MIN = 1.5
T20_RET10_MAX = -15.3917

D5_DD_MIN = -16.2879

# To preserve the original anti-artifact logic.
MAX_ABS_ADJ_CLOSE_JUMP_PCT = 100.0

# Need enough history for T-20 and its 10-day return / rolling features.
MIN_SIGNAL_INDEX = 40


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else np.nan
    except Exception:
        return np.nan


def pct(new, old):
    if old is None or old == 0 or not np.isfinite(old):
        return np.nan
    return 100.0 * (new / old - 1.0)


def add_features(raw_rows):
    """
    raw_rows columns:
      session_date, o, h, l, c, v, adj_c
    Reconstruct adjusted OHLC = raw OHLC * adj_close/raw_close.
    """
    df = pd.DataFrame(
        raw_rows,
        columns=["session_date", "o", "h", "l", "c", "v", "adj_c"]
    )
    if df.empty:
        return df

    for col in ["o", "h", "l", "c", "v", "adj_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["o", "h", "l", "c", "adj_c"])
    df = df[(df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) &
            (df["c"] > 0) & (df["adj_c"] > 0)]
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

    # Wilder-style ATR14.
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr14_pct"] = 100.0 * atr14 / df["ac"]

    df["ret5_pct"] = 100.0 * (df["ac"] / df["ac"].shift(5) - 1.0)
    df["ret10_pct"] = 100.0 * (df["ac"] / df["ac"].shift(10) - 1.0)

    sma20 = df["ac"].rolling(20).mean()
    std20 = df["ac"].rolling(20).std(ddof=0)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df["bb_width_pct"] = 100.0 * (upper - lower) / sma20

    # Match prior research convention: raw close * volume for dollar volume.
    dollar_volume = df["c"] * df["v"].fillna(0.0)
    dv5 = dollar_volume.rolling(5).mean()
    dv20 = dollar_volume.rolling(20).mean()
    df["dollarvol_ratio_5_20"] = dv5 / dv20.replace(0, np.nan)

    df["adj_close_jump_pct"] = 100.0 * (df["ac"] / df["ac"].shift(1) - 1.0)

    return df


def detector_passes(df, i):
    """
    i = signal-day row.
    Detector only sees information available through T-1 and T-20.
    """
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
    ]
    if any(pd.isna(x) or not np.isfinite(x) for x in vals):
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

    # Prior-only corporate-action artifact guard.
    prior = df.loc[max(1, i - 20):i - 1, "adj_close_jump_pct"].abs()
    if (prior > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any():
        return False

    return True


def first_touch_day(highs, base, threshold_pct):
    gains = 100.0 * (highs / base - 1.0)
    idx = np.flatnonzero(gains >= threshold_pct)
    return int(idx[0] + 1) if len(idx) else None


def analyze_symbol(symbol, raw_rows):
    df = add_features(raw_rows)
    if len(df) < 70:
        return []

    out = []

    # Need +5 for confirmation. +63 maturity handled separately.
    for i in range(MIN_SIGNAL_INDEX, len(df) - 5):
        signal_date = pd.Timestamp(df.at[i, "session_date"])
        if signal_date.year not in YEARS:
            continue

        if not detector_passes(df, i):
            continue

        sig_close = float(df.at[i, "ac"])
        sig_high = float(df.at[i, "ah"])
        if sig_close <= 0 or sig_high <= 0:
            continue

        first5 = df.iloc[i + 1:i + 6]
        d5_close = float(df.at[i + 5, "ac"])
        dd5 = pct(float(first5["al"].min()), sig_close)

        d5_above_signal_high = d5_close > sig_high
        confirmed = d5_above_signal_high and dd5 >= D5_DD_MIN

        mature63 = i + 63 < len(df)

        hit20 = hit50 = hit100 = hit200 = None
        max_gain63 = max_dd63 = close63 = None
        first200_day = None

        # Post-confirmation outcomes from D5 close.
        d5_hit20 = d5_hit50 = d5_hit100 = d5_hit200 = None
        d5_max_gain = d5_max_dd = d5_close63_ret = None

        if mature63:
            # Forward adjusted-close jump check is diagnostic / data-cleanliness only.
            forward_jumps = df.loc[i + 1:i + 63, "adj_close_jump_pct"].abs()
            clean_forward = not (forward_jumps > MAX_ABS_ADJ_CLOSE_JUMP_PCT).any()

            if clean_forward:
                f63 = df.iloc[i + 1:i + 64]
                highs = f63["ah"].to_numpy(dtype=float)
                lows = f63["al"].to_numpy(dtype=float)

                gains = 100.0 * (highs / sig_close - 1.0)
                dds = 100.0 * (lows / sig_close - 1.0)

                max_gain63 = float(np.nanmax(gains))
                max_dd63 = float(np.nanmin(dds))
                close63 = pct(float(df.at[i + 63, "ac"]), sig_close)

                hit20 = int(np.any(gains >= 20))
                hit50 = int(np.any(gains >= 50))
                hit100 = int(np.any(gains >= 100))
                hit200 = int(np.any(gains >= 200))
                first200_day = first_touch_day(highs, sig_close, 200)

                # From moment confirmation is known: D5 close -> through S+63.
                post = df.iloc[i + 6:i + 64]
                if len(post):
                    ph = post["ah"].to_numpy(dtype=float)
                    pl = post["al"].to_numpy(dtype=float)
                    pg = 100.0 * (ph / d5_close - 1.0)
                    pdn = 100.0 * (pl / d5_close - 1.0)

                    d5_max_gain = float(np.nanmax(pg))
                    d5_max_dd = float(np.nanmin(pdn))
                    d5_close63_ret = pct(float(df.at[i + 63, "ac"]), d5_close)

                    d5_hit20 = int(np.any(pg >= 20))
                    d5_hit50 = int(np.any(pg >= 50))
                    d5_hit100 = int(np.any(pg >= 100))
                    d5_hit200 = int(np.any(pg >= 200))
            else:
                mature63 = False

        out.append(
            {
                "symbol": symbol,
                "signal_date": signal_date.date().isoformat(),
                "year": int(signal_date.year),
                "signal_close": sig_close,

                "t1_atr14_pct": float(df.at[i - 1, "atr14_pct"]),
                "t1_ret5_pct": float(df.at[i - 1, "ret5_pct"]),
                "t1_bb_width_pct": float(df.at[i - 1, "bb_width_pct"]),
                "t1_dollarvol_ratio_5_20": float(df.at[i - 1, "dollarvol_ratio_5_20"]),
                "t20_ret10_pct": float(df.at[i - 20, "ret10_pct"]),

                "d5_close": d5_close,
                "d5_close_above_signal_high": int(d5_above_signal_high),
                "first5_max_drawdown_pct": float(dd5),
                "confirmed": int(confirmed),

                "mature_63d": int(mature63),
                "hit20": hit20,
                "hit50": hit50,
                "hit100": hit100,
                "hit200": hit200,
                "max_gain_63d_pct": max_gain63,
                "max_drawdown_63d_pct": max_dd63,
                "close_return_day63_pct": close63,
                "first200_day": first200_day,

                "d5_entry_hit20": d5_hit20,
                "d5_entry_hit50": d5_hit50,
                "d5_entry_hit100": d5_hit100,
                "d5_entry_hit200": d5_hit200,
                "d5_entry_max_gain_to_day63_pct": d5_max_gain,
                "d5_entry_max_drawdown_to_day63_pct": d5_max_dd,
                "d5_entry_close_return_day63_pct": d5_close63_ret,
            }
        )

    return out


def summarize(df, confirmed_only=False):
    x = df[df["confirmed"] == 1].copy() if confirmed_only else df.copy()
    mature = x[x["mature_63d"] == 1].copy()

    res = {
        "signals": int(len(x)),
        "mature": int(len(mature)),
        "unmatured": int(len(x) - len(mature)),
        "symbols": int(x["symbol"].nunique()) if len(x) else 0,
        "dates": int(x["signal_date"].nunique()) if len(x) else 0,
    }

    for th in (20, 50, 100, 200):
        col = f"hit{th}"
        if len(mature):
            n = int(mature[col].fillna(0).sum())
            res[f"hit{th}_n"] = n
            res[f"hit{th}_pct"] = float(100.0 * n / len(mature))
        else:
            res[f"hit{th}_n"] = 0
            res[f"hit{th}_pct"] = None

    # Trade-realistic D5-close outcomes only matter for confirmed cases.
    if confirmed_only:
        for th in (20, 50, 100, 200):
            col = f"d5_entry_hit{th}"
            if len(mature):
                n = int(mature[col].fillna(0).sum())
                res[f"d5_hit{th}_n"] = n
                res[f"d5_hit{th}_pct"] = float(100.0 * n / len(mature))
            else:
                res[f"d5_hit{th}_n"] = 0
                res[f"d5_hit{th}_pct"] = None

    return res


DB = database()

with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

    symbols = [
        r[0]
        for r in s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count() >= 80)
            .order_by(daily.c.symbol)
        ).all()
    ]

print("\nRAJIH — FULL US MARKET DETECTOR FROM SCRATCH")
print(f"Symbols with >=80 daily bars: {len(symbols)}")
print("Scanning every eligible stock/date in 2024-2026 ...")

all_rows = []
symbols_with_signal = set()

for n, symbol in enumerate(symbols, 1):
    try:
        with DB() as s:
            raw_rows = s.execute(
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

        rows = analyze_symbol(symbol, raw_rows)
        if rows:
            all_rows.extend(rows)
            symbols_with_signal.add(symbol)
            print(
                f"[{n}/{len(symbols)}] {symbol}: "
                f"pre={len(rows)} confirmed={sum(r['confirmed'] for r in rows)}"
            )
        elif n % 250 == 0:
            print(
                f"[{n}/{len(symbols)}] progress | "
                f"pre-signals={len(all_rows)} | symbols-hit={len(symbols_with_signal)}"
            )
    except Exception as e:
        print(f"[{n}/{len(symbols)}] {symbol} ERROR: {e}")

if not all_rows:
    raise SystemExit("No detector signals found.")

df = pd.DataFrame(all_rows)
df = df.sort_values(["signal_date", "symbol"]).reset_index(drop=True)

confirmed = df[df["confirmed"] == 1].copy()

df.to_csv(OUT_PRE, index=False, encoding="utf-8-sig")
confirmed.to_csv(OUT_CONF, index=False, encoding="utf-8-sig")

overall_pre = summarize(df, confirmed_only=False)
overall_conf = summarize(df, confirmed_only=True)

year_rows = []
for y in YEARS:
    yr = df[df["year"] == y]
    a = summarize(yr, confirmed_only=False)
    c = summarize(yr, confirmed_only=True)

    year_rows.append(
        {
            "year": y,
            "presignals": a["signals"],
            "presignal_mature": a["mature"],
            "presignal_hit200_n": a["hit200_n"],
            "presignal_hit200_pct": a["hit200_pct"],

            "confirmed": c["signals"],
            "confirmed_mature": c["mature"],
            "confirmed_unmatured": c["unmatured"],

            "signalbase_hit20_n": c["hit20_n"],
            "signalbase_hit20_pct": c["hit20_pct"],
            "signalbase_hit50_n": c["hit50_n"],
            "signalbase_hit50_pct": c["hit50_pct"],
            "signalbase_hit100_n": c["hit100_n"],
            "signalbase_hit100_pct": c["hit100_pct"],
            "signalbase_hit200_n": c["hit200_n"],
            "signalbase_hit200_pct": c["hit200_pct"],

            "d5entry_hit20_n": c.get("d5_hit20_n"),
            "d5entry_hit20_pct": c.get("d5_hit20_pct"),
            "d5entry_hit50_n": c.get("d5_hit50_n"),
            "d5entry_hit50_pct": c.get("d5_hit50_pct"),
            "d5entry_hit100_n": c.get("d5_hit100_n"),
            "d5entry_hit100_pct": c.get("d5_hit100_pct"),
            "d5entry_hit200_n": c.get("d5_hit200_n"),
            "d5entry_hit200_pct": c.get("d5_hit200_pct"),
        }
    )

year_df = pd.DataFrame(year_rows)
year_df.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

summary = {
    "method": "full market from scratch; no precomputed winner/signal list",
    "universe_symbols": len(symbols),
    "thresholds": {
        "t1_atr14_pct_gte": ATR14_MIN,
        "t1_ret5_pct_lte": RET5_MAX,
        "t1_bb_width_pct_gt": BB_WIDTH_MIN,
        "t1_dollarvol_ratio_5_20_gt": DOLLARVOL_RATIO_MIN,
        "t20_ret10_pct_lte": T20_RET10_MAX,
        "confirmation_d5_close_gt_signal_high": True,
        "confirmation_first5_max_drawdown_pct_gte": D5_DD_MIN,
    },
    "overall_presignal": overall_pre,
    "overall_confirmed": overall_conf,
    "yearly": year_rows,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== FULL MARKET OVERALL ===")
print(
    f"Pre-signals: {overall_pre['signals']} | "
    f"mature={overall_pre['mature']} | "
    f"+200={overall_pre['hit200_n']}/{overall_pre['mature']} "
    f"({overall_pre['hit200_pct']}%)"
)
print(
    f"Confirmed: {overall_conf['signals']} | "
    f"mature={overall_conf['mature']} | "
    f"signal-base +50={overall_conf['hit50_pct']}% | "
    f"+100={overall_conf['hit100_pct']}% | "
    f"+200={overall_conf['hit200_pct']}%"
)
print(
    f"From D5 confirmation close: "
    f"+20={overall_conf.get('d5_hit20_pct')}% | "
    f"+50={overall_conf.get('d5_hit50_pct')}% | "
    f"+100={overall_conf.get('d5_hit100_pct')}% | "
    f"+200={overall_conf.get('d5_hit200_pct')}%"
)

print("\n=== FULL MARKET YEARLY ===")
print(year_df.to_string(index=False))

print("\n=== CONFIRMED CASES ===")
show_cols = [
    "year", "signal_date", "symbol", "signal_close", "d5_close",
    "first5_max_drawdown_pct",
    "hit50", "hit100", "hit200", "max_gain_63d_pct",
    "d5_entry_hit20", "d5_entry_hit50", "d5_entry_hit100", "d5_entry_hit200",
    "d5_entry_max_gain_to_day63_pct",
]
print(confirmed[show_cols].to_string(index=False) if len(confirmed) else "None")

report = []
report.append("RAJIH — FULL US MARKET DETECTOR FROM SCRATCH")
report.append("=" * 110)
report.append(f"Universe symbols: {len(symbols)}")
report.append("")
report.append("OVERALL")
report.append(
    f"Pre-signals={overall_pre['signals']} mature={overall_pre['mature']} "
    f"+200={overall_pre['hit200_n']} ({overall_pre['hit200_pct']}%)"
)
report.append(
    f"Confirmed={overall_conf['signals']} mature={overall_conf['mature']} | "
    f"Signal-base +50={overall_conf['hit50_pct']}% "
    f"+100={overall_conf['hit100_pct']}% +200={overall_conf['hit200_pct']}%"
)
report.append(
    f"D5-entry +20={overall_conf.get('d5_hit20_pct')}% "
    f"+50={overall_conf.get('d5_hit50_pct')}% "
    f"+100={overall_conf.get('d5_hit100_pct')}% "
    f"+200={overall_conf.get('d5_hit200_pct')}%"
)
report.append("")
report.append("YEARLY")
report.append(year_df.to_string(index=False))
report.append("")
report.append("CONFIRMED CASES")
report.append(confirmed[show_cols].to_string(index=False) if len(confirmed) else "None")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_PRE, OUT_CONF, OUT_YEAR, OUT_JSON, OUT_REPORT]:
    print(" ", p)
