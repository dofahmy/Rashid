#!/usr/bin/env python3
"""
Rajih — find stocks that gained +200% within ~3 trading months,
but did NOT complete that +200% move in the first month.

Definition used:
- Start from a daily close.
- Look forward 63 trading sessions (~3 months).
- The stock must reach at least +200% versus the start close (3x price).
- It must NOT have reached +200% during the first 21 sessions.
  => first +200% touch must be on trading day 22..63.
- Uses adjusted prices/highs.
- Rejects obvious adjusted-price discontinuities >100% in the 20 sessions
  before the start, to reduce bad corporate-action/ticker artifacts.
- Requires at least 220 historical daily bars before/including the setup.
- Keeps ONE event per symbol: the event with the strongest 63-session peak
  among qualifying events.

No price cap is imposed. This scans the whole daily universe.

Outputs:
  /data/daily_200pct_3months_not_first_month.csv
  /data/daily_200pct_3months_not_first_month_summary.json

Run:
  python find_daily_200pct_3months_not_first_month.py
"""

from __future__ import annotations

import os, csv, json, math
from pathlib import Path
from statistics import median, mean

from sqlalchemy import MetaData, Table, select, func
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = DATA_DIR / "daily_200pct_3months_not_first_month.csv"
OUT_JSON = DATA_DIR / "daily_200pct_3months_not_first_month_summary.json"

TARGET_PCT = 200.0
MONTH1 = 21
MONTH2 = 42
MONTH3 = 63
MIN_HISTORY = 220
MAX_PRIOR_ADJ_JUMP_PCT = 100.0


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def load_symbol(DB, daily, sym):
    with DB() as s:
        raw = list(
            s.execute(
                select(
                    daily.c.session_date,
                    daily.c.o,
                    daily.c.h,
                    daily.c.l,
                    daily.c.c,
                    daily.c.v,
                    daily.c.adj_c,
                )
                .where(daily.c.symbol == sym)
                .order_by(daily.c.session_date)
            ).all()
        )

    out = []
    for d, o, h, l, c, v, ac in raw:
        ro, rh, rl, rc, adj = map(finite, (o, h, l, c, ac))
        if None in (ro, rh, rl, rc, adj) or min(ro, rh, rl, rc, adj) <= 0:
            continue
        fac = adj / rc
        out.append({
            "date": str(d),
            "raw_close": rc,
            "close": adj,
            "high": rh * fac,
            "low": rl * fac,
            "volume": float(v or 0),
        })
    return out


def prior_clean(rows, i):
    if i < 20:
        return False
    for j in range(i - 19, i + 1):
        if j <= 0:
            continue
        r = abs(pct(rows[j]["close"], rows[j - 1]["close"]))
        if r > MAX_PRIOR_ADJ_JUMP_PCT:
            return False
    return True


def avg_dollar_volume20(rows, i):
    if i < 19:
        return None
    vals = [
        rows[j]["raw_close"] * rows[j]["volume"]
        for j in range(i - 19, i + 1)
    ]
    return sum(vals) / len(vals)


DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())
    symbols = list(
        s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count() >= MIN_HISTORY + MONTH3)
            .order_by(daily.c.symbol)
        ).scalars().all()
    )

print("\nRajih — +200% WITHIN 3 MONTHS, NOT ALL IN MONTH 1")
print(f"Universe: {len(symbols)}")
print("Rule: first +200% touch must occur on trading day 22..63.\n")

best_by_symbol = {}
all_event_count = 0

for n, sym in enumerate(symbols, 1):
    rows = load_symbol(DB, daily, sym)
    if len(rows) < MIN_HISTORY + MONTH3:
        continue

    for i in range(MIN_HISTORY - 1, len(rows) - MONTH3):
        if not prior_clean(rows, i):
            continue

        start = rows[i]["close"]
        if start <= 0:
            continue

        # First month peak: sessions 1..21
        m1_slice = rows[i + 1 : i + MONTH1 + 1]
        m2_slice = rows[i + MONTH1 + 1 : i + MONTH2 + 1]
        m3_slice = rows[i + MONTH2 + 1 : i + MONTH3 + 1]
        full = rows[i + 1 : i + MONTH3 + 1]

        m1_peak = max(x["high"] for x in m1_slice)
        m2_peak = max(x["high"] for x in m2_slice)
        m3_peak = max(x["high"] for x in m3_slice)
        full_peak = max(x["high"] for x in full)

        m1_gain = pct(m1_peak, start)
        m2_gain = pct(m2_peak, start)
        m3_gain = pct(m3_peak, start)
        full_gain = pct(full_peak, start)

        # Must reach +200% within 63 sessions, but not during first 21.
        if full_gain < TARGET_PCT or m1_gain >= TARGET_PCT:
            continue

        first_hit = None
        peak_j = None
        peak_val = -1.0

        for j in range(i + 1, i + MONTH3 + 1):
            g = pct(rows[j]["high"], start)
            if first_hit is None and g >= TARGET_PCT:
                first_hit = j - i
            if rows[j]["high"] > peak_val:
                peak_val = rows[j]["high"]
                peak_j = j

        if first_hit is None or first_hit <= MONTH1:
            continue

        all_event_count += 1

        # Month-end close returns for trajectory, not just highs.
        close21 = pct(rows[i + MONTH1]["close"], start)
        close42 = pct(rows[i + MONTH2]["close"], start)
        close63 = pct(rows[i + MONTH3]["close"], start)

        event = {
            "symbol": sym,
            "start_date": rows[i]["date"],
            "start_raw_close": round(rows[i]["raw_close"], 6),
            "start_adjusted_close": round(start, 6),
            "avg_dollar_volume20": round(avg_dollar_volume20(rows, i) or 0, 2),
            "first_200_touch_day": first_hit,
            "first_200_touch_date": rows[i + first_hit]["date"],
            "peak_day_63": peak_j - i,
            "peak_date_63": rows[peak_j]["date"],
            "month1_max_gain_pct": round(m1_gain, 4),
            "month2_block_max_gain_pct": round(m2_gain, 4),
            "month3_block_max_gain_pct": round(m3_gain, 4),
            "max_gain_63d_pct": round(full_gain, 4),
            "close_return_day21_pct": round(close21, 4),
            "close_return_day42_pct": round(close42, 4),
            "close_return_day63_pct": round(close63, 4),
            "gain_added_after_month1_pct_points": round(full_gain - m1_gain, 4),
        }

        prev = best_by_symbol.get(sym)
        if prev is None or event["max_gain_63d_pct"] > prev["max_gain_63d_pct"]:
            best_by_symbol[sym] = event

    if n % 250 == 0 or n == len(symbols):
        print(
            f"Scanned {n}/{len(symbols)} | "
            f"qualifying events={all_event_count} | "
            f"distinct symbols={len(best_by_symbol)}"
        )

results = sorted(
    best_by_symbol.values(),
    key=lambda r: (r["max_gain_63d_pct"], r["avg_dollar_volume20"]),
    reverse=True
)

if not results:
    raise SystemExit("No qualifying stocks found.")

with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

first_days = [r["first_200_touch_day"] for r in results]
m1g = [r["month1_max_gain_pct"] for r in results]
fullg = [r["max_gain_63d_pct"] for r in results]

summary = {
    "definition": {
        "target_gain_pct": TARGET_PCT,
        "three_month_sessions": MONTH3,
        "first_month_sessions": MONTH1,
        "rule": "first +200% touch must occur after day 21 and on/before day 63",
        "one_event_per_symbol": "strongest qualifying 63-session peak",
    },
    "qualifying_events_before_dedup": all_event_count,
    "distinct_symbols": len(results),
    "median_first_200_touch_day": round(median(first_days), 2),
    "mean_first_200_touch_day": round(mean(first_days), 2),
    "median_month1_max_gain_pct": round(median(m1g), 4),
    "median_63d_max_gain_pct": round(median(fullg), 4),
    "outputs": [str(OUT_CSV), str(OUT_JSON)],
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== RESULTS ===")
print(f"Qualifying historical events: {all_event_count}")
print(f"Distinct symbols: {len(results)}")
print(f"Median first +200% touch: day {summary['median_first_200_touch_day']}")
print(f"Median month-1 max gain: {summary['median_month1_max_gain_pct']:.2f}%")
print(f"Median 3-month max gain: {summary['median_63d_max_gain_pct']:.2f}%")

print("\nTOP 30:")
print("rank | symbol | start | start$ | M1 max | first +200 day | 63d max | added after M1")
for rank, r in enumerate(results[:30], 1):
    print(
        f"{rank:>4} | {r['symbol']:<7} | {r['start_date']} | "
        f"${r['start_raw_close']:>8.2f} | "
        f"{r['month1_max_gain_pct']:>7.1f}% | "
        f"{r['first_200_touch_day']:>14} | "
        f"{r['max_gain_63d_pct']:>7.1f}% | "
        f"{r['gain_added_after_month1_pct_points']:>10.1f}pp"
    )

print(f"\nCreated:\n {OUT_CSV}\n {OUT_JSON}")
