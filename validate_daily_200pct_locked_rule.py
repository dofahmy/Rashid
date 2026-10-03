#!/usr/bin/env python3
"""
Rajih — OUT-OF-SAMPLE validation of the locked +200% runner rule.

LOCKED RULE (discovered on the separate 30-winner research sample):
    ATR14% at T-1 >= 10.96%
    5-day return at T-1 <= -9.22%

Eligibility / data-quality rules:
    - start raw close >= $1
    - prior 20d avg dollar volume >= $1,000,000
    - enough history for indicators
    - no >100% adjusted-close jump in prior 20 sessions
    - 63-session cooldown per symbol so one episode does not create many near-duplicate signals

IMPORTANT:
    The signal uses PRE-SIGNAL information only.
    Outcomes are measured AFTER the setup:
      +50%, +100%, +200% within 63 trading sessions
      first-touch day
      max gain / max drawdown
      final close return at day 63

Discovery sample:
    The 30 winner symbols used to derive the rule are excluded.
    If /data/daily_200pct_diverse30_winners.csv exists, symbols are loaded from it.
    A hardcoded fallback list is also included.

Also reports a more regime-diverse subset:
    one signal per distinct setup date, selecting the highest 20d dollar-volume candidate
    using pre-signal data only.

Outputs:
    /data/daily_200pct_locked_rule_all_signals.csv
    /data/daily_200pct_locked_rule_date_diverse.csv
    /data/daily_200pct_locked_rule_yearly.csv
    /data/daily_200pct_locked_rule_summary.json
    /data/daily_200pct_locked_rule_report.txt

Run:
    python validate_daily_200pct_locked_rule.py
"""

from __future__ import annotations

import os, csv, json, math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from sqlalchemy import MetaData, Table, select, func
from core import database


DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DISCOVERY_FILE = DATA_DIR / "daily_200pct_diverse30_winners.csv"

OUT_ALL = DATA_DIR / "daily_200pct_locked_rule_all_signals.csv"
OUT_DIVERSE = DATA_DIR / "daily_200pct_locked_rule_date_diverse.csv"
OUT_YEARLY = DATA_DIR / "daily_200pct_locked_rule_yearly.csv"
OUT_JSON = DATA_DIR / "daily_200pct_locked_rule_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_locked_rule_report.txt"

# Locked thresholds — DO NOT tune in this validation.
ATR_THRESHOLD = 10.96
RET5_THRESHOLD = -9.22

MIN_PRICE = 1.0
MIN_DOLLAR_VOL20 = 1_000_000.0
LOOKAHEAD = 63
COOLDOWN = 63
MIN_HISTORY = 220
MAX_PRIOR_ADJ_JUMP = 100.0

FALLBACK_DISCOVERY = {
    "LUNR","MSTR","DJT","BTDR","RDW","HIMS","LTBR","KTTA","TDUP","APLD",
    "SLDP","NEON","IXHL","ICU","RR","OPEN","SES","GRAL","NUVB","BIOA",
    "ASTI","CHRN","PRLD","AAOI","DOCN","OCC","CLOV","PAVS","ADVB","MF"
}


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def fmt(x, n=4):
    if x is None:
        return None
    try:
        y = float(x)
        return round(y, n) if math.isfinite(y) else None
    except Exception:
        return None


def load_discovery_symbols():
    syms = set(FALLBACK_DISCOVERY)
    if DISCOVERY_FILE.exists():
        try:
            with DISCOVERY_FILE.open(encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    s = (r.get("symbol") or "").strip().upper()
                    if s:
                        syms.add(s)
        except Exception:
            pass
    return syms


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

    rows = []
    for d, o, h, l, c, v, ac in raw:
        ro, rh, rl, rc, adj = map(finite, (o, h, l, c, ac))
        if None in (ro, rh, rl, rc, adj) or min(ro, rh, rl, rc, adj) <= 0:
            continue
        fac = adj / rc
        rows.append({
            "date": str(d),
            "raw_close": rc,
            "open": ro * fac,
            "high": rh * fac,
            "low": rl * fac,
            "close": adj,
            "volume": float(v or 0),
        })
    return rows


def atr_all(h, l, c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    tr = [None] * len(c)
    for i in range(1, len(c)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = sum(tr[1:n+1]) / n
    out[n] = a
    for i in range(n+1, len(c)):
        a = ((n-1)*a + tr[i]) / n
        out[i] = a
    return out


def build(rows):
    c = [r["close"] for r in rows]
    h = [r["high"] for r in rows]
    l = [r["low"] for r in rows]
    rc = [r["raw_close"] for r in rows]
    v = [r["volume"] for r in rows]

    atr14 = atr_all(h, l, c, 14)

    dollar = [rc[i] * v[i] for i in range(len(c))]
    avgdol20 = [None] * len(c)
    for i in range(19, len(c)):
        avgdol20[i] = mean(dollar[i-19:i+1])

    return {
        "c": c, "h": h, "l": l, "rc": rc,
        "atr14": atr14, "avgdol20": avgdol20,
    }


def prior_clean(rows, i):
    if i < 20:
        return False
    for j in range(i-19, i+1):
        if j <= 0:
            continue
        if abs(pct(rows[j]["close"], rows[j-1]["close"])) > MAX_PRIOR_ADJ_JUMP:
            return False
    return True


def outcome(rows, i):
    if i + LOOKAHEAD >= len(rows):
        return None

    start = rows[i]["close"]
    future = rows[i+1:i+LOOKAHEAD+1]

    max_gain = max(pct(r["high"], start) for r in future)
    max_dd = min(pct(r["low"], start) for r in future)
    final_ret = pct(rows[i+LOOKAHEAD]["close"], start)

    first = {}
    for th in (20, 50, 100, 200):
        first[th] = None

    peak_day = None
    peak_gain = -1e99

    for day, r in enumerate(future, 1):
        g = pct(r["high"], start)
        if g > peak_gain:
            peak_gain = g
            peak_day = day
        for th in first:
            if first[th] is None and g >= th:
                first[th] = day

    # Forward discontinuity is measured only as a diagnostic flag.
    # It is NOT used to generate/select the signal.
    future_discontinuity = 0
    for j in range(i+1, i+LOOKAHEAD+1):
        if abs(pct(rows[j]["close"], rows[j-1]["close"])) > 100:
            future_discontinuity = 1
            break

    return {
        "max_gain_63d_pct": fmt(max_gain),
        "max_drawdown_63d_pct": fmt(max_dd),
        "close_return_day63_pct": fmt(final_ret),
        "peak_day": peak_day,
        "hit20": int(max_gain >= 20),
        "hit50": int(max_gain >= 50),
        "hit100": int(max_gain >= 100),
        "hit200": int(max_gain >= 200),
        "first20_day": first[20],
        "first50_day": first[50],
        "first100_day": first[100],
        "first200_day": first[200],
        "future_discontinuity_flag": future_discontinuity,
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    if not rows:
        return {}
    n = len(rows)

    def hit(name):
        c = sum(int(r[name]) for r in rows)
        return c, 100*c/n

    h20 = hit("hit20")
    h50 = hit("hit50")
    h100 = hit("hit100")
    h200 = hit("hit200")

    g = [float(r["max_gain_63d_pct"]) for r in rows]
    dd = [float(r["max_drawdown_63d_pct"]) for r in rows]
    final = [float(r["close_return_day63_pct"]) for r in rows]
    d200 = [int(r["first200_day"]) for r in rows if r["first200_day"] not in (None, "")]

    return {
        "n": n,
        "hit20_n": h20[0], "hit20_rate_pct": fmt(h20[1]),
        "hit50_n": h50[0], "hit50_rate_pct": fmt(h50[1]),
        "hit100_n": h100[0], "hit100_rate_pct": fmt(h100[1]),
        "hit200_n": h200[0], "hit200_rate_pct": fmt(h200[1]),
        "median_max_gain_63d_pct": fmt(median(g)),
        "mean_max_gain_63d_pct": fmt(mean(g)),
        "median_max_drawdown_63d_pct": fmt(median(dd)),
        "median_close_return_day63_pct": fmt(median(final)),
        "median_first200_day": fmt(median(d200)) if d200 else None,
        "future_discontinuity_n": sum(int(r["future_discontinuity_flag"]) for r in rows),
    }


DISCOVERY = load_discovery_symbols()

DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())
    symbols = list(
        s.execute(
            select(daily.c.symbol)
            .group_by(daily.c.symbol)
            .having(func.count() >= MIN_HISTORY + LOOKAHEAD)
            .order_by(daily.c.symbol)
        ).scalars().all()
    )

print("\nRajih — LOCKED +200% RUNNER RULE VALIDATION")
print(f"Universe: {len(symbols)}")
print(f"Excluded discovery symbols: {len(DISCOVERY)}")
print(f"LOCKED: ATR14% T-1 >= {ATR_THRESHOLD:.2f}% AND 5d return T-1 <= {RET5_THRESHOLD:.2f}%")
print("Signal selection uses pre-signal data only.\n")

signals = []
by_date = defaultdict(list)

for n, sym in enumerate(symbols, 1):
    if sym in DISCOVERY:
        continue

    rows = load_symbol(DB, daily, sym)
    if len(rows) < MIN_HISTORY + LOOKAHEAD:
        continue

    F = build(rows)
    last_signal_i = -9999

    # setup date i; locked features are measured at i-1
    for i in range(MIN_HISTORY-1, len(rows)-LOOKAHEAD):
        if i - last_signal_i < COOLDOWN:
            continue

        t = i - 1
        if t < 20:
            continue

        if rows[i]["raw_close"] < MIN_PRICE:
            continue

        liq = F["avgdol20"][t]
        if liq is None or liq < MIN_DOLLAR_VOL20:
            continue

        if not prior_clean(rows, t):
            continue

        atrv = F["atr14"][t]
        if atrv is None or F["c"][t] <= 0:
            continue

        atr_pct = 100 * atrv / F["c"][t]
        ret5 = pct(F["c"][t], F["c"][t-5])

        if atr_pct < ATR_THRESHOLD:
            continue
        if ret5 > RET5_THRESHOLD:
            continue

        out = outcome(rows, i)
        if out is None:
            continue

        row = {
            "symbol": sym,
            "setup_date": rows[i]["date"],
            "setup_raw_close": fmt(rows[i]["raw_close"], 6),
            "t_minus_1_date": rows[t]["date"],
            "atr14_pct_tminus1": fmt(atr_pct),
            "ret5_pct_tminus1": fmt(ret5),
            "avg_dollar_volume20_tminus1": fmt(liq, 2),
            **out,
        }
        signals.append(row)
        by_date[row["setup_date"]].append(row)
        last_signal_i = i

    if n % 250 == 0 or n == len(symbols):
        print(
            f"Scanned {n}/{len(symbols)} | "
            f"signals={len(signals)} | distinct dates={len(by_date)}"
        )

# A regime-diverse view: one highest-liquidity signal per date.
diverse = []
for d in sorted(by_date):
    pool = sorted(
        by_date[d],
        key=lambda r: (float(r["avg_dollar_volume20_tminus1"]), r["symbol"]),
        reverse=True,
    )
    diverse.append(pool[0])

write_csv(OUT_ALL, signals)
write_csv(OUT_DIVERSE, diverse)

all_summary = summarize(signals)
div_summary = summarize(diverse)

# Yearly stats for both views.
yearly_rows = []
for label, rows_ in (("all_signals", signals), ("one_per_date", diverse)):
    buckets = defaultdict(list)
    for r in rows_:
        buckets[r["setup_date"][:4]].append(r)
    for year in sorted(buckets):
        s = summarize(buckets[year])
        yearly_rows.append({"sample": label, "year": year, **s})

write_csv(OUT_YEARLY, yearly_rows)

summary = {
    "locked_rule": {
        "atr14_pct_tminus1_min": ATR_THRESHOLD,
        "ret5_pct_tminus1_max": RET5_THRESHOLD,
    },
    "eligibility": {
        "min_setup_raw_close": MIN_PRICE,
        "min_avg_dollar_volume20_tminus1": MIN_DOLLAR_VOL20,
        "cooldown_sessions_per_symbol": COOLDOWN,
        "lookahead_sessions": LOOKAHEAD,
        "max_prior_adjusted_one_day_jump_pct": MAX_PRIOR_ADJ_JUMP,
    },
    "discovery_symbols_excluded": sorted(DISCOVERY),
    "all_signals": all_summary,
    "one_per_date": div_summary,
    "yearly": yearly_rows,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lines = []
lines.append("RAJIH — LOCKED +200% RUNNER RULE VALIDATION")
lines.append("="*82)
lines.append(
    f"LOCKED RULE: ATR14% T-1 >= {ATR_THRESHOLD:.2f}% "
    f"AND RET5 T-1 <= {RET5_THRESHOLD:.2f}%"
)
lines.append(f"Discovery symbols excluded: {len(DISCOVERY)}")
lines.append("")
for label, s in (("ALL SIGNALS", all_summary), ("ONE PER DATE", div_summary)):
    lines.append(label)
    lines.append("-"*82)
    lines.append(
        f"N={s.get('n',0)} | +20={s.get('hit20_rate_pct')}% | "
        f"+50={s.get('hit50_rate_pct')}% | +100={s.get('hit100_rate_pct')}% | "
        f"+200={s.get('hit200_rate_pct')}%"
    )
    lines.append(
        f"Median max63={s.get('median_max_gain_63d_pct')}% | "
        f"Median DD63={s.get('median_max_drawdown_63d_pct')}% | "
        f"Median close63={s.get('median_close_return_day63_pct')}% | "
        f"Median first +200 day={s.get('median_first200_day')}"
    )
    lines.append(
        f"Forward discontinuity flags (diagnostic only): "
        f"{s.get('future_discontinuity_n')}"
    )
    lines.append("")

lines.append("YEARLY")
lines.append("-"*82)
for r in yearly_rows:
    lines.append(
        f"{r['sample']} {r['year']} | N={r['n']} | "
        f"+50={r['hit50_rate_pct']}% | +100={r['hit100_rate_pct']}% | "
        f"+200={r['hit200_rate_pct']}% | median max63={r['median_max_gain_63d_pct']}%"
    )

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n=== LOCKED RULE RESULTS — ALL SIGNALS ===")
print(f"Signals: {all_summary.get('n',0)}")
print(f"+20% within 63d:  {all_summary.get('hit20_n',0)}/{all_summary.get('n',0)} = {all_summary.get('hit20_rate_pct')}%")
print(f"+50% within 63d:  {all_summary.get('hit50_n',0)}/{all_summary.get('n',0)} = {all_summary.get('hit50_rate_pct')}%")
print(f"+100% within 63d: {all_summary.get('hit100_n',0)}/{all_summary.get('n',0)} = {all_summary.get('hit100_rate_pct')}%")
print(f"+200% within 63d: {all_summary.get('hit200_n',0)}/{all_summary.get('n',0)} = {all_summary.get('hit200_rate_pct')}%")
print(f"Median max gain 63d: {all_summary.get('median_max_gain_63d_pct')}%")
print(f"Median max drawdown 63d: {all_summary.get('median_max_drawdown_63d_pct')}%")
print(f"Median close return day63: {all_summary.get('median_close_return_day63_pct')}%")
print(f"Median first +200 day: {all_summary.get('median_first200_day')}")

print("\n=== LOCKED RULE RESULTS — ONE PER DATE ===")
print(f"Distinct dates: {div_summary.get('n',0)}")
print(f"+20%:  {div_summary.get('hit20_rate_pct')}%")
print(f"+50%:  {div_summary.get('hit50_rate_pct')}%")
print(f"+100%: {div_summary.get('hit100_rate_pct')}%")
print(f"+200%: {div_summary.get('hit200_rate_pct')}%")
print(f"Median max gain 63d: {div_summary.get('median_max_gain_63d_pct')}%")
print(f"Median max drawdown 63d: {div_summary.get('median_max_drawdown_63d_pct')}%")

print("\n=== YEARLY ===")
for r in yearly_rows:
    print(
        f"{r['sample']:<12} {r['year']} | N={r['n']} | "
        f"+50={r['hit50_rate_pct']}% | +100={r['hit100_rate_pct']}% | "
        f"+200={r['hit200_rate_pct']}%"
    )

print("\nCreated:")
for p in [OUT_ALL, OUT_DIVERSE, OUT_YEARLY, OUT_JSON, OUT_REPORT]:
    print(" ", p)
