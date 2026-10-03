#!/usr/bin/env python3
"""
Rajih — test the discovered daily setup across ALL qualifying dates.

Purpose
-------
Use every distinct historical date that satisfies the locked BASE rule,
select ONE candidate per date using ONLY pre-signal information
(highest 20d average dollar volume), then evaluate what happened during
the next 21 trading sessions.

This turns the earlier 30-date validation into a ~501-date validation set.

BASE rule
---------
- $1 <= raw close < $20
- prior 20d average dollar volume >= $1,000,000
- worst opening gap in prior 20 sessions <= -9%
- average daily range over last 5 sessions >= 9.42%
- no >100% adjusted-close discontinuity in prior 20 sessions
- at least 220 daily bars
- exclude original discovery 30 symbols
- 21-session cooldown per symbol
- ONE setup per distinct date, chosen by highest prior 20d dollar volume

Pre-defined extra filters to test
---------------------------------
Based on the previous success-vs-fail comparison:
- EMA20 vs EMA50 spread: < 0, < -3, < -5, < -7
- ADX14: >= 30, 35, 40, 45
- Avg Range20: >= 15, 17.5, 20, 21.3
- ATR% upper cap: <= 15, 18, 20, 25
- Distance to 252d high: >= -80, -70, -60, -50
- combinations of the above

Outputs
-------
daily_rule_all_dates.csv
daily_rule_all_dates_filter_results.csv
daily_rule_all_dates_yearly.csv
daily_rule_all_dates_report.txt
daily_rule_all_dates_summary.json

Run:
    python test_daily_rule_all_501_dates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from sqlalchemy import MetaData, Table, select, func

from core import database


DISCOVERY_SYMBOLS = {
    "SMCI","PATH","OPEN","RIVN","HIMS","MARA","BMNR","WBD","F","HL",
    "RGTI","CLF","AG","SOFI","SOUN","JOBY","WULF","RDW","DJT","GAP",
    "PSKY","SBET","LYFT","SRPT","VFC","BBAI","AEO","U","AIFF","SNAP"
}


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def fmt(x, n=4):
    if x is None:
        return None
    try:
        y = float(x)
        return round(y, n) if math.isfinite(y) else None
    except Exception:
        return None


def ema_all(vals, n):
    if not vals:
        return []
    out = [None] * len(vals)
    a = 2 / (n + 1)
    e = float(vals[0])
    out[0] = e
    for i in range(1, len(vals)):
        e = a * float(vals[i]) + (1 - a) * e
        out[i] = e
    return out


def atr_all(h, l, c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    tr = [None] * len(c)
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    a = sum(tr[1:n+1]) / n
    out[n] = a
    for i in range(n+1, len(c)):
        a = ((n-1) * a + tr[i]) / n
        out[i] = a
    return out


def rsi_all(c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    gains, losses = [], []
    for i in range(1, len(c)):
        d = c[i] - c[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    out[n] = 100 if al == 0 and ag > 0 else (50 if al == 0 else 100 - 100/(1 + ag/al))
    for i in range(n+1, len(c)):
        ag = ((n-1)*ag + gains[i-1]) / n
        al = ((n-1)*al + losses[i-1]) / n
        out[i] = 100 if al == 0 and ag > 0 else (50 if al == 0 else 100 - 100/(1 + ag/al))
    return out


def adx_all(h, l, c, n=14):
    m = len(c)
    out = [None] * m
    if m < 2*n + 1:
        return out

    tr = [0.0] * m
    pdm = [0.0] * m
    mdm = [0.0] * m
    for i in range(1, m):
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        mdm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))

    atrs = sum(tr[1:n+1])
    ps = sum(pdm[1:n+1])
    ms = sum(mdm[1:n+1])
    dx = [None] * m

    for i in range(n, m):
        if i > n:
            atrs = atrs - atrs/n + tr[i]
            ps = ps - ps/n + pdm[i]
            ms = ms - ms/n + mdm[i]
        if atrs <= 0:
            continue
        pdi = 100 * ps / atrs
        mdi = 100 * ms / atrs
        den = pdi + mdi
        dx[i] = 0 if den == 0 else 100 * abs(pdi - mdi) / den

    seed = [x for x in dx[n:2*n] if x is not None]
    if len(seed) < n:
        return out

    a = sum(seed) / n
    out[2*n-1] = a
    for i in range(2*n, m):
        if dx[i] is not None:
            a = ((n-1)*a + dx[i]) / n
            out[i] = a
    return out


def stdev(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs) < 2:
        return None
    mu = mean(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))


def load_symbol(DB, daily, sym):
    with DB() as s:
        raw = list(
            s.execute(
                select(
                    daily.c.session_date, daily.c.o, daily.c.h, daily.c.l,
                    daily.c.c, daily.c.v, daily.c.adj_c
                )
                .where(daily.c.symbol == sym)
                .order_by(daily.c.session_date)
            ).all()
        )

    rows = []
    for x in raw:
        ac = finite(x[6])
        rc = finite(x[4])
        if ac is None or rc is None or ac <= 0 or rc <= 0:
            continue
        rows.append({
            "date": str(x[0]),
            "ro": float(x[1]),
            "rh": float(x[2]),
            "rl": float(x[3]),
            "rc": rc,
            "v": float(x[5] or 0),
            "c": ac,
        })
    return rows


def build(rows):
    ro = [r["ro"] for r in rows]
    rh = [r["rh"] for r in rows]
    rl = [r["rl"] for r in rows]
    rc = [r["rc"] for r in rows]
    v = [r["v"] for r in rows]
    c = [r["c"] for r in rows]

    fac = [c[i]/rc[i] for i in range(len(c))]
    o = [ro[i]*fac[i] for i in range(len(c))]
    h = [rh[i]*fac[i] for i in range(len(c))]
    l = [rl[i]*fac[i] for i in range(len(c))]

    atr = atr_all(h, l, c, 14)
    rsi = rsi_all(c, 14)
    adx = adx_all(h, l, c, 14)
    e20 = ema_all(c, 20)
    e50 = ema_all(c, 50)

    avgdol = [None] * len(c)
    r5 = [None] * len(c)
    r20 = [None] * len(c)
    mingap = [None] * len(c)
    rv = [None] * len(c)
    hi252 = [None] * len(c)

    ret = [None] * len(c)
    gap = [None] * len(c)
    dollar = [rc[i] * v[i] for i in range(len(c))]

    for i in range(1, len(c)):
        ret[i] = c[i]/c[i-1] - 1
        gap[i] = 100 * (o[i]/c[i-1] - 1)

    for i in range(len(c)):
        if i >= 4:
            r5[i] = mean(100*(h[j]-l[j])/c[j] for j in range(i-4, i+1))
        if i >= 19:
            avgdol[i] = mean(dollar[i-19:i+1])
            r20[i] = mean(100*(h[j]-l[j])/c[j] for j in range(i-19, i+1))
            gg = [gap[j] for j in range(i-19, i+1) if gap[j] is not None]
            mingap[i] = min(gg) if gg else None
            rr = [ret[j] for j in range(i-19, i+1) if ret[j] is not None]
            rv[i] = stdev(rr)
        if i >= 251:
            hi252[i] = max(h[i-251:i+1])

    return {
        "date": [r["date"] for r in rows],
        "rc": rc, "c": c, "h": h, "l": l,
        "atr": atr, "rsi": rsi, "adx": adx,
        "e20": e20, "e50": e50,
        "avgdol": avgdol, "r5": r5, "r20": r20,
        "mingap": mingap, "rv": rv, "hi252": hi252,
    }


def discontinuity_ok(f, i, max_abs):
    if i < 20:
        return False
    for j in range(i-19, i+1):
        if j <= 0:
            continue
        if abs(100*(f["c"][j]/f["c"][j-1]-1)) > max_abs:
            return False
    return True


def outcome(f, i, lookahead=21):
    j2 = min(len(f["c"])-1, i+lookahead)
    if j2 <= i:
        return None

    start = f["c"][i]
    fut = range(i+1, j2+1)

    def maxh(n):
        jj = min(j2, i+n)
        return max(100*(f["h"][j]/start - 1) for j in range(i+1, jj+1))

    def maxc(n):
        jj = min(j2, i+n)
        return max(100*(f["c"][j]/start - 1) for j in range(i+1, jj+1))

    def mae(n):
        jj = min(j2, i+n)
        return min(100*(f["l"][j]/start - 1) for j in range(i+1, jj+1))

    peak = max(fut, key=lambda j: f["h"][j])

    def dto(th):
        for j in fut:
            if 100*(f["h"][j]/start - 1) >= th:
                return j-i
        return None

    return {
        "max_high_5d_pct": fmt(maxh(5)),
        "max_high_10d_pct": fmt(maxh(10)),
        "max_high_21d_pct": fmt(maxh(21)),
        "max_close_21d_pct": fmt(maxc(21)),
        "mae_21d_pct": fmt(mae(21)),
        "days_to_20": dto(20),
        "days_to_30": dto(30),
        "days_to_50": dto(50),
        "days_to_100": dto(100),
        "days_to_peak": peak - i,
    }


def metrics(rows):
    if not rows:
        return {
            "n": 0, "hit20": 0, "hit30": 0, "hit50": 0, "hit100": 0,
            "hit20_rate": None, "hit30_rate": None, "hit50_rate": None, "hit100_rate": None,
            "mean_max21": None, "median_max21": None, "median_mae21": None,
        }

    n = len(rows)
    vals = [float(r["max_high_21d_pct"]) for r in rows]
    maes = [float(r["mae_21d_pct"]) for r in rows]

    def hit(th):
        return sum(v >= th for v in vals)

    return {
        "n": n,
        "hit20": hit(20),
        "hit30": hit(30),
        "hit50": hit(50),
        "hit100": hit(100),
        "hit20_rate": 100*hit(20)/n,
        "hit30_rate": 100*hit(30)/n,
        "hit50_rate": 100*hit(50)/n,
        "hit100_rate": 100*hit(100)/n,
        "mean_max21": mean(vals),
        "median_max21": median(vals),
        "median_mae21": median(maes),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-price", type=float, default=1.0)
    ap.add_argument("--max-price", type=float, default=20.0)
    ap.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    ap.add_argument("--gap-threshold", type=float, default=-9.0)
    ap.add_argument("--range5-threshold", type=float, default=9.42)
    ap.add_argument("--max-prior-jump", type=float, default=100.0)
    ap.add_argument("--lookahead", type=int, default=21)
    args = ap.parse_args()

    DB = database()
    with DB() as s:
        md = MetaData()
        daily = Table("market_candles_1d", md, autoload_with=s.get_bind())
        symbols = list(
            s.execute(
                select(daily.c.symbol)
                .group_by(daily.c.symbol)
                .having(func.count() >= 220)
                .order_by(daily.c.symbol)
            ).scalars().all()
        )

    by_date = defaultdict(list)

    print("\nRajih — ALL DISTINCT-DATE VALIDATION")
    print(f"Universe: {len(symbols)}")
    print(f"Excluded discovery symbols: {len(DISCOVERY_SYMBOLS)}")
    print("Collecting every base-rule date using only pre-signal information...\n")

    for n, sym in enumerate(symbols, 1):
        if sym in DISCOVERY_SYMBOLS:
            continue

        rows = load_symbol(DB, daily, sym)
        if len(rows) < 220:
            continue
        f = build(rows)

        last_match = -999
        for i in range(219, len(rows)-args.lookahead):
            # Avoid one long episode from generating near-duplicate setups.
            if i - last_match < 21:
                continue

            if not (args.min_price <= f["rc"][i] < args.max_price):
                continue
            if f["avgdol"][i] is None or f["avgdol"][i] < args.min_dollar_volume:
                continue
            if f["mingap"][i] is None or f["mingap"][i] > args.gap_threshold:
                continue
            if f["r5"][i] is None or f["r5"][i] < args.range5_threshold:
                continue
            if not discontinuity_ok(f, i, args.max_prior_jump):
                continue

            row = {
                "symbol": sym,
                "signal_date": f["date"][i],
                "signal_raw_close": fmt(f["rc"][i]),
                "avg_dollar_volume20": fmt(f["avgdol"][i], 0),
                "min_gap20_pct": fmt(f["mingap"][i]),
                "avg_range5_pct": fmt(f["r5"][i]),
                "avg_range20_pct": fmt(f["r20"][i]),
                "atr_pct": fmt(100*f["atr"][i]/f["c"][i] if f["atr"][i] else None),
                "rsi14": fmt(f["rsi"][i]),
                "adx14": fmt(f["adx"][i]),
                "ema20_vs_ema50_pct": fmt(
                    100*(f["e20"][i]/f["e50"][i]-1)
                    if f["e20"][i] and f["e50"][i] else None
                ),
                "realized_vol20_pct": fmt(
                    100*f["rv"][i] if f["rv"][i] is not None else None
                ),
                "distance_to_252d_high_pct": fmt(
                    100*(f["c"][i]/f["hi252"][i]-1)
                    if f["hi252"][i] else None
                ),
                "_i": i,
                "_f": f,
            }
            by_date[row["signal_date"]].append(row)
            last_match = i

        if n % 250 == 0 or n == len(symbols):
            print(f"Scanned {n}/{len(symbols)} | distinct qualifying dates={len(by_date)}")

    # Lock exactly one candidate per date BEFORE outcome:
    # highest prior 20d dollar volume.
    selected = []
    for d in sorted(by_date):
        pool = sorted(
            by_date[d],
            key=lambda r: (r["avg_dollar_volume20"] or 0, r["symbol"]),
            reverse=True,
        )
        selected.append(pool[0])

    # Now calculate outcomes.
    out = []
    for rank, r in enumerate(selected, 1):
        f = r.pop("_f")
        i = r.pop("_i")
        x = {"rank": rank, **r}
        x.update(outcome(f, i, args.lookahead))
        out.append(x)

    write_csv("daily_rule_all_dates.csv", out)

    base = metrics(out)

    # Fixed single filters.
    tests = []

    def add_test(name, fn):
        subset = [r for r in out if fn(r)]
        m = metrics(subset)
        tests.append({
            "filter": name,
            **{k: fmt(v) if isinstance(v, float) else v for k, v in m.items()},
            "lift50_vs_base": fmt(
                (m["hit50_rate"] / base["hit50_rate"])
                if m["hit50_rate"] is not None and base["hit50_rate"] else None
            ),
        })

    add_test("BASE", lambda r: True)

    ema_rules = [
        ("EMA20/50 < 0", lambda r: finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < 0),
        ("EMA20/50 < -3", lambda r: finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < -3),
        ("EMA20/50 < -5", lambda r: finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < -5),
        ("EMA20/50 < -7", lambda r: finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < -7),
    ]
    adx_rules = [
        ("ADX >= 30", lambda r: finite(r["adx14"]) is not None and r["adx14"] >= 30),
        ("ADX >= 35", lambda r: finite(r["adx14"]) is not None and r["adx14"] >= 35),
        ("ADX >= 40", lambda r: finite(r["adx14"]) is not None and r["adx14"] >= 40),
        ("ADX >= 45", lambda r: finite(r["adx14"]) is not None and r["adx14"] >= 45),
    ]
    range_rules = [
        ("Range20 >= 15", lambda r: finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 15),
        ("Range20 >= 17.5", lambda r: finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 17.5),
        ("Range20 >= 20", lambda r: finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 20),
        ("Range20 >= 21.3", lambda r: finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 21.3),
    ]
    atr_rules = [
        ("ATR% <= 15", lambda r: finite(r["atr_pct"]) is not None and r["atr_pct"] <= 15),
        ("ATR% <= 18", lambda r: finite(r["atr_pct"]) is not None and r["atr_pct"] <= 18),
        ("ATR% <= 20", lambda r: finite(r["atr_pct"]) is not None and r["atr_pct"] <= 20),
        ("ATR% <= 25", lambda r: finite(r["atr_pct"]) is not None and r["atr_pct"] <= 25),
    ]
    dist_rules = [
        ("Dist252 >= -80", lambda r: finite(r["distance_to_252d_high_pct"]) is not None and r["distance_to_252d_high_pct"] >= -80),
        ("Dist252 >= -70", lambda r: finite(r["distance_to_252d_high_pct"]) is not None and r["distance_to_252d_high_pct"] >= -70),
        ("Dist252 >= -60", lambda r: finite(r["distance_to_252d_high_pct"]) is not None and r["distance_to_252d_high_pct"] >= -60),
        ("Dist252 >= -50", lambda r: finite(r["distance_to_252d_high_pct"]) is not None and r["distance_to_252d_high_pct"] >= -50),
    ]

    all_single = ema_rules + adx_rules + range_rules + atr_rules + dist_rules
    for name, fn in all_single:
        add_test(name, fn)

    # Pair combinations across different families.
    families = [ema_rules, adx_rules, range_rules, atr_rules, dist_rules]
    for a in range(len(families)):
        for b in range(a+1, len(families)):
            for name1, fn1 in families[a]:
                for name2, fn2 in families[b]:
                    add_test(
                        f"{name1} AND {name2}",
                        lambda r, f1=fn1, f2=fn2: f1(r) and f2(r)
                    )

    # Selected 3-way combinations based on prior hypotheses.
    triples = [
        (
            "EMA<0 AND ADX>=30 AND Range20>=15",
            lambda r:
                finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < 0 and
                finite(r["adx14"]) is not None and r["adx14"] >= 30 and
                finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 15
        ),
        (
            "EMA<-3 AND ADX>=35 AND Range20>=17.5",
            lambda r:
                finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < -3 and
                finite(r["adx14"]) is not None and r["adx14"] >= 35 and
                finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 17.5
        ),
        (
            "EMA<0 AND ADX>=30 AND ATR<=20",
            lambda r:
                finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < 0 and
                finite(r["adx14"]) is not None and r["adx14"] >= 30 and
                finite(r["atr_pct"]) is not None and r["atr_pct"] <= 20
        ),
        (
            "EMA<0 AND Range20>=15 AND Dist252>=-70",
            lambda r:
                finite(r["ema20_vs_ema50_pct"]) is not None and r["ema20_vs_ema50_pct"] < 0 and
                finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 15 and
                finite(r["distance_to_252d_high_pct"]) is not None and r["distance_to_252d_high_pct"] >= -70
        ),
        (
            "ADX>=35 AND Range20>=17.5 AND ATR<=20",
            lambda r:
                finite(r["adx14"]) is not None and r["adx14"] >= 35 and
                finite(r["avg_range20_pct"]) is not None and r["avg_range20_pct"] >= 17.5 and
                finite(r["atr_pct"]) is not None and r["atr_pct"] <= 20
        ),
    ]
    for name, fn in triples:
        add_test(name, fn)

    # Keep ranking meaningful: show sample size and prioritize precision with reasonable N.
    tests_sorted = sorted(
        tests,
        key=lambda r: (
            r["hit50_rate"] if r["hit50_rate"] is not None else -1,
            r["n"],
        ),
        reverse=True
    )
    write_csv("daily_rule_all_dates_filter_results.csv", tests_sorted)

    # Yearly baseline.
    yearly = []
    buckets = defaultdict(list)
    for r in out:
        buckets[r["signal_date"][:4]].append(r)

    for year in sorted(buckets):
        m = metrics(buckets[year])
        yearly.append({
            "year": year,
            **{k: fmt(v) if isinstance(v, float) else v for k, v in m.items()},
        })
    write_csv("daily_rule_all_dates_yearly.csv", yearly)

    # Report only rules with at least 20 signals, plus at least 30 signals section.
    robust20 = [r for r in tests_sorted if r["n"] >= 20]
    robust30 = [r for r in tests_sorted if r["n"] >= 30]

    lines = []
    lines.append("RAJIH — ALL DISTINCT-DATE DAILY RULE VALIDATION")
    lines.append("=" * 78)
    lines.append(f"Distinct dates tested: {len(out)}")
    lines.append(
        f"BASE: +20={base['hit20_rate']:.2f}% | +30={base['hit30_rate']:.2f}% | "
        f"+50={base['hit50_rate']:.2f}% | +100={base['hit100_rate']:.2f}% | "
        f"median max21={base['median_max21']:.2f}% | median MAE21={base['median_mae21']:.2f}%"
    )
    lines.append("")
    lines.append("TOP FILTERS WITH N >= 30")
    lines.append("-" * 78)
    for r in robust30[:25]:
        lines.append(
            f"{r['filter']} | N={r['n']} | +50={r['hit50_rate']}% | "
            f"+30={r['hit30_rate']}% | +20={r['hit20_rate']}% | "
            f"lift50={r['lift50_vs_base']} | median max21={r['median_max21']}% | "
            f"median MAE21={r['median_mae21']}%"
        )

    lines.append("")
    lines.append("TOP FILTERS WITH N >= 20")
    lines.append("-" * 78)
    for r in robust20[:25]:
        lines.append(
            f"{r['filter']} | N={r['n']} | +50={r['hit50_rate']}% | "
            f"+30={r['hit30_rate']}% | +20={r['hit20_rate']}% | "
            f"lift50={r['lift50_vs_base']} | median max21={r['median_max21']}% | "
            f"median MAE21={r['median_mae21']}%"
        )

    lines.append("")
    lines.append("YEARLY BASELINE")
    lines.append("-" * 78)
    for r in yearly:
        lines.append(
            f"{r['year']} | N={r['n']} | +50={r['hit50_rate']}% | "
            f"+30={r['hit30_rate']}% | median max21={r['median_max21']}% | "
            f"median MAE21={r['median_mae21']}%"
        )

    Path("daily_rule_all_dates_report.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "distinct_dates": len(out),
        "base": {k: fmt(v) if isinstance(v, float) else v for k, v in base.items()},
        "top_filters_n_ge_30": robust30[:25],
        "top_filters_n_ge_20": robust20[:25],
        "yearly": yearly,
        "outputs": [
            "daily_rule_all_dates.csv",
            "daily_rule_all_dates_filter_results.csv",
            "daily_rule_all_dates_yearly.csv",
            "daily_rule_all_dates_report.txt",
        ],
    }
    Path("daily_rule_all_dates_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ALL-DATE BASELINE ===")
    print(f"Distinct dates: {len(out)}")
    print(f"+20% within 21d: {base['hit20']}/{base['n']} = {base['hit20_rate']:.2f}%")
    print(f"+30% within 21d: {base['hit30']}/{base['n']} = {base['hit30_rate']:.2f}%")
    print(f"+50% within 21d: {base['hit50']}/{base['n']} = {base['hit50_rate']:.2f}%")
    print(f"+100% within 21d: {base['hit100']}/{base['n']} = {base['hit100_rate']:.2f}%")
    print(f"Median max high 21d: {base['median_max21']:.2f}%")
    print(f"Median MAE 21d: {base['median_mae21']:.2f}%")

    print("\n=== TOP FILTERS N>=30 ===")
    for r in robust30[:15]:
        print(
            f"{r['filter']} | N={r['n']} | +50={r['hit50_rate']}% | "
            f"+30={r['hit30_rate']}% | lift50={r['lift50_vs_base']} | "
            f"MAE={r['median_mae21']}%"
        )

    print("\n=== YEARLY ===")
    for r in yearly:
        print(
            f"{r['year']} | N={r['n']} | +50={r['hit50_rate']}% | "
            f"+30={r['hit30_rate']}% | medianMax={r['median_max21']}%"
        )

    print("\nCreated:")
    for x in summary["outputs"]:
        print(" ", x)
    print("  daily_rule_all_dates_summary.json")


if __name__ == "__main__":
    main()
