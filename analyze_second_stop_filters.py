#!/usr/bin/env python3
"""
Rajih — SECOND_STOP_RECOVERY filter analysis.

Goal
----
Find what separates TARGET vs STOPPED after level-2 activation, then identify
simple robust filters (1, 2, or 3 conditions) that improve expectancy.

Base system
-----------
- Reconstruct ORIGINAL US 15m signals from candles.
- Level 1 = original stop.
- Level 2 = Level 1 - 1.2 ATR -> BUY LIMIT.
- Level 3 = Level 2 - 1.2 ATR -> protective stop.
- Target = Level 1.
- Wait up to 78 bars.
- Same execution assumptions as the reconstructed backtest.

Features analyzed
-----------------
At original signal:
- RSI
- ATR as % of original entry
- original risk distance % (entry -> first stop)
- volume ratio
- EMA20-vs-EMA50 spread %
- MACD-vs-signal spread in ATR units
- original price
- signal time of day

At level-2 activation:
- bars waited from signal to level 2
- hours waited
- activation time of day
- total pullback % from original entry to level 2

Robustness
----------
Signals are split chronologically:
- first 70% = TRAIN (filter discovery)
- last 30% = VALIDATION (out-of-sample check)

The script ranks:
1) single filters
2) two-condition combinations
3) three-condition combinations

A candidate must have enough validation trades. Default: 200.

Primary ranking on VALIDATION:
- positive Avg R
- higher Sum R
- higher win rate
- enough trades

READ ONLY.

Run:
    python analyze_second_stop_filters.py --days 30

Outputs:
    second_stop_filter_rows.csv
    second_stop_filter_analysis.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from sqlalchemy import select

from core import database
from monitor.models import Stock, Candle
from monitor.strategy import rounded

NY = ZoneInfo("America/New_York")
US_START = 570
US_END = 960


def iso_ny(ts):
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), NY).isoformat(timespec="minutes")


def local_parts(ts):
    dt = datetime.fromtimestamp(int(ts), NY)
    return dt.date(), dt.hour * 60 + dt.minute


def ema_series(values, n):
    if not values:
        return []
    alpha = 2.0 / (n + 1.0)
    v = float(values[0])
    out = []
    for x in values:
        v += (float(x) - v) * alpha
        out.append(v)
    return out


def indicators(bars):
    n = len(bars)
    closes = [float(b[4]) for b in bars]
    highs = [float(b[2]) for b in bars]
    lows = [float(b[3]) for b in bars]

    e12 = ema_series(closes, 12)
    e20 = ema_series(closes, 20)
    e26 = ema_series(closes, 26)
    e50 = ema_series(closes, 50)
    macd = [a - b for a, b in zip(e12, e26)]
    macd_sig = ema_series(macd, 9)

    atr = [None] * n
    rsi = [None] * n
    if n >= 15:
        trs, chgs = [], []
        for i in range(1, n):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            ))
            chgs.append(closes[i] - closes[i-1])

        a = sum(trs[:14]) / 14.0
        gain = sum(max(x, 0.0) for x in chgs[:14]) / 14.0
        loss = sum(max(-x, 0.0) for x in chgs[:14]) / 14.0
        atr[14] = a
        rsi[14] = 100 - 100/(1 + gain/loss) if loss else (100.0 if gain else 50.0)

        for i in range(15, n):
            a = (a * 13 + trs[i-1]) / 14.0
            gain = (gain * 13 + max(chgs[i-1], 0.0)) / 14.0
            loss = (loss * 13 + max(-chgs[i-1], 0.0)) / 14.0
            atr[i] = a
            rsi[i] = 100 - 100/(1 + gain/loss) if loss else (100.0 if gain else 50.0)

    return {
        "ema20": e20,
        "ema50": e50,
        "macd": macd,
        "macd_signal": macd_sig,
        "atr": atr,
        "rsi": rsi,
    }


def pivot_lows(bars):
    out = []
    for i in range(2, len(bars)-2):
        x = float(bars[i][3])
        left = [float(bars[i-2][3]), float(bars[i-1][3])]
        right = [float(bars[i+1][3]), float(bars[i+2][3])]
        if x <= min(left + right) and x < max(left) and x < max(right):
            out.append(i)
    return out


def volume_baseline(bars, ref_dates):
    by = {}
    for b in bars:
        d, minute = local_parts(b[0])
        by[(d, minute)] = float(b[5])

    out = {}
    for di, d in enumerate(ref_dates):
        if di < 20:
            continue
        prior = ref_dates[di-20:di]
        for minute in range(US_START, US_END, 15):
            vals = []
            ok = True
            for pd in prior:
                v = by.get((pd, minute))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if not ok:
                continue
            avg = sum(vals) / 20.0
            if avg <= 0 or (d, minute) not in by:
                continue
            dt = datetime(d.year, d.month, d.day, minute//60, minute%60, tzinfo=NY)
            out[int(dt.timestamp())] = avg
    return out


def reference_streaks(bars, ref_pos):
    out = [0] * len(bars)
    prev = None
    cur = 0
    for i, b in enumerate(bars):
        pos = ref_pos.get(int(b[0]))
        if pos is None:
            cur = 0
            prev = None
        else:
            cur = cur + 1 if prev is not None and pos == prev + 1 else 1
            prev = pos
        out[i] = cur
    return out


@dataclass
class PlanSim:
    symbol: str
    signal_ts: int
    original_entry: float
    first_stop: float
    original_target: float
    second_stop: float
    third_stop: float
    atr: float
    rsi: float
    volume_ratio: float
    ema_spread_pct: float
    macd_gap_atr: float
    risk_pct: float
    atr_pct: float
    pullback_pct: float
    signal_minute: int
    state: str = "WAITING"
    waiting_bars: int = 0
    last_ts: int = 0
    fill_ts: int | None = None
    fill_price: float | None = None
    result: str = ""
    exit_ts: int | None = None
    exit_price: float | None = None
    bars_after_fill: int = 0
    same_bar_ambiguous: bool = False


def build_signal(symbol, i, bars, ind, pivots, vol_avg, streak, corporate_flag, atr_mult):
    if corporate_flag or i < 199 or streak[i] < 21:
        return None

    atr = ind["atr"][i]
    rsi = ind["rsi"][i]
    if atr is None or rsi is None or atr <= 0:
        return None

    ts, o, h, l, c, v = bars[i]
    ts = int(ts)
    v = float(v)

    avg = vol_avg.get(ts)
    if avg is None or avg <= 0 or v <= 0:
        return None
    if not (35 <= rsi <= 75):
        return None
    if not (ind["ema20"][i] >= ind["ema50"][i] or ind["macd"][i] >= ind["macd_signal"][i]):
        return None

    entry, _ = rounded(float(h) + 0.05 * atr, "US", True)

    right = bisect.bisect_right(pivots, i - 2) - 1
    support_idx = None
    limit_price = min(float(c), entry)
    while right >= 0:
        j = pivots[right]
        if j < i - 79:
            break
        if float(bars[j][3]) < limit_price:
            support_idx = j
            break
        right -= 1
    if support_idx is None:
        return None

    support = float(bars[support_idx][3])
    first_stop, _ = rounded(support - 0.25 * atr, "US", False)
    risk = entry - first_stop
    if first_stop <= 0 or risk <= 0:
        return None

    selected_target = None
    for n_pct in (3, 4, 5):
        target, _ = rounded(entry * (1 + n_pct/100.0), "US", True)
        if (target - entry) / risk >= 1.5:
            selected_target = target
            break
    if selected_target is None:
        return None

    second_stop, _ = rounded(first_stop - atr_mult * atr, "US", False)
    third_stop, _ = rounded(second_stop - atr_mult * atr, "US", False)
    if third_stop <= 0 or not (third_stop < second_stop < first_stop):
        return None

    _, signal_minute = local_parts(ts)
    ema50 = float(ind["ema50"][i])
    ema_spread_pct = 100 * (float(ind["ema20"][i]) / ema50 - 1) if ema50 else 0.0
    macd_gap_atr = (float(ind["macd"][i]) - float(ind["macd_signal"][i])) / atr

    return PlanSim(
        symbol=symbol,
        signal_ts=ts,
        original_entry=entry,
        first_stop=first_stop,
        original_target=selected_target,
        second_stop=second_stop,
        third_stop=third_stop,
        atr=atr,
        rsi=rsi,
        volume_ratio=v/avg,
        ema_spread_pct=ema_spread_pct,
        macd_gap_atr=macd_gap_atr,
        risk_pct=100 * risk / entry,
        atr_pct=100 * atr / entry,
        pullback_pct=100 * (entry - second_stop) / entry,
        signal_minute=signal_minute,
        last_ts=ts,
    )


def row_from_plan(p):
    activation_minute = None
    if p.fill_ts:
        _, activation_minute = local_parts(p.fill_ts)

    risk = None
    pnl_r = None
    ret = None
    if p.fill_price is not None:
        risk = p.fill_price - p.third_stop
        if p.exit_price is not None and risk > 0:
            pnl_r = (p.exit_price - p.fill_price) / risk
            ret = 100 * (p.exit_price / p.fill_price - 1)

    return {
        "symbol": p.symbol,
        "signal_ts": p.signal_ts,
        "signal_ny": iso_ny(p.signal_ts),
        "result": p.result,
        "original_entry": p.original_entry,
        "first_stop": p.first_stop,
        "second_stop": p.second_stop,
        "third_stop": p.third_stop,
        "fill_price": p.fill_price,
        "fill_ts": p.fill_ts,
        "fill_ny": iso_ny(p.fill_ts),
        "exit_ny": iso_ny(p.exit_ts),
        "rsi": p.rsi,
        "atr_pct": p.atr_pct,
        "risk_pct": p.risk_pct,
        "pullback_pct": p.pullback_pct,
        "volume_ratio": p.volume_ratio,
        "ema_spread_pct": p.ema_spread_pct,
        "macd_gap_atr": p.macd_gap_atr,
        "price": p.original_entry,
        "signal_minute": p.signal_minute,
        "activation_minute": activation_minute,
        "waiting_bars": p.waiting_bars,
        "waiting_hours": p.waiting_bars * 0.25,
        "bars_after_fill": p.bars_after_fill,
        "pnl_r": pnl_r,
        "return_pct": ret,
        "same_bar_ambiguous": p.same_bar_ambiguous,
    }


def replay_symbol(symbol, bars, ref_pos, ref_dates, start_ts, end_ts, flag, atr_mult, waiting_limit):
    if len(bars) < 200:
        return []

    bars = [tuple(b) for b in bars if int(b[0]) <= end_ts]
    ind = indicators(bars)
    pivots = pivot_lows(bars)
    vol_avg = volume_baseline(bars, ref_dates)
    streak = reference_streaks(bars, ref_pos)
    times = [int(b[0]) for b in bars]
    start_i = bisect.bisect_left(times, start_ts)

    completed = []
    open_plan = None

    for i in range(start_i, len(bars)):
        ts, o, h, l, c, v = bars[i]
        ts = int(ts)
        if ts >= end_ts:
            break

        pos = ref_pos.get(ts)
        if pos is None:
            continue

        if open_plan is not None:
            prev_pos = ref_pos.get(open_plan.last_ts)
            contiguous = prev_pos is not None and pos == prev_pos + 1
            open_plan.last_ts = ts

            if not contiguous:
                open_plan.result = "DATA_GAP_ACTIVE" if open_plan.state == "ACTIVE" else "DATA_GAP_WAITING"
                if open_plan.state == "ACTIVE":
                    open_plan.exit_price = float(c)
                    open_plan.exit_ts = ts
                completed.append(row_from_plan(open_plan))
                open_plan = None

            elif open_plan.state == "WAITING":
                open_plan.waiting_bars += 1

                if float(o) <= open_plan.third_stop:
                    open_plan.result = "CANCELLED_GAP"
                    open_plan.exit_ts = ts
                    completed.append(row_from_plan(open_plan))
                    open_plan = None

                elif float(l) <= open_plan.second_stop:
                    fill = min(float(o), open_plan.second_stop) if float(o) <= open_plan.second_stop else open_plan.second_stop
                    fill, _ = rounded(fill, "US", True)
                    if fill <= open_plan.third_stop:
                        open_plan.result = "CANCELLED_INVALID_FILL"
                        open_plan.exit_ts = ts
                        completed.append(row_from_plan(open_plan))
                        open_plan = None
                    else:
                        open_plan.state = "ACTIVE"
                        open_plan.fill_ts = ts
                        open_plan.fill_price = fill
                        # As in the live engine: do not evaluate target/stop on fill candle.

                if open_plan is not None and open_plan.state == "WAITING" and open_plan.waiting_bars >= waiting_limit:
                    open_plan.result = "EXPIRED"
                    open_plan.exit_ts = ts
                    completed.append(row_from_plan(open_plan))
                    open_plan = None

            elif open_plan.state == "ACTIVE":
                open_plan.bars_after_fill += 1

                if float(l) <= open_plan.third_stop:
                    open_plan.result = "STOPPED"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = min(float(o), open_plan.third_stop)
                    if float(h) >= open_plan.first_stop:
                        open_plan.same_bar_ambiguous = True
                    completed.append(row_from_plan(open_plan))
                    open_plan = None

                elif float(h) >= open_plan.first_stop:
                    open_plan.result = "TARGET"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = open_plan.first_stop
                    completed.append(row_from_plan(open_plan))
                    open_plan = None

        if open_plan is None:
            sig = build_signal(
                symbol, i, bars, ind, pivots, vol_avg, streak,
                flag, atr_mult
            )
            if sig is not None:
                open_plan = sig

    if open_plan is not None:
        if open_plan.state == "ACTIVE" and bars:
            open_plan.result = "ACTIVE_END"
            open_plan.exit_price = float(bars[-1][4])
            open_plan.exit_ts = int(bars[-1][0])
        else:
            open_plan.result = "WAITING_END"
        completed.append(row_from_plan(open_plan))

    return completed


def metrics(rows):
    if not rows:
        return None
    target = sum(r["result"] == "TARGET" for r in rows)
    stopped = sum(r["result"] == "STOPPED" for r in rows)
    rvals = [float(r["pnl_r"]) for r in rows if r["pnl_r"] is not None]
    if not rvals:
        return None
    return {
        "n": len(rows),
        "target": target,
        "stopped": stopped,
        "win_rate": 100 * target / len(rows),
        "sum_r": sum(rvals),
        "avg_r": mean(rvals),
    }


def fmt(m):
    if not m:
        return "n/a"
    return f"n={m['n']} | win={m['win_rate']:.2f}% | avgR={m['avg_r']:+.3f} | sumR={m['sum_r']:+.1f}"


def candidate_rules(train):
    # Quantile-driven thresholds reduce arbitrary hand-picking.
    specs = [
        ("rsi", "low"), ("rsi", "high"),
        ("atr_pct", "low"), ("atr_pct", "high"),
        ("risk_pct", "low"), ("risk_pct", "high"),
        ("pullback_pct", "low"), ("pullback_pct", "high"),
        ("volume_ratio", "low"), ("volume_ratio", "high"),
        ("ema_spread_pct", "low"), ("ema_spread_pct", "high"),
        ("macd_gap_atr", "low"), ("macd_gap_atr", "high"),
        ("waiting_bars", "low"), ("waiting_bars", "high"),
        ("price", "low"), ("price", "high"),
        ("activation_minute", "low"), ("activation_minute", "high"),
    ]

    qs = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    rules = []
    for feature, direction in specs:
        vals = sorted(float(r[feature]) for r in train if r.get(feature) is not None)
        if len(vals) < 10:
            continue
        seen = set()
        for q in qs:
            idx = min(len(vals)-1, max(0, int(round((len(vals)-1)*q))))
            t = vals[idx]
            key = round(t, 8)
            if key in seen:
                continue
            seen.add(key)
            if direction == "low":
                desc = f"{feature} <= {t:.4g}"
                fn = lambda r, f=feature, x=t: r.get(f) is not None and float(r[f]) <= x
            else:
                desc = f"{feature} >= {t:.4g}"
                fn = lambda r, f=feature, x=t: r.get(f) is not None and float(r[f]) >= x
            rules.append((desc, fn, feature, direction, t))
    return rules


def apply_rules(rows, rules):
    return [r for r in rows if all(rule[1](r) for rule in rules)]


def search_filters(train, valid, min_train, min_valid):
    rules = candidate_rules(train)

    singles = []
    for rule in rules:
        tr = apply_rules(train, [rule])
        va = apply_rules(valid, [rule])
        if len(tr) < min_train or len(va) < min_valid:
            continue
        singles.append((rule, metrics(tr), metrics(va)))

    # Keep only stronger single-rule candidates to control combinatorics.
    singles.sort(key=lambda x: (x[2]["avg_r"], x[2]["sum_r"]), reverse=True)
    shortlist = [x[0] for x in singles[:30]]

    pairs = []
    for a, b in itertools.combinations(shortlist, 2):
        if a[2] == b[2]:
            continue
        tr = apply_rules(train, [a, b])
        va = apply_rules(valid, [a, b])
        if len(tr) < min_train or len(va) < min_valid:
            continue
        pairs.append(((a, b), metrics(tr), metrics(va)))
    pairs.sort(key=lambda x: (x[2]["avg_r"], x[2]["sum_r"]), reverse=True)

    pair_rules = [x[0] for x in pairs[:20]]
    triples = []
    for pair in pair_rules:
        used = {pair[0][2], pair[1][2]}
        for c in shortlist:
            if c[2] in used:
                continue
            combo = (pair[0], pair[1], c)
            tr = apply_rules(train, combo)
            va = apply_rules(valid, combo)
            if len(tr) < min_train or len(va) < min_valid:
                continue
            triples.append((combo, metrics(tr), metrics(va)))
    triples.sort(key=lambda x: (x[2]["avg_r"], x[2]["sum_r"]), reverse=True)

    return singles, pairs, triples


def print_rank(title, items, top=15):
    print(f"\n=== {title} ===")
    if not items:
        print("No candidates met minimum sample requirements.")
        return
    for i, (rules, tr, va) in enumerate(items[:top], 1):
        rs = rules if isinstance(rules, tuple) else (rules,)
        desc = " AND ".join(r[0] for r in rs)
        print(f"{i:2d}. {desc}")
        print(f"    TRAIN      {fmt(tr)}")
        print(f"    VALIDATION {fmt(va)}")


def feature_comparison(rows):
    wins = [r for r in rows if r["result"] == "TARGET"]
    losses = [r for r in rows if r["result"] == "STOPPED"]
    features = [
        "rsi", "atr_pct", "risk_pct", "pullback_pct", "volume_ratio",
        "ema_spread_pct", "macd_gap_atr", "waiting_bars",
        "waiting_hours", "price", "signal_minute", "activation_minute",
    ]

    print("\n=== TARGET vs STOPPED — FEATURE MEANS ===")
    print("feature              | target mean | stopped mean | difference")
    out = {}
    for f in features:
        w = [float(r[f]) for r in wins if r.get(f) is not None]
        l = [float(r[f]) for r in losses if r.get(f) is not None]
        if not w or not l:
            continue
        mw, ml = mean(w), mean(l)
        out[f] = {"target_mean": mw, "stopped_mean": ml, "difference": mw-ml}
        print(f"{f:20} | {mw:11.4f} | {ml:12.4f} | {mw-ml:+10.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--min-validation-trades", type=int, default=200)
    ap.add_argument("--min-train-trades", type=int, default=400)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--output", default="second_stop_filter_rows.csv")
    ap.add_argument("--summary-output", default="second_stop_filter_analysis.json")
    args = ap.parse_args()

    if not 0.5 <= args.train_frac <= 0.9:
        raise SystemExit("--train-frac should be between 0.5 and 0.9")

    DB = database()
    started = time.time()

    with DB() as s:
        ref = list(
            s.execute(
                select(Candle.ts, Candle.o, Candle.h, Candle.l, Candle.c, Candle.v)
                .where(Candle.symbol == "AAPL")
                .order_by(Candle.ts)
            ).all()
        )
        if not ref:
            raise SystemExit("AAPL reference candles missing.")

        latest = int(ref[-1][0])
        end_ts = latest + 900
        start_ts = end_ts - int(args.days * 86400)

        ref_times = [int(r[0]) for r in ref]
        ref_pos = {ts: i for i, ts in enumerate(ref_times)}
        ref_dates = sorted({local_parts(ts)[0] for ts in ref_times})

        symbols = list(
            s.execute(
                select(Stock.symbol)
                .where(Stock.market == "US")
                .order_by(Stock.symbol)
            ).scalars().all()
        )

    flags_path = Path(__file__).resolve().parent / "monitor" / "data" / "corporate_flags.json"
    try:
        flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except Exception:
        flags = {}

    rows = []
    total = len(symbols)
    errors = 0

    print("\nRajih — SECOND_STOP_RECOVERY filter analysis")
    print(f"Window: last {args.days:g} days ending {iso_ny(latest)}")
    print(f"US universe: {total}")
    print("Base target: FIRST STOP | protective stop: LEVEL 3")
    print("Reconstructing historical signals and collecting filter features...")

    for off in range(0, total, args.chunk_size):
        chunk = symbols[off:off+args.chunk_size]
        with DB() as s:
            raw = list(
                s.execute(
                    select(Candle.symbol, Candle.ts, Candle.o, Candle.h, Candle.l, Candle.c, Candle.v)
                    .where(Candle.symbol.in_(chunk))
                    .order_by(Candle.symbol, Candle.ts)
                ).all()
            )

        grouped = defaultdict(list)
        for sym, ts, o, h, l, c, v in raw:
            grouped[sym].append((int(ts), float(o), float(h), float(l), float(c), float(v)))

        for sym in chunk:
            try:
                rows.extend(
                    replay_symbol(
                        sym, grouped.get(sym, []), ref_pos, ref_dates,
                        start_ts, end_ts, sym in flags,
                        args.atr_mult, args.waiting_bars
                    )
                )
            except Exception as exc:
                errors += 1
                if errors <= 10:
                    print(f"WARNING {sym}: {type(exc).__name__}: {exc}")

        done = min(off+len(chunk), total)
        if done % 250 < args.chunk_size or done == total:
            print(f"Progress {done}/{total} | rows={len(rows)} | errors={errors} | elapsed={time.time()-started:.1f}s")

    # Only resolved filled trades are used to learn filters.
    resolved = [
        r for r in rows
        if r["result"] in ("TARGET", "STOPPED") and r["fill_price"] is not None and r["pnl_r"] is not None
    ]
    resolved.sort(key=lambda r: (r["signal_ts"], r["symbol"]))

    if len(resolved) < 1000:
        raise SystemExit(f"Only {len(resolved)} resolved trades; not enough for robust filter analysis.")

    split = int(len(resolved) * args.train_frac)
    train = resolved[:split]
    valid = resolved[split:]

    print("\n=== BASELINE ===")
    print(f"Resolved trades: {len(resolved)}")
    print(f"TRAIN      {fmt(metrics(train))}")
    print(f"VALIDATION {fmt(metrics(valid))}")

    feature_means = feature_comparison(resolved)

    singles, pairs, triples = search_filters(
        train, valid,
        args.min_train_trades,
        args.min_validation_trades,
    )

    print_rank("BEST SINGLE FILTERS", singles, 15)
    print_rank("BEST 2-CONDITION FILTERS", pairs, 15)
    print_rank("BEST 3-CONDITION FILTERS", triples, 15)

    # Robust candidates: positive avgR in BOTH train and validation.
    robust = []
    for size, items in [(1, singles), (2, pairs), (3, triples)]:
        for ruleset, tr, va in items:
            if tr["avg_r"] > 0 and va["avg_r"] > 0:
                rules_tuple = ruleset if isinstance(ruleset, tuple) else (ruleset,)
                robust.append((size, rules_tuple, tr, va))
    robust.sort(key=lambda x: (x[3]["avg_r"], x[3]["sum_r"]), reverse=True)

    print("\n=== ROBUST POSITIVE CANDIDATES (positive AvgR in TRAIN + VALIDATION) ===")
    if not robust:
        print("None found with the current minimum sample sizes.")
    else:
        for i, (size, ruleset, tr, va) in enumerate(robust[:20], 1):
            desc = " AND ".join(r[0] for r in ruleset)
            print(f"{i:2d}. [{size} rules] {desc}")
            print(f"    TRAIN      {fmt(tr)}")
            print(f"    VALIDATION {fmt(va)}")

    # Save all reconstructed rows.
    if rows:
        fields = list(rows[0].keys())
        with Path(args.output).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def serialize_item(item):
        ruleset, tr, va = item
        rs = ruleset if isinstance(ruleset, tuple) else (ruleset,)
        return {
            "rules": [r[0] for r in rs],
            "train": tr,
            "validation": va,
        }

    payload = {
        "window_days": args.days,
        "latest_ny": iso_ny(latest),
        "resolved_trades": len(resolved),
        "train_trades": len(train),
        "validation_trades": len(valid),
        "baseline_train": metrics(train),
        "baseline_validation": metrics(valid),
        "feature_means_target_vs_stopped": feature_means,
        "best_single": [serialize_item(x) for x in singles[:30]],
        "best_pairs": [serialize_item(x) for x in pairs[:30]],
        "best_triples": [serialize_item(x) for x in triples[:30]],
        "robust_positive": [
            {
                "size": size,
                "rules": [r[0] for r in ruleset],
                "train": tr,
                "validation": va,
            }
            for size, ruleset, tr, va in robust[:50]
        ],
        "notes": [
            "filters are discovered on chronological first 70% and checked on final 30%",
            "primary system target is first stop, because this was superior to original-entry target in prior backtest",
            "only resolved TARGET/STOPPED trades are used for filter discovery",
            "read-only",
        ],
    }

    Path(args.summary_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nCSV written: {args.output}")
    print(f"Analysis JSON written: {args.summary_output}")
    print(f"Total elapsed: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
