#!/usr/bin/env python3
"""
Rajih — FINAL HOLDOUT TEST for SECOND_STOP_RECOVERY fixed filter.

PURPOSE
-------
This is the FINAL out-of-sample holdout test for the fixed filter selected from
the prior discovery analysis.

The filter is LOCKED and is not optimized in this file:

    ATR% >= 0.668
    original entry price >= $93.49

The holdout window defaults to the 30 calendar days IMMEDIATELY BEFORE the
30-day discovery window.  In other words, with the latest reference candle as
the anchor:

    Discovery window used earlier: latest 30 days
    Final holdout here:          days -60 through -30

The current Plan table is NOT used. Original signals are reconstructed directly
from stored 15-minute candles.

Original signal rules reproduced here
--------------------------------------
- At least 200 complete regular-session 15m bars.
- Last 21 expected market slots present.
- 20 prior same-time sessions available with positive average volume.
- Signal-bar volume > 0.
- RSI(14) in [35, 75].
- EMA20 >= EMA50 OR MACD >= MACD signal.
- ATR(14) > 0.
- Recent confirmed pivot support in the last 80 bars.
- Original entry = signal high + 0.05 ATR, rounded up.
- Original stop = support - 0.25 ATR, rounded down.
- First available 3%/4%/5% target with RR >= 1.5.
- Current corporate-flag exclusions are honored.

SECOND_STOP_RECOVERY system
---------------------------
For every reconstructed original signal:
    level 1 / first_stop = original strategy stop
    level 2 / BUY LIMIT  = first_stop - 1.2 ATR
    level 3 / stop       = level 2 - 1.2 ATR
    target               = first_stop

Live-paper behavior is mirrored:
- A plan blocks a new plan for that symbol while WAITING or ACTIVE.
- WAITING expires after 78 processed 15m bars.
- Gap open at/below level 3 before a safe level-2 fill => CANCELLED.
- Safe gap below level 2 but above level 3 gets the better opening fill.
- A level-2 fill activates the trade; target/stop are evaluated starting with
  the NEXT bar, matching monitor.engine.advance().
- ACTIVE: stop is checked before target on an OHLC ambiguity.
- Default live time-exit is also mirrored: after 3 calendar days, exit only if
  current close is at least +0.5% above fill.
- Missing reference market slots after a plan is created are treated as DATA_GAP
  in the same conservative spirit as the live worker.

READ ONLY:
- Does not insert/update/delete DB rows.
- Does not send Telegram messages.

Run on Railway project root:
    python final_holdout_second_stop_filter.py --holdout-days 30 --discovery-days 30

Useful options:
    --days 60
    --atr-mult 1.2
    --waiting-bars 78
    --hold-days 3
    --hold-min-profit 0.5
    --chunk-size 50

Outputs:
    final_holdout_second_stop_filter.csv
    final_holdout_second_stop_filter_summary.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

from sqlalchemy import select, func

from core import database
from monitor.models import Stock, Candle
from monitor.strategy import rounded

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
US_START = 570   # 09:30 NY
US_END = 960     # 16:00 NY

# LOCKED FINAL FILTER. Do not optimize these in this script.
FILTER_MIN_ATR_PCT = 0.668
FILTER_MIN_PRICE = 93.49


def iso_ny(ts):
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), NY).isoformat(timespec="minutes")


def local_parts(ts):
    dt = datetime.fromtimestamp(int(ts), NY)
    return dt.date(), dt.hour * 60 + dt.minute


def percentile(values, p):
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def qstats(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p25": percentile(vals, .25),
        "p75": percentile(vals, .75),
        "p90": percentile(vals, .90),
        "min": min(vals),
        "max": max(vals),
    }


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


def rolling_indicators(bars):
    """Return exact-compatible indicator arrays used by the current strategy."""
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

    prefix = [0.0]
    for c in closes:
        prefix.append(prefix[-1] + c)

    sma20 = [None] * n
    sma50 = [None] * n
    sma200 = [None] * n
    for i in range(n):
        if i >= 19:
            sma20[i] = (prefix[i+1] - prefix[i+1-20]) / 20
        if i >= 49:
            sma50[i] = (prefix[i+1] - prefix[i+1-50]) / 50
        if i >= 199:
            sma200[i] = (prefix[i+1] - prefix[i+1-200]) / 200

    atr = [None] * n
    rsi = [None] * n
    if n >= 15:
        trs = []
        chgs = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            )
            trs.append(tr)
            chgs.append(closes[i] - closes[i-1])

        a = sum(trs[:14]) / 14.0
        gain = sum(max(x, 0.0) for x in chgs[:14]) / 14.0
        loss = sum(max(-x, 0.0) for x in chgs[:14]) / 14.0
        atr[14] = a
        rsi[14] = 100 - 100/(1 + gain/loss) if loss else (100.0 if gain else 50.0)

        for i in range(15, n):
            tr = trs[i-1]
            chg = chgs[i-1]
            a = (a * 13 + tr) / 14.0
            gain = (gain * 13 + max(chg, 0.0)) / 14.0
            loss = (loss * 13 + max(-chg, 0.0)) / 14.0
            atr[i] = a
            rsi[i] = 100 - 100/(1 + gain/loss) if loss else (100.0 if gain else 50.0)

    return {
        "close": closes,
        "high": highs,
        "low": lows,
        "ema20": e20,
        "ema50": e50,
        "macd": macd,
        "macd_signal": macd_sig,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
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


def volume_baseline_by_ts(bars, ref_dates):
    """
    Exact 20-prior-reference-date same-time baseline.
    Returns ts -> average prior volume, only when all 20 prior reference dates
    have that time slot and the average is positive.
    """
    by_key = {}
    for b in bars:
        d, minute = local_parts(b[0])
        by_key[(d, minute)] = float(b[5])

    result = {}
    for di, d in enumerate(ref_dates):
        if di < 20:
            continue
        prior = ref_dates[di-20:di]
        for minute in range(US_START, US_END, 15):
            vals = []
            ok = True
            for pd in prior:
                v = by_key.get((pd, minute))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if not ok:
                continue
            avg = sum(vals) / 20.0
            if avg <= 0:
                continue
            # Only store if current stock bar exists.
            current_ts = None
            # by_key is date/minute keyed, but we need the real stock ts.
            # Convert deterministically using NY timezone.
            dt = datetime(d.year, d.month, d.day, minute//60, minute%60, tzinfo=NY)
            current_ts = int(dt.timestamp())
            if (d, minute) in by_key:
                result[current_ts] = avg
    return result


def consecutive_reference_streak(bars, ref_pos):
    streak = [0] * len(bars)
    prev_pos = None
    current = 0
    for i, b in enumerate(bars):
        pos = ref_pos.get(int(b[0]))
        if pos is None:
            current = 0
            prev_pos = None
        else:
            if prev_pos is not None and pos == prev_pos + 1:
                current += 1
            else:
                current = 1
            prev_pos = pos
        streak[i] = current
    return streak


@dataclass
class Signal:
    symbol: str
    signal_ts: int
    signal_ny: str
    original_entry: float
    first_stop: float
    original_target: float
    atr: float
    rsi: float
    volume_ratio: float
    risk_pct: float
    selected_target_pct: int
    atr_pct: float
    filter_pass: bool
    second_stop: float
    third_stop: float
    state: str = "WAITING"
    waiting_bars: int = 0
    last_ts: int = 0
    fill_ts: int | None = None
    fill_price: float | None = None
    activation_ts: int | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    result: str = ""
    bars_after_fill: int = 0
    same_bar_ambiguous: bool = False


def build_signal(symbol, i, bars, ind, pivots, vol_avg, streak, corporate_flag, atr_mult):
    if corporate_flag:
        return None

    if i < 199 or streak[i] < 21:
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

    # evaluate() sees only bars through i, and pivot confirmation requires two
    # later bars, therefore latest usable pivot is i-2.
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
    selected_n = None
    for n_pct in (3, 4, 5):
        target, _ = rounded(entry * (1 + n_pct/100.0), "US", True)
        rr = (target - entry) / risk
        if rr >= 1.5:
            selected_target = target
            selected_n = n_pct
            break

    if selected_target is None:
        return None

    second_stop, _ = rounded(first_stop - atr_mult * atr, "US", False)
    third_stop, _ = rounded(second_stop - atr_mult * atr, "US", False)
    if third_stop <= 0 or not (third_stop < second_stop < first_stop):
        return None

    atr_pct = 100 * atr / entry
    filter_pass = (atr_pct >= FILTER_MIN_ATR_PCT and entry >= FILTER_MIN_PRICE)

    return Signal(
        symbol=symbol,
        signal_ts=ts,
        signal_ny=iso_ny(ts),
        original_entry=entry,
        first_stop=first_stop,
        original_target=selected_target,
        atr=atr,
        rsi=rsi,
        volume_ratio=v/avg,
        risk_pct=100*risk/entry,
        selected_target_pct=selected_n,
        atr_pct=atr_pct,
        filter_pass=filter_pass,
        second_stop=second_stop,
        third_stop=third_stop,
        last_ts=ts,
    )


def finalize_row(sig, latest_close=None):
    if not sig.result:
        sig.result = "WAITING_END" if sig.state == "WAITING" else "ACTIVE_END"
        if sig.state == "ACTIVE" and latest_close is not None:
            sig.exit_price = float(latest_close)

    risk = None
    pnl_r = None
    ret = None
    if sig.fill_price is not None:
        risk = sig.fill_price - sig.third_stop
        if sig.exit_price is not None and risk > 0:
            pnl_r = (sig.exit_price - sig.fill_price) / risk
            ret = 100 * (sig.exit_price / sig.fill_price - 1)

    return {
        "symbol": sig.symbol,
        "signal_ts": sig.signal_ts,
        "signal_ny": sig.signal_ny,
        "original_entry": sig.original_entry,
        "first_stop": sig.first_stop,
        "original_target": sig.original_target,
        "atr": sig.atr,
        "rsi": sig.rsi,
        "volume_ratio": sig.volume_ratio,
        "risk_pct": sig.risk_pct,
        "selected_target_pct": sig.selected_target_pct,
        "atr_pct": sig.atr_pct,
        "filter_pass": sig.filter_pass,
        "second_stop_entry": sig.second_stop,
        "third_stop_protective": sig.third_stop,
        "fill_ts_ny": iso_ny(sig.fill_ts),
        "fill_price": sig.fill_price,
        "exit_ts_ny": iso_ny(sig.exit_ts),
        "exit_price": sig.exit_price,
        "result": sig.result,
        "waiting_bars": sig.waiting_bars,
        "bars_after_fill": sig.bars_after_fill,
        "same_bar_ambiguous": sig.same_bar_ambiguous,
        "return_pct": ret,
        "pnl_r": pnl_r,
    }


def replay_symbol(
    symbol,
    bars,
    ref_pos,
    ref_dates,
    window_start,
    window_end,
    corporate_flag,
    atr_mult,
    waiting_limit,
    hold_days,
    hold_min_profit,
):
    if len(bars) < 200:
        return [], {"bars": len(bars), "signals": 0}

    bars = [tuple(b) for b in bars if int(b[0]) <= window_end]
    ind = rolling_indicators(bars)
    piv = pivot_lows(bars)
    vol_avg = volume_baseline_by_ts(bars, ref_dates)
    streak = consecutive_reference_streak(bars, ref_pos)

    completed = []
    open_plan = None
    signals = 0

    # Process only bars in the requested signal window, but indicator history
    # naturally includes older stored bars.
    start_i = bisect.bisect_left([int(b[0]) for b in bars], window_start)

    for i in range(start_i, len(bars)):
        ts, o, h, l, c, v = bars[i]
        ts = int(ts)
        if ts >= window_end:
            break

        # Only replay bars that belong to the reference market timeline.
        current_pos = ref_pos.get(ts)
        if current_pos is None:
            continue

        # 1) Advance existing plan first, exactly like the worker.
        if open_plan is not None:
            prev_pos = ref_pos.get(open_plan.last_ts)
            contiguous = prev_pos is not None and current_pos == prev_pos + 1
            open_plan.last_ts = ts

            if not contiguous:
                if open_plan.state == "ACTIVE":
                    open_plan.result = "DATA_GAP_ACTIVE"
                    completed.append(finalize_row(open_plan, latest_close=float(c)))
                    # Live new_plan blocks DATA_GAP with a paper entry; keep it
                    # blocked for the rest of this historical replay.
                    open_plan = None
                    break
                else:
                    open_plan.result = "DATA_GAP_WAITING"
                    completed.append(finalize_row(open_plan))
                    open_plan = None

            elif open_plan.state == "WAITING":
                open_plan.waiting_bars += 1

                if float(o) <= open_plan.third_stop:
                    open_plan.result = "CANCELLED_GAP"
                    open_plan.exit_ts = ts
                    completed.append(finalize_row(open_plan))
                    open_plan = None

                elif float(l) <= open_plan.second_stop:
                    fill = min(float(o), open_plan.second_stop) if float(o) <= open_plan.second_stop else open_plan.second_stop
                    fill, _ = rounded(fill, "US", True)
                    if fill <= open_plan.third_stop or fill >= open_plan.first_stop:
                        open_plan.result = "CANCELLED_INVALID_FILL"
                        open_plan.exit_ts = ts
                        completed.append(finalize_row(open_plan))
                        open_plan = None
                    else:
                        open_plan.state = "ACTIVE"
                        open_plan.fill_ts = ts
                        open_plan.activation_ts = ts
                        open_plan.fill_price = fill
                        # IMPORTANT: live engine returns immediately after fill;
                        # target/stop are NOT evaluated on this fill candle.

                if open_plan is not None and open_plan.state == "WAITING" and open_plan.waiting_bars >= waiting_limit:
                    open_plan.result = "EXPIRED"
                    open_plan.exit_ts = ts
                    completed.append(finalize_row(open_plan))
                    open_plan = None

            elif open_plan.state == "ACTIVE":
                open_plan.bars_after_fill += 1

                if float(l) <= open_plan.third_stop:
                    open_plan.result = "STOPPED"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = min(float(o), open_plan.third_stop)
                    if float(h) >= open_plan.first_stop:
                        open_plan.same_bar_ambiguous = True
                    completed.append(finalize_row(open_plan))
                    open_plan = None

                elif float(h) >= open_plan.first_stop:
                    open_plan.result = "TARGET"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = open_plan.first_stop
                    completed.append(finalize_row(open_plan))
                    open_plan = None

                elif (
                    open_plan.activation_ts is not None
                    and open_plan.fill_price is not None
                    and ts - open_plan.activation_ts >= hold_days * 86400
                    and (float(c) / open_plan.fill_price - 1) * 100 + 1e-9 >= hold_min_profit
                ):
                    open_plan.result = "TIME_EXIT"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = float(c)
                    completed.append(finalize_row(open_plan))
                    open_plan = None

        # 2) If symbol is free after advancement, evaluate THIS bar for a new
        # original conditional signal.
        if open_plan is None:
            sig = build_signal(
                symbol, i, bars, ind, piv, vol_avg, streak,
                corporate_flag, atr_mult
            )
            if sig is not None:
                signals += 1
                open_plan = sig

    if open_plan is not None:
        latest_close = float(bars[-1][4]) if bars else None
        completed.append(finalize_row(open_plan, latest_close=latest_close))

    return completed, {"bars": len(bars), "signals": signals}


def summarize(rows):
    states = Counter(r["result"] for r in rows)
    filled = [r for r in rows if r["fill_price"] is not None]
    resolved = [r for r in filled if r["result"] in ("TARGET", "STOPPED", "TIME_EXIT")]
    targets = [r for r in resolved if r["result"] == "TARGET"]
    stops = [r for r in resolved if r["result"] == "STOPPED"]
    time_exits = [r for r in resolved if r["result"] == "TIME_EXIT"]
    opens = [r for r in filled if r["result"] in ("ACTIVE_END", "DATA_GAP_ACTIVE")]

    rvals = [r["pnl_r"] for r in resolved if r["pnl_r"] is not None]
    returns = [r["return_pct"] for r in resolved if r["return_pct"] is not None]

    return {
        "signals": len(rows),
        "filled": len(filled),
        "fill_rate_pct": round(100 * len(filled) / len(rows), 2) if rows else 0,
        "target": len(targets),
        "stopped": len(stops),
        "time_exit": len(time_exits),
        "resolved": len(resolved),
        "open_or_data_gap_active": len(opens),
        "resolved_target_win_rate_pct": round(100 * len(targets) / len(resolved), 2) if resolved else None,
        "positive_resolved_pct": round(
            100 * sum((r["pnl_r"] or 0) > 0 for r in resolved) / len(resolved), 2
        ) if resolved else None,
        "sum_R_resolved": round(sum(rvals), 3) if rvals else 0,
        "avg_R_resolved": round(mean(rvals), 3) if rvals else None,
        "avg_return_pct_resolved": round(mean(returns), 3) if returns else None,
        "waiting_bars_to_fill": qstats([r["waiting_bars"] for r in filled]),
        "bars_after_fill_resolved": qstats([r["bars_after_fill"] for r in resolved]),
        "same_bar_ambiguous": sum(bool(r["same_bar_ambiguous"]) for r in rows),
        "results": dict(states),
    }


def filtered_subset(rows):
    return [r for r in rows if r.get("filter_pass")]


def print_summary(title, s):
    print(f"\n=== {title} ===")
    print(
        f"Signals: {s['signals']} | Filled: {s['filled']} ({s['fill_rate_pct']}%)"
    )
    print(
        f"Target: {s['target']} | Stop: {s['stopped']} | Time exit: {s['time_exit']} | "
        f"Resolved: {s['resolved']} | Open/data-gap active: {s['open_or_data_gap_active']}"
    )
    print(
        f"Resolved target win rate: {s['resolved_target_win_rate_pct']}% | "
        f"Positive resolved: {s['positive_resolved_pct']}% | "
        f"Sum R: {s['sum_R_resolved']} | Avg R: {s['avg_R_resolved']} | "
        f"Avg return: {s['avg_return_pct_resolved']}%"
    )
    if s["waiting_bars_to_fill"]:
        x = s["waiting_bars_to_fill"]
        print(
            f"Waiting bars to fill: median={x['median']:.1f} | P75={x['p75']:.1f} | P90={x['p90']:.1f}"
        )
    if s["bars_after_fill_resolved"]:
        x = s["bars_after_fill_resolved"]
        print(
            f"Bars after fill to resolution: median={x['median']:.1f} | P75={x['p75']:.1f} | P90={x['p90']:.1f}"
        )
    print("Result states:", s["results"])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-days", type=float, default=30.0,
                    help="Length of the untouched final holdout window.")
    ap.add_argument("--discovery-days", type=float, default=30.0,
                    help="Days reserved after the holdout (the prior discovery window).")
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--hold-days", type=float, default=3.0)
    ap.add_argument("--hold-min-profit", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--output", default="final_holdout_second_stop_filter.csv")
    ap.add_argument("--summary-output", default="final_holdout_second_stop_filter_summary.json")
    args = ap.parse_args()

    if args.holdout_days <= 0 or args.discovery_days < 0:
        raise SystemExit("Invalid holdout/discovery window.")

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
            raise SystemExit("AAPL reference candles are missing.")

        latest = int(ref[-1][0])

        # Final holdout is the untouched block BEFORE the prior discovery window.
        holdout_end = latest + 900 - int(args.discovery_days * 86400)
        holdout_start = holdout_end - int(args.holdout_days * 86400)

        ref_times = [int(r[0]) for r in ref]
        ref_pos = {ts: i for i, ts in enumerate(ref_times)}
        ref_dates = sorted({local_parts(ts)[0] for ts in ref_times})

        stocks = list(
            s.execute(
                select(Stock.symbol)
                .where(Stock.market == "US")
                .order_by(Stock.symbol)
            ).scalars().all()
        )

    flags_path = Path(__file__).resolve().parent / "monitor" / "data" / "corporate_flags.json"
    try:
        corporate_flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except Exception:
        corporate_flags = {}

    all_rows = []
    counters = Counter()
    total = len(stocks)

    print("\nRajih — FINAL HOLDOUT TEST — fixed SECOND_STOP_RECOVERY filter")
    print(f"Latest reference candle: {iso_ny(latest)}")
    print(f"Prior discovery window reserved: latest {args.discovery_days:g} days")
    print(
        f"FINAL HOLDOUT: {iso_ny(holdout_start)} -> {iso_ny(holdout_end)} "
        f"({args.holdout_days:g} calendar days)"
    )
    print(f"US universe: {total}")
    print("LOCKED FILTER (no optimization in this file):")
    print(f"  ATR% >= {FILTER_MIN_ATR_PCT}")
    print(f"  Original entry price >= ${FILTER_MIN_PRICE:.2f}")
    print(
        f"System: level2=first_stop-{args.atr_mult:g}ATR | target=first_stop | "
        f"level3=level2-{args.atr_mult:g}ATR | waiting={args.waiting_bars} bars"
    )
    print("Reconstructing original signals from candles; current Plan table is not used.")

    for offset in range(0, total, args.chunk_size):
        chunk = stocks[offset:offset + args.chunk_size]

        with DB() as s:
            raw_rows = list(
                s.execute(
                    select(Candle.symbol, Candle.ts, Candle.o, Candle.h, Candle.l, Candle.c, Candle.v)
                    .where(Candle.symbol.in_(chunk))
                    .order_by(Candle.symbol, Candle.ts)
                ).all()
            )

        grouped = defaultdict(list)
        for sym, ts, o, h, l, c, v in raw_rows:
            grouped[sym].append((int(ts), float(o), float(h), float(l), float(c), float(v)))

        for sym in chunk:
            bars = grouped.get(sym, [])
            try:
                rows, info = replay_symbol(
                    sym, bars, ref_pos, ref_dates,
                    holdout_start, holdout_end,
                    sym in corporate_flags,
                    args.atr_mult,
                    args.waiting_bars,
                    args.hold_days,
                    args.hold_min_profit,
                )
                all_rows.extend(rows)
                counters["signals"] += info["signals"]
                counters["symbols_ok"] += 1
                if info["signals"]:
                    counters["symbols_with_signals"] += 1
            except Exception as exc:
                counters["symbols_error"] += 1
                if counters["symbols_error"] <= 10:
                    print(f"WARNING {sym}: {type(exc).__name__}: {exc}")

        done = min(offset + len(chunk), total)
        if done % 250 < args.chunk_size or done == total:
            print(
                f"Progress {done}/{total} | signals={counters['signals']} | "
                f"errors={counters['symbols_error']} | elapsed={time.time()-started:.1f}s"
            )

    if not all_rows:
        print("No reconstructed signals in holdout window.")
        return 2

    baseline = summarize(all_rows)
    filt_rows = filtered_subset(all_rows)
    filtered = summarize(filt_rows)

    print_summary("HOLDOUT BASELINE — NO FILTER", baseline)
    print_summary(
        f"HOLDOUT FIXED FILTER — ATR% >= {FILTER_MIN_ATR_PCT} AND PRICE >= ${FILTER_MIN_PRICE:.2f}",
        filtered
    )

    # The decision check is informational only; the filter itself remains fixed.
    print("\n=== HOLDOUT CHECK ===")
    enough = filtered["resolved"] >= 200
    positive = filtered["avg_R_resolved"] is not None and filtered["avg_R_resolved"] > 0
    win55 = filtered["resolved_target_win_rate_pct"] is not None and filtered["resolved_target_win_rate_pct"] >= 55
    print(f"Resolved filtered trades >= 200: {'YES' if enough else 'NO'} ({filtered['resolved']})")
    print(f"Avg R > 0: {'YES' if positive else 'NO'} ({filtered['avg_R_resolved']})")
    print(f"Win rate >= 55%: {'YES' if win55 else 'NO'} ({filtered['resolved_target_win_rate_pct']}%)")

    fields = list(all_rows[0].keys())
    with Path(args.output).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    payload = {
        "latest_reference_ny": iso_ny(latest),
        "holdout_start_ny": iso_ny(holdout_start),
        "holdout_end_ny": iso_ny(holdout_end),
        "holdout_days": args.holdout_days,
        "discovery_days_reserved_after_holdout": args.discovery_days,
        "locked_filter": {
            "atr_pct_min": FILTER_MIN_ATR_PCT,
            "original_entry_price_min": FILTER_MIN_PRICE,
        },
        "baseline": baseline,
        "filtered": filtered,
        "holdout_check": {
            "resolved_at_least_200": enough,
            "avg_r_positive": positive,
            "win_rate_at_least_55_pct": win55,
        },
        "notes": [
            "filter thresholds are locked from the prior discovery analysis",
            "this script does not search or optimize thresholds",
            "holdout defaults to the 30 days immediately preceding the prior 30-day discovery window",
            "read-only",
        ],
    }

    Path(args.summary_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nCSV written: {args.output}")
    print(f"Summary JSON written: {args.summary_output}")
    print(f"Total elapsed: {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
