"""
Rajih — post-stop depth + wider-stop + timed-exit analysis.

Fixed entry system
------------------
Entry = BULLISH_RECLAIM close
Target = Stop1
Original stop = Stop3

This analysis answers TWO questions:

A) For trades that hit Stop3 and later recovered to Stop1:
   - how far BELOW Stop3 did price go before recovery?
   - measured in dollars, %, ATR, and original R

B) If we do NOT exit immediately at Stop3:
   - what happens if we give the trade 1 / 2 / 3 US sessions to recover?
   - if target is not reached by the deadline, close at that deadline close
   - what is the resulting loss / return / R?

It also simulates WIDER hard stops from the original confirmation entry:
    Stop3 - 0.5 ATR
    Stop3 - 1.0 ATR
    Stop3 - 1.5 ATR
    Stop3 - 2.0 ATR

For each wider stop, it reports results with time exits at:
    26 bars (~1 US session)
    52 bars (~2 US sessions)
    78 bars (~3 US sessions)

Important
---------
- Target remains Stop1.
- Same-bar stop+target is treated stop-first conservatively.
- No target/stop evaluation on the confirmation entry candle.
- Current Plan table is NOT used.
- Two independent 30-day periods are analyzed and also combined.

Run:
    python analyze_stop_depth_time_exit.py

Outputs:
    stop_depth_time_exit_rows.csv
    stop_depth_time_exit_summary.json
"""




from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
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

VARIANTS = ("STOP1",)


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
    a = 2.0 / (n + 1.0)
    v = float(values[0])
    out = []
    for x in values:
        v += (float(x) - v) * a
        out.append(v)
    return out


def indicators(bars):
    n = len(bars)
    c = [float(b[4]) for b in bars]
    h = [float(b[2]) for b in bars]
    l = [float(b[3]) for b in bars]

    e12 = ema_series(c, 12)
    e20 = ema_series(c, 20)
    e26 = ema_series(c, 26)
    e50 = ema_series(c, 50)
    macd = [x - y for x, y in zip(e12, e26)]
    macd_sig = ema_series(macd, 9)

    atr = [None] * n
    rsi = [None] * n
    if n >= 15:
        trs, ch = [], []
        for i in range(1, n):
            trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
            ch.append(c[i]-c[i-1])

        av = sum(trs[:14]) / 14.0
        g = sum(max(x, 0) for x in ch[:14]) / 14.0
        d = sum(max(-x, 0) for x in ch[:14]) / 14.0
        atr[14] = av
        rsi[14] = 100 - 100/(1+g/d) if d else (100.0 if g else 50.0)

        for i in range(15, n):
            av = (av*13 + trs[i-1]) / 14.0
            g = (g*13 + max(ch[i-1], 0)) / 14.0
            d = (d*13 + max(-ch[i-1], 0)) / 14.0
            atr[i] = av
            rsi[i] = 100 - 100/(1+g/d) if d else (100.0 if g else 50.0)

    return {
        "ema20": e20, "ema50": e50,
        "macd": macd, "macd_signal": macd_sig,
        "atr": atr, "rsi": rsi,
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
            avg = sum(vals)/20.0
            if avg <= 0 or (d, minute) not in by:
                continue
            dt = datetime(d.year, d.month, d.day, minute//60, minute%60, tzinfo=NY)
            out[int(dt.timestamp())] = avg
    return out


def streaks(bars, ref_pos):
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
class Signal:
    symbol: str
    signal_ts: int
    original_entry: float
    first_stop: float
    second_stop: float
    third_stop: float
    atr: float
    state: str = "WAITING_TOUCH"
    waiting_bars: int = 0
    touch_ts: int | None = None
    confirm_wait_bars: int = 0
    fill_ts: int | None = None
    fill_price: float | None = None
    target_price: float | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    result: str = ""
    bars_after_fill: int = 0
    last_ts: int = 0

    # Post-stop recovery tracking
    stopped_ts: int | None = None
    post_stop_recovered: bool = False
    post_stop_recovery_ts: int | None = None
    post_stop_bars_to_target: int | None = None
    post_stop_max_high: float | None = None
    post_stop_min_low: float | None = None
    post_stop_bar_closes: list | None = None


def build_signal(symbol, i, bars, ind, pivs, volavg, streak, corporate_flag, mult):
    if corporate_flag or i < 199 or streak[i] < 21:
        return None

    atr = ind["atr"][i]
    rsi = ind["rsi"][i]
    if atr is None or rsi is None or atr <= 0:
        return None

    ts, o, h, l, c, v = bars[i]
    ts = int(ts)
    v = float(v)
    avg = volavg.get(ts)

    if avg is None or avg <= 0 or v <= 0:
        return None
    if not (35 <= rsi <= 75):
        return None
    if not (ind["ema20"][i] >= ind["ema50"][i] or ind["macd"][i] >= ind["macd_signal"][i]):
        return None

    entry, _ = rounded(float(h) + 0.05*atr, "US", True)

    right = bisect.bisect_right(pivs, i-2) - 1
    support_idx = None
    limit_price = min(float(c), entry)
    while right >= 0:
        j = pivs[right]
        if j < i-79:
            break
        if float(bars[j][3]) < limit_price:
            support_idx = j
            break
        right -= 1
    if support_idx is None:
        return None

    support = float(bars[support_idx][3])
    first_stop, _ = rounded(support - 0.25*atr, "US", False)
    risk = entry - first_stop
    if first_stop <= 0 or risk <= 0:
        return None

    ok = False
    for pct in (3,4,5):
        target, _ = rounded(entry*(1+pct/100.0), "US", True)
        if (target-entry)/risk >= 1.5:
            ok = True
            break
    if not ok:
        return None

    second, _ = rounded(first_stop - mult*atr, "US", False)
    third, _ = rounded(second - mult*atr, "US", False)
    if third <= 0 or not (third < second < first_stop):
        return None

    return Signal(
        symbol=symbol,
        signal_ts=ts,
        original_entry=entry,
        first_stop=first_stop,
        second_stop=second,
        third_stop=third,
        atr=atr,
        last_ts=ts,
    )


def clone_signal(s):
    return Signal(
        symbol=s.symbol,
        signal_ts=s.signal_ts,
        original_entry=s.original_entry,
        first_stop=s.first_stop,
        second_stop=s.second_stop,
        third_stop=s.third_stop,
        atr=s.atr,
        last_ts=s.last_ts,
    )


def confirmation_fill(variant, p, bar, confirm_window):
    ts, o, h, l, c, v = bar
    ts = int(ts)
    o, h, l, c = map(float, (o, h, l, c))

    if p.state == "WAITING_TOUCH":
        if l > p.second_stop:
            return None
        p.touch_ts = ts
        p.state = "WAITING_CONFIRM"
        p.confirm_wait_bars = 0

        # Touch candle itself can be the bullish reclaim.
        if c >= p.second_stop and c > o:
            return c
        return None

    if p.state == "WAITING_CONFIRM":
        p.confirm_wait_bars += 1

        if c >= p.second_stop and c > o:
            return c

        if p.confirm_wait_bars >= confirm_window:
            p.result = "CONFIRM_EXPIRED"
        return None

    return None


def target_price_for_variant(variant, p):
    """
    Target is calculated only AFTER confirmation fill exists.
    """
    fill = float(p.fill_price)
    stop = float(p.third_stop)
    stop1 = float(p.first_stop)
    original = float(p.original_entry)

    if variant == "STOP1":
        return stop1
    if variant == "T25":
        return stop1 + 0.25 * (original - stop1)
    if variant == "T50":
        return stop1 + 0.50 * (original - stop1)
    if variant == "T75":
        return stop1 + 0.75 * (original - stop1)
    if variant == "ORIGINAL_ENTRY":
        return original

    risk = fill - stop
    if risk <= 0:
        return None

    mult = {
        "R_075": 0.75,
        "R_100": 1.00,
        "R_125": 1.25,
        "R_150": 1.50,
    }.get(variant)

    if mult is None:
        return None
    return fill + mult * risk


def finalize(variant, period_name, p, latest_close=None):
    if not p.result:
        if p.fill_price is not None:
            p.result = "ACTIVE_END"
            if latest_close is not None:
                p.exit_price = float(latest_close)
        elif p.state == "WAITING_CONFIRM":
            p.result = "CONFIRM_END"
        else:
            p.result = "WAITING_END"

    pnl_r = None
    ret = None
    if p.fill_price is not None and p.exit_price is not None:
        risk = p.fill_price - p.third_stop
        if risk > 0:
            pnl_r = (p.exit_price-p.fill_price)/risk
            ret = 100*(p.exit_price/p.fill_price-1)

    return {
        "period": period_name,
        "variant": variant,
        "symbol": p.symbol,
        "signal_ts": p.signal_ts,
        "signal_ny": iso_ny(p.signal_ts),
        "touch_ny": iso_ny(p.touch_ts),
        "fill_ny": iso_ny(p.fill_ts),
        "exit_ny": iso_ny(p.exit_ts),
        "original_entry": p.original_entry,
        "first_stop": p.first_stop,
        "second_stop": p.second_stop,
        "third_stop": p.third_stop,
        "atr": p.atr,
        "waiting_bars": p.waiting_bars,
        "confirm_wait_bars": p.confirm_wait_bars,
        "fill_price": p.fill_price,
        "target_price": p.target_price,
        "exit_price": p.exit_price,
        "result": p.result,
        "bars_after_fill": p.bars_after_fill,
        "pnl_r": pnl_r,
        "return_pct": ret,
        "post_stop_recovered": False,
        "post_stop_recovery_ny": "",
        "post_stop_bars_to_target": None,
        "post_stop_hours_to_target": None,
        "post_stop_max_high": None,
        "post_stop_min_low": None,
        "post_stop_max_rebound_pct_from_stop": None,
        "post_stop_depth_pct_below_stop": None,
        "post_stop_depth_atr_below_stop": None,
        "post_stop_mae_pct_from_entry": None,
        "post_stop_mae_r_from_entry": None,
        "timeexit_26_outcome": "",
        "timeexit_26_exit_price": None,
        "timeexit_26_return_pct": None,
        "timeexit_26_r": None,
        "timeexit_52_outcome": "",
        "timeexit_52_exit_price": None,
        "timeexit_52_return_pct": None,
        "timeexit_52_r": None,
        "timeexit_78_outcome": "",
        "timeexit_78_exit_price": None,
        "timeexit_78_return_pct": None,
        "timeexit_78_r": None,
    }


def replay_variant(symbol, bars, ref_pos, ref_dates, start_ts, end_ts, flag,
                   mult, waiting_limit, confirm_window, variant):
    if len(bars) < 200:
        return []

    bars = [tuple(b) for b in bars if int(b[0]) <= end_ts]
    ind = indicators(bars)
    pivs = pivot_lows(bars)
    volavg = volume_baseline(bars, ref_dates)
    streak = streaks(bars, ref_pos)

    times = [int(b[0]) for b in bars]
    start_i = bisect.bisect_left(times, start_ts)
    rows = []
    open_plan = None
    post_stop_trackers = []
    all_stopped_trackers = []

    for i in range(start_i, len(bars)):
        ts, o, h, l, c, v = bars[i]
        ts = int(ts)
        if ts >= end_ts:
            break
        pos = ref_pos.get(ts)
        if pos is None:
            continue

        # Continue observing stopped trades to see whether price later reaches
        # the original target. This does NOT change the trade result.
        if post_stop_trackers:
            still_tracking = []
            for tr in post_stop_trackers:
                if tr.post_stop_recovered:
                    continue

                # Count only bars AFTER the stop bar.
                bars_since = 0
                prev_pos = ref_pos.get(tr.stopped_ts)
                if prev_pos is not None:
                    bars_since = max(0, pos - prev_pos)

                if tr.post_stop_max_high is None or float(h) > tr.post_stop_max_high:
                    tr.post_stop_max_high = float(h)
                if tr.post_stop_min_low is None or float(l) < tr.post_stop_min_low:
                    tr.post_stop_min_low = float(l)

                if tr.post_stop_bar_closes is None:
                    tr.post_stop_bar_closes = []
                tr.post_stop_bar_closes.append((bars_since, float(c), float(h), float(l), int(ts)))

                if float(h) >= float(tr.target_price):
                    tr.post_stop_recovered = True
                    tr.post_stop_recovery_ts = ts
                    tr.post_stop_bars_to_target = bars_since
                else:
                    still_tracking.append(tr)

            post_stop_trackers = still_tracking

        if open_plan is not None:
            prev = ref_pos.get(open_plan.last_ts)
            contiguous = prev is not None and pos == prev + 1
            open_plan.last_ts = ts

            if not contiguous:
                open_plan.result = "DATA_GAP_ACTIVE" if open_plan.fill_price is not None else "DATA_GAP_WAITING"
                if open_plan.fill_price is not None:
                    open_plan.exit_ts = ts
                    open_plan.exit_price = float(c)
                rows.append(finalize(variant, "", open_plan))
                open_plan = None

            elif open_plan.fill_price is None:
                open_plan.waiting_bars += 1

                # Unsafe gap below protective stop before a fill.
                if float(o) <= open_plan.third_stop:
                    open_plan.result = "CANCELLED_GAP"
                    open_plan.exit_ts = ts
                    rows.append(finalize(variant, "", open_plan))
                    open_plan = None
                else:
                    fill = confirmation_fill(
                        variant, open_plan, (ts,o,h,l,c,v), confirm_window
                    )

                    if open_plan is not None and open_plan.result in ("CONFIRM_FAILED", "CONFIRM_EXPIRED"):
                        open_plan.exit_ts = ts
                        rows.append(finalize(variant, "", open_plan))
                        open_plan = None

                    elif open_plan is not None and fill is not None:
                        fill, _ = rounded(float(fill), "US", True)
                        if fill <= open_plan.third_stop:
                            open_plan.result = "CANCELLED_INVALID_CONFIRM"
                            open_plan.exit_ts = ts
                            rows.append(finalize(variant, "", open_plan))
                            open_plan = None
                        else:
                            open_plan.fill_price = fill
                            open_plan.fill_ts = ts
                            open_plan.target_price = target_price_for_variant(variant, open_plan)

                            if (
                                open_plan.target_price is None
                                or open_plan.target_price <= fill
                            ):
                                open_plan.result = "CANCELLED_INVALID_TARGET"
                                open_plan.exit_ts = ts
                                rows.append(finalize(variant, "", open_plan))
                                open_plan = None
                            else:
                                open_plan.state = "ACTIVE"

                    if open_plan is not None and open_plan.fill_price is None and open_plan.waiting_bars >= waiting_limit:
                        open_plan.result = "EXPIRED"
                        open_plan.exit_ts = ts
                        rows.append(finalize(variant, "", open_plan))
                        open_plan = None

            else:
                open_plan.bars_after_fill += 1

                # Conservative stop-first on ambiguous later candles.
                target = float(open_plan.target_price)

                if float(l) <= open_plan.third_stop:
                    open_plan.result = "STOPPED"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = min(float(o), open_plan.third_stop)
                    open_plan.stopped_ts = ts
                    open_plan.post_stop_max_high = float(h)
                    open_plan.post_stop_min_low = float(l)
                    open_plan.post_stop_bar_closes = []

                    # If the SAME stop candle also traded at/above target, we still
                    # count STOP first conservatively, but recovery time is 0 bars.
                    if float(h) >= target:
                        open_plan.post_stop_recovered = True
                        open_plan.post_stop_recovery_ts = ts
                        open_plan.post_stop_bars_to_target = 0

                    rows.append(finalize(variant, "", open_plan))

                    # Create a lightweight tracker for post-stop price evolution.
                    tracker = open_plan
                    open_plan = None

                    # Store tracker in a local list attached to rows via hidden key
                    # handled below by replay function state.
                    post_stop_trackers.append(tracker)
                    all_stopped_trackers.append(tracker)

                elif float(h) >= target:
                    open_plan.result = "TARGET"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = target
                    rows.append(finalize(variant, "", open_plan))
                    open_plan = None

        if open_plan is None:
            sig = build_signal(
                symbol, i, bars, ind, pivs, volavg, streak, flag, mult
            )
            if sig is not None:
                open_plan = sig

    if open_plan is not None:
        rows.append(finalize(
            variant, "", open_plan,
            float(bars[-1][4]) if bars and open_plan.fill_price is not None else None
        ))

    tracker_map = {(t.symbol, t.signal_ts): t for t in all_stopped_trackers}
    for r in rows:
        if r.get("result") != "STOPPED":
            continue
        tr = tracker_map.get((r.get("symbol"), r.get("signal_ts")))
        if tr is None:
            continue
        r["post_stop_recovered"] = bool(tr.post_stop_recovered)
        r["post_stop_recovery_ny"] = iso_ny(tr.post_stop_recovery_ts)
        r["post_stop_bars_to_target"] = tr.post_stop_bars_to_target
        r["post_stop_hours_to_target"] = (
            tr.post_stop_bars_to_target * 0.25
            if tr.post_stop_bars_to_target is not None else None
        )
        r["post_stop_max_high"] = tr.post_stop_max_high
        r["post_stop_min_low"] = tr.post_stop_min_low

        if tr.post_stop_max_high is not None and tr.exit_price:
            r["post_stop_max_rebound_pct_from_stop"] = (
                100 * (tr.post_stop_max_high / tr.exit_price - 1)
            )
        else:
            r["post_stop_max_rebound_pct_from_stop"] = None

        if tr.post_stop_min_low is not None and tr.exit_price:
            stop_px = float(tr.exit_price)
            atr = float(tr.atr)
            fill = float(tr.fill_price) if tr.fill_price is not None else None
            original_risk = (fill - stop_px) if fill is not None else None

            r["post_stop_depth_pct_below_stop"] = 100 * (stop_px - tr.post_stop_min_low) / stop_px
            r["post_stop_depth_atr_below_stop"] = (
                (stop_px - tr.post_stop_min_low) / atr if atr > 0 else None
            )
            r["post_stop_mae_pct_from_entry"] = (
                100 * (tr.post_stop_min_low / fill - 1) if fill else None
            )
            r["post_stop_mae_r_from_entry"] = (
                (tr.post_stop_min_low - fill) / original_risk
                if original_risk and original_risk > 0 else None
            )
        else:
            r["post_stop_depth_pct_below_stop"] = None
            r["post_stop_depth_atr_below_stop"] = None
            r["post_stop_mae_pct_from_entry"] = None
            r["post_stop_mae_r_from_entry"] = None

        # Snapshot outcomes if we ignore Stop3 and instead give the trade
        # 1/2/3 sessions after the original stop to recover.
        seq = tr.post_stop_bar_closes or []
        for deadline in (26, 52, 78):
            target_hit = False
            deadline_close = None
            actual_bar = None
            for bnum, close_px, hi_px, lo_px, bts in seq:
                if hi_px >= float(tr.target_price):
                    target_hit = True
                    break
                if bnum >= deadline:
                    deadline_close = close_px
                    actual_bar = bnum
                    break

            key = f"timeexit_{deadline}"
            if target_hit:
                r[key + "_outcome"] = "TARGET_BEFORE_DEADLINE"
                r[key + "_exit_price"] = float(tr.target_price)
            elif deadline_close is not None:
                r[key + "_outcome"] = "TIME_EXIT"
                r[key + "_exit_price"] = deadline_close
            else:
                r[key + "_outcome"] = "NO_DATA_TO_DEADLINE"
                r[key + "_exit_price"] = None

            ep = r[key + "_exit_price"]
            if ep is not None and tr.fill_price is not None:
                risk0 = float(tr.fill_price) - float(tr.exit_price)
                r[key + "_return_pct"] = 100 * (ep / float(tr.fill_price) - 1)
                r[key + "_r"] = (
                    (ep - float(tr.fill_price)) / risk0 if risk0 > 0 else None
                )
            else:
                r[key + "_return_pct"] = None
                r[key + "_r"] = None

    return rows


def qstats(vals):
    vals = sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
    if not vals:
        return {}
    def q(p):
        if len(vals) == 1:
            return vals[0]
        k = (len(vals)-1)*p
        lo, hi = math.floor(k), math.ceil(k)
        if lo == hi:
            return vals[lo]
        return vals[lo]*(hi-k)+vals[hi]*(k-lo)
    return {"mean":mean(vals), "median":q(.5), "p75":q(.75), "p90":q(.9)}


def summarize(rows):
    stopped = [r for r in rows if r["result"] == "STOPPED"]
    recovered = [r for r in stopped if r.get("post_stop_recovered")]
    unrecovered = [r for r in stopped if not r.get("post_stop_recovered")]

    bars = [
        int(r["post_stop_bars_to_target"])
        for r in recovered
        if r.get("post_stop_bars_to_target") is not None
    ]
    hours = [b * 0.25 for b in bars]

    depth_pct = [
        float(r["post_stop_depth_pct_below_stop"])
        for r in recovered
        if r.get("post_stop_depth_pct_below_stop") is not None
    ]
    depth_atr = [
        float(r["post_stop_depth_atr_below_stop"])
        for r in recovered
        if r.get("post_stop_depth_atr_below_stop") is not None
    ]
    mae_r = [
        float(r["post_stop_mae_r_from_entry"])
        for r in recovered
        if r.get("post_stop_mae_r_from_entry") is not None
    ]

    timed = {}
    for deadline in (26, 52, 78):
        k = f"timeexit_{deadline}"
        usable = [r for r in stopped if r.get(k + "_outcome") != "NO_DATA_TO_DEADLINE"]
        wins = [r for r in usable if r.get(k + "_outcome") == "TARGET_BEFORE_DEADLINE"]
        exits = [r for r in usable if r.get(k + "_outcome") == "TIME_EXIT"]
        rvals = [float(r[k + "_r"]) for r in usable if r.get(k + "_r") is not None]
        rets = [float(r[k + "_return_pct"]) for r in usable if r.get(k + "_return_pct") is not None]
        exit_only_rets = [float(r[k + "_return_pct"]) for r in exits if r.get(k + "_return_pct") is not None]
        exit_only_r = [float(r[k + "_r"]) for r in exits if r.get(k + "_r") is not None]

        timed[str(deadline)] = {
            "usable_stopped_trades": len(usable),
            "target_before_deadline": len(wins),
            "time_exit_count": len(exits),
            "target_rate_pct": round(100*len(wins)/len(usable),2) if usable else None,
            "combined_avg_r": round(mean(rvals),3) if rvals else None,
            "combined_sum_r": round(sum(rvals),3) if rvals else None,
            "combined_avg_return_pct": round(mean(rets),3) if rets else None,
            "time_exit_only_avg_return_pct": round(mean(exit_only_rets),3) if exit_only_rets else None,
            "time_exit_only_median_return_pct": qstats(exit_only_rets).get("median") if exit_only_rets else None,
            "time_exit_only_avg_r": round(mean(exit_only_r),3) if exit_only_r else None,
        }

    return {
        "stopped": len(stopped),
        "recovered_to_target": len(recovered),
        "recovery_pct": round(100*len(recovered)/len(stopped),2) if stopped else None,
        "unrecovered": len(unrecovered),
        "recovery_hours": qstats(hours),
        "recovered_depth_pct_below_stop": qstats(depth_pct),
        "recovered_depth_atr_below_stop": qstats(depth_atr),
        "recovered_mae_r_from_entry": qstats(mae_r),
        "time_exit_scenarios": timed,
    }



def print_summary(period, variant, s):
    print(f"\n[{period}] STOP DEPTH + TIME EXIT — {variant}")
    print(
        f"Stopped={s['stopped']} | Recovered to target={s['recovered_to_target']} "
        f"({s['recovery_pct']}%) | Unrecovered={s['unrecovered']}"
    )

    if s["recovery_hours"]:
        x = s["recovery_hours"]
        print(
            f"Recovery time: mean={x['mean']:.2f}h | median={x['median']:.2f}h | "
            f"P75={x['p75']:.2f}h | P90={x['p90']:.2f}h"
        )

    if s["recovered_depth_atr_below_stop"]:
        a = s["recovered_depth_atr_below_stop"]
        p = s["recovered_depth_pct_below_stop"]
        r = s["recovered_mae_r_from_entry"]
        print("Recovered trades — MAX adverse excursion below original Stop3 before target:")
        print(
            f"  ATR below Stop3: mean={a['mean']:.2f} | median={a['median']:.2f} | "
            f"P75={a['p75']:.2f} | P90={a['p90']:.2f}"
        )
        print(
            f"  % below Stop3:   mean={p['mean']:.2f}% | median={p['median']:.2f}% | "
            f"P75={p['p75']:.2f}% | P90={p['p90']:.2f}%"
        )
        print(
            f"  MAE in original R from entry: mean={r['mean']:.2f}R | median={r['median']:.2f}R | "
            f"P75={r['p75']:.2f}R | P90={r['p90']:.2f}R"
        )

    print("\nIf Stop3 is ignored and we wait for target OR force-close at deadline:")
    labels = {26:"~1 US session", 52:"~2 US sessions", 78:"~3 US sessions"}
    for deadline in (26,52,78):
        t = s["time_exit_scenarios"][str(deadline)]
        print(
            f"  {labels[deadline]} ({deadline} bars): usable={t['usable_stopped_trades']} | "
            f"target before deadline={t['target_before_deadline']} ({t['target_rate_pct']}%) | "
            f"time exits={t['time_exit_count']}"
        )
        print(
            f"      ALL cohort: AvgR={t['combined_avg_r']} | SumR={t['combined_sum_r']} | "
            f"AvgReturn={t['combined_avg_return_pct']}%"
        )
        print(
            f"      TIME-EXIT losers only: AvgReturn={t['time_exit_only_avg_return_pct']}% | "
            f"MedianReturn={t['time_exit_only_median_return_pct']}% | "
            f"AvgR={t['time_exit_only_avg_r']}"
        )



def simulate_wider_stops_from_rows(rows):
    """
    Approximate wider-stop scenarios using the observed post-stop path data.

    Since these rows only retain post-stop extrema / deadline snapshots rather
    than every intra-bar event after Stop3, this section gives a conservative
    screening estimate:
      - if observed post-stop minimum breached the wider stop, classify STOP
      - otherwise if target was later reached, TARGET
      - otherwise use the selected time-exit snapshot

    Exact same-bar ordering beyond original Stop3 cannot be reconstructed from
    the compact row alone; use this as a screening table, not final execution.
    """
    stopped = [r for r in rows if r["result"] == "STOPPED"]
    out = {}

    for extra_atr in (0.5, 1.0, 1.5, 2.0):
        out[str(extra_atr)] = {}
        for deadline in (26, 52, 78):
            outcomes = []
            for r in stopped:
                if r.get("post_stop_min_low") is None:
                    continue
                atr = float(r["atr"])
                stop3 = float(r["exit_price"])
                fill = float(r["fill_price"])
                target = float(r["target_price"])
                wider_stop = stop3 - extra_atr*atr
                risk = fill - wider_stop
                if risk <= 0:
                    continue

                # If the recorded minimum ever breached the wider stop, count it
                # as stopped for this screening estimate.
                if float(r["post_stop_min_low"]) <= wider_stop:
                    exit_px = wider_stop
                    outcome = "WIDER_STOP"
                else:
                    k = f"timeexit_{deadline}"
                    if r.get(k + "_outcome") == "TARGET_BEFORE_DEADLINE":
                        exit_px = target
                        outcome = "TARGET"
                    elif r.get(k + "_exit_price") is not None:
                        exit_px = float(r[k + "_exit_price"])
                        outcome = "TIME_EXIT"
                    else:
                        continue

                rr = (exit_px - fill)/risk
                ret = 100*(exit_px/fill - 1)
                outcomes.append((outcome, rr, ret))

            if outcomes:
                counts = Counter(x[0] for x in outcomes)
                rvals = [x[1] for x in outcomes]
                rets = [x[2] for x in outcomes]
                out[str(extra_atr)][str(deadline)] = {
                    "n": len(outcomes),
                    "target": counts.get("TARGET",0),
                    "wider_stop": counts.get("WIDER_STOP",0),
                    "time_exit": counts.get("TIME_EXIT",0),
                    "avg_r": round(mean(rvals),3),
                    "sum_r": round(sum(rvals),3),
                    "avg_return_pct": round(mean(rets),3),
                }
            else:
                out[str(extra_atr)][str(deadline)] = {}
    return out


def print_wider_stop_table(title, table):
    print(f"\n=== {title} — WIDER STOP SCREENING ===")
    for extra_atr in ("0.5","1.0","1.5","2.0"):
        for deadline in ("26","52","78"):
            x = table.get(extra_atr,{}).get(deadline,{})
            if not x:
                continue
            print(
                f"Stop3-{extra_atr}ATR | exit {deadline} bars: n={x['n']} | "
                f"Target={x['target']} | WiderStop={x['wider_stop']} | TimeExit={x['time_exit']} | "
                f"AvgR={x['avg_r']} | SumR={x['sum_r']} | AvgReturn={x['avg_return_pct']}%"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-days", type=int, default=30)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--confirm-window", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--output", default="stop_depth_time_exit_rows.csv")
    ap.add_argument("--summary-output", default="stop_depth_time_exit_summary.json")
    args = ap.parse_args()

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
            raise SystemExit("AAPL reference candles missing")

        latest = int(ref[-1][0])
        end_latest = latest + 900
        one = args.period_days * 86400

        periods = [
            ("PREVIOUS_30D", end_latest - 2*one, end_latest - one),
            ("LATEST_30D", end_latest - one, end_latest),
        ]

        ref_times = [int(r[0]) for r in ref]
        ref_pos = {ts:i for i,ts in enumerate(ref_times)}
        ref_dates = sorted({local_parts(ts)[0] for ts in ref_times})

        stocks = list(
            s.execute(
                select(Stock.symbol).where(Stock.market=="US").order_by(Stock.symbol)
            ).scalars().all()
        )

    flags_path = Path(__file__).resolve().parent / "monitor" / "data" / "corporate_flags.json"
    try:
        flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except Exception:
        flags = {}

    print("\nRajih — STOP DEPTH + TIME EXIT analysis")
    print(f"Latest candle: {iso_ny(latest)}")
    print(f"US universe: {len(stocks)}")
    print("System: BULLISH_RECLAIM entry | target=Stop1 | stop=Stop3")
    print(f"BULLISH_RECLAIM confirmation window: {args.confirm_window} bars")

    all_rows = []
    summaries = {}

    for period_name, start_ts, end_ts in periods:
        print(f"\n=== PERIOD {period_name}: {iso_ny(start_ts)} -> {iso_ny(end_ts)} ===")
        period_rows_by_variant = {v: [] for v in VARIANTS}

        for off in range(0, len(stocks), args.chunk_size):
            chunk = stocks[off:off+args.chunk_size]
            with DB() as s:
                raw = list(
                    s.execute(
                        select(Candle.symbol,Candle.ts,Candle.o,Candle.h,Candle.l,Candle.c,Candle.v)
                        .where(Candle.symbol.in_(chunk))
                        .order_by(Candle.symbol,Candle.ts)
                    ).all()
                )

            grouped = defaultdict(list)
            for sym,ts,o,h,l,c,v in raw:
                grouped[sym].append((int(ts),float(o),float(h),float(l),float(c),float(v)))

            for sym in chunk:
                bars = grouped.get(sym, [])
                for variant in VARIANTS:
                    rows = replay_variant(
                        sym, bars, ref_pos, ref_dates,
                        start_ts, end_ts, sym in flags,
                        args.atr_mult, args.waiting_bars,
                        args.confirm_window, variant
                    )
                    for r in rows:
                        r["period"] = period_name
                    period_rows_by_variant[variant].extend(rows)
                    all_rows.extend(rows)

            done = min(off+len(chunk), len(stocks))
            if done % 500 < args.chunk_size or done == len(stocks):
                print(f"Progress {done}/{len(stocks)} | elapsed={time.time()-started:.1f}s")

        summaries[period_name] = {}
        for variant in VARIANTS:
            s = summarize(period_rows_by_variant[variant])
            summaries[period_name][variant] = s
            print_summary(period_name, variant, s)

    print("\n=== COMBINED 60-DAY STOP DEPTH + TIME EXIT ===")
    combined_rows = [
        r for r in all_rows
        if r["variant"] == "STOP1"
    ]
    combined_summary = summarize(combined_rows)
    print_summary("COMBINED_60D", "STOP1", combined_summary)

    wider_stop_screening = simulate_wider_stops_from_rows(combined_rows)
    print_wider_stop_table("COMBINED_60D", wider_stop_screening)

    if all_rows:
        fields = list(all_rows[0].keys())
        with Path(args.output).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    payload = {
        "latest_reference_ny": iso_ny(latest),
        "parameters": vars(args),
        "variants": list(VARIANTS),
        "summaries": summaries,
        "combined_60d": combined_summary,
        "wider_stop_screening_combined_60d": wider_stop_screening,
        "notes": [
            "analysis measures post-stop depth, timed exits, and screens wider-stop alternatives",
            "target remains first_stop and protective stop remains third_stop",
            "two independent 30-day periods are reported separately",
            "read-only",
        ],
    }
    Path(args.summary_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nCSV written: {args.output}")
    print(f"Summary JSON written: {args.summary_output}")
    print(f"Total elapsed: {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
