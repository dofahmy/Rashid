"""
Rajih — post-stop recovery analysis for BULLISH_RECLAIM.

Question answered
-----------------
For trades that:
1) entered after BULLISH_RECLAIM,
2) used target = Stop1,
3) then hit the current protective Stop3,

how often did price LATER recover all the way to the original target (Stop1),
and how long did that recovery take AFTER the stop was hit?

The script reports:
- stopped trades
- recovered-to-target count and %
- median / mean / P75 / P90 recovery time
- recovery within 1, 2, 4, 8, 16, 26, 52, 78 bars
- recovery within 15m, 30m, 1h, 2h, 4h, 1 session-ish, 2 sessions-ish, 3 sessions-ish
- non-recovered count
- maximum favorable rebound after stop for trades that never reached target

Two independent 30-day periods are analyzed:
A) previous 30 days
B) latest 30 days

IMPORTANT
---------
The trade is still counted as STOPPED at Stop3.
This analysis only asks what happened to price AFTER that stop.

Run:
    python analyze_after_stop_recovery.py

Outputs:
    after_stop_recovery_rows.csv
    after_stop_recovery_summary.json
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
        "post_stop_max_rebound_pct_from_stop": None,
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
        if tr.post_stop_max_high is not None and tr.exit_price:
            r["post_stop_max_rebound_pct_from_stop"] = (
                100 * (tr.post_stop_max_high / tr.exit_price - 1)
            )
        else:
            r["post_stop_max_rebound_pct_from_stop"] = None

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

    windows = {}
    for n in (0, 1, 2, 4, 8, 16, 26, 52, 78):
        cnt = sum(b <= n for b in bars)
        windows[str(n)] = {
            "count": cnt,
            "pct_of_stopped": round(100*cnt/len(stopped), 2) if stopped else 0,
            "pct_of_recovered": round(100*cnt/len(recovered), 2) if recovered else 0,
        }

    rebounds = [
        float(r["post_stop_max_rebound_pct_from_stop"])
        for r in unrecovered
        if r.get("post_stop_max_rebound_pct_from_stop") is not None
    ]

    return {
        "stopped": len(stopped),
        "recovered_to_target_after_stop": len(recovered),
        "recovery_pct_of_stopped": round(100*len(recovered)/len(stopped), 2) if stopped else None,
        "not_recovered": len(unrecovered),
        "bars_to_target": qstats(bars),
        "hours_to_target": qstats(hours),
        "recovery_windows": windows,
        "unrecovered_max_rebound_pct_from_stop": qstats(rebounds),
    }



def print_summary(period, variant, s):
    print(f"\n[{period}] POST-STOP RECOVERY — {variant}")
    print(
        f"Stopped={s['stopped']} | Later reached target={s['recovered_to_target_after_stop']} "
        f"({s['recovery_pct_of_stopped']}%) | Never reached target={s['not_recovered']}"
    )

    if s["bars_to_target"]:
        b = s["bars_to_target"]
        h = s["hours_to_target"]
        print(
            f"Recovery time after STOP: mean={h['mean']:.2f}h | median={h['median']:.2f}h | "
            f"P75={h['p75']:.2f}h | P90={h['p90']:.2f}h"
        )
        print(
            f"Bars: mean={b['mean']:.2f} | median={b['median']:.1f} | "
            f"P75={b['p75']:.1f} | P90={b['p90']:.1f}"
        )

    print("Reached target after stop within:")
    labels = {
        0: "same stop candle",
        1: "15 min",
        2: "30 min",
        4: "1 hour",
        8: "2 hours",
        16: "4 hours",
        26: "~1 US session",
        52: "~2 US sessions",
        78: "~3 US sessions",
    }
    for n in (0,1,2,4,8,16,26,52,78):
        w = s["recovery_windows"][str(n)]
        print(
            f"  {labels[n]:16} : {w['count']} "
            f"({w['pct_of_stopped']}% of all stopped)"
        )

    if s["unrecovered_max_rebound_pct_from_stop"]:
        x = s["unrecovered_max_rebound_pct_from_stop"]
        print(
            f"Unrecovered max rebound from stop: median={x['median']:.2f}% | "
            f"P75={x['p75']:.2f}% | P90={x['p90']:.2f}%"
        )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-days", type=int, default=30)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--confirm-window", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--output", default="after_stop_recovery_rows.csv")
    ap.add_argument("--summary-output", default="after_stop_recovery_summary.json")
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

    print("\nRajih — POST-STOP RECOVERY analysis")
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

    print("\n=== COMBINED 60-DAY POST-STOP RECOVERY ===")
    combined_rows = [
        r for r in all_rows
        if r["variant"] == "STOP1"
    ]
    combined_summary = summarize(combined_rows)
    print_summary("COMBINED_60D", "STOP1", combined_summary)

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
        "notes": [
            "analysis tracks whether STOPPED bullish-reclaim trades later reached the same Stop1 target",
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
