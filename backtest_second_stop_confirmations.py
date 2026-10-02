#!/usr/bin/env python3
"""
Rajih — SECOND_STOP_RECOVERY confirmation-entry backtest.

Instead of buying immediately on the first touch of Level 2, this script tests
several predefined confirmation rules.

Base levels
-----------
Level 1 (target) = original strategy stop
Level 2          = Level 1 - 1.2 ATR
Level 3 (stop)   = Level 2 - 1.2 ATR

Confirmation variants
---------------------
1) TOUCH_BASELINE
   Old behavior: buy immediately when Level 2 is touched.

2) RECLAIM_CLOSE
   Level 2 is touched, and the SAME 15m candle closes back >= Level 2.
   Entry = that candle close.

3) BULLISH_RECLAIM
   Level 2 is touched, then within 3 bars we wait for a bullish candle
   (close > open) that closes >= Level 2.
   Entry = confirmation candle close.

4) WICK_RECLAIM
   Level 2 is touched, and the touch candle:
   - closes >= Level 2
   - is bullish or neutral
   - has lower wick >= candle body
   Entry = touch candle close.

5) DEEP_SWEEP_RECLAIM
   Price sweeps at least 0.25 ATR below Level 2, but closes back >= Level 2
   on the same candle.
   Entry = touch candle close.

Execution notes
---------------
- If confirmation entry is already >= Level 1 target, the setup is skipped.
- Protective stop remains Level 3.
- Target remains Level 1.
- No target/stop evaluation on the confirmation/fill candle itself, matching
  the live engine's conservative fill-candle behavior.
- Same-bar target+stop on later candles is counted stop-first.
- Waiting for first Level-2 touch is limited to 78 bars.
- Confirmation window for multi-bar rules defaults to 3 bars.
- Current Plan table is NOT used.

The script evaluates TWO independent periods:
A) previous 30 days (older holdout): days -60 to -30
B) latest 30 days: days -30 to 0

Run:
    python backtest_second_stop_confirmations.py

Outputs:
    second_stop_confirmation_results.csv
    second_stop_confirmation_summary.json
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

VARIANTS = (
    "TOUCH_BASELINE",
    "RECLAIM_CLOSE",
    "BULLISH_RECLAIM",
    "WICK_RECLAIM",
    "DEEP_SWEEP_RECLAIM",
)


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
    exit_ts: int | None = None
    exit_price: float | None = None
    result: str = ""
    bars_after_fill: int = 0
    last_ts: int = 0


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
    o, h, l, c = map(float, (o,h,l,c))

    if variant == "TOUCH_BASELINE":
        if l <= p.second_stop:
            fill = min(o, p.second_stop) if o <= p.second_stop else p.second_stop
            return fill
        return None

    # For all confirmation variants, first wait for Level-2 touch.
    if p.state == "WAITING_TOUCH":
        if l > p.second_stop:
            return None

        p.touch_ts = int(ts)
        p.state = "WAITING_CONFIRM"
        p.confirm_wait_bars = 0

        body = abs(c-o)
        lower_wick = max(0.0, min(o,c)-l)

        if variant == "RECLAIM_CLOSE":
            return c if c >= p.second_stop else None

        if variant == "WICK_RECLAIM":
            bullish_or_neutral = c >= o
            if c >= p.second_stop and bullish_or_neutral and lower_wick >= body:
                return c
            return None

        if variant == "DEEP_SWEEP_RECLAIM":
            deep_level = p.second_stop - 0.25*p.atr
            if l <= deep_level and c >= p.second_stop:
                return c
            return None

        if variant == "BULLISH_RECLAIM":
            if c >= p.second_stop and c > o:
                return c
            return None

    elif p.state == "WAITING_CONFIRM":
        p.confirm_wait_bars += 1

        if variant == "BULLISH_RECLAIM":
            if c >= p.second_stop and c > o:
                return c
            if p.confirm_wait_bars >= confirm_window:
                p.result = "CONFIRM_EXPIRED"
            return None

        # Same-candle-only variants expire immediately after the touch candle.
        if variant in ("RECLAIM_CLOSE", "WICK_RECLAIM", "DEEP_SWEEP_RECLAIM"):
            p.result = "CONFIRM_FAILED"
            return None

    return None


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
        "exit_price": p.exit_price,
        "result": p.result,
        "bars_after_fill": p.bars_after_fill,
        "pnl_r": pnl_r,
        "return_pct": ret,
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

    for i in range(start_i, len(bars)):
        ts, o, h, l, c, v = bars[i]
        ts = int(ts)
        if ts >= end_ts:
            break
        pos = ref_pos.get(ts)
        if pos is None:
            continue

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
                        if fill <= open_plan.third_stop or fill >= open_plan.first_stop:
                            open_plan.result = "CANCELLED_INVALID_CONFIRM"
                            open_plan.exit_ts = ts
                            rows.append(finalize(variant, "", open_plan))
                            open_plan = None
                        else:
                            open_plan.fill_price = fill
                            open_plan.fill_ts = ts
                            open_plan.state = "ACTIVE"

                    if open_plan is not None and open_plan.fill_price is None and open_plan.waiting_bars >= waiting_limit:
                        open_plan.result = "EXPIRED"
                        open_plan.exit_ts = ts
                        rows.append(finalize(variant, "", open_plan))
                        open_plan = None

            else:
                open_plan.bars_after_fill += 1

                # Conservative stop-first on ambiguous later candles.
                if float(l) <= open_plan.third_stop:
                    open_plan.result = "STOPPED"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = min(float(o), open_plan.third_stop)
                    rows.append(finalize(variant, "", open_plan))
                    open_plan = None

                elif float(h) >= open_plan.first_stop:
                    open_plan.result = "TARGET"
                    open_plan.exit_ts = ts
                    open_plan.exit_price = open_plan.first_stop
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
    filled = [r for r in rows if r["fill_price"] is not None]
    resolved = [r for r in filled if r["result"] in ("TARGET","STOPPED")]
    targets = [r for r in resolved if r["result"] == "TARGET"]
    stops = [r for r in resolved if r["result"] == "STOPPED"]
    rvals = [r["pnl_r"] for r in resolved if r["pnl_r"] is not None]
    returns = [r["return_pct"] for r in resolved if r["return_pct"] is not None]

    return {
        "signals": len(rows),
        "filled": len(filled),
        "fill_rate_pct": round(100*len(filled)/len(rows),2) if rows else 0,
        "resolved": len(resolved),
        "target": len(targets),
        "stopped": len(stops),
        "win_rate_pct": round(100*len(targets)/len(resolved),2) if resolved else None,
        "sum_r": round(sum(rvals),3) if rvals else 0,
        "avg_r": round(mean(rvals),3) if rvals else None,
        "avg_return_pct": round(mean(returns),3) if returns else None,
        "bars_to_resolution": qstats([r["bars_after_fill"] for r in resolved]),
        "states": dict(Counter(r["result"] for r in rows)),
    }


def print_summary(period, variant, s):
    print(f"\n[{period}] {variant}")
    print(
        f"Signals={s['signals']} | Filled={s['filled']} ({s['fill_rate_pct']}%) | "
        f"Resolved={s['resolved']}"
    )
    print(
        f"Target={s['target']} | Stop={s['stopped']} | Win={s['win_rate_pct']}% | "
        f"AvgR={s['avg_r']} | SumR={s['sum_r']} | AvgReturn={s['avg_return_pct']}%"
    )
    if s["bars_to_resolution"]:
        x = s["bars_to_resolution"]
        print(
            f"After fill: mean={x['mean']:.2f} bars | median={x['median']:.1f} | "
            f"P75={x['p75']:.1f} | P90={x['p90']:.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-days", type=int, default=30)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--confirm-window", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--output", default="second_stop_confirmation_results.csv")
    ap.add_argument("--summary-output", default="second_stop_confirmation_summary.json")
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

    print("\nRajih — SECOND_STOP_RECOVERY confirmation comparison")
    print(f"Latest candle: {iso_ny(latest)}")
    print(f"US universe: {len(stocks)}")
    print(f"Variants: {', '.join(VARIANTS)}")
    print(f"Confirmation window for BULLISH_RECLAIM: {args.confirm_window} bars")

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

    print("\n=== CONSISTENCY CHECK ===")
    for variant in VARIANTS:
        a = summaries["PREVIOUS_30D"][variant]
        b = summaries["LATEST_30D"][variant]
        both_pos = (
            a["avg_r"] is not None and b["avg_r"] is not None
            and a["avg_r"] > 0 and b["avg_r"] > 0
        )
        both_55 = (
            a["win_rate_pct"] is not None and b["win_rate_pct"] is not None
            and a["win_rate_pct"] >= 55 and b["win_rate_pct"] >= 55
        )
        print(
            f"{variant:20} | prev AvgR={a['avg_r']} Win={a['win_rate_pct']}% | "
            f"latest AvgR={b['avg_r']} Win={b['win_rate_pct']}% | "
            f"AvgR+ both={'YES' if both_pos else 'NO'} | Win>=55 both={'YES' if both_55 else 'NO'}"
        )

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
        "notes": [
            "confirmation rules are predefined; no threshold search is performed",
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
