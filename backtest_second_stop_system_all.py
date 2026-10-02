#!/usr/bin/env python3
"""
Rajih — backtest of SECOND_STOP_RECOVERY on ALL stored US recommendations.

System tested
-------------
For every stored US recommendation in the selected window:

    first_stop  = original strategy stop
    second_stop = first_stop - 1.2 * signal ATR      -> BUY LIMIT
    third_stop  = second_stop - 1.2 * signal ATR     -> protective stop
    target      = first_stop                         -> recovery target

Execution assumptions match the live paper logic:
- Wait for price to reach second_stop.
- If a candle opens at/below third_stop before a safe fill, cancel the setup.
- If price reaches second_stop safely, fill at second_stop.
- If candle opens below second_stop but above third_stop, fill at the better open.
- After fill, if target and third_stop are both possible in the same OHLC bar,
  assume third_stop first (conservative).
- No commissions or extra slippage.
- READ ONLY: does not modify the database.

Run:
    python backtest_second_stop_system_all.py --days 30

Optional:
    python backtest_second_stop_system_all.py --days 60
    python backtest_second_stop_system_all.py --days 30 --atr-mult 1.2

Outputs:
    second_stop_backtest_all.csv
    second_stop_backtest_all_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

from sqlalchemy import select, func

from core import database
from monitor.models import Plan, Candle
from monitor.strategy import rounded

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def iso_ny(ts):
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), NY).isoformat(timespec="minutes")


def q(values, p):
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals)-1)*p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo]*(hi-k) + vals[hi]*(k-lo)


def qstats(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p25": q(vals,.25),
        "p75": q(vals,.75),
        "p90": q(vals,.90),
        "min": min(vals),
        "max": max(vals),
    }


@dataclass
class Result:
    plan_id: int
    symbol: str
    signal_ny: str
    original_state: str
    score: float
    atr: float

    original_entry: float
    first_stop: float
    original_target: float

    second_stop: float
    third_stop: float
    recovery_target: float

    second_stop_touched: bool
    fill_ts_ny: str
    fill_price: float | None

    result: str
    exit_ts_ny: str
    exit_price: float | None

    return_pct: float | None
    pnl_r: float | None
    bars_to_fill: int | None
    bars_after_fill: int | None
    same_bar_ambiguous: bool


def original_levels(plan):
    """
    Works with both:
    - historical/legacy plans where plan.stop is the original first stop
    - new SECOND_STOP_RECOVERY plans where original levels are stored in context
    """
    ctx = {}
    try:
        ctx = json.loads(plan.context_json or "{}")
    except Exception:
        pass

    if ctx.get("entry_system") == "SECOND_STOP_RECOVERY":
        original_entry = float(ctx.get("original_entry_reference", plan.entry))
        first_stop = float(ctx.get("original_first_stop", plan.target))
        original_target = float(ctx.get("original_selected_target", plan.target))
    else:
        original_entry = float(plan.entry)
        first_stop = float(plan.stop)
        original_target = float(plan.target)

    return original_entry, first_stop, original_target


def simulate(plan, bars, atr_mult):
    atr = float(plan.atr or 0)
    original_entry, first_stop, original_target = original_levels(plan)

    if atr <= 0 or first_stop <= 0:
        return None

    second_stop, _ = rounded(first_stop - atr_mult * atr, "US", False)
    third_stop, _ = rounded(second_stop - atr_mult * atr, "US", False)
    recovery_target = first_stop

    if not (0 < third_stop < second_stop < recovery_target):
        return None

    fill_i = None
    fill = None
    fill_ts = None
    cancelled_gap = False
    same_bar = False

    for i, row in enumerate(bars):
        ts, o, h, l, c, v = row
        o, h, l = float(o), float(h), float(l)

        # Match live logic: unsafe gap through protective stop before a safe fill.
        if o <= third_stop:
            cancelled_gap = True
            return Result(
                plan_id=int(plan.id),
                symbol=plan.symbol,
                signal_ny=iso_ny(plan.signal_ts),
                original_state=plan.state,
                score=float(plan.score or 0),
                atr=atr,
                original_entry=original_entry,
                first_stop=first_stop,
                original_target=original_target,
                second_stop=second_stop,
                third_stop=third_stop,
                recovery_target=recovery_target,
                second_stop_touched=True,
                fill_ts_ny="",
                fill_price=None,
                result="CANCELLED_GAP_BELOW_THIRD",
                exit_ts_ny=iso_ny(ts),
                exit_price=None,
                return_pct=None,
                pnl_r=None,
                bars_to_fill=i+1,
                bars_after_fill=None,
                same_bar_ambiguous=False,
            )

        if l <= second_stop:
            fill_i = i
            fill_ts = int(ts)
            fill = min(o, second_stop) if o <= second_stop else second_stop
            break

    if fill_i is None:
        return Result(
            plan_id=int(plan.id),
            symbol=plan.symbol,
            signal_ny=iso_ny(plan.signal_ts),
            original_state=plan.state,
            score=float(plan.score or 0),
            atr=atr,
            original_entry=original_entry,
            first_stop=first_stop,
            original_target=original_target,
            second_stop=second_stop,
            third_stop=third_stop,
            recovery_target=recovery_target,
            second_stop_touched=False,
            fill_ts_ny="",
            fill_price=None,
            result="UNFILLED",
            exit_ts_ny="",
            exit_price=None,
            return_pct=None,
            pnl_r=None,
            bars_to_fill=None,
            bars_after_fill=None,
            same_bar_ambiguous=False,
        )

    state = "OPEN_END"
    exit_ts = None
    exit_price = None
    bars_after_fill = 0

    risk = fill - third_stop
    if risk <= 0:
        return None

    for j in range(fill_i, len(bars)):
        ts, o, h, l, c, v = bars[j]
        o, h, l = float(o), float(h), float(l)
        bars_after_fill = j - fill_i + 1

        hit_target = h >= recovery_target
        hit_stop = l <= third_stop

        if hit_target and hit_stop:
            same_bar = True
            state = "STOPPED"
            exit_price = third_stop if j == fill_i else min(o, third_stop)
            exit_ts = int(ts)
            break

        if hit_stop:
            state = "STOPPED"
            exit_price = third_stop if j == fill_i else min(o, third_stop)
            exit_ts = int(ts)
            break

        if hit_target:
            state = "TARGET"
            exit_price = recovery_target
            exit_ts = int(ts)
            break

    if state == "OPEN_END":
        if bars:
            exit_price = float(bars[-1][4])
            exit_ts = int(bars[-1][0])

    ret = 100*(exit_price/fill - 1) if exit_price is not None else None
    pnl_r = (exit_price-fill)/risk if exit_price is not None else None

    return Result(
        plan_id=int(plan.id),
        symbol=plan.symbol,
        signal_ny=iso_ny(plan.signal_ts),
        original_state=plan.state,
        score=float(plan.score or 0),
        atr=atr,
        original_entry=original_entry,
        first_stop=first_stop,
        original_target=original_target,
        second_stop=second_stop,
        third_stop=third_stop,
        recovery_target=recovery_target,
        second_stop_touched=True,
        fill_ts_ny=iso_ny(fill_ts),
        fill_price=fill,
        result=state,
        exit_ts_ny=iso_ny(exit_ts),
        exit_price=exit_price,
        return_pct=ret,
        pnl_r=pnl_r,
        bars_to_fill=fill_i+1,
        bars_after_fill=bars_after_fill,
        same_bar_ambiguous=same_bar,
    )


def summarize(rows):
    states = Counter(r.result for r in rows)
    filled = [r for r in rows if r.fill_price is not None]
    resolved = [r for r in filled if r.result in ("TARGET","STOPPED")]
    targets = [r for r in resolved if r.result == "TARGET"]
    stops = [r for r in resolved if r.result == "STOPPED"]
    opens = [r for r in filled if r.result == "OPEN_END"]
    rvals = [r.pnl_r for r in resolved if r.pnl_r is not None]
    rets = [r.return_pct for r in resolved if r.return_pct is not None]

    return {
        "plans": len(rows),
        "unfilled": states.get("UNFILLED",0),
        "cancelled_gap": states.get("CANCELLED_GAP_BELOW_THIRD",0),
        "filled": len(filled),
        "fill_rate_pct": round(100*len(filled)/len(rows),2) if rows else 0,
        "target": len(targets),
        "stopped": len(stops),
        "open_end": len(opens),
        "resolved": len(resolved),
        "resolved_win_rate_pct": round(100*len(targets)/len(resolved),2) if resolved else None,
        "sum_R_resolved": round(sum(rvals),3) if rvals else 0,
        "avg_R_resolved": round(mean(rvals),3) if rvals else None,
        "avg_return_pct_resolved": round(mean(rets),3) if rets else None,
        "bars_to_fill": qstats([r.bars_to_fill for r in filled if r.bars_to_fill is not None]),
        "bars_after_fill": qstats([r.bars_after_fill for r in resolved if r.bars_after_fill is not None]),
        "same_bar_ambiguous": sum(r.same_bar_ambiguous for r in rows),
        "states": dict(states),
    }


def print_summary(title, s):
    print(f"\n=== {title} ===")
    print(
        f"Plans: {s['plans']} | Filled: {s['filled']} ({s['fill_rate_pct']}%) | "
        f"Unfilled: {s['unfilled']} | Gap-cancelled: {s['cancelled_gap']}"
    )
    print(
        f"Target: {s['target']} | Stop: {s['stopped']} | Open/end: {s['open_end']} | "
        f"Resolved: {s['resolved']}"
    )
    print(
        f"Resolved win rate: {s['resolved_win_rate_pct']}% | "
        f"Sum R: {s['sum_R_resolved']} | Avg R: {s['avg_R_resolved']} | "
        f"Avg return: {s['avg_return_pct_resolved']}%"
    )
    if s["bars_to_fill"]:
        print(
            f"Bars to fill: median={s['bars_to_fill']['median']:.1f} | "
            f"P75={s['bars_to_fill']['p75']:.1f} | P90={s['bars_to_fill']['p90']:.1f}"
        )
    if s["bars_after_fill"]:
        print(
            f"Bars after fill to resolution: median={s['bars_after_fill']['median']:.1f} | "
            f"P75={s['bars_after_fill']['p75']:.1f} | P90={s['bars_after_fill']['p90']:.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--output", default="second_stop_backtest_all.csv")
    ap.add_argument("--summary-output", default="second_stop_backtest_all_summary.json")
    args = ap.parse_args()

    if args.days <= 0 or args.atr_mult <= 0:
        raise SystemExit("--days and --atr-mult must be > 0")

    DB = database()
    with DB() as s:
        latest = s.scalar(select(func.max(Candle.ts)))
        if not latest:
            latest = int(datetime.now(UTC).timestamp())
        end_ts = int(latest) + 900
        start_ts = end_ts - int(args.days*86400)

        plans = s.scalars(
            select(Plan)
            .where(
                Plan.market == "US",
                Plan.signal_ts >= start_ts,
                Plan.signal_ts < end_ts,
            )
            .order_by(Plan.signal_ts, Plan.symbol)
        ).all()

        rows = []
        skipped = 0

        for p in plans:
            bars = list(
                s.execute(
                    select(Candle.ts, Candle.o, Candle.h, Candle.l, Candle.c, Candle.v)
                    .where(
                        Candle.symbol == p.symbol,
                        Candle.ts > int(p.signal_ts),
                        Candle.ts < end_ts,
                    )
                    .order_by(Candle.ts)
                ).all()
            )
            r = simulate(p, bars, args.atr_mult)
            if r is None:
                skipped += 1
            else:
                rows.append(r)

    if not rows:
        print("No valid US recommendations found.")
        return 2

    overall = summarize(rows)

    by_original_state = defaultdict(list)
    for r in rows:
        by_original_state[r.original_state].append(r)

    print("\nRajih — SECOND_STOP_RECOVERY backtest on ALL US recommendations")
    print(f"Window: last {args.days:g} days ending {iso_ny(latest)}")
    print(f"Selected plans: {len(plans)} | valid tested: {len(rows)} | skipped invalid: {skipped}")
    print(
        f"System: buy at first_stop - {args.atr_mult:g} ATR | "
        f"target=first_stop | protective stop=another {args.atr_mult:g} ATR lower"
    )
    print("OHLC ambiguity rule: STOP first if target and protective stop are both possible in the same candle.")

    print_summary("ALL RECOMMENDATIONS", overall)

    print("\n=== BY ORIGINAL PLAN STATE ===")
    print("state      | plans | filled | target | stop | open | win% | sumR")
    for state in sorted(by_original_state):
        s = summarize(by_original_state[state])
        wr = "n/a" if s["resolved_win_rate_pct"] is None else f"{s['resolved_win_rate_pct']:.2f}"
        print(
            f"{state:10} | {s['plans']:5d} | {s['filled']:6d} | {s['target']:6d} | "
            f"{s['stopped']:4d} | {s['open_end']:4d} | {wr:>5} | {s['sum_R_resolved']:7.3f}"
        )

    out = Path(args.output)
    fields = list(asdict(rows[0]).keys())
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    payload = {
        "window_days": args.days,
        "latest_candle_ny": iso_ny(latest),
        "selected_plans": len(plans),
        "valid_tested": len(rows),
        "skipped_invalid": skipped,
        "atr_mult": args.atr_mult,
        "overall": overall,
        "by_original_state": {k: summarize(v) for k,v in by_original_state.items()},
        "assumptions": [
            "all stored US recommendations in the window are tested",
            "second stop = original first stop - ATR multiplier * original signal ATR",
            "third protective stop = second stop - same ATR multiplier * original signal ATR",
            "target = original first stop",
            "unsafe gap at/below third stop before safe fill cancels the setup",
            "resting buy-limit receives opening-price improvement when opening below second stop but above third stop",
            "stop-first conservative rule on same-bar OHLC ambiguity",
            "stored 15-minute candles only",
            "read-only; no database changes",
        ],
    }

    Path(args.summary_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nCSV written: {out}")
    print(f"Summary JSON written: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
