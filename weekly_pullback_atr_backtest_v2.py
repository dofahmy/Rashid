#!/usr/bin/env python3
"""
Rajih 7-day comparison backtest for three pullback-entry variants.

All three variants start from the same idea:
    old stop-loss = pullback reference level

Then compare:

A) DIRECT_LIMIT
   Resting BUY LIMIT exactly at the old stop.

B) BOUNCE_CONFIRM
   Price must first touch/trade through the old stop.
   Then wait for a completed 15m bullish rebound candle:
       close > open AND close > old stop
   Entry is at the NEXT 15m candle open.
   This deliberately avoids using the confirmation candle close as an
   executable fill and avoids intrabar look-ahead.

C) RECLAIM_02ATR
   Price must first touch/trade through the old stop.
   Starting from the NEXT 15m candle, place a BUY STOP at:
       old stop + 0.20 * signal ATR
   If a candle opens above the trigger, fill at the open (worse fill);
   otherwise fill at the trigger when high reaches it.

After any entry:
    new stop   = actual fill - STOP_ATR * signal ATR
    new target = actual fill + TARGET_ATR * signal ATR

Defaults:
    STOP_ATR = 1.2
    TARGET_ATR = 2.4
    RECLAIM_ATR = 0.20

The script is READ-ONLY. It does not modify recommendations, customers,
candles, outbox, or settings.

Run:
    python weekly_pullback_atr_backtest_v2.py --days 7

Optional:
    python weekly_pullback_atr_backtest_v2.py --start 2026-09-25 --end 2026-10-02
    python weekly_pullback_atr_backtest_v2.py --days 7 --only-old-stopped

Outputs:
    weekly_pullback_atr_backtest_v2.csv
    weekly_pullback_atr_backtest_v2_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select, func

from core import database
from monitor.models import Plan, Candle
from monitor.strategy import rounded

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

VARIANTS = ("DIRECT_LIMIT", "BOUNCE_CONFIRM", "RECLAIM_02ATR")


@dataclass
class Result:
    variant: str
    plan_id: int
    symbol: str
    signal_ny: str
    score: float
    original_state: str
    original_entry: float
    original_stop: float
    original_target: float
    signal_atr: float
    pullback_reference: float
    touch_ts_ny: str
    entry_trigger: float | None
    filled: bool
    fill_ts_ny: str
    fill_price: float | None
    new_stop: float | None
    new_target: float | None
    new_state: str
    exit_ts_ny: str
    exit_price: float | None
    pnl_pct: float | None
    pnl_r: float | None
    bars_after_signal: int
    bars_after_touch: int
    bars_after_fill: int
    same_bar_ambiguous: bool
    note: str


def iso_ny(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), NY).isoformat(timespec="minutes")


def parse_date_ny(text: str, *, end: bool = False) -> int:
    d = datetime.strptime(text, "%Y-%m-%d").date()
    dt = datetime.combine(
        d + (timedelta(days=1) if end else timedelta(0)),
        dtime.min,
        tzinfo=NY,
    )
    return int(dt.timestamp())


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def safe_round(v: float | None, n: int = 6):
    return None if v is None else round(float(v), n)


def choose_window(session, args) -> tuple[int, int, str]:
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("Use --start and --end together (YYYY-MM-DD, New York dates).")
        start = parse_date_ny(args.start)
        end = parse_date_ny(args.end, end=True)
        return start, end, f"{args.start} .. {args.end} (New York)"

    latest = session.scalar(select(func.max(Candle.ts)))
    if not latest:
        latest = int(datetime.now(UTC).timestamp())
    end = int(latest) + 900
    start = end - int(args.days * 86400)
    return start, end, f"last {args.days:g} days ending {iso_ny(latest)}"


def make_base(plan: Plan, variant: str, reference: float, atr: float) -> dict:
    return dict(
        variant=variant,
        plan_id=int(plan.id),
        symbol=plan.symbol,
        signal_ny=iso_ny(plan.signal_ts),
        score=float(plan.score or 0),
        original_state=plan.state,
        original_entry=float(plan.entry),
        original_stop=float(plan.stop),
        original_target=float(plan.target),
        signal_atr=atr,
        pullback_reference=reference,
    )


def invalid_result(base: dict, state: str, note: str, bars_count: int) -> Result:
    return Result(
        **base,
        touch_ts_ny="",
        entry_trigger=None,
        filled=False,
        fill_ts_ny="",
        fill_price=None,
        new_stop=None,
        new_target=None,
        new_state=state,
        exit_ts_ny="",
        exit_price=None,
        pnl_pct=None,
        pnl_r=None,
        bars_after_signal=bars_count,
        bars_after_touch=0,
        bars_after_fill=0,
        same_bar_ambiguous=False,
        note=note,
    )


def find_touch(bars: list[tuple], reference: float):
    for i, row in enumerate(bars):
        ts, o, h, l, c, v = row
        if float(l) <= reference:
            return i
    return None


def determine_entry(
    variant: str,
    bars: list[tuple],
    reference: float,
    atr: float,
    reclaim_atr: float,
):
    """
    Returns:
      touch_i, fill_i, fill_ts, fill_price, entry_trigger, note
    or fill_i=None if no entry.
    """
    touch_i = find_touch(bars, reference)
    if touch_i is None:
        return None, None, None, None, None, "Old stop/pullback reference was never touched"

    touch_ts = int(bars[touch_i][0])

    if variant == "DIRECT_LIMIT":
        ts, o, h, l, c, v = bars[touch_i]
        o = float(o)
        # Resting buy limit may receive price improvement on a gap below.
        fill = min(o, reference) if o <= reference else reference
        return touch_i, touch_i, int(ts), fill, reference, "Direct resting buy-limit at old stop"

    if variant == "BOUNCE_CONFIRM":
        # A confirmation is only known at candle close. Enter on NEXT candle open.
        confirm_i = None
        for j in range(touch_i, len(bars)):
            ts, o, h, l, c, v = bars[j]
            o, c = float(o), float(c)
            if c > o and c > reference:
                confirm_i = j
                break
        if confirm_i is None:
            return touch_i, None, None, None, reference, "Touched old stop but no completed bullish rebound candle"
        entry_i = confirm_i + 1
        if entry_i >= len(bars):
            return touch_i, None, None, None, reference, "Bullish rebound confirmed but no next stored candle for entry"
        ts, o, h, l, c, v = bars[entry_i]
        return touch_i, entry_i, int(ts), float(o), reference, "Bullish 15m rebound confirmed; entered next candle open"

    if variant == "RECLAIM_02ATR":
        trigger, _ = rounded(reference + reclaim_atr * atr, "US", True)
        # Require reclaim AFTER the touch candle to avoid assuming intrabar order.
        for j in range(touch_i + 1, len(bars)):
            ts, o, h, l, c, v = bars[j]
            o, h = float(o), float(h)
            if o >= trigger:
                return touch_i, j, int(ts), o, trigger, f"Reclaim buy-stop triggered; gap/open above {reclaim_atr:g} ATR reclaim"
            if h >= trigger:
                return touch_i, j, int(ts), trigger, trigger, f"Reclaim buy-stop triggered at old stop + {reclaim_atr:g} ATR"
        return touch_i, None, None, None, trigger, f"Touched old stop but never reclaimed +{reclaim_atr:g} ATR"

    raise ValueError(f"Unknown variant: {variant}")


def simulate(
    plan: Plan,
    bars: list[tuple],
    variant: str,
    stop_atr: float,
    target_atr: float,
    reclaim_atr: float,
) -> Result:
    reference, _ = rounded(float(plan.stop), "US", False)
    atr = float(plan.atr or 0.0)
    base = make_base(plan, variant, reference, atr)

    if atr <= 0 or not math.isfinite(atr):
        return invalid_result(base, "INVALID_ATR", "Original plan ATR is missing/invalid", len(bars))

    touch_i, fill_i, fill_ts, fill, trigger, entry_note = determine_entry(
        variant, bars, reference, atr, reclaim_atr
    )

    if touch_i is None:
        last = float(bars[-1][4]) if bars else None
        note = entry_note
        if last is not None:
            note += f"; last close vs reference={pct(last, reference):+.2f}%"
        return Result(
            **base,
            touch_ts_ny="",
            entry_trigger=trigger,
            filled=False,
            fill_ts_ny="",
            fill_price=None,
            new_stop=None,
            new_target=None,
            new_state="UNTOUCHED",
            exit_ts_ny="",
            exit_price=None,
            pnl_pct=None,
            pnl_r=None,
            bars_after_signal=len(bars),
            bars_after_touch=0,
            bars_after_fill=0,
            same_bar_ambiguous=False,
            note=note,
        )

    touch_ts = int(bars[touch_i][0])

    if fill_i is None:
        return Result(
            **base,
            touch_ts_ny=iso_ny(touch_ts),
            entry_trigger=trigger,
            filled=False,
            fill_ts_ny="",
            fill_price=None,
            new_stop=None,
            new_target=None,
            new_state="TOUCHED_NOT_ENTERED",
            exit_ts_ny="",
            exit_price=None,
            pnl_pct=None,
            pnl_r=None,
            bars_after_signal=len(bars),
            bars_after_touch=len(bars) - touch_i,
            bars_after_fill=0,
            same_bar_ambiguous=False,
            note=entry_note,
        )

    fill = float(fill)
    new_stop, _ = rounded(fill - stop_atr * atr, "US", False)
    new_target, _ = rounded(fill + target_atr * atr, "US", True)

    if new_stop <= 0 or new_stop >= fill or new_target <= fill:
        return Result(
            **base,
            touch_ts_ny=iso_ny(touch_ts),
            entry_trigger=trigger,
            filled=True,
            fill_ts_ny=iso_ny(fill_ts),
            fill_price=fill,
            new_stop=new_stop,
            new_target=new_target,
            new_state="INVALID_LEVELS",
            exit_ts_ny="",
            exit_price=None,
            pnl_pct=None,
            pnl_r=None,
            bars_after_signal=len(bars),
            bars_after_touch=len(bars) - touch_i,
            bars_after_fill=len(bars) - fill_i,
            same_bar_ambiguous=False,
            note="Calculated ATR bracket is invalid",
        )

    risk = fill - new_stop
    state = "OPEN"
    exit_ts = None
    exit_price = None
    same_bar_ambiguous = False
    note = entry_note

    # Entry semantics:
    # DIRECT_LIMIT and RECLAIM can fill intrabar, so same-bar OHLC path is ambiguous.
    # BOUNCE_CONFIRM fills at next candle OPEN, so stop/target checks on that candle are valid
    # but if both occur we still conservatively assume stop first.
    for j in range(fill_i, len(bars)):
        ts, o, h, l, c, v = bars[j]
        o, h, l, c = map(float, (o, h, l, c))
        stop_hit = l <= new_stop
        target_hit = h >= new_target
        if j == fill_i and stop_hit and target_hit:
            same_bar_ambiguous = True
        if stop_hit:
            state = "STOPPED"
            # If later candle gaps below stop use the open; on fill candle use stop.
            exit_price = min(o, new_stop) if j > fill_i else new_stop
            exit_ts = int(ts)
            note += "; stop hit"
            if target_hit:
                note += " (stop-first conservative OHLC ambiguity)"
            break
        if target_hit:
            state = "TARGET"
            exit_price = new_target
            exit_ts = int(ts)
            note += "; target hit"
            break

    if state == "OPEN":
        if bars:
            exit_price = float(bars[-1][4])
            exit_ts = int(bars[-1][0])
            note += "; still open, marked to latest stored close"
        else:
            note += "; filled but no later stored candles"

    pnl_pct = pct(exit_price, fill) if exit_price is not None else None
    pnl_r = (exit_price - fill) / risk if exit_price is not None and risk > 0 else None

    return Result(
        **base,
        touch_ts_ny=iso_ny(touch_ts),
        entry_trigger=trigger,
        filled=True,
        fill_ts_ny=iso_ny(fill_ts),
        fill_price=fill,
        new_stop=new_stop,
        new_target=new_target,
        new_state=state,
        exit_ts_ny=iso_ny(exit_ts),
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        bars_after_signal=len(bars),
        bars_after_touch=len(bars) - touch_i,
        bars_after_fill=len(bars) - fill_i,
        same_bar_ambiguous=same_bar_ambiguous,
        note=note,
    )


def summarize(rows: list[Result]) -> dict:
    states = Counter(r.new_state for r in rows)
    orig = Counter(r.original_state for r in rows)
    touched = [r for r in rows if r.touch_ts_ny]
    filled = [r for r in rows if r.filled and r.new_state not in ("INVALID_ATR", "INVALID_LEVELS")]
    realized = [r for r in filled if r.new_state in ("TARGET", "STOPPED")]
    wins = [r for r in realized if r.new_state == "TARGET"]
    losses = [r for r in realized if r.new_state == "STOPPED"]
    rvals = [r.pnl_r for r in realized if r.pnl_r is not None]
    returns = [r.pnl_pct for r in realized if r.pnl_pct is not None]
    return {
        "plans": len(rows),
        "touched": len(touched),
        "touch_rate_pct": round(100 * len(touched) / len(rows), 2) if rows else 0,
        "filled": len(filled),
        "fill_rate_all_pct": round(100 * len(filled) / len(rows), 2) if rows else 0,
        "fill_rate_after_touch_pct": round(100 * len(filled) / len(touched), 2) if touched else 0,
        "realized": len(realized),
        "wins": len(wins),
        "losses": len(losses),
        "open": states.get("OPEN", 0),
        "untouched": states.get("UNTOUCHED", 0),
        "touched_not_entered": states.get("TOUCHED_NOT_ENTERED", 0),
        "win_rate_realized_pct": round(100 * len(wins) / len(realized), 2) if realized else None,
        "sum_realized_R": round(sum(rvals), 3) if rvals else 0,
        "avg_realized_R": round(sum(rvals) / len(rvals), 3) if rvals else None,
        "avg_realized_return_pct": round(sum(returns) / len(returns), 3) if returns else None,
        "same_bar_ambiguous": sum(r.same_bar_ambiguous for r in rows),
        "new_state_counts": dict(states),
        "original_state_counts": dict(orig),
    }


def print_summary(title: str, s: dict):
    print(f"\n=== {title} ===")
    print(
        f"Plans: {s['plans']} | Touched: {s['touched']} ({s['touch_rate_pct']}%) | "
        f"Entered: {s['filled']} ({s['fill_rate_all_pct']}% of all / {s['fill_rate_after_touch_pct']}% after touch)"
    )
    print(
        f"Target: {s['wins']} | Stop: {s['losses']} | Open: {s['open']} | "
        f"Touched-not-entered: {s['touched_not_entered']} | Untouched: {s['untouched']}"
    )
    print(
        f"Realized win rate: {s['win_rate_realized_pct']}% | "
        f"Sum R: {s['sum_realized_R']} | Avg R: {s['avg_realized_R']} | "
        f"Avg return: {s['avg_realized_return_pct']}%"
    )
    if s["same_bar_ambiguous"]:
        print(f"Conservative same-bar OHLC ambiguity cases: {s['same_bar_ambiguous']}")


def main():
    ap = argparse.ArgumentParser(description="Compare 3 pullback-entry variants with ATR exits")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--start", help="YYYY-MM-DD in New York")
    ap.add_argument("--end", help="YYYY-MM-DD in New York, inclusive")
    ap.add_argument("--stop-atr", type=float, default=1.2)
    ap.add_argument("--target-atr", type=float, default=2.4)
    ap.add_argument("--reclaim-atr", type=float, default=0.20)
    ap.add_argument("--only-old-stopped", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--output", default="weekly_pullback_atr_backtest_v2.csv")
    ap.add_argument("--summary-output", default="weekly_pullback_atr_backtest_v2_summary.json")
    args = ap.parse_args()

    if args.days <= 0 or args.stop_atr <= 0 or args.target_atr <= 0 or args.reclaim_atr <= 0:
        raise SystemExit("days/ATR multipliers must be > 0")

    DB = database()
    rows_by_variant = {v: [] for v in VARIANTS}

    with DB() as s:
        start_ts, end_ts, window_label = choose_window(s, args)
        q = (
            select(Plan)
            .where(
                Plan.market == "US",
                Plan.signal_ts >= start_ts,
                Plan.signal_ts < end_ts,
                Plan.score >= args.min_score,
            )
            .order_by(Plan.signal_ts, Plan.symbol)
        )
        if args.only_old_stopped:
            q = q.where(Plan.state == "STOPPED")
        plans = s.scalars(q).all()

        if not plans:
            print("No US Plan rows were found for the requested window.")
            print("This script backtests stored recommendations only.")
            print(f"Window: {window_label}")
            return 2

        for p in plans:
            bars = s.execute(
                select(Candle.ts, Candle.o, Candle.h, Candle.l, Candle.c, Candle.v)
                .where(
                    Candle.symbol == p.symbol,
                    Candle.ts > int(p.signal_ts),
                    Candle.ts < end_ts,
                )
                .order_by(Candle.ts)
            ).all()
            bars = list(bars)
            for variant in VARIANTS:
                rows_by_variant[variant].append(
                    simulate(
                        p,
                        bars,
                        variant,
                        args.stop_atr,
                        args.target_atr,
                        args.reclaim_atr,
                    )
                )

    summaries = {}
    stopped_summaries = {}

    print("\nRajih weekly pullback/ATR comparison backtest")
    print(f"Window: {window_label}")
    print(f"Exit bracket: stop={args.stop_atr:g} ATR | target={args.target_atr:g} ATR")
    print(
        "Variants: DIRECT_LIMIT=old stop; "
        "BOUNCE_CONFIRM=bullish 15m close above old stop then next-open entry; "
        f"RECLAIM_02ATR=touch then buy-stop at old stop + {args.reclaim_atr:g} ATR from next candle."
    )
    print("OHLC ambiguity rule: conservative stop-first when both stop and target are possible in one candle.")

    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        summaries[variant] = summarize(rows)
        stopped_rows = [r for r in rows if r.original_state == "STOPPED"]
        stopped_summaries[variant] = summarize(stopped_rows)
        print_summary(f"{variant} — ALL STORED RECOMMENDATIONS", summaries[variant])
        print_summary(f"{variant} — ORIGINAL STOPPED SUBSET", stopped_summaries[variant])

    print("\n=== SIDE-BY-SIDE: ORIGINAL STOPPED SUBSET ===")
    print("variant          | entered | target | stop | open | win%   | sum_R")
    for variant in VARIANTS:
        s = stopped_summaries[variant]
        wr = "n/a" if s["win_rate_realized_pct"] is None else f"{s['win_rate_realized_pct']:.2f}"
        print(
            f"{variant:16} | {s['filled']:7d} | {s['wins']:6d} | {s['losses']:4d} | "
            f"{s['open']:4d} | {wr:>6} | {s['sum_realized_R']:7.3f}"
        )

    # Combined CSV: one row per plan per variant.
    all_rows = []
    for variant in VARIANTS:
        all_rows.extend(rows_by_variant[variant])

    out = Path(args.output)
    fields = list(asdict(all_rows[0]).keys())
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            d = asdict(r)
            for k in (
                "score", "original_entry", "original_stop", "original_target",
                "signal_atr", "pullback_reference", "entry_trigger", "fill_price",
                "new_stop", "new_target", "exit_price", "pnl_pct", "pnl_r",
            ):
                d[k] = safe_round(d[k])
            w.writerow(d)

    payload = {
        "window": window_label,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "stop_atr": args.stop_atr,
        "target_atr": args.target_atr,
        "reclaim_atr": args.reclaim_atr,
        "variant_definitions": {
            "DIRECT_LIMIT": "resting buy limit at original stop",
            "BOUNCE_CONFIRM": "touch original stop, then bullish 15m close above original stop; enter next candle open",
            "RECLAIM_02ATR": f"touch original stop, then from next candle buy-stop at original stop + {args.reclaim_atr:g} ATR",
        },
        "assumptions": [
            "stored US recommendations and stored 15-minute candles only",
            "ATR frozen from original recommendation",
            "new stop/target calculated from actual new entry",
            "DIRECT_LIMIT receives opening-price improvement on a gap below the limit",
            "BOUNCE_CONFIRM enters only after confirmation candle is fully closed, at next candle open",
            "RECLAIM variant starts only on the candle after the touch to avoid assuming intrabar order",
            "RECLAIM buy-stop fills at candle open if it gaps above trigger; otherwise at trigger",
            "stop evaluated before target when OHLC path is ambiguous",
            "OPEN uses latest stored close as mark-to-market only",
            "no commissions or general slippage beyond explicit gap rules",
        ],
        "all_recommendations": summaries,
        "original_stopped_subset": stopped_summaries,
        "csv": str(out),
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
