#!/usr/bin/env python3
"""
Rajih 30-day Bounce-Confirm ATR backtest.

Purpose
-------
Test only the best entry idea from v2:

1) Original stop-loss becomes the pullback reference.
2) Price must touch/trade through that old stop.
3) Wait for a COMPLETED bullish 15m candle:
       close > open
       close > old stop
4) Enter at the NEXT 15m candle open.
5) Test three ATR exit brackets:
       A: Stop 1.0 ATR / Target 2.0 ATR
       B: Stop 1.2 ATR / Target 2.4 ATR
       C: Stop 1.5 ATR / Target 3.0 ATR
6) Maximum holding time defaults to 78 x 15m bars ~= 3 regular US sessions.
   If neither target nor stop is hit by then, close at that bar's close.
7) Show results overall, on originally STOPPED recommendations, and by score bucket.

Read-only:
- Does NOT modify plans, candles, customers, outbox, or settings.

Run:
    python weekly_pullback_atr_backtest_v3.py --days 30

Optional:
    python weekly_pullback_atr_backtest_v3.py --days 30 --max-hold-bars 78
    python weekly_pullback_atr_backtest_v3.py --start 2026-09-01 --end 2026-10-02
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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

BRACKETS = [
    ("ATR_1.0_2.0", 1.0, 2.0),
    ("ATR_1.2_2.4", 1.2, 2.4),
    ("ATR_1.5_3.0", 1.5, 3.0),
]


@dataclass
class Result:
    bracket: str
    stop_atr_mult: float
    target_atr_mult: float
    plan_id: int
    symbol: str
    signal_ny: str
    score: float
    score_bucket: str
    original_state: str
    original_entry: float
    original_stop: float
    original_target: float
    signal_atr: float
    pullback_reference: float
    touched: bool
    touch_ts_ny: str
    confirmed: bool
    confirm_ts_ny: str
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
    bars_held: int
    timed_exit: bool
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


def choose_window(session, args):
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


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def score_bucket(score: float) -> str:
    if score < 60:
        return "<60"
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    if score < 90:
        return "80-89"
    return "90+"


def find_touch_and_confirmation(bars, reference):
    touch_i = None
    confirm_i = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v = row
        o, l, c = float(o), float(l), float(c)

        if touch_i is None and l <= reference:
            touch_i = i

        if touch_i is not None and i >= touch_i:
            if c > o and c > reference:
                confirm_i = i
                break

    return touch_i, confirm_i


def simulate(plan, bars, name, stop_mult, target_mult, max_hold_bars):
    reference, _ = rounded(float(plan.stop), "US", False)
    atr = float(plan.atr or 0.0)
    score = float(plan.score or 0.0)

    base = dict(
        bracket=name,
        stop_atr_mult=stop_mult,
        target_atr_mult=target_mult,
        plan_id=int(plan.id),
        symbol=plan.symbol,
        signal_ny=iso_ny(plan.signal_ts),
        score=score,
        score_bucket=score_bucket(score),
        original_state=plan.state,
        original_entry=float(plan.entry),
        original_stop=float(plan.stop),
        original_target=float(plan.target),
        signal_atr=atr,
        pullback_reference=reference,
    )

    if atr <= 0 or not math.isfinite(atr):
        return Result(
            **base,
            touched=False, touch_ts_ny="", confirmed=False, confirm_ts_ny="",
            filled=False, fill_ts_ny="", fill_price=None,
            new_stop=None, new_target=None, new_state="INVALID_ATR",
            exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
            bars_held=0, timed_exit=False, same_bar_ambiguous=False,
            note="Original plan ATR missing/invalid",
        )

    touch_i, confirm_i = find_touch_and_confirmation(bars, reference)

    if touch_i is None:
        return Result(
            **base,
            touched=False, touch_ts_ny="", confirmed=False, confirm_ts_ny="",
            filled=False, fill_ts_ny="", fill_price=None,
            new_stop=None, new_target=None, new_state="UNTOUCHED",
            exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
            bars_held=0, timed_exit=False, same_bar_ambiguous=False,
            note="Pullback reference was never touched",
        )

    touch_ts = int(bars[touch_i][0])

    if confirm_i is None:
        return Result(
            **base,
            touched=True, touch_ts_ny=iso_ny(touch_ts),
            confirmed=False, confirm_ts_ny="",
            filled=False, fill_ts_ny="", fill_price=None,
            new_stop=None, new_target=None, new_state="TOUCHED_NO_CONFIRM",
            exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
            bars_held=0, timed_exit=False, same_bar_ambiguous=False,
            note="Touched old stop but no completed bullish 15m candle closed above it",
        )

    confirm_ts = int(bars[confirm_i][0])
    entry_i = confirm_i + 1
    if entry_i >= len(bars):
        return Result(
            **base,
            touched=True, touch_ts_ny=iso_ny(touch_ts),
            confirmed=True, confirm_ts_ny=iso_ny(confirm_ts),
            filled=False, fill_ts_ny="", fill_price=None,
            new_stop=None, new_target=None, new_state="CONFIRMED_NO_NEXT_BAR",
            exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
            bars_held=0, timed_exit=False, same_bar_ambiguous=False,
            note="Confirmation exists but no next stored candle to enter",
        )

    fill_ts, o, h, l, c, v = bars[entry_i]
    fill = float(o)

    new_stop, _ = rounded(fill - stop_mult * atr, "US", False)
    new_target, _ = rounded(fill + target_mult * atr, "US", True)
    risk = fill - new_stop

    if new_stop <= 0 or new_stop >= fill or new_target <= fill or risk <= 0:
        return Result(
            **base,
            touched=True, touch_ts_ny=iso_ny(touch_ts),
            confirmed=True, confirm_ts_ny=iso_ny(confirm_ts),
            filled=True, fill_ts_ny=iso_ny(fill_ts), fill_price=fill,
            new_stop=new_stop, new_target=new_target, new_state="INVALID_LEVELS",
            exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
            bars_held=0, timed_exit=False, same_bar_ambiguous=False,
            note="Invalid ATR bracket",
        )

    state = "OPEN"
    exit_ts = None
    exit_price = None
    same_bar_ambiguous = False
    timed_exit = False
    bars_held = 0
    note = "Bullish 15m rebound confirmed; entered next candle open"

    final_i = min(len(bars) - 1, entry_i + max_hold_bars - 1)

    for j in range(entry_i, final_i + 1):
        ts, o, h, l, c, v = bars[j]
        o, h, l, c = map(float, (o, h, l, c))
        bars_held = j - entry_i + 1

        stop_hit = l <= new_stop
        target_hit = h >= new_target

        if stop_hit and target_hit:
            same_bar_ambiguous = True

        if stop_hit:
            state = "STOPPED"
            exit_price = new_stop if j == entry_i else min(o, new_stop)
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
        ts, o, h, l, c, v = bars[final_i]
        exit_ts = int(ts)
        exit_price = float(c)
        bars_held = final_i - entry_i + 1

        if final_i < len(bars) - 1 or bars_held >= max_hold_bars:
            state = "TIME_EXIT"
            timed_exit = True
            note += f"; max holding period reached ({max_hold_bars} bars), exited at close"
        else:
            state = "OPEN_END"
            note += "; dataset ended before max holding period, marked to last stored close"

    pnl_pct = pct(exit_price, fill) if exit_price is not None else None
    pnl_r = (exit_price - fill) / risk if exit_price is not None else None

    return Result(
        **base,
        touched=True, touch_ts_ny=iso_ny(touch_ts),
        confirmed=True, confirm_ts_ny=iso_ny(confirm_ts),
        filled=True, fill_ts_ny=iso_ny(fill_ts), fill_price=fill,
        new_stop=new_stop, new_target=new_target, new_state=state,
        exit_ts_ny=iso_ny(exit_ts), exit_price=exit_price,
        pnl_pct=pnl_pct, pnl_r=pnl_r, bars_held=bars_held,
        timed_exit=timed_exit, same_bar_ambiguous=same_bar_ambiguous,
        note=note,
    )


def summarize(rows):
    states = Counter(r.new_state for r in rows)
    touched = [r for r in rows if r.touched]
    filled = [r for r in rows if r.filled and r.new_state not in ("INVALID_ATR", "INVALID_LEVELS")]
    targets = [r for r in filled if r.new_state == "TARGET"]
    stops = [r for r in filled if r.new_state == "STOPPED"]
    timed = [r for r in filled if r.new_state == "TIME_EXIT"]
    completed = [r for r in filled if r.new_state in ("TARGET", "STOPPED", "TIME_EXIT")]
    positive = [r for r in completed if (r.pnl_r or 0) > 0]
    rvals = [r.pnl_r for r in completed if r.pnl_r is not None]
    returns = [r.pnl_pct for r in completed if r.pnl_pct is not None]

    return {
        "plans": len(rows),
        "touched": len(touched),
        "entered": len(filled),
        "target": len(targets),
        "stop": len(stops),
        "time_exit": len(timed),
        "open_end": states.get("OPEN_END", 0),
        "touched_no_confirm": states.get("TOUCHED_NO_CONFIRM", 0),
        "untouched": states.get("UNTOUCHED", 0),
        "positive_completed": len(positive),
        "positive_completed_pct": round(100 * len(positive) / len(completed), 2) if completed else None,
        "target_vs_stop_win_pct": round(100 * len(targets) / (len(targets) + len(stops)), 2) if (len(targets) + len(stops)) else None,
        "sum_R": round(sum(rvals), 3) if rvals else 0,
        "avg_R": round(sum(rvals) / len(rvals), 3) if rvals else None,
        "avg_return_pct": round(sum(returns) / len(returns), 3) if returns else None,
        "same_bar_ambiguous": sum(r.same_bar_ambiguous for r in rows),
        "states": dict(states),
    }


def print_summary(title, s):
    print(f"\n=== {title} ===")
    print(
        f"Plans: {s['plans']} | Touched: {s['touched']} | Entered: {s['entered']} | "
        f"Target: {s['target']} | Stop: {s['stop']} | Time-exit: {s['time_exit']} | Open-end: {s['open_end']}"
    )
    print(
        f"No-confirm after touch: {s['touched_no_confirm']} | Untouched: {s['untouched']} | "
        f"Target-vs-stop win%: {s['target_vs_stop_win_pct']} | "
        f"Positive completed% incl. time exits: {s['positive_completed_pct']}"
    )
    print(
        f"Sum R: {s['sum_R']} | Avg R: {s['avg_R']} | Avg return: {s['avg_return_pct']}%"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-hold-bars", type=int, default=78,
                    help="78 x 15m bars ~= 3 regular US sessions")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--output", default="weekly_pullback_atr_backtest_v3.csv")
    ap.add_argument("--summary-output", default="weekly_pullback_atr_backtest_v3_summary.json")
    args = ap.parse_args()

    DB = database()
    results_by_bracket = {name: [] for name, _, _ in BRACKETS}

    with DB() as s:
        start_ts, end_ts, label = choose_window(s, args)

        plans = s.scalars(
            select(Plan)
            .where(
                Plan.market == "US",
                Plan.signal_ts >= start_ts,
                Plan.signal_ts < end_ts,
                Plan.score >= args.min_score,
            )
            .order_by(Plan.signal_ts, Plan.symbol)
        ).all()

        if not plans:
            print("No US recommendations found in requested window.")
            return 2

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

            for name, sm, tm in BRACKETS:
                results_by_bracket[name].append(
                    simulate(p, bars, name, sm, tm, args.max_hold_bars)
                )

    print("\nRajih 30-day Bounce-Confirm ATR backtest")
    print(f"Window: {label}")
    print(f"Entry: touch old stop -> bullish 15m close above it -> enter next candle open")
    print(f"Max hold: {args.max_hold_bars} bars (~{args.max_hold_bars/26:.1f} regular sessions)")
    print("Exit brackets: 1.0/2.0 ATR, 1.2/2.4 ATR, 1.5/3.0 ATR")

    overall = {}
    stopped = {}
    score_tables = {}

    for name, _, _ in BRACKETS:
        rows = results_by_bracket[name]
        overall[name] = summarize(rows)
        stopped_rows = [r for r in rows if r.original_state == "STOPPED"]
        stopped[name] = summarize(stopped_rows)

        print_summary(f"{name} — ALL RECOMMENDATIONS", overall[name])
        print_summary(f"{name} — ORIGINAL STOPPED SUBSET", stopped[name])

        by_bucket = defaultdict(list)
        for r in rows:
            by_bucket[r.score_bucket].append(r)

        score_tables[name] = {k: summarize(v) for k, v in by_bucket.items()}

    print("\n=== SIDE-BY-SIDE — ALL RECOMMENDATIONS ===")
    print("bracket      | entered | target | stop | time | positive% | sum_R")
    for name, _, _ in BRACKETS:
        s = overall[name]
        pos = "n/a" if s["positive_completed_pct"] is None else f"{s['positive_completed_pct']:.2f}"
        print(
            f"{name:12} | {s['entered']:7d} | {s['target']:6d} | {s['stop']:4d} | "
            f"{s['time_exit']:4d} | {pos:>9} | {s['sum_R']:7.3f}"
        )

    print("\n=== SIDE-BY-SIDE — ORIGINAL STOPPED SUBSET ===")
    print("bracket      | entered | target | stop | time | positive% | sum_R")
    for name, _, _ in BRACKETS:
        s = stopped[name]
        pos = "n/a" if s["positive_completed_pct"] is None else f"{s['positive_completed_pct']:.2f}"
        print(
            f"{name:12} | {s['entered']:7d} | {s['target']:6d} | {s['stop']:4d} | "
            f"{s['time_exit']:4d} | {pos:>9} | {s['sum_R']:7.3f}"
        )

    print("\n=== SCORE BUCKETS — ALL RECOMMENDATIONS ===")
    for name, _, _ in BRACKETS:
        print(f"\n{name}")
        print("score   | plans | entered | target | stop | time | positive% | sum_R")
        for bucket in ("<60", "60-69", "70-79", "80-89", "90+"):
            if bucket not in score_tables[name]:
                continue
            s = score_tables[name][bucket]
            pos = "n/a" if s["positive_completed_pct"] is None else f"{s['positive_completed_pct']:.2f}"
            print(
                f"{bucket:7} | {s['plans']:5d} | {s['entered']:7d} | {s['target']:6d} | "
                f"{s['stop']:4d} | {s['time_exit']:4d} | {pos:>9} | {s['sum_R']:7.3f}"
            )

    all_rows = []
    for name, _, _ in BRACKETS:
        all_rows.extend(results_by_bracket[name])

    out = Path(args.output)
    fields = list(asdict(all_rows[0]).keys())
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))

    payload = {
        "window": label,
        "max_hold_bars": args.max_hold_bars,
        "brackets": [{"name": n, "stop_atr": s, "target_atr": t} for n, s, t in BRACKETS],
        "overall": overall,
        "original_stopped_subset": stopped,
        "score_buckets": score_tables,
        "assumptions": [
            "entry only after completed bullish 15m candle closes above old stop",
            "entry at next 15m candle open",
            "ATR frozen from original recommendation",
            "time exit after max_hold_bars, at that bar's close",
            "stop-first if same candle can hit both stop and target",
            "stored candles only",
            "no commissions/slippage beyond gap behavior for stops",
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
