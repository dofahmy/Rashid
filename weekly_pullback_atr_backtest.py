#!/usr/bin/env python3
"""
One-off 7-day backtest for Rajih US recommendations.

Hypothesis tested
-----------------
Instead of buying at the original recommendation entry, place a BUY LIMIT at the
ORIGINAL stop-loss level.  Once that limit is filled, define a fresh ATR bracket:

    new_stop   = actual_fill - STOP_ATR * signal_ATR
    new_target = actual_fill + TARGET_ATR * signal_ATR

Defaults requested in chat:
    STOP_ATR   = 1.2
    TARGET_ATR = 2.4        # 2R target

The script is READ-ONLY: it does not modify plans, candles, customers or outbox.
It uses stored 15-minute candles and the ATR frozen on each original Plan.

Run from the Railway project root:
    python weekly_pullback_atr_backtest.py --days 7

Optional explicit New York date window:
    python weekly_pullback_atr_backtest.py --start 2026-09-25 --end 2026-10-02

Useful filters:
    --only-old-stopped      only plans whose original result is STOPPED
    --min-score 60          only original plans with score >= 60

Outputs:
    - console summary
    - weekly_pullback_atr_backtest.csv
    - weekly_pullback_atr_backtest_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select, func

# Run from repository root so these imports resolve against the deployed project.
from core import database
from monitor.models import Plan, Candle
from monitor.strategy import rounded

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass
class Result:
    plan_id: int
    symbol: str
    signal_ny: str
    score: float
    original_state: str
    original_entry: float
    original_stop: float
    original_target: float
    signal_atr: float
    new_limit: float
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
    bars_after_fill: int
    same_bar_ambiguous: bool
    note: str


def iso_ny(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), NY).isoformat(timespec="minutes")


def parse_date_ny(text: str, *, end: bool = False) -> int:
    d = datetime.strptime(text, "%Y-%m-%d").date()
    # --end is inclusive through the whole NY calendar day.
    dt = datetime.combine(d + (timedelta(days=1) if end else timedelta(0)), dtime.min, tzinfo=NY)
    return int(dt.timestamp())


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def safe_round(v: float | None, n: int = 4):
    return None if v is None else round(float(v), n)


def choose_window(session, args) -> tuple[int, int, str]:
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("Use --start and --end together (YYYY-MM-DD, New York dates).")
        start = parse_date_ny(args.start)
        end = parse_date_ny(args.end, end=True)
        return start, end, f"{args.start} .. {args.end} (New York)"

    # Anchor to latest US candle in DB instead of wall-clock time.  This makes a
    # Friday/weekend run deterministic and avoids counting empty future hours.
    latest = session.scalar(select(func.max(Candle.ts)).join(Plan, Plan.symbol == Candle.symbol, isouter=True))
    if not latest:
        latest = int(datetime.now(UTC).timestamp())
    end = int(latest) + 900
    start = end - int(args.days * 86400)
    label = f"last {args.days:g} days ending {iso_ny(latest)}"
    return start, end, label


def simulate(plan: Plan, bars: list[tuple], stop_atr: float, target_atr: float) -> Result:
    """Simulate a resting buy-limit at the original stop from immediately after signal."""
    limit_price, _ = rounded(float(plan.stop), "US", False)
    atr = float(plan.atr or 0.0)

    base = dict(
        plan_id=int(plan.id), symbol=plan.symbol, signal_ny=iso_ny(plan.signal_ts),
        score=float(plan.score or 0), original_state=plan.state,
        original_entry=float(plan.entry), original_stop=float(plan.stop),
        original_target=float(plan.target), signal_atr=atr, new_limit=limit_price,
    )

    if atr <= 0 or not math.isfinite(atr):
        return Result(**base, filled=False, fill_ts_ny="", fill_price=None,
                      new_stop=None, new_target=None, new_state="INVALID_ATR",
                      exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
                      bars_after_signal=len(bars), bars_after_fill=0,
                      same_bar_ambiguous=False, note="Original plan ATR is missing/invalid")

    fill_i = None
    fill = None
    fill_ts = None
    for i, (ts, o, h, l, c, v) in enumerate(bars):
        if l <= limit_price:
            # Resting buy-limit gets price improvement on a gap below the limit.
            fill = min(float(o), limit_price) if float(o) <= limit_price else limit_price
            fill_i = i
            fill_ts = int(ts)
            break

    if fill_i is None:
        last = float(bars[-1][4]) if bars else None
        note = "Buy-limit at old stop never traded"
        if last is not None:
            note += f"; last close vs limit={pct(last, limit_price):+.2f}%"
        return Result(**base, filled=False, fill_ts_ny="", fill_price=None,
                      new_stop=None, new_target=None, new_state="UNFILLED",
                      exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
                      bars_after_signal=len(bars), bars_after_fill=0,
                      same_bar_ambiguous=False, note=note)

    # Recalculate the ATR bracket from the actual fill, as requested.
    new_stop, _ = rounded(fill - stop_atr * atr, "US", False)
    new_target, _ = rounded(fill + target_atr * atr, "US", True)
    if new_stop <= 0 or new_stop >= fill or new_target <= fill:
        return Result(**base, filled=True, fill_ts_ny=iso_ny(fill_ts), fill_price=fill,
                      new_stop=new_stop, new_target=new_target, new_state="INVALID_LEVELS",
                      exit_ts_ny="", exit_price=None, pnl_pct=None, pnl_r=None,
                      bars_after_signal=len(bars), bars_after_fill=len(bars)-fill_i,
                      same_bar_ambiguous=False, note="Calculated ATR bracket is invalid")

    risk = fill - new_stop
    same_bar_ambiguous = False
    exit_ts = None
    exit_price = None
    state = "OPEN"
    note = ""

    # Start with the fill candle. Because OHLC does not reveal the path within
    # the candle, if both stop and target are possible after the limit touch we
    # use the conservative convention: STOP first.
    for j in range(fill_i, len(bars)):
        ts, o, h, l, c, v = bars[j]
        o, h, l, c = map(float, (o, h, l, c))
        stop_hit = l <= new_stop
        target_hit = h >= new_target
        if j == fill_i and (stop_hit or target_hit):
            same_bar_ambiguous = True
        if stop_hit:
            state = "STOPPED"
            # If a later candle gaps below stop, use open as conservative exit.
            exit_price = min(o, new_stop) if j > fill_i else new_stop
            exit_ts = int(ts)
            note = "Stop hit; conservative stop-first rule on OHLC ambiguity" if target_hit else "Stop hit"
            break
        if target_hit:
            state = "TARGET"
            exit_price = new_target
            exit_ts = int(ts)
            note = "Target hit"
            break

    if state == "OPEN":
        if bars:
            exit_price = float(bars[-1][4])  # mark-to-market only
            exit_ts = int(bars[-1][0])
            note = "Still open; P/L is mark-to-market at latest stored close"
        else:
            note = "Filled but no subsequent stored candles"

    pnl_pct = pct(exit_price, fill) if exit_price is not None else None
    pnl_r = (exit_price - fill) / risk if exit_price is not None and risk > 0 else None

    return Result(**base, filled=True, fill_ts_ny=iso_ny(fill_ts), fill_price=fill,
                  new_stop=new_stop, new_target=new_target, new_state=state,
                  exit_ts_ny=iso_ny(exit_ts), exit_price=exit_price,
                  pnl_pct=pnl_pct, pnl_r=pnl_r,
                  bars_after_signal=len(bars), bars_after_fill=len(bars)-fill_i,
                  same_bar_ambiguous=same_bar_ambiguous, note=note)


def summarize(rows: list[Result]) -> dict:
    states = Counter(r.new_state for r in rows)
    orig = Counter(r.original_state for r in rows)
    filled = [r for r in rows if r.filled and r.new_state not in ("INVALID_ATR", "INVALID_LEVELS")]
    realized = [r for r in filled if r.new_state in ("TARGET", "STOPPED")]
    pnl = [r.pnl_pct for r in realized if r.pnl_pct is not None]
    rvals = [r.pnl_r for r in realized if r.pnl_r is not None]
    wins = sum(r.new_state == "TARGET" for r in realized)
    losses = sum(r.new_state == "STOPPED" for r in realized)
    return {
        "plans": len(rows),
        "original_state_counts": dict(orig),
        "new_state_counts": dict(states),
        "filled": len(filled),
        "fill_rate_pct": round(100 * len(filled) / len(rows), 2) if rows else 0,
        "realized": len(realized),
        "wins": wins,
        "losses": losses,
        "win_rate_realized_pct": round(100 * wins / len(realized), 2) if realized else None,
        "sum_realized_R": round(sum(rvals), 3) if rvals else 0,
        "avg_realized_R": round(sum(rvals) / len(rvals), 3) if rvals else None,
        "avg_realized_return_pct": round(sum(pnl) / len(pnl), 3) if pnl else None,
        "same_bar_ambiguous": sum(r.same_bar_ambiguous for r in rows),
    }


def print_summary(title: str, s: dict):
    print(f"\n=== {title} ===")
    print(f"Plans: {s['plans']} | Filled: {s['filled']} ({s['fill_rate_pct']}%) | Realized: {s['realized']}")
    print(f"Target: {s['wins']} | Stop: {s['losses']} | Open: {s['new_state_counts'].get('OPEN',0)} | Unfilled: {s['new_state_counts'].get('UNFILLED',0)}")
    print(f"Realized win rate: {s['win_rate_realized_pct']}% | Sum R: {s['sum_realized_R']} | Avg R: {s['avg_realized_R']} | Avg return: {s['avg_realized_return_pct']}%")
    print(f"New states: {s['new_state_counts']}")
    print(f"Original states: {s['original_state_counts']}")
    if s['same_bar_ambiguous']:
        print(f"Conservative same-bar OHLC ambiguity cases: {s['same_bar_ambiguous']}")


def main():
    ap = argparse.ArgumentParser(description="Backtest old-stop-as-buy-limit + ATR stop/target")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--start", help="YYYY-MM-DD in New York")
    ap.add_argument("--end", help="YYYY-MM-DD in New York, inclusive")
    ap.add_argument("--stop-atr", type=float, default=1.2)
    ap.add_argument("--target-atr", type=float, default=2.4)
    ap.add_argument("--only-old-stopped", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--output", default="weekly_pullback_atr_backtest.csv")
    ap.add_argument("--summary-output", default="weekly_pullback_atr_backtest_summary.json")
    args = ap.parse_args()
    if args.days <= 0 or args.stop_atr <= 0 or args.target_atr <= 0:
        raise SystemExit("days/ATR multipliers must be > 0")

    DB = database()
    with DB() as s:
        start_ts, end_ts, window_label = choose_window(s, args)
        q = select(Plan).where(
            Plan.market == "US",
            Plan.signal_ts >= start_ts,
            Plan.signal_ts < end_ts,
            Plan.score >= args.min_score,
        ).order_by(Plan.signal_ts, Plan.symbol)
        if args.only_old_stopped:
            q = q.where(Plan.state == "STOPPED")
        plans = s.scalars(q).all()

        if not plans:
            print("No US Plan rows were found for the requested window.")
            print("This script intentionally backtests real stored recommendations, not reconstructed signals.")
            print("If Development Reset deleted those historical Plan rows, exact past recommendations cannot be recovered from this table.")
            print(f"Window: {window_label}")
            return 2

        rows: list[Result] = []
        for p in plans:
            bars = s.execute(
                select(Candle.ts, Candle.o, Candle.h, Candle.l, Candle.c, Candle.v)
                .where(Candle.symbol == p.symbol,
                       Candle.ts > int(p.signal_ts),
                       Candle.ts < end_ts)
                .order_by(Candle.ts)
            ).all()
            rows.append(simulate(p, list(bars), args.stop_atr, args.target_atr))

    summary_all = summarize(rows)
    old_stopped = [r for r in rows if r.original_state == "STOPPED"]
    summary_stopped = summarize(old_stopped)

    print("\nRajih weekly pullback/ATR backtest")
    print(f"Window: {window_label}")
    print(f"Hypothesis: buy-limit=old stop | new stop={args.stop_atr:g} ATR | new target={args.target_atr:g} ATR")
    print("Execution assumptions: resting limit; gap gets better fill; stop-first if OHLC path is ambiguous.")
    print_summary("ALL STORED RECOMMENDATIONS", summary_all)
    print_summary("ORIGINAL STOPPED SUBSET", summary_stopped)

    # Small human-readable sample, prioritizing the old-stopped subset.
    print("\n--- Old STOPPED recommendations under the new rule ---")
    print("symbol | old_entry | old_stop->new_limit | fill | new_stop | new_target | result | pnl% | R")
    for r in old_stopped[:100]:
        print(f"{r.symbol:6} | {r.original_entry:9.3f} | {r.new_limit:9.3f} | "
              f"{(r.fill_price if r.fill_price is not None else float('nan')):9.3f} | "
              f"{(r.new_stop if r.new_stop is not None else float('nan')):9.3f} | "
              f"{(r.new_target if r.new_target is not None else float('nan')):10.3f} | "
              f"{r.new_state:9} | "
              f"{(r.pnl_pct if r.pnl_pct is not None else float('nan')):+7.2f} | "
              f"{(r.pnl_r if r.pnl_r is not None else float('nan')):+6.2f}")

    out = Path(args.output)
    fields = list(asdict(rows[0]).keys())
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            for k in ("score","original_entry","original_stop","original_target","signal_atr","new_limit","fill_price","new_stop","new_target","exit_price","pnl_pct","pnl_r"):
                d[k] = safe_round(d[k], 6)
            w.writerow(d)

    payload = {
        "window": window_label,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "stop_atr": args.stop_atr,
        "target_atr": args.target_atr,
        "assumptions": [
            "buy limit equals original stop",
            "new ATR bracket calculated from actual fill",
            "resting buy-limit receives opening-price improvement on gap below limit",
            "stop is evaluated before target when OHLC cannot reveal intrabar order",
            "OPEN result uses latest stored close only as mark-to-market",
            "no commissions/slippage except explicit gap fill logic",
        ],
        "all": summary_all,
        "original_stopped_subset": summary_stopped,
        "csv": str(out),
    }
    Path(args.summary_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCSV written: {out}")
    print(f"Summary JSON written: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
