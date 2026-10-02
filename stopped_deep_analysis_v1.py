#!/usr/bin/env python3
"""
Rajih — deep analysis of ORIGINAL STOPPED US recommendations.

Questions answered:
1) What is common among the recommendations that stopped out?
2) Before the FIRST/original stop was hit, how far did price rise on average?
   - MFE in %
   - MFE in ATR
   - median / quartiles / 90th percentile
   - how many reached +0.25%, +0.5%, +1%, +1R before stopping
3) Alternative entry:
      first_stop  = original stop
      second_stop = first_stop - SECOND_STOP_ATR * signal ATR
   Hypothesis:
      do NOT buy at original entry or first stop;
      place BUY LIMIT at second_stop;
      target = first_stop.
   Also test a symmetric protective "third stop":
      third_stop = second_stop - SECOND_STOP_ATR * signal ATR
   and report whether target or third stop came first.
4) Show results by score bucket, original stop distance bucket, and ATR% bucket.

READ ONLY. Does not modify DB.

Run:
    python stopped_deep_analysis_v1.py --days 30

Optional:
    python stopped_deep_analysis_v1.py --days 30 --second-stop-atr 1.2
    python stopped_deep_analysis_v1.py --days 60 --second-stop-atr 1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
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


def pct_change(a, b):
    return 100.0 * (float(a) / float(b) - 1.0)


def percentile(values, p):
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def qstats(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
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


def score_bucket(x):
    x = float(x or 0)
    if x < 60: return "<60"
    if x < 70: return "60-69"
    if x < 80: return "70-79"
    if x < 90: return "80-89"
    return "90+"


def stopdist_bucket(x):
    # x is positive percent risk from entry to first stop
    if x < .25: return "<0.25%"
    if x < .50: return "0.25-0.49%"
    if x < .75: return "0.50-0.74%"
    if x < 1.0: return "0.75-0.99%"
    if x < 1.5: return "1.00-1.49%"
    if x < 2.0: return "1.50-1.99%"
    return "2.00%+"


def atrpct_bucket(x):
    if x < .25: return "<0.25%"
    if x < .50: return "0.25-0.49%"
    if x < .75: return "0.50-0.74%"
    if x < 1.0: return "0.75-0.99%"
    if x < 1.5: return "1.00-1.49%"
    return "1.50%+"


@dataclass
class Row:
    plan_id: int
    symbol: str
    signal_ny: str
    score: float
    entry: float
    first_stop: float
    target: float
    atr: float
    stop_distance_pct: float
    stop_distance_atr: float
    atr_pct: float

    first_stop_hit_ts_ny: str
    bars_to_first_stop: int
    hours_to_first_stop: float

    mfe_before_first_stop_pct: float
    mfe_before_first_stop_atr: float
    max_high_before_first_stop: float
    reached_plus_025pct_before_stop: bool
    reached_plus_050pct_before_stop: bool
    reached_plus_100pct_before_stop: bool
    reached_plus_1R_before_stop: bool

    second_stop: float
    second_stop_touched: bool
    second_stop_touch_ts_ny: str
    second_stop_gap_fill: float | None

    recovery_target: float
    recovery_target_after_second: bool
    recovery_target_ts_ny: str
    bars_second_to_recovery: int | None

    third_stop: float
    third_stop_hit_after_second: bool
    third_stop_ts_ny: str

    second_strategy_state: str
    second_strategy_exit: float | None
    second_strategy_return_pct: float | None
    second_strategy_R: float | None
    same_bar_ambiguous: bool

    score_bucket: str
    stopdist_bucket: str
    atrpct_bucket: str


def analyze_plan(p, bars, second_atr):
    entry = float(p.entry)
    first_stop = float(p.stop)
    target = float(p.target)
    atr = float(p.atr or 0)
    score = float(p.score or 0)

    if not (entry > 0 and first_stop > 0 and atr > 0):
        return None

    stop_distance = entry - first_stop
    stop_distance_pct = 100 * stop_distance / entry
    stop_distance_atr = stop_distance / atr
    atr_pct = 100 * atr / entry

    # Locate FIRST candle that actually touches original stop.
    first_i = None
    for i, bar in enumerate(bars):
        ts, o, h, l, c, v = bar
        if float(l) <= first_stop:
            first_i = i
            break

    # Only analyze truly reproducible stopped events from stored candles.
    if first_i is None:
        return None

    # MFE before first stop.
    # For the stop candle we cannot know whether its high occurred before or after stop,
    # so exclude the stop candle from strict MFE to avoid look-ahead/intrabar-path bias.
    pre = bars[:first_i]
    if pre:
        max_high = max(float(b[2]) for b in pre)
    else:
        max_high = entry

    max_high = max(max_high, entry)
    mfe_pct = max(0.0, 100 * (max_high / entry - 1))
    mfe_atr = max(0.0, (max_high - entry) / atr)

    stop_ts = int(bars[first_i][0])
    bars_to_stop = first_i + 1
    hours_to_stop = max(0.0, (stop_ts - int(p.signal_ts)) / 3600.0)

    second_stop, _ = rounded(first_stop - second_atr * atr, "US", False)
    third_stop, _ = rounded(second_stop - second_atr * atr, "US", False)
    recovery_target = first_stop

    second_i = None
    fill = None
    for j in range(first_i, len(bars)):
        ts, o, h, l, c, v = bars[j]
        o, l = float(o), float(l)
        if l <= second_stop:
            second_i = j
            # resting buy limit: price improvement if gap opens below it
            fill = min(o, second_stop) if o <= second_stop else second_stop
            break

    second_touched = second_i is not None
    second_touch_ts = int(bars[second_i][0]) if second_touched else None

    recovery = False
    recovery_ts = None
    third_hit = False
    third_ts = None
    state = "SECOND_NOT_TOUCHED"
    exit_price = None
    same_bar = False
    bars_to_recovery = None

    if second_touched:
        state = "OPEN_END"
        # From second-stop fill forward:
        # target = first stop; protective third stop = same ATR spacing below second.
        for j in range(second_i, len(bars)):
            ts, o, h, l, c, v = bars[j]
            o, h, l = float(o), float(h), float(l)
            hit_target = h >= recovery_target
            hit_third = l <= third_stop

            if hit_target and hit_third:
                same_bar = True
                # Conservative: third stop first when OHLC path is ambiguous.
                third_hit = True
                third_ts = int(ts)
                state = "THIRD_STOP"
                exit_price = third_stop if j == second_i else min(o, third_stop)
                break
            if hit_third:
                third_hit = True
                third_ts = int(ts)
                state = "THIRD_STOP"
                exit_price = third_stop if j == second_i else min(o, third_stop)
                break
            if hit_target:
                recovery = True
                recovery_ts = int(ts)
                bars_to_recovery = j - second_i + 1
                state = "RECOVERED_FIRST_STOP"
                exit_price = recovery_target
                break

        if state == "OPEN_END" and bars:
            exit_price = float(bars[-1][4])

    ret = 100 * (exit_price / fill - 1) if (fill and exit_price) else None
    risk = fill - third_stop if fill else None
    rr = (exit_price - fill) / risk if (fill and exit_price is not None and risk and risk > 0) else None

    return Row(
        plan_id=int(p.id),
        symbol=p.symbol,
        signal_ny=iso_ny(p.signal_ts),
        score=score,
        entry=entry,
        first_stop=first_stop,
        target=target,
        atr=atr,
        stop_distance_pct=stop_distance_pct,
        stop_distance_atr=stop_distance_atr,
        atr_pct=atr_pct,

        first_stop_hit_ts_ny=iso_ny(stop_ts),
        bars_to_first_stop=bars_to_stop,
        hours_to_first_stop=hours_to_stop,

        mfe_before_first_stop_pct=mfe_pct,
        mfe_before_first_stop_atr=mfe_atr,
        max_high_before_first_stop=max_high,
        reached_plus_025pct_before_stop=mfe_pct >= .25,
        reached_plus_050pct_before_stop=mfe_pct >= .50,
        reached_plus_100pct_before_stop=mfe_pct >= 1.00,
        reached_plus_1R_before_stop=(max_high - entry) >= stop_distance,

        second_stop=second_stop,
        second_stop_touched=second_touched,
        second_stop_touch_ts_ny=iso_ny(second_touch_ts),
        second_stop_gap_fill=fill,

        recovery_target=recovery_target,
        recovery_target_after_second=recovery,
        recovery_target_ts_ny=iso_ny(recovery_ts),
        bars_second_to_recovery=bars_to_recovery,

        third_stop=third_stop,
        third_stop_hit_after_second=third_hit,
        third_stop_ts_ny=iso_ny(third_ts),

        second_strategy_state=state,
        second_strategy_exit=exit_price,
        second_strategy_return_pct=ret,
        second_strategy_R=rr,
        same_bar_ambiguous=same_bar,

        score_bucket=score_bucket(score),
        stopdist_bucket=stopdist_bucket(stop_distance_pct),
        atrpct_bucket=atrpct_bucket(atr_pct),
    )


def group_summary(rows):
    if not rows:
        return {}
    n = len(rows)

    second_touched = [r for r in rows if r.second_stop_touched]
    recovered = [r for r in second_touched if r.second_strategy_state == "RECOVERED_FIRST_STOP"]
    third = [r for r in second_touched if r.second_strategy_state == "THIRD_STOP"]
    open_end = [r for r in second_touched if r.second_strategy_state == "OPEN_END"]

    return {
        "n": n,
        "score": qstats([r.score for r in rows]),
        "original_stop_distance_pct": qstats([r.stop_distance_pct for r in rows]),
        "original_stop_distance_atr": qstats([r.stop_distance_atr for r in rows]),
        "atr_pct_of_entry": qstats([r.atr_pct for r in rows]),
        "bars_to_first_stop": qstats([r.bars_to_first_stop for r in rows]),
        "hours_to_first_stop": qstats([r.hours_to_first_stop for r in rows]),
        "mfe_before_first_stop_pct": qstats([r.mfe_before_first_stop_pct for r in rows]),
        "mfe_before_first_stop_atr": qstats([r.mfe_before_first_stop_atr for r in rows]),
        "reached_before_first_stop": {
            "+0.25%": sum(r.reached_plus_025pct_before_stop for r in rows),
            "+0.50%": sum(r.reached_plus_050pct_before_stop for r in rows),
            "+1.00%": sum(r.reached_plus_100pct_before_stop for r in rows),
            "+1R": sum(r.reached_plus_1R_before_stop for r in rows),
        },
        "reached_before_first_stop_pct": {
            "+0.25%": round(100*sum(r.reached_plus_025pct_before_stop for r in rows)/n, 2),
            "+0.50%": round(100*sum(r.reached_plus_050pct_before_stop for r in rows)/n, 2),
            "+1.00%": round(100*sum(r.reached_plus_100pct_before_stop for r in rows)/n, 2),
            "+1R": round(100*sum(r.reached_plus_1R_before_stop for r in rows)/n, 2),
        },
        "second_stop": {
            "touched": len(second_touched),
            "touched_pct_of_original_stops": round(100*len(second_touched)/n, 2),
            "recovered_to_first_stop": len(recovered),
            "third_stop_hit_first": len(third),
            "open_end": len(open_end),
            "recovery_rate_of_second_entries_pct": round(100*len(recovered)/len(second_touched), 2) if second_touched else None,
            "third_stop_rate_of_second_entries_pct": round(100*len(third)/len(second_touched), 2) if second_touched else None,
            "avg_return_pct_resolved": (
                mean([r.second_strategy_return_pct for r in recovered+third if r.second_strategy_return_pct is not None])
                if recovered or third else None
            ),
            "sum_R_resolved": sum(r.second_strategy_R for r in recovered+third if r.second_strategy_R is not None),
            "avg_R_resolved": (
                mean([r.second_strategy_R for r in recovered+third if r.second_strategy_R is not None])
                if recovered or third else None
            ),
            "bars_to_recovery": qstats([r.bars_second_to_recovery for r in recovered if r.bars_second_to_recovery is not None]),
            "same_bar_ambiguous": sum(r.same_bar_ambiguous for r in rows),
        }
    }


def print_q(label, s, suffix=""):
    if not s:
        print(f"{label}: n/a")
        return
    print(
        f"{label}: mean={s['mean']:.3f}{suffix} | median={s['median']:.3f}{suffix} | "
        f"P25={s['p25']:.3f}{suffix} | P75={s['p75']:.3f}{suffix} | P90={s['p90']:.3f}{suffix}"
    )


def print_summary(title, s):
    if not s:
        return
    print(f"\n=== {title} ===")
    print(f"Reproducible original STOPPED plans: {s['n']}")
    print_q("Original stop distance", s["original_stop_distance_pct"], "%")
    print_q("Original stop distance", s["original_stop_distance_atr"], " ATR")
    print_q("ATR / entry", s["atr_pct_of_entry"], "%")
    print_q("Time to original stop", s["hours_to_first_stop"], " h")
    print_q("STRICT MFE before original stop", s["mfe_before_first_stop_pct"], "%")
    print_q("STRICT MFE before original stop", s["mfe_before_first_stop_atr"], " ATR")

    rp = s["reached_before_first_stop_pct"]
    print(
        "Before stopping, share that had already traded up from entry: "
        f"+0.25%={rp['+0.25%']}% | +0.50%={rp['+0.50%']}% | "
        f"+1.00%={rp['+1.00%']}% | +1R={rp['+1R']}%"
    )

    x = s["second_stop"]
    print(
        f"Second-stop touched: {x['touched']} ({x['touched_pct_of_original_stops']}% of original stops)"
    )
    if x["touched"]:
        print(
            f"After buying second stop: recovered to FIRST stop={x['recovered_to_first_stop']} "
            f"({x['recovery_rate_of_second_entries_pct']}%) | "
            f"third stop hit first={x['third_stop_hit_first']} ({x['third_stop_rate_of_second_entries_pct']}%) | "
            f"open/end={x['open_end']}"
        )
        print(
            f"Resolved second-stop strategy: Sum R={x['sum_R_resolved']:.3f} | "
            f"Avg R={x['avg_R_resolved']:.3f} | Avg return={x['avg_return_pct_resolved']:.3f}%"
        )
        print_q("Bars from second-stop entry to recovery target", x["bars_to_recovery"], " bars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--second-stop-atr", type=float, default=1.2)
    ap.add_argument("--output", default="stopped_deep_analysis_v1.csv")
    ap.add_argument("--summary-output", default="stopped_deep_analysis_v1_summary.json")
    args = ap.parse_args()

    if args.days <= 0 or args.second_stop_atr <= 0:
        raise SystemExit("--days and --second-stop-atr must be > 0")

    DB = database()
    with DB() as s:
        latest = s.scalar(select(func.max(Candle.ts)))
        if not latest:
            latest = int(datetime.now(UTC).timestamp())
        end_ts = int(latest) + 900
        start_ts = end_ts - int(args.days * 86400)

        plans = s.scalars(
            select(Plan)
            .where(
                Plan.market == "US",
                Plan.state == "STOPPED",
                Plan.signal_ts >= start_ts,
                Plan.signal_ts < end_ts,
            )
            .order_by(Plan.signal_ts, Plan.symbol)
        ).all()

        rows = []
        skipped_no_repro = 0

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
            r = analyze_plan(p, bars, args.second_stop_atr)
            if r is None:
                skipped_no_repro += 1
            else:
                rows.append(r)

    if not rows:
        print("No reproducible STOPPED US recommendations found.")
        return 2

    overall = group_summary(rows)

    by_score = defaultdict(list)
    by_stopdist = defaultdict(list)
    by_atrpct = defaultdict(list)
    for r in rows:
        by_score[r.score_bucket].append(r)
        by_stopdist[r.stopdist_bucket].append(r)
        by_atrpct[r.atrpct_bucket].append(r)

    print("\nRajih — deep analysis of original STOPPED recommendations")
    print(f"Window: last {args.days:g} days ending {iso_ny(latest)}")
    print(f"Stored STOPPED plans selected: {len(plans)}")
    print(f"Reproducible from stored candles: {len(rows)} | skipped/no first-stop reproduction: {skipped_no_repro}")
    print(
        f"Second-stop definition: first stop - {args.second_stop_atr:g} ATR; "
        f"recovery target = first stop; third protective stop = second stop - {args.second_stop_atr:g} ATR"
    )
    print("MFE is STRICT: stop candle high is excluded because intrabar path is unknown.")

    print_summary("OVERALL", overall)

    print("\n=== BY ORIGINAL STOP-DISTANCE BUCKET ===")
    order_stop = ["<0.25%","0.25-0.49%","0.50-0.74%","0.75-0.99%","1.00-1.49%","1.50-1.99%","2.00%+"]
    print("bucket       | n | MFEmean% | MFE med% | +0.5% before stop | 2nd touched% | recovery% | third-stop%")
    for k in order_stop:
        rs = by_stopdist.get(k, [])
        if not rs: continue
        ss = group_summary(rs)
        print(
            f"{k:12} | {ss['n']:3d} | {ss['mfe_before_first_stop_pct']['mean']:8.3f} | "
            f"{ss['mfe_before_first_stop_pct']['median']:8.3f} | "
            f"{ss['reached_before_first_stop_pct']['+0.50%']:17.2f} | "
            f"{ss['second_stop']['touched_pct_of_original_stops']:11.2f} | "
            f"{(ss['second_stop']['recovery_rate_of_second_entries_pct'] or 0):9.2f} | "
            f"{(ss['second_stop']['third_stop_rate_of_second_entries_pct'] or 0):11.2f}"
        )

    print("\n=== BY ATR% BUCKET ===")
    order_atr = ["<0.25%","0.25-0.49%","0.50-0.74%","0.75-0.99%","1.00-1.49%","1.50%+"]
    print("bucket       | n | stop mean% | MFEmean% | 2nd touched% | recovery% | third-stop%")
    for k in order_atr:
        rs = by_atrpct.get(k, [])
        if not rs: continue
        ss = group_summary(rs)
        print(
            f"{k:12} | {ss['n']:3d} | {ss['original_stop_distance_pct']['mean']:10.3f} | "
            f"{ss['mfe_before_first_stop_pct']['mean']:8.3f} | "
            f"{ss['second_stop']['touched_pct_of_original_stops']:11.2f} | "
            f"{(ss['second_stop']['recovery_rate_of_second_entries_pct'] or 0):9.2f} | "
            f"{(ss['second_stop']['third_stop_rate_of_second_entries_pct'] or 0):11.2f}"
        )

    print("\n=== BY SCORE BUCKET ===")
    print("score | n | stop mean% | MFEmean% | 2nd touched% | recovery% | sumR")
    for k in ["<60","60-69","70-79","80-89","90+"]:
        rs = by_score.get(k, [])
        if not rs: continue
        ss = group_summary(rs)
        print(
            f"{k:5} | {ss['n']:3d} | {ss['original_stop_distance_pct']['mean']:10.3f} | "
            f"{ss['mfe_before_first_stop_pct']['mean']:8.3f} | "
            f"{ss['second_stop']['touched_pct_of_original_stops']:11.2f} | "
            f"{(ss['second_stop']['recovery_rate_of_second_entries_pct'] or 0):9.2f} | "
            f"{ss['second_stop']['sum_R_resolved']:7.3f}"
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
        "selected_stopped_plans": len(plans),
        "reproducible_rows": len(rows),
        "skipped_no_reproduction": skipped_no_repro,
        "second_stop_atr": args.second_stop_atr,
        "overall": overall,
        "by_score": {k: group_summary(v) for k,v in by_score.items()},
        "by_original_stop_distance_pct": {k: group_summary(v) for k,v in by_stopdist.items()},
        "by_atr_pct": {k: group_summary(v) for k,v in by_atrpct.items()},
        "assumptions": [
            "strict MFE excludes the first-stop candle high because OHLC cannot reveal whether high happened before or after stop",
            "second stop = original first stop - multiplier * original signal ATR",
            "second-stop entry is a resting buy limit and may get opening gap price improvement",
            "target after second-stop entry = original first stop",
            "third protective stop = second stop - same multiplier * original signal ATR",
            "if target and third stop are both possible in the same candle, third stop is assumed first (conservative)",
            "stored 15-minute candles only",
        ],
    }
    Path(args.summary_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nCSV written: {out}")
    print(f"Summary JSON written: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
