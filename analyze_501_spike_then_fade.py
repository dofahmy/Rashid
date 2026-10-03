#!/usr/bin/env python3
"""
Rajih — analyze "spike then fade" behavior across the 501 all-date setups.

Reads:
    daily_rule_all_dates.csv
    market_candles_1d

Answers:
- How many touched +20% / +30% / +50% / +100% during 21 sessions?
- Of those, how many FINISHED the 21st session below the signal price?
- How many gave back most/all of the move?
- Median/mean days to:
    +20%, +30%, +50%, +100%, peak
- Median final close return after 21 sessions
- Median drawdown after the peak
- Peak-to-final giveback
- Breakdown by threshold

Outputs:
    daily_501_spike_then_fade.csv
    daily_501_spike_then_fade_summary.json
    daily_501_spike_then_fade_report.txt

Run:
    python analyze_501_spike_then_fade.py
"""

from __future__ import annotations

import os

import csv
import json
import math
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
from statistics import mean, median

from sqlalchemy import MetaData, Table, select

from core import database


INPUT = DATA_DIR / "daily_rule_all_dates.csv"


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def fmt(x, n=4):
    if x is None:
        return None
    try:
        y = float(x)
        return round(y, n) if math.isfinite(y) else None
    except Exception:
        return None


def q(vals, p):
    vals = sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
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


def load_bars(DB, daily, sym):
    with DB() as s:
        raw = list(
            s.execute(
                select(
                    daily.c.session_date,
                    daily.c.h,
                    daily.c.l,
                    daily.c.c,
                    daily.c.adj_c,
                )
                .where(daily.c.symbol == sym)
                .order_by(daily.c.session_date)
            ).all()
        )

    out = []
    for d, h, l, c, adj_c in raw:
        rc = finite(c)
        ac = finite(adj_c)
        hh = finite(h)
        ll = finite(l)
        if None in (rc, ac, hh, ll) or min(rc, ac, hh, ll) <= 0:
            continue
        factor = ac / rc
        out.append({
            "date": str(d),
            "c": ac,
            "h": hh * factor,
            "l": ll * factor,
        })
    return out


def first_day_to(bars, i, j2, start, threshold):
    for j in range(i+1, j2+1):
        if 100*(bars[j]["h"]/start - 1) >= threshold:
            return j-i
    return None


def analyze():
    p = Path(INPUT)
    if not p.exists():
        raise SystemExit(f"Missing {INPUT}. Run test_daily_rule_all_501_dates.py first.")

    with p.open(encoding="utf-8-sig", newline="") as fh:
        setups = list(csv.DictReader(fh))

    DB = database()
    with DB() as s:
        md = MetaData()
        daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

    rows = []

    for n, r in enumerate(setups, 1):
        sym = r["symbol"]
        signal_date = r["signal_date"]

        bars = load_bars(DB, daily, sym)
        dates = [x["date"] for x in bars]
        try:
            i = dates.index(signal_date)
        except ValueError:
            continue

        j2 = min(len(bars)-1, i+21)
        if j2 <= i:
            continue

        start = bars[i]["c"]
        final_close = bars[j2]["c"]

        peak_j = max(range(i+1, j2+1), key=lambda j: bars[j]["h"])
        peak_high = bars[peak_j]["h"]
        peak_gain = 100*(peak_high/start - 1)
        final_ret = 100*(final_close/start - 1)

        post_peak_min_low = min(bars[j]["l"] for j in range(peak_j, j2+1))
        post_peak_drawdown_from_peak = 100*(post_peak_min_low/peak_high - 1)

        giveback_points = peak_gain - final_ret
        giveback_pct_of_peak = None
        if peak_gain > 0:
            giveback_pct_of_peak = 100 * giveback_points / peak_gain

        x = {
            "symbol": sym,
            "signal_date": signal_date,
            "peak_gain_21d_pct": fmt(peak_gain),
            "days_to_peak": peak_j-i,
            "final_close_21d_return_pct": fmt(final_ret),
            "finished_up": int(final_ret > 0),
            "finished_down": int(final_ret < 0),
            "days_to_20": first_day_to(bars, i, j2, start, 20),
            "days_to_30": first_day_to(bars, i, j2, start, 30),
            "days_to_50": first_day_to(bars, i, j2, start, 50),
            "days_to_100": first_day_to(bars, i, j2, start, 100),
            "post_peak_drawdown_pct": fmt(post_peak_drawdown_from_peak),
            "peak_to_final_giveback_points": fmt(giveback_points),
            "giveback_pct_of_peak_gain": fmt(giveback_pct_of_peak),
        }

        for th in (20, 30, 50, 100):
            touched = peak_gain >= th
            x[f"touched_{th}"] = int(touched)
            x[f"touched_{th}_finished_down"] = int(touched and final_ret < 0)
            x[f"touched_{th}_finished_below_{th/2:.0f}"] = int(
                touched and final_ret < th/2
            )

        rows.append(x)

        if n % 100 == 0 or n == len(setups):
            print(f"Processed {n}/{len(setups)}")

    if not rows:
        raise SystemExit("No usable setups.")

    fields = list(rows[0].keys())
    with (DATA_DIR / "daily_501_spike_then_fade.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {"total": len(rows), "thresholds": {}}

    for th in (20, 30, 50, 100):
        touched_rows = [r for r in rows if r[f"touched_{th}"]]
        finished_down = [r for r in touched_rows if r["finished_down"]]
        finished_below_half = [
            r for r in touched_rows if r["final_close_21d_return_pct"] < th/2
        ]

        days = [r[f"days_to_{th}"] for r in touched_rows if r[f"days_to_{th}"] is not None]
        peak_days = [r["days_to_peak"] for r in touched_rows]
        final_rets = [r["final_close_21d_return_pct"] for r in touched_rows]
        givebacks = [r["giveback_pct_of_peak_gain"] for r in touched_rows if r["giveback_pct_of_peak_gain"] is not None]
        post_peak_dd = [r["post_peak_drawdown_pct"] for r in touched_rows if r["post_peak_drawdown_pct"] is not None]

        summary["thresholds"][str(th)] = {
            "touched_n": len(touched_rows),
            "touched_rate_pct": fmt(100*len(touched_rows)/len(rows)),
            "finished_down_n": len(finished_down),
            "finished_down_pct_of_touched": fmt(
                100*len(finished_down)/len(touched_rows) if touched_rows else None
            ),
            "finished_below_half_threshold_n": len(finished_below_half),
            "finished_below_half_threshold_pct_of_touched": fmt(
                100*len(finished_below_half)/len(touched_rows) if touched_rows else None
            ),
            "median_days_to_threshold": fmt(median(days) if days else None),
            "mean_days_to_threshold": fmt(mean(days) if days else None),
            "p25_days_to_threshold": fmt(q(days, .25)),
            "p75_days_to_threshold": fmt(q(days, .75)),
            "median_days_to_peak": fmt(median(peak_days) if peak_days else None),
            "median_final_return_pct": fmt(median(final_rets) if final_rets else None),
            "median_giveback_pct_of_peak_gain": fmt(median(givebacks) if givebacks else None),
            "median_post_peak_drawdown_pct": fmt(median(post_peak_dd) if post_peak_dd else None),
        }

    all_final = [r["final_close_21d_return_pct"] for r in rows]
    all_peak_days = [r["days_to_peak"] for r in rows]
    all_giveback = [r["giveback_pct_of_peak_gain"] for r in rows if r["giveback_pct_of_peak_gain"] is not None]

    summary["overall"] = {
        "finished_up_n": sum(r["finished_up"] for r in rows),
        "finished_down_n": sum(r["finished_down"] for r in rows),
        "median_final_return_pct": fmt(median(all_final)),
        "mean_final_return_pct": fmt(mean(all_final)),
        "median_days_to_peak": fmt(median(all_peak_days)),
        "median_giveback_pct_of_peak_gain": fmt(median(all_giveback)),
    }

    (DATA_DIR / "daily_501_spike_then_fade_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    lines = []
    lines.append("RAJIH — 501 SPIKE-THEN-FADE ANALYSIS")
    lines.append("="*72)
    lines.append(
        f"Overall: UP={summary['overall']['finished_up_n']} | "
        f"DOWN={summary['overall']['finished_down_n']} | "
        f"Median final={summary['overall']['median_final_return_pct']}% | "
        f"Median days to peak={summary['overall']['median_days_to_peak']}"
    )
    lines.append("")

    for th in (20, 30, 50, 100):
        z = summary["thresholds"][str(th)]
        lines.append(f"TOUCHED +{th}%")
        lines.append("-"*72)
        lines.append(
            f"{z['touched_n']}/{len(rows)} = {z['touched_rate_pct']}%"
        )
        lines.append(
            f"Finished DOWN after touching +{th}: "
            f"{z['finished_down_n']} = {z['finished_down_pct_of_touched']}%"
        )
        lines.append(
            f"Finished below +{th/2:.0f}% after touching +{th}: "
            f"{z['finished_below_half_threshold_n']} = "
            f"{z['finished_below_half_threshold_pct_of_touched']}%"
        )
        lines.append(
            f"Median days to +{th}: {z['median_days_to_threshold']} | "
            f"P25={z['p25_days_to_threshold']} | P75={z['p75_days_to_threshold']}"
        )
        lines.append(
            f"Median days to peak: {z['median_days_to_peak']} | "
            f"Median final return: {z['median_final_return_pct']}%"
        )
        lines.append(
            f"Median giveback of peak gain: {z['median_giveback_pct_of_peak_gain']}% | "
            f"Median post-peak drawdown: {z['median_post_peak_drawdown_pct']}%"
        )
        lines.append("")

    (DATA_DIR / "daily_501_spike_then_fade_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n=== SPIKE THEN FADE ===")
    print(
        f"Overall: UP {summary['overall']['finished_up_n']}/{len(rows)} | "
        f"DOWN {summary['overall']['finished_down_n']}/{len(rows)}"
    )
    print(
        f"Median final return: {summary['overall']['median_final_return_pct']:.2f}% | "
        f"Median days to peak: {summary['overall']['median_days_to_peak']:.2f}"
    )

    for th in (20, 30, 50, 100):
        z = summary["thresholds"][str(th)]
        print(f"\nTouched +{th}%: {z['touched_n']}/{len(rows)} = {z['touched_rate_pct']:.2f}%")
        print(
            f"Then finished DOWN: {z['finished_down_n']}/{z['touched_n']} "
            f"= {z['finished_down_pct_of_touched']:.2f}%"
        )
        print(
            f"Finished below +{th/2:.0f}%: "
            f"{z['finished_below_half_threshold_n']}/{z['touched_n']} "
            f"= {z['finished_below_half_threshold_pct_of_touched']:.2f}%"
        )
        print(
            f"Median days to +{th}: {z['median_days_to_threshold']} | "
            f"Median days to peak: {z['median_days_to_peak']}"
        )
        print(
            f"Median final return: {z['median_final_return_pct']:.2f}% | "
            f"Median giveback of peak: {z['median_giveback_pct_of_peak_gain']:.2f}%"
        )

    print("\nCreated:")
    print(" daily_501_spike_then_fade.csv")
    print(" daily_501_spike_then_fade_summary.json")
    print(" daily_501_spike_then_fade_report.txt")


if __name__ == "__main__":
    analyze()
