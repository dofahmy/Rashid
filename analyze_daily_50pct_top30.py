#!/usr/bin/env python3
"""
Rajih — discover 30 sub-$20 US stocks that gained >=50% within ~1 trading month,
then profile what they looked like BEFORE the move.

Data source
-----------
Reads ONLY the already stored table:
    market_candles_1d

No web requests.
No changes to the trading bot.
No changes to stored candles.

Event definition
----------------
For every eligible daily bar:
- start price = that session's CLOSE
- stock must be < $20 at the start
- look forward up to 21 trading sessions
- event qualifies if the maximum future HIGH reaches >= +50%
- require enough prior history for indicators
- choose ONE strongest qualifying event per symbol
- rank distinct symbols by forward return
- keep top 30 distinct stocks

Why HIGH?
---------
The question is whether the stock "rose at least 50% within a month".
Using the future daily HIGH captures that the price actually traded there.
The script also reports forward max CLOSE return separately.

Pre-move snapshots
------------------
For each selected event it calculates indicators at:
- 1 trading month before start: t-20
- 5 trading days before start: t-5
- 1 trading day before start: t-1

It also calculates aggregates over the PRECEDING 20 sessions ending t-1.

Features
--------
Price / trend:
- Close
- 5d / 20d returns
- SMA20, SMA50, SMA200
- distance from SMA20/50/200
- EMA20, EMA50, EMA20-vs-EMA50 spread
- 20d high/low position
- distance to 20d high

Volatility:
- ATR14
- ATR14 as % of price
- 20d realized daily-return volatility
- 20d average range %

Momentum:
- RSI14
- MACD(12,26), signal(9), MACD gap
- ADX14

Volume:
- current volume
- avg volume 20
- volume ratio vs avg20
- dollar volume
- avg dollar volume 20

Outputs
-------
daily_50pct_top30.csv
daily_50pct_top30_summary.json
daily_50pct_commonality.csv

Run:
    python analyze_daily_50pct_top30.py

Optional:
    python analyze_daily_50pct_top30.py --top 30 --max-price 20 --min-gain 50
    python analyze_daily_50pct_top30.py --lookahead 21
"""

from __future__ import annotations

import os

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
from statistics import mean, median

from sqlalchemy import MetaData, Table, select

from core import database


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return 100.0 * (float(a) / float(b) - 1.0)


def sma(vals, i, n):
    if i + 1 < n:
        return None
    xs = vals[i+1-n:i+1]
    if any(x is None for x in xs):
        return None
    return sum(xs) / n


def ema_all(vals, n):
    out = [None] * len(vals)
    clean_start = None
    for i, x in enumerate(vals):
        if x is not None:
            clean_start = i
            break
    if clean_start is None:
        return out
    alpha = 2.0 / (n + 1.0)
    e = float(vals[clean_start])
    out[clean_start] = e
    for i in range(clean_start + 1, len(vals)):
        x = vals[i]
        if x is None:
            out[i] = e
        else:
            e = alpha * float(x) + (1-alpha) * e
            out[i] = e
    return out


def atr_all(h, l, c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    tr = [None] * len(c)
    for i in range(1, len(c)):
        tr[i] = max(
            h[i]-l[i],
            abs(h[i]-c[i-1]),
            abs(l[i]-c[i-1]),
        )
    seed = tr[1:n+1]
    if any(x is None for x in seed):
        return out
    a = sum(seed)/n
    out[n] = a
    for i in range(n+1, len(c)):
        a = ((n-1)*a + tr[i]) / n
        out[i] = a
    return out


def rsi_all(c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    gains, losses = [], []
    for i in range(1, len(c)):
        d = c[i]-c[i-1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:n])/n
    al = sum(losses[:n])/n
    out[n] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100 - 100/(1+ag/al))
    for i in range(n+1, len(c)):
        ag = ((n-1)*ag + gains[i-1]) / n
        al = ((n-1)*al + losses[i-1]) / n
        out[i] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100 - 100/(1+ag/al))
    return out


def adx_all(h, l, c, n=14):
    m = len(c)
    out = [None] * m
    if m < 2*n + 1:
        return out

    tr = [0.0] * m
    pdm = [0.0] * m
    mdm = [0.0] * m
    for i in range(1, m):
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        mdm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))

    atr_sum = sum(tr[1:n+1])
    p_sum = sum(pdm[1:n+1])
    m_sum = sum(mdm[1:n+1])
    dx = [None] * m

    for i in range(n, m):
        if i > n:
            atr_sum = atr_sum - atr_sum/n + tr[i]
            p_sum = p_sum - p_sum/n + pdm[i]
            m_sum = m_sum - m_sum/n + mdm[i]
        if atr_sum <= 0:
            continue
        pdi = 100*p_sum/atr_sum
        mdi = 100*m_sum/atr_sum
        denom = pdi + mdi
        dx[i] = 0.0 if denom == 0 else 100*abs(pdi-mdi)/denom

    seed = [x for x in dx[n:2*n] if x is not None]
    if len(seed) < n:
        return out
    a = sum(seed)/n
    out[2*n-1] = a
    for i in range(2*n, m):
        if dx[i] is not None:
            a = ((n-1)*a + dx[i]) / n
            out[i] = a
    return out


def stdev(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs) < 2:
        return None
    mu = sum(xs)/len(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))


def q(values, p):
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
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


def fmt(x, n=4):
    return None if x is None else round(float(x), n)


def build_features(rows):
    dates = [r["session_date"] for r in rows]
    o = [float(r["o"]) for r in rows]
    h = [float(r["h"]) for r in rows]
    l = [float(r["l"]) for r in rows]
    c = [float(r["c"]) for r in rows]
    v = [float(r["v"]) for r in rows]

    e12 = ema_all(c, 12)
    e20 = ema_all(c, 20)
    e26 = ema_all(c, 26)
    e50 = ema_all(c, 50)
    macd = [None if e12[i] is None or e26[i] is None else e12[i]-e26[i] for i in range(len(c))]
    macd_clean = [0.0 if x is None else x for x in macd]
    macd_sig = ema_all(macd_clean, 9)
    atr14 = atr_all(h, l, c, 14)
    rsi14 = rsi_all(c, 14)
    adx14 = adx_all(h, l, c, 14)

    sma20 = [sma(c, i, 20) for i in range(len(c))]
    sma50 = [sma(c, i, 50) for i in range(len(c))]
    sma200 = [sma(c, i, 200) for i in range(len(c))]

    avgvol20 = [sma(v, i, 20) for i in range(len(v))]
    dollar = [c[i]*v[i] for i in range(len(c))]
    avgdol20 = [sma(dollar, i, 20) for i in range(len(c))]

    ret1 = [None] * len(c)
    for i in range(1, len(c)):
        ret1[i] = c[i]/c[i-1]-1.0

    vol20 = [None] * len(c)
    avg_range20 = [None] * len(c)
    high20 = [None] * len(c)
    low20 = [None] * len(c)
    for i in range(len(c)):
        if i >= 19:
            vol20[i] = stdev(ret1[i-19:i+1])
            avg_range20[i] = mean(100*(h[j]-l[j])/c[j] for j in range(i-19, i+1) if c[j] > 0)
            high20[i] = max(h[i-19:i+1])
            low20[i] = min(l[i-19:i+1])

    return {
        "date": dates, "o": o, "h": h, "l": l, "c": c, "v": v,
        "ema20": e20, "ema50": e50,
        "macd": macd, "macd_signal": macd_sig,
        "atr14": atr14, "rsi14": rsi14, "adx14": adx14,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "avgvol20": avgvol20, "avgdol20": avgdol20,
        "retvol20": vol20, "avg_range20": avg_range20,
        "high20": high20, "low20": low20,
    }


def snapshot(f, i):
    if i < 0 or i >= len(f["c"]):
        return {}
    c = f["c"][i]
    atr = f["atr14"][i]
    av = f["avgvol20"][i]
    s20, s50, s200 = f["sma20"][i], f["sma50"][i], f["sma200"][i]
    e20, e50 = f["ema20"][i], f["ema50"][i]
    macd, msig = f["macd"][i], f["macd_signal"][i]
    hi20, lo20 = f["high20"][i], f["low20"][i]

    pos20 = None
    if hi20 is not None and lo20 is not None and hi20 > lo20:
        pos20 = 100*(c-lo20)/(hi20-lo20)

    return {
        "date": f["date"][i],
        "close": fmt(c),
        "ret_5d_pct": fmt(pct(c, f["c"][i-5]) if i >= 5 else None),
        "ret_20d_pct": fmt(pct(c, f["c"][i-20]) if i >= 20 else None),
        "atr14": fmt(atr),
        "atr14_pct": fmt(100*atr/c if atr is not None and c > 0 else None),
        "rsi14": fmt(f["rsi14"][i]),
        "adx14": fmt(f["adx14"][i]),
        "sma20": fmt(s20),
        "sma50": fmt(s50),
        "sma200": fmt(s200),
        "dist_sma20_pct": fmt(pct(c, s20)),
        "dist_sma50_pct": fmt(pct(c, s50)),
        "dist_sma200_pct": fmt(pct(c, s200)),
        "ema20": fmt(e20),
        "ema50": fmt(e50),
        "ema20_vs_ema50_pct": fmt(pct(e20, e50)),
        "macd": fmt(macd),
        "macd_signal": fmt(msig),
        "macd_gap": fmt(macd-msig if macd is not None and msig is not None else None),
        "volume": fmt(f["v"][i], 0),
        "avg_volume20": fmt(av, 0),
        "volume_ratio20": fmt(f["v"][i]/av if av and av > 0 else None),
        "dollar_volume": fmt(c*f["v"][i], 0),
        "avg_dollar_volume20": fmt(f["avgdol20"][i], 0),
        "realized_vol20_pct": fmt(100*f["retvol20"][i] if f["retvol20"][i] is not None else None),
        "avg_range20_pct": fmt(f["avg_range20"][i]),
        "position_in_20d_range_pct": fmt(pos20),
        "distance_to_20d_high_pct": fmt(pct(c, hi20)),
    }


def prior20_aggregate(f, start_i):
    a = max(0, start_i-20)
    b = start_i
    if b-a < 20:
        return {}
    idx = list(range(a, b))
    closes = [f["c"][i] for i in idx]
    vols = [f["v"][i] for i in idx]
    atrp = [
        100*f["atr14"][i]/f["c"][i]
        for i in idx if f["atr14"][i] is not None and f["c"][i] > 0
    ]
    rsis = [f["rsi14"][i] for i in idx if f["rsi14"][i] is not None]
    adxs = [f["adx14"][i] for i in idx if f["adx14"][i] is not None]
    volrat = [
        f["v"][i]/f["avgvol20"][i]
        for i in idx if f["avgvol20"][i] is not None and f["avgvol20"][i] > 0
    ]
    return {
        "month_avg_atr_pct": fmt(mean(atrp) if atrp else None),
        "month_median_atr_pct": fmt(median(atrp) if atrp else None),
        "month_avg_rsi": fmt(mean(rsis) if rsis else None),
        "month_avg_adx": fmt(mean(adxs) if adxs else None),
        "month_avg_volume_ratio": fmt(mean(volrat) if volrat else None),
        "month_avg_volume": fmt(mean(vols) if vols else None, 0),
        "month_price_change_pct": fmt(pct(closes[-1], closes[0]) if closes else None),
        "month_low_to_high_pct": fmt(100*(max(closes)/min(closes)-1) if closes and min(closes)>0 else None),
    }


def flatten(prefix, d, out):
    for k, v in d.items():
        out[f"{prefix}_{k}"] = v


def summarize_numeric(rows, columns):
    out = []
    for col in columns:
        vals = []
        for r in rows:
            x = r.get(col)
            try:
                x = float(x)
                if math.isfinite(x):
                    vals.append(x)
            except (TypeError, ValueError):
                pass
        if not vals:
            continue
        out.append({
            "metric": col,
            "n": len(vals),
            "mean": fmt(mean(vals)),
            "median": fmt(median(vals)),
            "p25": fmt(q(vals, .25)),
            "p75": fmt(q(vals, .75)),
            "min": fmt(min(vals)),
            "max": fmt(max(vals)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-price", type=float, default=20.0)
    ap.add_argument("--min-gain", type=float, default=50.0)
    ap.add_argument("--lookahead", type=int, default=21)
    ap.add_argument("--min-history", type=int, default=220)
    ap.add_argument("--output", default=str(DATA_DIR / "daily_50pct_top30.csv"))
    ap.add_argument("--summary-output", default=str(DATA_DIR / "daily_50pct_top30_summary.json"))
    ap.add_argument("--commonality-output", default=str(DATA_DIR / "daily_50pct_commonality.csv"))
    args = ap.parse_args()

    if args.top <= 0 or args.lookahead <= 0:
        raise SystemExit("Invalid positive parameter")

    DB = database()
    with DB() as s:
        md = MetaData()
        daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

    with DB() as s:
        symbols = list(
            s.execute(
                select(daily.c.symbol)
                .group_by(daily.c.symbol)
                .having(__import__("sqlalchemy").func.count() >= args.min_history)
                .order_by(daily.c.symbol)
            ).scalars().all()
        )

    print("\nRajih — Daily +50% move discovery")
    print(f"Eligible symbols with >= {args.min_history} daily bars: {len(symbols)}")
    print(f"Start price < ${args.max_price:g}")
    print(f"Forward move >= {args.min_gain:g}% within {args.lookahead} trading sessions")
    print("Selecting one strongest event per symbol, then top distinct symbols.\n")

    candidates = []
    for n, sym in enumerate(symbols, 1):
        with DB() as s:
            raw = list(
                s.execute(
                    select(
                        daily.c.session_date,
                        daily.c.o, daily.c.h, daily.c.l,
                        daily.c.c, daily.c.v
                    )
                    .where(daily.c.symbol == sym)
                    .order_by(daily.c.session_date)
                ).all()
            )
        rows = [
            {
                "session_date": str(x[0]),
                "o": float(x[1]), "h": float(x[2]),
                "l": float(x[3]), "c": float(x[4]),
                "v": float(x[5]),
            }
            for x in raw
        ]
        if len(rows) < args.min_history:
            continue

        f = build_features(rows)
        best = None

        # Need 200-day indicators available before the event.
        first_i = max(args.min_history-1, 200)
        last_i = len(rows) - 2
        for i in range(first_i, last_i+1):
            start_close = f["c"][i]
            if not (0 < start_close < args.max_price):
                continue

            j2 = min(len(rows)-1, i + args.lookahead)
            if j2 <= i:
                continue

            future_highs = f["h"][i+1:j2+1]
            if not future_highs:
                continue

            max_high = max(future_highs)
            max_high_rel_idx = future_highs.index(max_high) + 1
            peak_i = i + max_high_rel_idx
            gain_high = 100*(max_high/start_close - 1.0)
            if gain_high < args.min_gain:
                continue

            future_closes = f["c"][i+1:j2+1]
            max_close = max(future_closes)
            gain_close = 100*(max_close/start_close - 1.0)

            event = {
                "symbol": sym,
                "start_i": i,
                "peak_i": peak_i,
                "start_date": f["date"][i],
                "peak_date": f["date"][peak_i],
                "start_close": fmt(start_close),
                "peak_high": fmt(max_high),
                "forward_max_high_gain_pct": fmt(gain_high),
                "forward_max_close_gain_pct": fmt(gain_close),
                "trading_days_to_peak": peak_i-i,
            }
            if best is None or gain_high > best["forward_max_high_gain_pct"]:
                best = event

        if best:
            best["_features"] = f
            candidates.append(best)

        if n % 250 == 0 or n == len(symbols):
            print(f"Scanned {n}/{len(symbols)} | qualifying distinct symbols={len(candidates)}")

    candidates.sort(key=lambda x: x["forward_max_high_gain_pct"], reverse=True)
    selected = candidates[:args.top]

    out_rows = []
    for rank, ev in enumerate(selected, 1):
        f = ev.pop("_features")
        i = ev["start_i"]

        row = {
            "rank": rank,
            "symbol": ev["symbol"],
            "move_start_date": ev["start_date"],
            "peak_date": ev["peak_date"],
            "start_close": ev["start_close"],
            "peak_high": ev["peak_high"],
            "forward_max_high_gain_pct": ev["forward_max_high_gain_pct"],
            "forward_max_close_gain_pct": ev["forward_max_close_gain_pct"],
            "trading_days_to_peak": ev["trading_days_to_peak"],
        }

        flatten("t_minus_20", snapshot(f, i-20), row)
        flatten("t_minus_5", snapshot(f, i-5), row)
        flatten("t_minus_1", snapshot(f, i-1), row)
        row.update(prior20_aggregate(f, i))
        out_rows.append(row)

    if not out_rows:
        print("No qualifying events found.")
        return 2

    fields = list(out_rows[0].keys())
    with Path(args.output).open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    numeric_cols = [
        x for x in fields
        if x not in {
            "rank","symbol","move_start_date","peak_date",
            "t_minus_20_date","t_minus_5_date","t_minus_1_date"
        }
    ]
    common = summarize_numeric(out_rows, numeric_cols)
    with Path(args.commonality_output).open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["metric","n","mean","median","p25","p75","min","max"]
        )
        w.writeheader()
        w.writerows(common)

    def prop(pred):
        return round(100*sum(1 for r in out_rows if pred(r))/len(out_rows), 2)

    summary = {
        "definition": {
            "distinct_symbols": True,
            "price_rule": f"start close < ${args.max_price:g}",
            "move_rule": f"future max HIGH >= +{args.min_gain:g}% within {args.lookahead} trading sessions",
            "one_event_per_symbol": "strongest qualifying event",
            "selection": f"top {args.top} distinct symbols by forward max HIGH return",
        },
        "selected_count": len(out_rows),
        "qualifying_distinct_symbols_total": len(candidates),
        "move_stats": {
            "mean_gain_high_pct": fmt(mean(r["forward_max_high_gain_pct"] for r in out_rows)),
            "median_gain_high_pct": fmt(median(r["forward_max_high_gain_pct"] for r in out_rows)),
            "mean_days_to_peak": fmt(mean(r["trading_days_to_peak"] for r in out_rows)),
            "median_days_to_peak": fmt(median(r["trading_days_to_peak"] for r in out_rows)),
        },
        "common_patterns_t_minus_1": {
            "pct_close_above_sma20": prop(lambda r: (r.get("t_minus_1_dist_sma20_pct") or -999) > 0),
            "pct_close_above_sma50": prop(lambda r: (r.get("t_minus_1_dist_sma50_pct") or -999) > 0),
            "pct_close_above_sma200": prop(lambda r: (r.get("t_minus_1_dist_sma200_pct") or -999) > 0),
            "pct_ema20_above_ema50": prop(lambda r: (r.get("t_minus_1_ema20_vs_ema50_pct") or -999) > 0),
            "pct_macd_above_signal": prop(lambda r: (r.get("t_minus_1_macd_gap") or -999) > 0),
            "pct_rsi_above_50": prop(lambda r: (r.get("t_minus_1_rsi14") or -999) > 50),
            "pct_adx_above_20": prop(lambda r: (r.get("t_minus_1_adx14") or -999) >= 20),
            "pct_volume_ratio_above_1": prop(lambda r: (r.get("t_minus_1_volume_ratio20") or -999) >= 1),
            "pct_volume_ratio_above_1_5": prop(lambda r: (r.get("t_minus_1_volume_ratio20") or -999) >= 1.5),
            "pct_atr_pct_above_5": prop(lambda r: (r.get("t_minus_1_atr14_pct") or -999) >= 5),
            "pct_within_10pct_of_20d_high": prop(lambda r: (r.get("t_minus_1_distance_to_20d_high_pct") or -999) >= -10),
        },
        "files": {
            "events": args.output,
            "commonality": args.commonality_output,
        },
    }

    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== TOP 30 FOUND ===")
    print(f"Qualifying distinct symbols total: {len(candidates)}")
    print(f"Selected: {len(out_rows)}")
    print("\nrank | symbol | start | start$ | peak | gain% | days")
    for r in out_rows:
        print(
            f"{r['rank']:>4} | {r['symbol']:<7} | {r['move_start_date']} | "
            f"{r['start_close']:>7.2f} | {r['peak_high']:>8.2f} | "
            f"{r['forward_max_high_gain_pct']:>7.2f}% | {r['trading_days_to_peak']:>4}"
        )

    print("\n=== COMMON PATTERNS AT T-1 ===")
    for k, v in summary["common_patterns_t_minus_1"].items():
        print(f"{k}: {v:.2f}%")

    print(f"\nDetailed events CSV: {args.output}")
    print(f"Commonality CSV: {args.commonality_output}")
    print(f"Summary JSON: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
