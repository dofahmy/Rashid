#!/usr/bin/env python3
"""
Rajih — Level2 / Level3 excursion-to-recovery analysis.

Question
--------
If we DO NOT enter immediately at Level2 or Level3, after touching either level:
- how far below that level does price typically travel before later reaching target (Stop1)?
- what ATR depth covers 70/75/80/85/90/95% of the trades that eventually recover?
- for each depth threshold (0.5, 1, 1.5 ... 8 ATR), what % of ALL touched trades
  eventually recovered to target without first exceeding that depth?

Definitions
-----------
Level1 = original strategy stop = recovery target
Level2 = Level1 - 1.2 ATR
Level3 = Level2 - 1.2 ATR

For each reconstructed original US signal:
- wait up to 78 bars for first Level2 touch
- once Level2 is touched, track:
    * whether Level1 is later reached
    * maximum adverse excursion below Level2 before recovery
    * recovery time
- if Level3 is touched, separately track the same statistics from Level3

The analysis is purely observational. It does NOT assume an entry at Level2 or Level3.

Two independent 30-day periods are analyzed plus a combined 60-day view.

Run:
    python analyze_level2_level3_recovery_depth.py

Outputs:
    level2_level3_recovery_depth_rows.csv
    level2_level3_recovery_depth_summary.json
"""

from __future__ import annotations

import argparse, bisect, csv, json, math, time
from collections import defaultdict
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
    macd = [x-y for x,y in zip(e12,e26)]
    macd_sig = ema_series(macd, 9)

    atr = [None]*n
    rsi = [None]*n
    if n >= 15:
        trs, ch = [], []
        for i in range(1,n):
            trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
            ch.append(c[i]-c[i-1])
        av = sum(trs[:14])/14.0
        g = sum(max(x,0) for x in ch[:14])/14.0
        d = sum(max(-x,0) for x in ch[:14])/14.0
        atr[14] = av
        rsi[14] = 100-100/(1+g/d) if d else (100.0 if g else 50.0)
        for i in range(15,n):
            av = (av*13 + trs[i-1])/14.0
            g = (g*13 + max(ch[i-1],0))/14.0
            d = (d*13 + max(-ch[i-1],0))/14.0
            atr[i] = av
            rsi[i] = 100-100/(1+g/d) if d else (100.0 if g else 50.0)

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
        if x <= min(left+right) and x < max(left) and x < max(right):
            out.append(i)
    return out


def volume_baseline(bars, ref_dates):
    by = {}
    for b in bars:
        d,m = local_parts(b[0])
        by[(d,m)] = float(b[5])
    out = {}
    for di,d in enumerate(ref_dates):
        if di < 20:
            continue
        prior = ref_dates[di-20:di]
        for m in range(US_START, US_END, 15):
            vals = []
            ok = True
            for pd in prior:
                v = by.get((pd,m))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if not ok:
                continue
            avg = sum(vals)/20.0
            if avg <= 0 or (d,m) not in by:
                continue
            dt = datetime(d.year,d.month,d.day,m//60,m%60,tzinfo=NY)
            out[int(dt.timestamp())] = avg
    return out


def streaks(bars, ref_pos):
    out = [0]*len(bars)
    prev = None
    cur = 0
    for i,b in enumerate(bars):
        pos = ref_pos.get(int(b[0]))
        if pos is None:
            cur = 0
            prev = None
        else:
            cur = cur+1 if prev is not None and pos == prev+1 else 1
            prev = pos
        out[i] = cur
    return out


def build_signal(symbol, i, bars, ind, pivs, volavg, streak, flag, atr_mult):
    if flag or i < 199 or streak[i] < 21:
        return None

    atr = ind["atr"][i]
    rsi = ind["rsi"][i]
    if atr is None or rsi is None or atr <= 0:
        return None

    ts,o,h,l,c,v = bars[i]
    ts = int(ts)
    v = float(v)
    avg = volavg.get(ts)
    if avg is None or avg <= 0 or v <= 0:
        return None
    if not (35 <= rsi <= 75):
        return None
    if not (ind["ema20"][i] >= ind["ema50"][i] or ind["macd"][i] >= ind["macd_signal"][i]):
        return None

    entry,_ = rounded(float(h) + 0.05*atr, "US", True)

    right = bisect.bisect_right(pivs, i-2)-1
    support_idx = None
    limit = min(float(c), entry)
    while right >= 0:
        j = pivs[right]
        if j < i-79:
            break
        if float(bars[j][3]) < limit:
            support_idx = j
            break
        right -= 1
    if support_idx is None:
        return None

    support = float(bars[support_idx][3])
    level1,_ = rounded(support - 0.25*atr, "US", False)
    risk = entry - level1
    if level1 <= 0 or risk <= 0:
        return None

    ok = False
    for pct in (3,4,5):
        tgt,_ = rounded(entry*(1+pct/100.0), "US", True)
        if (tgt-entry)/risk >= 1.5:
            ok = True
            break
    if not ok:
        return None

    level2,_ = rounded(level1 - atr_mult*atr, "US", False)
    level3,_ = rounded(level2 - atr_mult*atr, "US", False)
    if level3 <= 0 or not (level3 < level2 < level1):
        return None

    return {
        "symbol": symbol,
        "signal_ts": ts,
        "entry": entry,
        "level1": level1,
        "level2": level2,
        "level3": level3,
        "atr": atr,
    }


def quantiles(vals):
    vals = sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
    if not vals:
        return {}

    def q(p):
        if len(vals) == 1:
            return vals[0]
        k = (len(vals)-1)*p
        lo,hi = math.floor(k), math.ceil(k)
        if lo == hi:
            return vals[lo]
        return vals[lo]*(hi-k) + vals[hi]*(k-lo)

    out = {
        "mean": mean(vals),
        "median": q(.50),
        "p70": q(.70),
        "p75": q(.75),
        "p80": q(.80),
        "p85": q(.85),
        "p90": q(.90),
        "p95": q(.95),
    }
    return out


def analyze_level(level_name, level_px, level_touch_pos, bars, ref_pos, target, atr, period_end_pos):
    """
    Starting from first touch of a level, observe until target or period end.
    """
    min_low = float("inf")
    recovered = False
    recovery_bars = None
    recovery_ts = None

    # Include the touch candle itself.
    start_pos = level_touch_pos
    for j in range(start_pos, min(len(bars), period_end_pos)):
        ts,o,h,l,c,v = bars[j]
        l = float(l); h = float(h)
        if l < min_low:
            min_low = l

        bars_since = j - start_pos

        # Conservative: if same candle both reaches deeper and target, we still
        # record the full candle low as adverse excursion.
        if h >= target:
            recovered = True
            recovery_bars = bars_since
            recovery_ts = int(ts)
            break

    if min_low == float("inf"):
        min_low = None

    depth_abs = (level_px - min_low) if min_low is not None else None
    depth_atr = (depth_abs / atr) if depth_abs is not None and atr > 0 else None
    depth_pct = (100*depth_abs/level_px) if depth_abs is not None and level_px > 0 else None

    return {
        "level": level_name,
        "touched": True,
        "recovered": recovered,
        "recovery_bars": recovery_bars,
        "recovery_hours": recovery_bars*0.25 if recovery_bars is not None else None,
        "recovery_ts": recovery_ts,
        "min_low_before_recovery_or_end": min_low,
        "depth_abs": depth_abs,
        "depth_atr": depth_atr,
        "depth_pct": depth_pct,
    }


def replay_symbol(symbol, bars, ref_pos, ref_dates, start_ts, end_ts, flag, atr_mult, waiting_limit):
    if len(bars) < 200:
        return []

    bars = [tuple(b) for b in bars if int(b[0]) <= end_ts]
    times = [int(b[0]) for b in bars]
    ind = indicators(bars)
    pivs = pivot_lows(bars)
    volavg = volume_baseline(bars, ref_dates)
    streak = streaks(bars, ref_pos)

    start_i = bisect.bisect_left(times, start_ts)
    end_i = bisect.bisect_left(times, end_ts)

    out = []
    i = start_i
    while i < end_i:
        sig = build_signal(symbol, i, bars, ind, pivs, volavg, streak, flag, atr_mult)
        if sig is None:
            i += 1
            continue

        # Wait up to waiting_limit bars for first Level2 touch.
        touch2_idx = None
        max_j = min(end_i, i + waiting_limit + 1)
        for j in range(i+1, max_j):
            if float(bars[j][3]) <= sig["level2"]:
                touch2_idx = j
                break

        if touch2_idx is None:
            i += 1
            continue

        a2 = analyze_level(
            "LEVEL2", sig["level2"], touch2_idx, bars, ref_pos,
            sig["level1"], sig["atr"], end_i
        )

        # First Level3 touch, if any, starting from Level2 touch.
        touch3_idx = None
        for j in range(touch2_idx, end_i):
            if float(bars[j][3]) <= sig["level3"]:
                touch3_idx = j
                break
            if float(bars[j][2]) >= sig["level1"]:
                # Target reached before Level3; no Level3-touch sample.
                break

        row2 = {
            "symbol": symbol,
            "signal_ny": iso_ny(sig["signal_ts"]),
            "level": "LEVEL2",
            "level_price": sig["level2"],
            "target_price": sig["level1"],
            "atr": sig["atr"],
            "touch_ny": iso_ny(int(bars[touch2_idx][0])),
            **a2,
        }
        out.append(row2)

        if touch3_idx is not None:
            a3 = analyze_level(
                "LEVEL3", sig["level3"], touch3_idx, bars, ref_pos,
                sig["level1"], sig["atr"], end_i
            )
            row3 = {
                "symbol": symbol,
                "signal_ny": iso_ny(sig["signal_ts"]),
                "level": "LEVEL3",
                "level_price": sig["level3"],
                "target_price": sig["level1"],
                "atr": sig["atr"],
                "touch_ny": iso_ny(int(bars[touch3_idx][0])),
                **a3,
            }
            out.append(row3)

        # Avoid immediately creating overlapping duplicate signals while this
        # observational path is active. Move forward to target recovery if it
        # happened, otherwise at least one bar.
        if a2["recovered"] and a2["recovery_ts"] is not None:
            next_i = bisect.bisect_right(times, a2["recovery_ts"])
            i = max(i+1, next_i)
        else:
            i += 1

    return out


def summarize(rows, level):
    rs = [r for r in rows if r["level"] == level]
    rec = [r for r in rs if r["recovered"]]
    nonrec = [r for r in rs if not r["recovered"]]

    depths_rec = [r["depth_atr"] for r in rec if r["depth_atr"] is not None]
    times_rec = [r["recovery_hours"] for r in rec if r["recovery_hours"] is not None]

    depth_thresholds = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8]
    threshold_table = []
    for d in depth_thresholds:
        # recovered without exceeding d ATR below touched level
        count = sum(
            r["recovered"] and r["depth_atr"] is not None and r["depth_atr"] <= d
            for r in rs
        )
        threshold_table.append({
            "depth_atr": d,
            "recovered_count": count,
            "pct_of_all_touched": round(100*count/len(rs),2) if rs else None,
            "pct_of_eventual_recoveries": round(100*count/len(rec),2) if rec else None,
        })

    return {
        "level": level,
        "touched": len(rs),
        "recovered": len(rec),
        "not_recovered": len(nonrec),
        "recovery_pct_all_touched": round(100*len(rec)/len(rs),2) if rs else None,
        "depth_atr_eventual_recoveries": quantiles(depths_rec),
        "recovery_hours_eventual_recoveries": quantiles(times_rec),
        "threshold_table": threshold_table,
    }


def print_summary(period, s):
    print(f"\n[{period}] {s['level']}")
    print(
        f"Touched={s['touched']} | Later reached target={s['recovered']} "
        f"({s['recovery_pct_all_touched']}%) | Not recovered={s['not_recovered']}"
    )

    q = s["depth_atr_eventual_recoveries"]
    if q:
        print("ATR depth below touched level BEFORE later recovery to target:")
        print(
            f"  P70={q['p70']:.2f} | P75={q['p75']:.2f} | P80={q['p80']:.2f} | "
            f"P85={q['p85']:.2f} | P90={q['p90']:.2f} | P95={q['p95']:.2f}"
        )
        print(
            f"  median={q['median']:.2f} | mean={q['mean']:.2f}"
        )

    t = s["recovery_hours_eventual_recoveries"]
    if t:
        print(
            f"Recovery time: median={t['median']:.2f}h | P75={t['p75']:.2f}h | "
            f"P80={t['p80']:.2f}h | P90={t['p90']:.2f}h"
        )

    print("Depth threshold -> recoveries captured:")
    for x in s["threshold_table"]:
        print(
            f"  <= {x['depth_atr']:>3} ATR : "
            f"{x['recovered_count']:>5} | "
            f"{x['pct_of_all_touched']:>6}% of ALL touched | "
            f"{x['pct_of_eventual_recoveries']:>6}% of eventual recoveries"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-days", type=int, default=30)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--output", default="level2_level3_recovery_depth_rows.csv")
    ap.add_argument("--summary-output", default="level2_level3_recovery_depth_summary.json")
    args = ap.parse_args()

    DB = database()
    started = time.time()

    with DB() as s:
        ref = list(
            s.execute(
                select(Candle.ts,Candle.o,Candle.h,Candle.l,Candle.c,Candle.v)
                .where(Candle.symbol=="AAPL")
                .order_by(Candle.ts)
            ).all()
        )
        if not ref:
            raise SystemExit("AAPL reference candles missing")

        latest = int(ref[-1][0])
        end_latest = latest + 900
        one = args.period_days*86400
        periods = [
            ("PREVIOUS_30D", end_latest-2*one, end_latest-one),
            ("LATEST_30D", end_latest-one, end_latest),
        ]

        ref_times = [int(r[0]) for r in ref]
        ref_pos = {ts:i for i,ts in enumerate(ref_times)}
        ref_dates = sorted({local_parts(ts)[0] for ts in ref_times})

        stocks = list(
            s.execute(
                select(Stock.symbol).where(Stock.market=="US").order_by(Stock.symbol)
            ).scalars().all()
        )

    flags_path = Path(__file__).resolve().parent/"monitor"/"data"/"corporate_flags.json"
    try:
        flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except Exception:
        flags = {}

    print("\nRajih — LEVEL2 / LEVEL3 recovery-depth analysis")
    print(f"Latest candle: {iso_ny(latest)}")
    print(f"US universe: {len(stocks)}")
    print(f"Levels: L2=L1-{args.atr_mult:g}ATR | L3=L2-{args.atr_mult:g}ATR | target=L1")
    print("No entry is assumed at L2 or L3; this is observational.")

    all_rows = []
    summaries = {}

    for period_name,start_ts,end_ts in periods:
        print(f"\n=== PERIOD {period_name}: {iso_ny(start_ts)} -> {iso_ny(end_ts)} ===")
        period_rows = []

        for off in range(0,len(stocks),args.chunk_size):
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
                try:
                    rows = replay_symbol(
                        sym, grouped.get(sym,[]), ref_pos, ref_dates,
                        start_ts, end_ts, sym in flags,
                        args.atr_mult, args.waiting_bars
                    )
                    for r in rows:
                        r["period"] = period_name
                    period_rows.extend(rows)
                    all_rows.extend(rows)
                except Exception as exc:
                    print(f"WARNING {sym}: {type(exc).__name__}: {exc}")

            done = min(off+len(chunk),len(stocks))
            if done % 500 < args.chunk_size or done == len(stocks):
                print(f"Progress {done}/{len(stocks)} | rows={len(period_rows)} | elapsed={time.time()-started:.1f}s")

        summaries[period_name] = {}
        for level in ("LEVEL2","LEVEL3"):
            s = summarize(period_rows, level)
            summaries[period_name][level] = s
            print_summary(period_name, s)

    print("\n=== COMBINED 60-DAY ===")
    combined = {}
    for level in ("LEVEL2","LEVEL3"):
        s = summarize(all_rows, level)
        combined[level] = s
        print_summary("COMBINED_60D", s)

    if all_rows:
        fields = list(all_rows[0].keys())
        with Path(args.output).open("w",newline="",encoding="utf-8-sig") as f:
            w = csv.DictWriter(f,fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    payload = {
        "latest_reference_ny": iso_ny(latest),
        "parameters": vars(args),
        "summaries": summaries,
        "combined_60d": combined,
        "notes": [
            "P80 means 80% of eventual recoveries stayed within that ATR depth below the touched level",
            "pct_of_all_touched is the unconditional percentage of all touches that both stayed within the depth and later recovered",
            "no entry is assumed at Level2 or Level3",
            "read-only",
        ],
    }
    Path(args.summary_output).write_text(
        json.dumps(payload,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )

    print(f"\nCSV written: {args.output}")
    print(f"Summary JSON written: {args.summary_output}")
    print(f"Total elapsed: {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
