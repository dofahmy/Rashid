#!/usr/bin/env python3
"""
Rajih — Level2 deeper-entry grid backtest (FAST + RESUME v2, non-overlapping).

Purpose
-------
Test entering BELOW Level2 instead of at Level2 itself.

Entry candidates:
    Level2 - 0.5 ATR
    Level2 - 1.0 ATR
    Level2 - 1.5 ATR
    Level2 - 2.0 ATR

For each entry candidate, test protective stops:
    Entry - 0.5 ATR
    Entry - 1.0 ATR
    Entry - 1.5 ATR
    Entry - 2.0 ATR

Target:
    Level1 (original strategy stop)

Execution:
- Reconstruct original US 15m signals from stored candles.
- Wait up to 78 bars after the signal for the candidate entry to be touched.
- BUY LIMIT at the candidate entry price.
- Safe gap fill: if bar opens below entry but above stop, fill at open.
- If bar opens at/below stop before fill, cancel as unsafe gap.
- No target/stop evaluation on the fill candle.
- After fill, if stop and target are both touched on the same candle, count STOP first.
- Current Plan table is NOT used.

Two independent periods are tested:
A) previous 30 days
B) latest 30 days

Outputs:
    level2_deeper_entry_grid_summary_fast.json
    level2_deeper_entry_checkpoint.json

Run:
    python backtest_level2_deeper_entries_fast.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections import Counter, defaultdict
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

ENTRY_OFFSETS = (2.0,)
STOP_OFFSETS = (0.5,)


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
        "ema20": e20,
        "ema50": e50,
        "macd": macd,
        "macd_signal": macd_sig,
        "atr": atr,
        "rsi": rsi,
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
        d, m = local_parts(b[0])
        by[(d, m)] = float(b[5])

    out = {}
    for di, d in enumerate(ref_dates):
        if di < 20:
            continue
        prior = ref_dates[di-20:di]
        for m in range(US_START, US_END, 15):
            vals = []
            ok = True
            for pd in prior:
                v = by.get((pd, m))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if not ok:
                continue
            avg = sum(vals) / 20.0
            if avg <= 0 or (d, m) not in by:
                continue
            dt = datetime(d.year, d.month, d.day, m//60, m%60, tzinfo=NY)
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


def build_signal(symbol, i, bars, ind, pivs, volavg, streak, flag, atr_mult):
    if flag or i < 199 or streak[i] < 21:
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

    original_entry, _ = rounded(float(h) + 0.05*atr, "US", True)

    right = bisect.bisect_right(pivs, i-2) - 1
    support_idx = None
    limit = min(float(c), original_entry)
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
    level1, _ = rounded(support - 0.25*atr, "US", False)
    risk = original_entry - level1
    if level1 <= 0 or risk <= 0:
        return None

    eligible = False
    for pct in (3, 4, 5):
        tgt, _ = rounded(original_entry*(1+pct/100.0), "US", True)
        if (tgt-original_entry)/risk >= 1.5:
            eligible = True
            break
    if not eligible:
        return None

    level2, _ = rounded(level1 - atr_mult*atr, "US", False)
    level3, _ = rounded(level2 - atr_mult*atr, "US", False)
    if level3 <= 0 or not (level3 < level2 < level1):
        return None

    return {
        "symbol": symbol,
        "signal_ts": ts,
        "original_entry": original_entry,
        "level1": level1,
        "level2": level2,
        "level3": level3,
        "atr": atr,
    }


def simulate_trade(sig, bars, start_idx, end_idx, entry_offset, stop_offset, waiting_limit):
    atr = float(sig["atr"])
    target = float(sig["level1"])

    entry_raw = float(sig["level2"]) - entry_offset * atr
    stop_raw = entry_raw - stop_offset * atr
    entry, _ = rounded(entry_raw, "US", False)
    stop, _ = rounded(stop_raw, "US", False)

    if stop <= 0 or not (stop < entry < target):
        return None

    fill_idx = None
    fill_price = None
    waiting_bars = 0
    state = "WAITING"

    max_wait_idx = min(end_idx, start_idx + waiting_limit + 1)

    for j in range(start_idx + 1, max_wait_idx):
        ts, o, h, l, c, v = bars[j]
        o = float(o); h = float(h); l = float(l); c = float(c)
        waiting_bars += 1

        # Unsafe gap through stop before fill.
        if o <= stop:
            return {
                "result": "CANCELLED_GAP",
                "entry_offset_atr": entry_offset,
                "stop_offset_atr": stop_offset,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "waiting_bars": waiting_bars,
                "fill_price": None,
                "fill_ts": None,
                "exit_price": None,
                "exit_ts": int(ts),
                "bars_after_fill": 0,
                "pnl_r": None,
                "return_pct": None,
            }

        if l <= entry:
            fill = min(o, entry) if o <= entry else entry
            fill, _ = rounded(fill, "US", True)

            if fill <= stop:
                return {
                    "result": "CANCELLED_INVALID_FILL",
                    "entry_offset_atr": entry_offset,
                    "stop_offset_atr": stop_offset,
                    "entry_price": entry,
                    "stop_price": stop,
                    "target_price": target,
                    "waiting_bars": waiting_bars,
                    "fill_price": None,
                    "fill_ts": None,
                    "exit_price": None,
                    "exit_ts": int(ts),
                    "bars_after_fill": 0,
                    "pnl_r": None,
                    "return_pct": None,
                }

            fill_idx = j
            fill_price = fill
            state = "ACTIVE"
            break

    if fill_idx is None:
        return {
            "result": "EXPIRED",
            "entry_offset_atr": entry_offset,
            "stop_offset_atr": stop_offset,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "waiting_bars": waiting_bars,
            "fill_price": None,
            "fill_ts": None,
            "exit_price": None,
            "exit_ts": None,
            "bars_after_fill": 0,
            "pnl_r": None,
            "return_pct": None,
        }

    # Evaluate only from NEXT candle after fill.
    for j in range(fill_idx + 1, end_idx):
        ts, o, h, l, c, v = bars[j]
        o = float(o); h = float(h); l = float(l)
        bars_after = j - fill_idx

        # Stop-first on ambiguous bar.
        if l <= stop:
            exit_price = min(o, stop)
            risk = fill_price - stop
            pnl_r = (exit_price - fill_price) / risk if risk > 0 else None
            ret = 100 * (exit_price / fill_price - 1)
            return {
                "result": "STOPPED",
                "entry_offset_atr": entry_offset,
                "stop_offset_atr": stop_offset,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "waiting_bars": waiting_bars,
                "fill_price": fill_price,
                "fill_ts": int(bars[fill_idx][0]),
                "exit_price": exit_price,
                "exit_ts": int(ts),
                "bars_after_fill": bars_after,
                "pnl_r": pnl_r,
                "return_pct": ret,
            }

        if h >= target:
            exit_price = target
            risk = fill_price - stop
            pnl_r = (exit_price - fill_price) / risk if risk > 0 else None
            ret = 100 * (exit_price / fill_price - 1)
            return {
                "result": "TARGET",
                "entry_offset_atr": entry_offset,
                "stop_offset_atr": stop_offset,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "waiting_bars": waiting_bars,
                "fill_price": fill_price,
                "fill_ts": int(bars[fill_idx][0]),
                "exit_price": exit_price,
                "exit_ts": int(ts),
                "bars_after_fill": bars_after,
                "pnl_r": pnl_r,
                "return_pct": ret,
            }

    # End of period.
    last_close = float(bars[end_idx-1][4]) if end_idx > 0 else None
    risk = fill_price - stop
    pnl_r = ((last_close-fill_price)/risk) if (last_close is not None and risk > 0) else None
    ret = 100*(last_close/fill_price-1) if last_close is not None else None

    return {
        "result": "ACTIVE_END",
        "entry_offset_atr": entry_offset,
        "stop_offset_atr": stop_offset,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "waiting_bars": waiting_bars,
        "fill_price": fill_price,
        "fill_ts": int(bars[fill_idx][0]),
        "exit_price": last_close,
        "exit_ts": int(bars[end_idx-1][0]) if end_idx > 0 else None,
        "bars_after_fill": end_idx - 1 - fill_idx if end_idx > fill_idx else 0,
        "pnl_r": pnl_r,
        "return_pct": ret,
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

    rows = []

    for i in range(start_i, end_i):
        sig = build_signal(symbol, i, bars, ind, pivs, volavg, streak, flag, atr_mult)
        if sig is None:
            continue

        for entry_offset in ENTRY_OFFSETS:
            for stop_offset in STOP_OFFSETS:
                tr = simulate_trade(
                    sig, bars, i, end_i,
                    entry_offset, stop_offset, waiting_limit
                )
                if tr is None:
                    continue

                rows.append({
                    "symbol": symbol,
                    "signal_ts": sig["signal_ts"],
                    "signal_ny": iso_ny(sig["signal_ts"]),
                    "level1": sig["level1"],
                    "level2": sig["level2"],
                    "level3": sig["level3"],
                    "atr": sig["atr"],
                    **tr,
                    "fill_ny": iso_ny(tr["fill_ts"]),
                    "exit_ny": iso_ny(tr["exit_ts"]),
                })

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
        return vals[lo]*(hi-k) + vals[hi]*(k-lo)

    return {
        "mean": mean(vals),
        "median": q(.5),
        "p75": q(.75),
        "p90": q(.90),
    }



def new_agg():
    return {
        "signals": 0,
        "filled": 0,
        "resolved": 0,
        "target": 0,
        "stopped": 0,
        "sum_r": 0.0,
        "sum_return": 0.0,
        "resolved_with_r": 0,
        "resolved_with_return": 0,
        "waiting_bars_values": [],
        "bars_after_fill_values": [],
        "states": Counter(),
    }


def add_trade_to_agg(agg, tr):
    agg["signals"] += 1
    agg["states"][tr["result"]] += 1

    if tr["fill_price"] is not None:
        agg["filled"] += 1
        agg["waiting_bars_values"].append(tr["waiting_bars"])

    if tr["result"] in ("TARGET", "STOPPED"):
        agg["resolved"] += 1
        if tr["result"] == "TARGET":
            agg["target"] += 1
        else:
            agg["stopped"] += 1

        agg["bars_after_fill_values"].append(tr["bars_after_fill"])

        if tr["pnl_r"] is not None:
            agg["sum_r"] += float(tr["pnl_r"])
            agg["resolved_with_r"] += 1

        if tr["return_pct"] is not None:
            agg["sum_return"] += float(tr["return_pct"])
            agg["resolved_with_return"] += 1


def finalize_agg(entry_offset, stop_offset, agg):
    resolved = agg["resolved"]
    return {
        "entry_offset_atr": entry_offset,
        "stop_offset_atr": stop_offset,
        "signals": agg["signals"],
        "filled": agg["filled"],
        "fill_rate_pct": round(100*agg["filled"]/agg["signals"], 2) if agg["signals"] else None,
        "resolved": resolved,
        "target": agg["target"],
        "stopped": agg["stopped"],
        "win_rate_pct": round(100*agg["target"]/resolved, 2) if resolved else None,
        "avg_r": round(agg["sum_r"]/agg["resolved_with_r"], 3) if agg["resolved_with_r"] else None,
        "sum_r": round(agg["sum_r"], 3),
        "avg_return_pct": round(
            agg["sum_return"]/agg["resolved_with_return"], 3
        ) if agg["resolved_with_return"] else None,
        "waiting_bars": qstats(agg["waiting_bars_values"]),
        "bars_after_fill": qstats(agg["bars_after_fill_values"]),
        "states": dict(agg["states"]),
    }


def serialize_agg_map(agg_map):
    out = {}
    for (eo, so), agg in agg_map.items():
        key = f"{eo:.1f}|{so:.1f}"
        out[key] = {
            "signals": agg["signals"],
            "filled": agg["filled"],
            "resolved": agg["resolved"],
            "target": agg["target"],
            "stopped": agg["stopped"],
            "sum_r": agg["sum_r"],
            "sum_return": agg["sum_return"],
            "resolved_with_r": agg["resolved_with_r"],
            "resolved_with_return": agg["resolved_with_return"],
            "waiting_bars_values": agg["waiting_bars_values"],
            "bars_after_fill_values": agg["bars_after_fill_values"],
            "states": dict(agg["states"]),
        }
    return out


def deserialize_agg_map(data):
    agg_map = {}
    for eo in ENTRY_OFFSETS:
        for so in STOP_OFFSETS:
            key = f"{eo:.1f}|{so:.1f}"
            raw = data.get(key)
            if raw is None:
                agg_map[(eo, so)] = new_agg()
                continue
            agg = new_agg()
            for k in (
                "signals","filled","resolved","target","stopped",
                "sum_r","sum_return","resolved_with_r","resolved_with_return"
            ):
                agg[k] = raw.get(k, agg[k])
            agg["waiting_bars_values"] = raw.get("waiting_bars_values", [])
            agg["bars_after_fill_values"] = raw.get("bars_after_fill_values", [])
            agg["states"] = Counter(raw.get("states", {}))
            agg_map[(eo, so)] = agg
    return agg_map


def print_table(period_name, summaries):
    print(f"\n=== {period_name} — GRID RESULTS ===")
    ordered = sorted(
        summaries,
        key=lambda x: (
            x["avg_r"] if x["avg_r"] is not None else -999,
            x["sum_r"] if x["sum_r"] is not None else -999,
        ),
        reverse=True,
    )

    for i, s in enumerate(ordered, 1):
        print(
            f"{i:2d}. Entry=L2-{s['entry_offset_atr']:.1f}ATR | "
            f"Stop=Entry-{s['stop_offset_atr']:.1f}ATR | "
            f"Filled={s['filled']} ({s['fill_rate_pct']}%) | "
            f"Resolved={s['resolved']} | Win={s['win_rate_pct']}% | "
            f"AvgR={s['avg_r']} | SumR={s['sum_r']} | "
            f"AvgReturn={s['avg_return_pct']}%"
        )
        if s["bars_after_fill"]:
            b = s["bars_after_fill"]
            print(
                f"    Close time after fill: mean={b['mean']*0.25:.2f}h | "
                f"median={b['median']*0.25:.2f}h | "
                f"P75={b['p75']*0.25:.2f}h | P90={b['p90']*0.25:.2f}h"
            )




def process_symbol_nonoverlap(
    symbol, bars, ref_pos, ref_dates, start_ts, end_ts, flag,
    atr_mult, waiting_limit, agg_map
):
    """
    Correct replay: for EACH entry/stop combo, allow only one live setup at a time
    for a symbol. This mirrors the earlier reconstructed backtests and prevents
    creating a fresh overlapping signal on every eligible 15m candle.
    """
    if len(bars) < 200:
        return

    times = [int(b[0]) for b in bars]
    ind = indicators(bars)
    pivs = pivot_lows(bars)
    volavg = volume_baseline(bars, ref_dates)
    streak = streaks(bars, ref_pos)

    start_i = bisect.bisect_left(times, start_ts)
    end_i = bisect.bisect_left(times, end_ts)

    combos = [(eo, so) for eo in ENTRY_OFFSETS for so in STOP_OFFSETS]
    next_free_i = {combo: start_i for combo in combos}

    # Cache original signal reconstruction once per bar.
    signal_cache = {}

    for i in range(start_i, end_i):
        needed = [combo for combo in combos if i >= next_free_i[combo]]
        if not needed:
            continue

        sig = signal_cache.get(i, "__MISSING__")
        if sig == "__MISSING__":
            sig = build_signal(
                symbol, i, bars, ind, pivs, volavg, streak,
                flag, atr_mult
            )
            signal_cache[i] = sig

        if sig is None:
            continue

        for eo, so in needed:
            tr = simulate_trade(
                sig, bars, i, end_i,
                eo, so, waiting_limit
            )
            if tr is None:
                continue

            add_trade_to_agg(agg_map[(eo, so)], tr)

            # Block new signals for this combo until the current setup has ended.
            # For EXPIRED/CANCELLED with no exit_ts, waiting_limit is the end.
            if tr.get("exit_ts") is not None:
                j_end = bisect.bisect_right(times, int(tr["exit_ts"]))
            elif tr.get("fill_ts") is not None and tr.get("result") == "ACTIVE_END":
                j_end = end_i
            else:
                j_end = min(end_i, i + waiting_limit + 1)

            next_free_i[(eo, so)] = max(i + 1, j_end)



def compute_equity_stats(trades, slippage_bps=0.0, commission_bps=0.0, capital_per_trade=1000.0):
    """
    Reprice resolved trades with simple execution costs:
      - slippage_bps applied adversely on entry and exit
      - commission_bps applied on both entry and exit notional
    Returns net R, net %, dollar P&L, max drawdown and losing streak.
    """
    rows = []
    for tr in trades:
        if tr["result"] not in ("TARGET", "STOPPED"):
            continue
        fill = float(tr["fill_price"])
        exit_px = float(tr["exit_price"])
        stop = float(tr["stop_price"])
        if fill <= 0 or stop >= fill:
            continue

        slip = slippage_bps / 10000.0
        comm = commission_bps / 10000.0

        # Long trade: worse entry = higher; worse exit = lower.
        adj_entry = fill * (1 + slip)
        adj_exit = exit_px * (1 - slip)

        gross_ret = adj_exit / adj_entry - 1.0
        commission_cost = comm * (1.0 + adj_exit / adj_entry)
        net_ret = gross_ret - commission_cost

        # R is based on original technical risk distance.
        risk_pct = (fill - stop) / fill
        net_r = net_ret / risk_pct if risk_pct > 0 else None

        pnl_dollars = capital_per_trade * net_ret

        rows.append({
            **tr,
            "net_return_pct": net_ret * 100.0,
            "net_r": net_r,
            "pnl_dollars": pnl_dollars,
        })

    rows.sort(key=lambda x: (x["exit_ts"] or 0, x["symbol"]))

    equity_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0
    equity_d = 0.0
    peak_d = 0.0
    max_dd_d = 0.0
    losing_streak = 0
    max_losing_streak = 0

    for r in rows:
        nr = float(r["net_r"])
        pd = float(r["pnl_dollars"])

        equity_r += nr
        peak_r = max(peak_r, equity_r)
        max_dd_r = max(max_dd_r, peak_r - equity_r)

        equity_d += pd
        peak_d = max(peak_d, equity_d)
        max_dd_d = max(max_dd_d, peak_d - equity_d)

        if nr < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    vals_r = [float(r["net_r"]) for r in rows]
    vals_ret = [float(r["net_return_pct"]) for r in rows]
    wins = [r for r in rows if float(r["net_r"]) > 0]

    return {
        "resolved": len(rows),
        "net_win_rate_pct": round(100*len(wins)/len(rows), 2) if rows else None,
        "avg_net_r": round(sum(vals_r)/len(vals_r), 4) if vals_r else None,
        "sum_net_r": round(sum(vals_r), 3) if vals_r else None,
        "avg_net_return_pct": round(sum(vals_ret)/len(vals_ret), 4) if vals_ret else None,
        "total_pnl_dollars": round(sum(float(r["pnl_dollars"]) for r in rows), 2),
        "max_drawdown_r": round(max_dd_r, 3),
        "max_drawdown_dollars": round(max_dd_d, 2),
        "max_consecutive_losses": max_losing_streak,
    }


def run_fixed_combo_symbol(
    symbol, bars, ref_pos, ref_dates, start_ts, end_ts, flag,
    atr_mult, waiting_limit
):
    """
    One live setup at a time for the fixed combo:
      Entry = L2 - 2.0 ATR
      Stop  = Entry - 0.5 ATR
    """
    if len(bars) < 200:
        return []

    times = [int(b[0]) for b in bars]
    ind = indicators(bars)
    pivs = pivot_lows(bars)
    volavg = volume_baseline(bars, ref_dates)
    streak = streaks(bars, ref_pos)

    start_i = bisect.bisect_left(times, start_ts)
    end_i = bisect.bisect_left(times, end_ts)

    results = []
    i = start_i

    while i < end_i:
        sig = build_signal(
            symbol, i, bars, ind, pivs, volavg, streak,
            flag, atr_mult
        )
        if sig is None:
            i += 1
            continue

        tr = simulate_trade(
            sig, bars, i, end_i,
            2.0, 0.5, waiting_limit
        )
        if tr is None:
            i += 1
            continue

        tr["symbol"] = symbol
        tr["signal_ts"] = sig["signal_ts"]
        tr["signal_ny"] = iso_ny(sig["signal_ts"])
        tr["atr"] = sig["atr"]
        tr["level1"] = sig["level1"]
        tr["level2"] = sig["level2"]
        results.append(tr)

        if tr.get("exit_ts") is not None:
            i = max(i + 1, bisect.bisect_right(times, int(tr["exit_ts"])))
        elif tr.get("fill_ts") is not None and tr.get("result") == "ACTIVE_END":
            break
        else:
            i = min(end_i, i + waiting_limit + 1)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-days", type=int, default=30)
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--slippage-bps", type=float, default=2.0,
                    help="Adverse slippage in basis points on EACH side. Default 2 bps.")
    ap.add_argument("--commission-bps", type=float, default=0.5,
                    help="Commission in basis points on EACH side. Default 0.5 bps.")
    ap.add_argument("--capital-per-trade", type=float, default=1000.0,
                    help="Fixed notional per trade for dollar P&L stats.")
    ap.add_argument("--checkpoint", default="third_holdout_l2minus2_checkpoint.json")
    ap.add_argument("--summary-output", default="third_holdout_l2minus2_summary.json")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    DB = database()
    started = time.time()

    with DB() as s:
        ref = list(
            s.execute(
                select(Candle.ts,Candle.o,Candle.h,Candle.l,Candle.c,Candle.v)
                .where(Candle.symbol == "AAPL")
                .order_by(Candle.ts)
            ).all()
        )
        if not ref:
            raise SystemExit("AAPL reference candles missing")

        latest = int(ref[-1][0])
        end_latest = latest + 900
        one = args.period_days * 86400

        # TRUE THIRD HOLDOUT: the 30 calendar days immediately BEFORE the two
        # periods already used in discovery/validation.
        start_ts = end_latest - 3*one
        end_ts = end_latest - 2*one
        period_name = "THIRD_HOLDOUT_30D"

        ref_times = [int(r[0]) for r in ref]
        ref_pos = {ts:i for i,ts in enumerate(ref_times)}
        ref_dates = sorted({local_parts(ts)[0] for ts in ref_times})

        stocks = list(
            s.execute(
                select(Stock.symbol)
                .where(Stock.market == "US")
                .order_by(Stock.symbol)
            ).scalars().all()
        )

    flags_path = Path(__file__).resolve().parent / "monitor" / "data" / "corporate_flags.json"
    try:
        flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except Exception:
        flags = {}

    checkpoint_path = Path(args.checkpoint)
    checkpoint = {
        "version": 1,
        "completed": 0,
        "trades": [],
        "period": period_name,
        "start_ny": iso_ny(start_ts),
        "end_ny": iso_ny(end_ts),
    }

    if checkpoint_path.exists() and not args.fresh:
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            print(f"RESUME: loaded checkpoint, completed {checkpoint.get('completed',0)}/{len(stocks)} symbols")
        except Exception:
            print("WARNING: checkpoint unreadable; starting fresh.")

    print("\nRajih — TRUE THIRD HOLDOUT")
    print(f"Period: {iso_ny(start_ts)} -> {iso_ny(end_ts)}")
    print(f"US universe: {len(stocks)}")
    print("FIXED BEFORE TEST:")
    print("  Entry = Level2 - 2.0 ATR")
    print("  Stop  = Entry - 0.5 ATR")
    print("  Target = Level1")
    print(f"  Waiting = {args.waiting_bars} bars")
    print(f"Costs: slippage={args.slippage_bps} bps/side | commission={args.commission_bps} bps/side")
    print(f"Fixed notional per trade: ${args.capital_per_trade:,.2f}")

    done_until = int(checkpoint.get("completed", 0))
    trades = list(checkpoint.get("trades", []))
    errors = 0

    for off in range(done_until, len(stocks), args.chunk_size):
        chunk = stocks[off:off+args.chunk_size]

        with DB() as s:
            raw = list(
                s.execute(
                    select(
                        Candle.symbol,Candle.ts,Candle.o,Candle.h,
                        Candle.l,Candle.c,Candle.v
                    )
                    .where(Candle.symbol.in_(chunk))
                    .order_by(Candle.symbol,Candle.ts)
                ).all()
            )

        grouped = defaultdict(list)
        for sym,ts,o,h,l,c,v in raw:
            grouped[sym].append(
                (int(ts),float(o),float(h),float(l),float(c),float(v))
            )

        for sym in chunk:
            try:
                trs = run_fixed_combo_symbol(
                    sym, grouped.get(sym, []),
                    ref_pos, ref_dates,
                    start_ts, end_ts,
                    sym in flags,
                    args.atr_mult,
                    args.waiting_bars,
                )
                trades.extend(trs)
            except Exception as exc:
                errors += 1
                if errors <= 10:
                    print(f"WARNING {sym}: {type(exc).__name__}: {exc}")

        done = min(off + len(chunk), len(stocks))
        checkpoint["completed"] = done
        checkpoint["trades"] = trades
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Progress {done}/{len(stocks)} | trades={len(trades)} | "
            f"errors={errors} | elapsed={time.time()-started:.1f}s | checkpoint saved"
        )

    # Raw technical result, no costs.
    resolved = [t for t in trades if t["result"] in ("TARGET","STOPPED")]
    targets = [t for t in resolved if t["result"] == "TARGET"]
    stops = [t for t in resolved if t["result"] == "STOPPED"]
    rvals = [float(t["pnl_r"]) for t in resolved if t["pnl_r"] is not None]
    rets = [float(t["return_pct"]) for t in resolved if t["return_pct"] is not None]

    gross = {
        "signals_or_setups": len(trades),
        "resolved": len(resolved),
        "target": len(targets),
        "stopped": len(stops),
        "win_rate_pct": round(100*len(targets)/len(resolved),2) if resolved else None,
        "avg_r": round(sum(rvals)/len(rvals),4) if rvals else None,
        "sum_r": round(sum(rvals),3) if rvals else None,
        "avg_return_pct": round(sum(rets)/len(rets),4) if rets else None,
    }

    net = compute_equity_stats(
        trades,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        capital_per_trade=args.capital_per_trade,
    )

    # Cost sensitivity table.
    cost_scenarios = []
    for slip, comm in ((0,0),(1,0),(2,0.5),(3,0.5),(5,1.0),(10,1.0)):
        s = compute_equity_stats(
            trades, slippage_bps=slip, commission_bps=comm,
            capital_per_trade=args.capital_per_trade
        )
        s["slippage_bps_each_side"] = slip
        s["commission_bps_each_side"] = comm
        cost_scenarios.append(s)

    print("\n=== THIRD HOLDOUT — GROSS ===")
    print(
        f"Resolved={gross['resolved']} | Target={gross['target']} | Stop={gross['stopped']} | "
        f"Win={gross['win_rate_pct']}% | AvgR={gross['avg_r']} | "
        f"SumR={gross['sum_r']} | AvgReturn={gross['avg_return_pct']}%"
    )

    print("\n=== THIRD HOLDOUT — WITH DEFAULT COSTS ===")
    print(
        f"NetWin={net['net_win_rate_pct']}% | AvgNetR={net['avg_net_r']} | "
        f"SumNetR={net['sum_net_r']} | AvgNetReturn={net['avg_net_return_pct']}%"
    )
    print(
        f"MaxDD={net['max_drawdown_r']}R | "
        f"MaxDD(${args.capital_per_trade:,.0f}/trade)=${net['max_drawdown_dollars']:,.2f} | "
        f"Max consecutive losses={net['max_consecutive_losses']}"
    )
    print(
        f"Total P&L at ${args.capital_per_trade:,.0f} fixed notional/trade: "
        f"${net['total_pnl_dollars']:,.2f}"
    )

    print("\n=== COST SENSITIVITY ===")
    for s in cost_scenarios:
        print(
            f"slip={s['slippage_bps_each_side']}bps/side "
            f"comm={s['commission_bps_each_side']}bps/side | "
            f"AvgNetR={s['avg_net_r']} | SumNetR={s['sum_net_r']} | "
            f"AvgNetReturn={s['avg_net_return_pct']}% | "
            f"MaxDD={s['max_drawdown_r']}R"
        )

    payload = {
        "period": period_name,
        "start_ny": iso_ny(start_ts),
        "end_ny": iso_ny(end_ts),
        "fixed_system": {
            "entry": "LEVEL2 - 2.0 ATR",
            "stop": "ENTRY - 0.5 ATR",
            "target": "LEVEL1",
            "waiting_bars": args.waiting_bars,
        },
        "gross": gross,
        "default_costs": {
            "slippage_bps_each_side": args.slippage_bps,
            "commission_bps_each_side": args.commission_bps,
            "capital_per_trade": args.capital_per_trade,
            **net,
        },
        "cost_sensitivity": cost_scenarios,
        "notes": [
            "This is the third 30-day period preceding the two periods already used.",
            "One live setup per symbol at a time.",
            "Same-bar stop+target after fill is stop-first.",
            "No target/stop evaluation on fill candle.",
            "Dollar P&L assumes fixed notional per trade and ignores portfolio capital overlap.",
        ],
    }

    Path(args.summary_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary JSON written: {args.summary_output}")
    print(f"Checkpoint kept at: {checkpoint_path}")
    print(f"Total elapsed: {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
