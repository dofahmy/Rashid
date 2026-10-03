#!/usr/bin/env python3
"""
Rajih — fixed rule detailed two-month backtest

Fixed rule:
    Entry  = Level2 - 2.0 ATR
    Stop   = Entry - 0.5 ATR
    Target = Level1
    Waiting = 78 bars

Periods:
    PREVIOUS_30D: 2026-08-03 16:00 NY -> 2026-09-02 16:00 NY
    LATEST_30D:   2026-09-02 16:00 NY -> 2026-10-02 16:00 NY

Outputs exact monthly metrics including:
- trades / wins / losses / win rate
- gross and net average winner/loser
- avg R / sum R
- total P&L with fixed $1,000 notional per trade
- commissions + slippage assumptions
- profit factor
- max drawdown
- maximum single loss
- maximum single win
- longest losing streak
- cost sensitivity

Default costs:
    slippage = 2 bps each side
    commission = 0.5 bps each side

Run:
    python backtest_fixed_rule_two_months_detailed.py --fresh

Resume:
    python backtest_fixed_rule_two_months_detailed.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

from core import database
from monitor.models import Stock, Candle
from monitor.strategy import rounded

NY = ZoneInfo("America/New_York")
US_START = 570
US_END = 960

ENTRY_OFFSET_ATR = 2.0
STOP_OFFSET_ATR = 0.5


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
    if level2 <= 0 or not (level2 < level1):
        return None

    return {
        "symbol": symbol,
        "signal_ts": ts,
        "level1": level1,
        "level2": level2,
        "atr": atr,
    }


def simulate_trade(sig, bars, start_idx, end_idx, waiting_limit):
    atr = float(sig["atr"])
    target = float(sig["level1"])

    entry_raw = float(sig["level2"]) - ENTRY_OFFSET_ATR * atr
    stop_raw = entry_raw - STOP_OFFSET_ATR * atr
    entry, _ = rounded(entry_raw, "US", False)
    stop, _ = rounded(stop_raw, "US", False)

    if stop <= 0 or not (stop < entry < target):
        return None

    fill_idx = None
    fill_price = None
    waiting_bars = 0

    max_wait_idx = min(end_idx, start_idx + waiting_limit + 1)

    for j in range(start_idx + 1, max_wait_idx):
        ts, o, h, l, c, v = bars[j]
        o = float(o); l = float(l)
        waiting_bars += 1

        if o <= stop:
            return {
                "result": "CANCELLED_GAP",
                "waiting_bars": waiting_bars,
                "fill_price": None,
                "fill_ts": None,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "exit_price": None,
                "exit_ts": int(ts),
            }

        if l <= entry:
            fill = min(o, entry) if o <= entry else entry
            fill, _ = rounded(fill, "US", True)

            if fill <= stop:
                return {
                    "result": "CANCELLED_INVALID_FILL",
                    "waiting_bars": waiting_bars,
                    "fill_price": None,
                    "fill_ts": None,
                    "entry_price": entry,
                    "stop_price": stop,
                    "target_price": target,
                    "exit_price": None,
                    "exit_ts": int(ts),
                }

            fill_idx = j
            fill_price = fill
            break

    if fill_idx is None:
        return {
            "result": "EXPIRED",
            "waiting_bars": waiting_bars,
            "fill_price": None,
            "fill_ts": None,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "exit_price": None,
            "exit_ts": None,
        }

    # Do not evaluate target/stop on fill candle.
    for j in range(fill_idx + 1, end_idx):
        ts, o, h, l, c, v = bars[j]
        o = float(o); h = float(h); l = float(l)

        # Conservative: stop first on same-bar ambiguity.
        if l <= stop:
            exit_price = min(o, stop)
            return {
                "result": "STOPPED",
                "waiting_bars": waiting_bars,
                "fill_price": fill_price,
                "fill_ts": int(bars[fill_idx][0]),
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "exit_price": exit_price,
                "exit_ts": int(ts),
            }

        if h >= target:
            return {
                "result": "TARGET",
                "waiting_bars": waiting_bars,
                "fill_price": fill_price,
                "fill_ts": int(bars[fill_idx][0]),
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "exit_price": target,
                "exit_ts": int(ts),
            }

    return {
        "result": "ACTIVE_END",
        "waiting_bars": waiting_bars,
        "fill_price": fill_price,
        "fill_ts": int(bars[fill_idx][0]),
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "exit_price": float(bars[end_idx-1][4]),
        "exit_ts": int(bars[end_idx-1][0]),
    }


def replay_symbol(symbol, bars, ref_pos, ref_dates, start_ts, end_ts, flag, atr_mult, waiting_limit):
    if len(bars) < 200:
        return []

    times = [int(b[0]) for b in bars]
    ind = indicators(bars)
    pivs = pivot_lows(bars)
    volavg = volume_baseline(bars, ref_dates)
    streak = streaks(bars, ref_pos)

    start_i = bisect.bisect_left(times, start_ts)
    end_i = bisect.bisect_left(times, end_ts)

    rows = []
    i = start_i

    # One live setup at a time per symbol.
    while i < end_i:
        sig = build_signal(
            symbol, i, bars, ind, pivs, volavg, streak,
            flag, atr_mult
        )
        if sig is None:
            i += 1
            continue

        tr = simulate_trade(sig, bars, i, end_i, waiting_limit)
        if tr is None:
            i += 1
            continue

        tr["symbol"] = symbol
        tr["signal_ts"] = sig["signal_ts"]
        tr["signal_ny"] = iso_ny(sig["signal_ts"])
        tr["atr"] = sig["atr"]
        tr["level1"] = sig["level1"]
        tr["level2"] = sig["level2"]
        rows.append(tr)

        if tr.get("exit_ts") is not None:
            i = max(i+1, bisect.bisect_right(times, int(tr["exit_ts"])))
        elif tr.get("fill_ts") is not None and tr["result"] == "ACTIVE_END":
            break
        else:
            i = min(end_i, i + waiting_limit + 1)

    return rows


def enrich_costs(trade, slippage_bps, commission_bps, capital_per_trade):
    if trade["result"] not in ("TARGET", "STOPPED"):
        return trade

    fill = float(trade["fill_price"])
    exit_px = float(trade["exit_price"])
    stop = float(trade["stop_price"])

    slip = slippage_bps / 10000.0
    comm = commission_bps / 10000.0

    adj_entry = fill * (1 + slip)
    adj_exit = exit_px * (1 - slip)

    gross_ret = exit_px / fill - 1.0
    net_ret_before_comm = adj_exit / adj_entry - 1.0

    # Commission charged on entry and exit notional.
    commission_frac = comm * (1.0 + adj_exit / adj_entry)
    net_ret = net_ret_before_comm - commission_frac

    technical_risk_pct = (fill - stop) / fill
    gross_r = gross_ret / technical_risk_pct if technical_risk_pct > 0 else None
    net_r = net_ret / technical_risk_pct if technical_risk_pct > 0 else None

    trade = dict(trade)
    trade.update({
        "gross_return_pct": gross_ret * 100.0,
        "gross_r": gross_r,
        "net_return_pct": net_ret * 100.0,
        "net_r": net_r,
        "gross_pnl_dollars": capital_per_trade * gross_ret,
        "net_pnl_dollars": capital_per_trade * net_ret,
        "slippage_cost_dollars": capital_per_trade * (
            (fill*(1+slip)-fill)/fill +
            (exit_px-exit_px*(1-slip))/fill
        ),
        "commission_cost_dollars": capital_per_trade * commission_frac,
    })
    return trade


def stats_from_resolved(resolved):
    if not resolved:
        return {}

    gross_w = [t for t in resolved if float(t["gross_pnl_dollars"]) > 0]
    gross_l = [t for t in resolved if float(t["gross_pnl_dollars"]) <= 0]
    net_w = [t for t in resolved if float(t["net_pnl_dollars"]) > 0]
    net_l = [t for t in resolved if float(t["net_pnl_dollars"]) <= 0]

    def avg(vals):
        return sum(vals)/len(vals) if vals else 0.0

    gross_profit = sum(max(0.0, float(t["gross_pnl_dollars"])) for t in resolved)
    gross_loss_abs = abs(sum(min(0.0, float(t["gross_pnl_dollars"])) for t in resolved))

    net_profit = sum(max(0.0, float(t["net_pnl_dollars"])) for t in resolved)
    net_loss_abs = abs(sum(min(0.0, float(t["net_pnl_dollars"])) for t in resolved))

    ordered = sorted(resolved, key=lambda t: (t["exit_ts"], t["symbol"]))

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    equity_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0

    streak = 0
    max_streak = 0

    for t in ordered:
        pnl = float(t["net_pnl_dollars"])
        nr = float(t["net_r"])

        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak-equity)

        equity_r += nr
        peak_r = max(peak_r, equity_r)
        max_dd_r = max(max_dd_r, peak_r-equity_r)

        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "resolved": len(resolved),
        "gross": {
            "wins": len(gross_w),
            "losses": len(gross_l),
            "win_rate_pct": 100*len(gross_w)/len(resolved),
            "avg_win_dollars": avg([float(t["gross_pnl_dollars"]) for t in gross_w]),
            "avg_loss_dollars": avg([float(t["gross_pnl_dollars"]) for t in gross_l]),
            "avg_win_pct": avg([float(t["gross_return_pct"]) for t in gross_w]),
            "avg_loss_pct": avg([float(t["gross_return_pct"]) for t in gross_l]),
            "avg_win_r": avg([float(t["gross_r"]) for t in gross_w]),
            "avg_loss_r": avg([float(t["gross_r"]) for t in gross_l]),
            "gross_profit_dollars": gross_profit,
            "gross_loss_dollars_abs": gross_loss_abs,
            "profit_factor": gross_profit/gross_loss_abs if gross_loss_abs else None,
            "total_pnl_dollars": sum(float(t["gross_pnl_dollars"]) for t in resolved),
            "avg_r": avg([float(t["gross_r"]) for t in resolved]),
            "sum_r": sum(float(t["gross_r"]) for t in resolved),
        },
        "net": {
            "wins": len(net_w),
            "losses": len(net_l),
            "win_rate_pct": 100*len(net_w)/len(resolved),
            "avg_win_dollars": avg([float(t["net_pnl_dollars"]) for t in net_w]),
            "avg_loss_dollars": avg([float(t["net_pnl_dollars"]) for t in net_l]),
            "avg_win_pct": avg([float(t["net_return_pct"]) for t in net_w]),
            "avg_loss_pct": avg([float(t["net_return_pct"]) for t in net_l]),
            "avg_win_r": avg([float(t["net_r"]) for t in net_w]),
            "avg_loss_r": avg([float(t["net_r"]) for t in net_l]),
            "gross_profit_dollars": net_profit,
            "gross_loss_dollars_abs": net_loss_abs,
            "profit_factor": net_profit/net_loss_abs if net_loss_abs else None,
            "total_pnl_dollars": sum(float(t["net_pnl_dollars"]) for t in resolved),
            "avg_r": avg([float(t["net_r"]) for t in resolved]),
            "sum_r": sum(float(t["net_r"]) for t in resolved),
            "total_slippage_cost_dollars": sum(float(t["slippage_cost_dollars"]) for t in resolved),
            "total_commission_cost_dollars": sum(float(t["commission_cost_dollars"]) for t in resolved),
            "total_costs_dollars": sum(
                float(t["slippage_cost_dollars"]) + float(t["commission_cost_dollars"])
                for t in resolved
            ),
            "max_drawdown_dollars": max_dd,
            "max_drawdown_r": max_dd_r,
            "max_consecutive_losses": max_streak,
            "max_single_loss_dollars": min(float(t["net_pnl_dollars"]) for t in resolved),
            "max_single_win_dollars": max(float(t["net_pnl_dollars"]) for t in resolved),
        }
    }


def print_month(name, st, capital_per_trade, slippage_bps, commission_bps):
    g = st["gross"]
    n = st["net"]

    print(f"\n=== {name} ===")
    print(f"Resolved trades: {st['resolved']}")
    print("\nGROSS (before costs)")
    print(
        f"Wins={g['wins']} | Losses={g['losses']} | Win rate={g['win_rate_pct']:.2f}% | "
        f"Avg win=${g['avg_win_dollars']:.2f} ({g['avg_win_pct']:.4f}%, {g['avg_win_r']:.3f}R) | "
        f"Avg loss=${g['avg_loss_dollars']:.2f} ({g['avg_loss_pct']:.4f}%, {g['avg_loss_r']:.3f}R)"
    )
    print(
        f"Profit factor={g['profit_factor']:.3f} | AvgR={g['avg_r']:.4f} | SumR={g['sum_r']:.2f} | "
        f"Total P&L=${g['total_pnl_dollars']:.2f}"
    )

    print(f"\nNET after costs (slippage={slippage_bps}bps/side, commission={commission_bps}bps/side)")
    print(
        f"Wins={n['wins']} | Losses={n['losses']} | Win rate={n['win_rate_pct']:.2f}% | "
        f"Avg win=${n['avg_win_dollars']:.2f} ({n['avg_win_pct']:.4f}%, {n['avg_win_r']:.3f}R) | "
        f"Avg loss=${n['avg_loss_dollars']:.2f} ({n['avg_loss_pct']:.4f}%, {n['avg_loss_r']:.3f}R)"
    )
    print(
        f"Profit factor={n['profit_factor']:.3f} | AvgR={n['avg_r']:.4f} | SumR={n['sum_r']:.2f} | "
        f"Total P&L=${n['total_pnl_dollars']:.2f}"
    )
    print(
        f"Total slippage cost=${n['total_slippage_cost_dollars']:.2f} | "
        f"Total commission=${n['total_commission_cost_dollars']:.2f} | "
        f"Total costs=${n['total_costs_dollars']:.2f}"
    )
    print(
        f"Max drawdown=${n['max_drawdown_dollars']:.2f} ({n['max_drawdown_r']:.2f}R) | "
        f"Max single loss=${n['max_single_loss_dollars']:.2f} | "
        f"Max single win=${n['max_single_win_dollars']:.2f} | "
        f"Longest losing streak={n['max_consecutive_losses']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atr-mult", type=float, default=1.2)
    ap.add_argument("--waiting-bars", type=int, default=78)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--capital-per-trade", type=float, default=1000.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--commission-bps", type=float, default=0.5)
    ap.add_argument("--checkpoint", default="fixed_rule_two_months_checkpoint.json")
    ap.add_argument("--summary-output", default="fixed_rule_two_months_detailed_summary.json")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    DB = database()

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
        one = 30 * 86400

        periods = [
            ("PREVIOUS_30D", end_latest - 2*one, end_latest - one),
            ("LATEST_30D", end_latest - one, end_latest),
        ]

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

    cp_path = Path(args.checkpoint)
    cp = {
        "version": 1,
        "periods": {},
    }

    if cp_path.exists() and not args.fresh:
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
            print(f"RESUME: loaded {cp_path}")
        except Exception:
            print("WARNING: bad checkpoint; starting fresh.")

    summary = {
        "rule": {
            "entry": "Level2 - 2.0 ATR",
            "stop": "Entry - 0.5 ATR",
            "target": "Level1",
            "waiting_bars": args.waiting_bars,
        },
        "costs": {
            "capital_per_trade": args.capital_per_trade,
            "slippage_bps_each_side": args.slippage_bps,
            "commission_bps_each_side": args.commission_bps,
        },
        "periods": {},
    }

    for pname, start_ts, end_ts in periods:
        print(f"\n=== RUN {pname}: {iso_ny(start_ts)} -> {iso_ny(end_ts)} ===")

        pcp = cp["periods"].setdefault(pname, {"next_index": 0, "trades": []})
        trades = list(pcp.get("trades", []))
        start_idx = int(pcp.get("next_index", 0))
        started = time.time()

        for off in range(start_idx, len(stocks), args.chunk_size):
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

            errors = 0
            for sym in chunk:
                try:
                    rs = replay_symbol(
                        sym, grouped.get(sym, []),
                        ref_pos, ref_dates,
                        start_ts, end_ts,
                        sym in flags,
                        args.atr_mult,
                        args.waiting_bars,
                    )
                    trades.extend(rs)
                except Exception as exc:
                    errors += 1
                    if errors <= 5:
                        print(f"WARNING {sym}: {type(exc).__name__}: {exc}")

            done = min(off + len(chunk), len(stocks))
            pcp["next_index"] = done
            pcp["trades"] = trades
            cp_path.write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")

            print(
                f"Progress {done}/{len(stocks)} | setups={len(trades)} | "
                f"elapsed={time.time()-started:.1f}s | checkpoint saved"
            )

        enriched = [
            enrich_costs(
                t, args.slippage_bps, args.commission_bps, args.capital_per_trade
            )
            for t in trades
        ]
        resolved = [t for t in enriched if t["result"] in ("TARGET","STOPPED")]
        st = stats_from_resolved(resolved)
        summary["periods"][pname] = {
            "start_ny": iso_ny(start_ts),
            "end_ny": iso_ny(end_ts),
            "setups_total": len(trades),
            "resolved": len(resolved),
            "states": dict(Counter(t["result"] for t in trades)),
            "stats": st,
        }
        print_month(
            pname, st,
            args.capital_per_trade,
            args.slippage_bps,
            args.commission_bps,
        )

    # Combined 60d
    combined_trades = []
    for pname in ("PREVIOUS_30D","LATEST_30D"):
        combined_trades.extend(cp["periods"][pname]["trades"])

    combined_enriched = [
        enrich_costs(t, args.slippage_bps, args.commission_bps, args.capital_per_trade)
        for t in combined_trades
    ]
    combined_resolved = [t for t in combined_enriched if t["result"] in ("TARGET","STOPPED")]
    combined_stats = stats_from_resolved(combined_resolved)

    summary["combined_60d"] = {
        "resolved": len(combined_resolved),
        "stats": combined_stats,
    }

    print_month(
        "COMBINED_60D", combined_stats,
        args.capital_per_trade,
        args.slippage_bps,
        args.commission_bps,
    )

    # Cost sensitivity
    scenarios = []
    print("\n=== COST SENSITIVITY — COMBINED 60D ===")
    for slip, comm in ((0,0),(1,0),(2,0.5),(3,0.5),(5,1.0),(10,1.0)):
        enr = [enrich_costs(t, slip, comm, args.capital_per_trade) for t in combined_trades]
        res = [t for t in enr if t["result"] in ("TARGET","STOPPED")]
        st = stats_from_resolved(res)
        scenarios.append({
            "slippage_bps_each_side": slip,
            "commission_bps_each_side": comm,
            "stats": st,
        })
        print(
            f"slip={slip}bps/side comm={comm}bps/side | "
            f"Net PF={st['net']['profit_factor']:.3f} | "
            f"Net AvgR={st['net']['avg_r']:.4f} | "
            f"Net P&L=${st['net']['total_pnl_dollars']:.2f} | "
            f"MaxDD=${st['net']['max_drawdown_dollars']:.2f} | "
            f"LosingStreak={st['net']['max_consecutive_losses']}"
        )

    summary["cost_sensitivity_combined_60d"] = scenarios

    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary JSON written: {args.summary_output}")
    print(f"Checkpoint kept at: {cp_path}")


if __name__ == "__main__":
    main()
