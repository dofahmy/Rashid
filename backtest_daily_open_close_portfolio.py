#!/usr/bin/env python3
"""
Rajih — same-day open-to-close portfolio backtest.

Assumption:
- Use every row in /data/daily_rule_all_dates.csv as a signal/setup date.
- On the NEXT trading session after the signal:
    Buy at that day's adjusted OPEN.
    Sell at that same day's adjusted CLOSE.
- If multiple setups occur on the same trading day, split capital equally across them.
- Start portfolio = $10,000.
- No leverage.
- No transaction costs by default.
- Also reports an optional simple cost scenario (10 bps round-trip = 0.10%).

Why next session?
The signal is based on end-of-day data. Buying that same day's open would use information
not known yet (look-ahead). So this backtest enters at the following day's open.

Outputs:
  /data/daily_intraday_open_close_trades.csv
  /data/daily_intraday_open_close_portfolio.csv
  /data/daily_intraday_open_close_summary.json
"""

import os, csv, json, math
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

from sqlalchemy import MetaData, Table, select
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_rule_all_dates.csv"

START_CAPITAL = 10_000.0
ROUND_TRIP_COST_BPS = 10.0  # 0.10% total; set to 0 if you want gross only.

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run test_daily_rule_all_501_dates_PERSISTENT.py first.")

def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except:
        return None

with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    setups = list(csv.DictReader(fh))

DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

# Cache each symbol once.
symbols = sorted(set(r["symbol"] for r in setups))
cache = {}

print(f"Loading daily bars for {len(symbols)} symbols...")
for idx, sym in enumerate(symbols, 1):
    with DB() as s:
        raw = list(s.execute(
            select(
                daily.c.session_date,
                daily.c.o,
                daily.c.c,
                daily.c.adj_c
            )
            .where(daily.c.symbol == sym)
            .order_by(daily.c.session_date)
        ).all())

    bars = []
    for d, o, c, ac in raw:
        ro = fnum(o); rc = fnum(c); adj = fnum(ac)
        if None in (ro, rc, adj) or min(ro, rc, adj) <= 0:
            continue
        fac = adj / rc
        bars.append({
            "date": str(d),
            "open": ro * fac,
            "close": adj,
        })
    cache[sym] = bars

    if idx % 100 == 0 or idx == len(symbols):
        print(f"Loaded {idx}/{len(symbols)}")

trades = []
for r in setups:
    sym = r["symbol"]
    signal_date = r["signal_date"]
    bars = cache.get(sym, [])
    dates = [b["date"] for b in bars]

    try:
        i = dates.index(signal_date)
    except ValueError:
        continue

    # Enter next session to avoid look-ahead.
    if i + 1 >= len(bars):
        continue

    b = bars[i + 1]
    op = b["open"]
    cl = b["close"]
    gross_ret = 100 * (cl / op - 1)
    net_ret = gross_ret - ROUND_TRIP_COST_BPS / 100.0

    trades.append({
        "symbol": sym,
        "signal_date": signal_date,
        "trade_date": b["date"],
        "open": round(op, 6),
        "close": round(cl, 6),
        "gross_return_pct": round(gross_ret, 6),
        "net_return_pct": round(net_ret, 6),
        "win_gross": 1 if gross_ret > 0 else 0,
        "win_net": 1 if net_ret > 0 else 0,
    })

if not trades:
    raise SystemExit("No usable trades.")

# Equal-weight all same-day trades; one portfolio return per trade date.
by_date = defaultdict(list)
for t in trades:
    by_date[t["trade_date"]].append(t)

portfolio_rows = []
capital_gross = START_CAPITAL
capital_net = START_CAPITAL
peak_gross = START_CAPITAL
peak_net = START_CAPITAL
max_dd_gross = 0.0
max_dd_net = 0.0

for d in sorted(by_date):
    ts = by_date[d]
    gross_day = mean(t["gross_return_pct"] for t in ts)
    net_day = mean(t["net_return_pct"] for t in ts)

    capital_gross *= (1 + gross_day / 100.0)
    capital_net *= (1 + net_day / 100.0)

    peak_gross = max(peak_gross, capital_gross)
    peak_net = max(peak_net, capital_net)

    dd_g = 100 * (capital_gross / peak_gross - 1)
    dd_n = 100 * (capital_net / peak_net - 1)

    max_dd_gross = min(max_dd_gross, dd_g)
    max_dd_net = min(max_dd_net, dd_n)

    portfolio_rows.append({
        "trade_date": d,
        "positions": len(ts),
        "gross_day_return_pct": round(gross_day, 6),
        "net_day_return_pct": round(net_day, 6),
        "gross_equity": round(capital_gross, 2),
        "net_equity": round(capital_net, 2),
        "gross_drawdown_pct": round(dd_g, 6),
        "net_drawdown_pct": round(dd_n, 6),
    })

gross_returns = [t["gross_return_pct"] for t in trades]
net_returns = [t["net_return_pct"] for t in trades]
gross_days = [r["gross_day_return_pct"] for r in portfolio_rows]
net_days = [r["net_day_return_pct"] for r in portfolio_rows]

summary = {
    "assumption": "signal at close, enter next trading day open, exit same day close, equal weight same-day signals",
    "start_capital": START_CAPITAL,
    "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    "signals_input": len(setups),
    "usable_trades": len(trades),
    "distinct_trade_days": len(portfolio_rows),

    "trade_level": {
        "gross_win_rate_pct": round(100 * sum(x > 0 for x in gross_returns) / len(gross_returns), 4),
        "net_win_rate_pct": round(100 * sum(x > 0 for x in net_returns) / len(net_returns), 4),
        "gross_mean_return_pct": round(mean(gross_returns), 4),
        "gross_median_return_pct": round(median(gross_returns), 4),
        "net_mean_return_pct": round(mean(net_returns), 4),
        "net_median_return_pct": round(median(net_returns), 4),
    },

    "portfolio": {
        "gross_final_equity": round(capital_gross, 2),
        "gross_total_return_pct": round(100 * (capital_gross / START_CAPITAL - 1), 4),
        "net_final_equity": round(capital_net, 2),
        "net_total_return_pct": round(100 * (capital_net / START_CAPITAL - 1), 4),
        "gross_profitable_days_pct": round(100 * sum(x > 0 for x in gross_days) / len(gross_days), 4),
        "net_profitable_days_pct": round(100 * sum(x > 0 for x in net_days) / len(net_days), 4),
        "gross_avg_day_return_pct": round(mean(gross_days), 4),
        "gross_median_day_return_pct": round(median(gross_days), 4),
        "net_avg_day_return_pct": round(mean(net_days), 4),
        "net_median_day_return_pct": round(median(net_days), 4),
        "gross_max_drawdown_pct": round(max_dd_gross, 4),
        "net_max_drawdown_pct": round(max_dd_net, 4),
    }
}

trades_csv = DATA_DIR / "daily_intraday_open_close_trades.csv"
with trades_csv.open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
    w.writeheader(); w.writerows(trades)

portfolio_csv = DATA_DIR / "daily_intraday_open_close_portfolio.csv"
with portfolio_csv.open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(portfolio_rows[0].keys()))
    w.writeheader(); w.writerows(portfolio_rows)

summary_json = DATA_DIR / "daily_intraday_open_close_summary.json"
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== OPEN -> CLOSE PORTFOLIO BACKTEST ===")
print("Entry: NEXT trading day's open after signal")
print("Exit: same day's close")
print("Weighting: equal weight among signals on the same day")
print(f"Usable trades: {summary['usable_trades']}")
print(f"Distinct trade days: {summary['distinct_trade_days']}")

print("\nTRADE LEVEL — GROSS")
print(f"Win rate: {summary['trade_level']['gross_win_rate_pct']:.2f}%")
print(f"Mean trade: {summary['trade_level']['gross_mean_return_pct']:.3f}%")
print(f"Median trade: {summary['trade_level']['gross_median_return_pct']:.3f}%")

print("\nPORTFOLIO — GROSS")
print(f"Start: ${START_CAPITAL:,.2f}")
print(f"Final: ${summary['portfolio']['gross_final_equity']:,.2f}")
print(f"Total return: {summary['portfolio']['gross_total_return_pct']:.2f}%")
print(f"Profitable trading days: {summary['portfolio']['gross_profitable_days_pct']:.2f}%")
print(f"Avg daily portfolio return: {summary['portfolio']['gross_avg_day_return_pct']:.3f}%")
print(f"Median daily portfolio return: {summary['portfolio']['gross_median_day_return_pct']:.3f}%")
print(f"Max drawdown: {summary['portfolio']['gross_max_drawdown_pct']:.2f}%")

print(f"\nPORTFOLIO — NET ({ROUND_TRIP_COST_BPS:.0f} bps round-trip)")
print(f"Final: ${summary['portfolio']['net_final_equity']:,.2f}")
print(f"Total return: {summary['portfolio']['net_total_return_pct']:.2f}%")
print(f"Profitable trading days: {summary['portfolio']['net_profitable_days_pct']:.2f}%")
print(f"Avg daily portfolio return: {summary['portfolio']['net_avg_day_return_pct']:.3f}%")
print(f"Median daily portfolio return: {summary['portfolio']['net_median_day_return_pct']:.3f}%")
print(f"Max drawdown: {summary['portfolio']['net_max_drawdown_pct']:.2f}%")

print("\nCreated:")
print(f" {trades_csv}")
print(f" {portfolio_csv}")
print(f" {summary_json}")
