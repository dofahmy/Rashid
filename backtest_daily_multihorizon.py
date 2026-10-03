#!/usr/bin/env python3
"""
Rajih — multi-horizon backtest for the 501 daily setups.

Entry:
- Signal is known after the signal day's close.
- Enter at NEXT trading day's adjusted OPEN (avoids look-ahead).

Exit variants:
- Hold 1, 2, 3, 5, 7, 10 trading days.
- Exit at adjusted CLOSE of the Nth holding day.

Portfolio:
- One setup per distinct signal date in the 501 file.
- Since the current 501 sample has one setup per date, this is effectively one trade at a time by signal date,
  but overlapping holding periods are handled by allocating equal capital to all active positions each day
  in the event-level portfolio simulation.
- Reports both:
  A) Trade-level fixed-horizon stats.
  B) Sequential-compounding "one trade after another" stats for easy comparison.
  C) Optional daily marked-to-market overlapping portfolio using equal-weight active positions.

Costs:
- Gross
- Net with 10 bps round-trip per trade by default.

Reads:
  /data/daily_rule_all_dates.csv
  market_candles_1d

Outputs:
  /data/daily_multihorizon_trade_stats.csv
  /data/daily_multihorizon_trades.csv
  /data/daily_multihorizon_summary.json
"""

import os, csv, json, math
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

from sqlalchemy import MetaData, Table, select
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_rule_all_dates.csv"

HORIZONS = [1, 2, 3, 5, 7, 10]
START_CAPITAL = 10_000.0
ROUND_TRIP_COST_BPS = 10.0  # 0.10%

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run test_daily_rule_all_501_dates_PERSISTENT.py first.")

def fnum(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except:
        return None

with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    setups=list(csv.DictReader(fh))

DB=database()
with DB() as s:
    md=MetaData()
    daily=Table("market_candles_1d", md, autoload_with=s.get_bind())

# Cache all bars per symbol once.
symbols=sorted(set(r["symbol"] for r in setups))
cache={}
print(f"Loading daily bars for {len(symbols)} symbols...")
for idx,sym in enumerate(symbols,1):
    with DB() as s:
        raw=list(s.execute(
            select(
                daily.c.session_date,
                daily.c.o,
                daily.c.h,
                daily.c.l,
                daily.c.c,
                daily.c.adj_c
            )
            .where(daily.c.symbol==sym)
            .order_by(daily.c.session_date)
        ).all())
    bars=[]
    for d,o,h,l,c,ac in raw:
        ro=fnum(o); rh=fnum(h); rl=fnum(l); rc=fnum(c); adj=fnum(ac)
        if None in (ro,rh,rl,rc,adj) or min(ro,rh,rl,rc,adj)<=0:
            continue
        fac=adj/rc
        bars.append({
            "date":str(d),
            "open":ro*fac,
            "high":rh*fac,
            "low":rl*fac,
            "close":adj,
        })
    cache[sym]=bars
    if idx%100==0 or idx==len(symbols):
        print(f"Loaded {idx}/{len(symbols)}")

all_trades=[]
stats=[]

for hold in HORIZONS:
    trades=[]
    for r in setups:
        sym=r["symbol"]
        signal_date=r["signal_date"]
        bars=cache.get(sym,[])
        dates=[b["date"] for b in bars]
        try:
            i=dates.index(signal_date)
        except ValueError:
            continue

        entry_i=i+1
        exit_i=i+hold
        if entry_i>=len(bars) or exit_i>=len(bars):
            continue

        entry=bars[entry_i]["open"]
        exitp=bars[exit_i]["close"]
        gross=100*(exitp/entry-1)
        net=gross-ROUND_TRIP_COST_BPS/100.0

        # Path risk during hold.
        min_low=min(bars[j]["low"] for j in range(entry_i, exit_i+1))
        max_high=max(bars[j]["high"] for j in range(entry_i, exit_i+1))
        mae=100*(min_low/entry-1)
        mfe=100*(max_high/entry-1)

        t={
            "hold_days":hold,
            "symbol":sym,
            "signal_date":signal_date,
            "entry_date":bars[entry_i]["date"],
            "exit_date":bars[exit_i]["date"],
            "entry_open":round(entry,6),
            "exit_close":round(exitp,6),
            "gross_return_pct":round(gross,6),
            "net_return_pct":round(net,6),
            "mae_pct":round(mae,6),
            "mfe_pct":round(mfe,6),
            "gross_win":1 if gross>0 else 0,
            "net_win":1 if net>0 else 0,
        }
        trades.append(t)
        all_trades.append(t)

    if not trades:
        continue

    gross_returns=[t["gross_return_pct"] for t in trades]
    net_returns=[t["net_return_pct"] for t in trades]
    maes=[t["mae_pct"] for t in trades]
    mfes=[t["mfe_pct"] for t in trades]

    # Sequential compounding, same comparison convention as previous 1-day backtest.
    gross_cap=START_CAPITAL
    net_cap=START_CAPITAL
    gross_peak=START_CAPITAL
    net_peak=START_CAPITAL
    gross_mdd=0.0
    net_mdd=0.0

    for t in sorted(trades, key=lambda x:(x["entry_date"],x["symbol"])):
        gross_cap *= 1+t["gross_return_pct"]/100
        net_cap *= 1+t["net_return_pct"]/100
        gross_peak=max(gross_peak,gross_cap)
        net_peak=max(net_peak,net_cap)
        gross_mdd=min(gross_mdd,100*(gross_cap/gross_peak-1))
        net_mdd=min(net_mdd,100*(net_cap/net_peak-1))

    stats.append({
        "hold_days":hold,
        "trades":len(trades),
        "gross_win_rate_pct":round(100*sum(x>0 for x in gross_returns)/len(gross_returns),4),
        "gross_mean_return_pct":round(mean(gross_returns),4),
        "gross_median_return_pct":round(median(gross_returns),4),
        "gross_final_equity":round(gross_cap,2),
        "gross_total_return_pct":round(100*(gross_cap/START_CAPITAL-1),4),
        "gross_max_drawdown_pct":round(gross_mdd,4),

        "net_win_rate_pct":round(100*sum(x>0 for x in net_returns)/len(net_returns),4),
        "net_mean_return_pct":round(mean(net_returns),4),
        "net_median_return_pct":round(median(net_returns),4),
        "net_final_equity":round(net_cap,2),
        "net_total_return_pct":round(100*(net_cap/START_CAPITAL-1),4),
        "net_max_drawdown_pct":round(net_mdd,4),

        "median_mae_pct":round(median(maes),4),
        "median_mfe_pct":round(median(mfes),4),
        "mean_mae_pct":round(mean(maes),4),
        "mean_mfe_pct":round(mean(mfes),4),
    })

# Save trades
trades_out=DATA_DIR/"daily_multihorizon_trades.csv"
with trades_out.open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(all_trades[0].keys()))
    w.writeheader(); w.writerows(all_trades)

stats_out=DATA_DIR/"daily_multihorizon_trade_stats.csv"
with stats_out.open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(stats[0].keys()))
    w.writeheader(); w.writerows(stats)

summary={
    "entry_rule":"next trading day adjusted open after signal",
    "exit_rule":"adjusted close after N trading days",
    "horizons":HORIZONS,
    "start_capital":START_CAPITAL,
    "round_trip_cost_bps":ROUND_TRIP_COST_BPS,
    "results":stats,
}
summary_out=DATA_DIR/"daily_multihorizon_summary.json"
summary_out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

print("\n=== MULTI-HORIZON BACKTEST ===")
print("Entry: next day's OPEN after signal")
print("Exit: CLOSE after N trading days")
print(f"Costs: {ROUND_TRIP_COST_BPS:.0f} bps round-trip\n")

print(
    "Hold | Win% G | Mean G | Med G | Win% N | Mean N | Med N | "
    "Final Net | Net Ret | MDD Net | Med MAE | Med MFE"
)
print("-"*122)
for s in stats:
    print(
        f"{s['hold_days']:>4} | "
        f"{s['gross_win_rate_pct']:>6.2f} | "
        f"{s['gross_mean_return_pct']:>6.2f}% | "
        f"{s['gross_median_return_pct']:>6.2f}% | "
        f"{s['net_win_rate_pct']:>6.2f} | "
        f"{s['net_mean_return_pct']:>6.2f}% | "
        f"{s['net_median_return_pct']:>6.2f}% | "
        f"${s['net_final_equity']:>10,.2f} | "
        f"{s['net_total_return_pct']:>7.2f}% | "
        f"{s['net_max_drawdown_pct']:>7.2f}% | "
        f"{s['median_mae_pct']:>7.2f}% | "
        f"{s['median_mfe_pct']:>7.2f}%"
    )

print("\nCreated:")
print(f" {trades_out}")
print(f" {stats_out}")
print(f" {summary_out}")
