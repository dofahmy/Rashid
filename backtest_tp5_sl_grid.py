#!/usr/bin/env python3
"""
Rajih — TP/SL grid backtest for the 501 daily setups.

Entry:
- Signal known after signal-day close.
- Enter NEXT trading day's adjusted OPEN.

Exit:
- Take Profit +5%
- Stop Loss grid: -3%, -5%, -7%, -10%
- Max hold: 1, 2, 3, 5 trading days
- If neither TP nor SL hits, exit at CLOSE of final holding day.

Daily-bar ambiguity:
If both TP and SL fall inside the same day's high/low range, daily bars cannot tell which came first.
So results are calculated twice:
- optimistic = TP first
- conservative = SL first

Costs:
- 10 bps round-trip per trade.

Reads:
  /data/daily_rule_all_dates.csv

Outputs:
  /data/daily_tp5_sl_grid_results.csv
  /data/daily_tp5_sl_grid_trades.csv
  /data/daily_tp5_sl_grid_summary.json
"""

import os, csv, json, math
from pathlib import Path
from statistics import mean, median
from sqlalchemy import MetaData, Table, select
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_rule_all_dates.csv"

TP_PCT = 5.0
STOP_GRID = [3.0, 5.0, 7.0, 10.0]
HOLD_GRID = [1, 2, 3, 5]
ROUND_TRIP_COST_BPS = 10.0

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

def run_trade(bars, signal_i, stop_pct, max_hold, scenario):
    entry_i=signal_i+1
    if entry_i>=len(bars):
        return None

    entry=bars[entry_i]["open"]
    tp_price=entry*(1+TP_PCT/100)
    sl_price=entry*(1-stop_pct/100)
    last_i=min(len(bars)-1, entry_i+max_hold-1)

    exit_price=None
    exit_i=None
    reason=None
    ambiguous=0

    for j in range(entry_i,last_i+1):
        hi=bars[j]["high"]
        lo=bars[j]["low"]
        hit_tp=hi>=tp_price
        hit_sl=lo<=sl_price

        if hit_tp and hit_sl:
            ambiguous=1
            if scenario=="optimistic":
                exit_price=tp_price
                reason="TP"
            else:
                exit_price=sl_price
                reason="SL"
            exit_i=j
            break
        if hit_tp:
            exit_price=tp_price
            exit_i=j
            reason="TP"
            break
        if hit_sl:
            exit_price=sl_price
            exit_i=j
            reason="SL"
            break

    if exit_price is None:
        exit_i=last_i
        exit_price=bars[exit_i]["close"]
        reason="TIME"

    gross=100*(exit_price/entry-1)
    net=gross-ROUND_TRIP_COST_BPS/100.0

    return {
        "entry_date":bars[entry_i]["date"],
        "exit_date":bars[exit_i]["date"],
        "entry_price":round(entry,6),
        "exit_price":round(exit_price,6),
        "gross_return_pct":round(gross,6),
        "net_return_pct":round(net,6),
        "reason":reason,
        "ambiguous_same_day":ambiguous,
        "days_held":exit_i-entry_i+1,
    }

all_trade_rows=[]
result_rows=[]

for stop in STOP_GRID:
    for hold in HOLD_GRID:
        for scenario in ("optimistic","conservative"):
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

                tr=run_trade(bars,i,stop,hold,scenario)
                if tr is None:
                    continue

                row={
                    "stop_pct":stop,
                    "max_hold_days":hold,
                    "scenario":scenario,
                    "symbol":sym,
                    "signal_date":signal_date,
                    **tr,
                }
                trades.append(row)
                all_trade_rows.append(row)

            if not trades:
                continue

            gross=[t["gross_return_pct"] for t in trades]
            net=[t["net_return_pct"] for t in trades]
            tp_n=sum(t["reason"]=="TP" for t in trades)
            sl_n=sum(t["reason"]=="SL" for t in trades)
            time_n=sum(t["reason"]=="TIME" for t in trades)
            amb_n=sum(t["ambiguous_same_day"] for t in trades)

            cap=10000.0
            peak=cap
            mdd=0.0
            for t in sorted(trades,key=lambda x:(x["entry_date"],x["symbol"])):
                cap*=1+t["net_return_pct"]/100
                peak=max(peak,cap)
                mdd=min(mdd,100*(cap/peak-1))

            result_rows.append({
                "stop_pct":stop,
                "max_hold_days":hold,
                "scenario":scenario,
                "trades":len(trades),
                "tp_n":tp_n,
                "tp_rate_pct":round(100*tp_n/len(trades),4),
                "sl_n":sl_n,
                "sl_rate_pct":round(100*sl_n/len(trades),4),
                "time_exit_n":time_n,
                "time_exit_rate_pct":round(100*time_n/len(trades),4),
                "ambiguous_same_day_n":amb_n,
                "ambiguous_same_day_pct":round(100*amb_n/len(trades),4),
                "gross_win_rate_pct":round(100*sum(x>0 for x in gross)/len(gross),4),
                "net_win_rate_pct":round(100*sum(x>0 for x in net)/len(net),4),
                "gross_mean_return_pct":round(mean(gross),4),
                "gross_median_return_pct":round(median(gross),4),
                "net_mean_return_pct":round(mean(net),4),
                "net_median_return_pct":round(median(net),4),
                "net_final_equity":round(cap,2),
                "net_total_return_pct":round(100*(cap/10000-1),4),
                "net_max_drawdown_pct":round(mdd,4),
                "median_days_held":round(median(t["days_held"] for t in trades),2),
            })

trades_out=DATA_DIR/"daily_tp5_sl_grid_trades.csv"
with trades_out.open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(all_trade_rows[0].keys()))
    w.writeheader(); w.writerows(all_trade_rows)

results_out=DATA_DIR/"daily_tp5_sl_grid_results.csv"
with results_out.open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(result_rows[0].keys()))
    w.writeheader(); w.writerows(result_rows)

summary_out=DATA_DIR/"daily_tp5_sl_grid_summary.json"
summary_out.write_text(
    json.dumps({
        "tp_pct":TP_PCT,
        "stop_grid_pct":STOP_GRID,
        "hold_grid_days":HOLD_GRID,
        "round_trip_cost_bps":ROUND_TRIP_COST_BPS,
        "intraday_ambiguity_note":"optimistic=TP first if TP and SL both touched same daily bar; conservative=SL first",
        "results":result_rows,
    },ensure_ascii=False,indent=2),
    encoding="utf-8"
)

print("\n=== TP +5% / SL GRID BACKTEST ===")
print("Entry: next day's OPEN after signal")
print("TP: +5%")
print("Stops: -3%, -5%, -7%, -10%")
print("Max holds: 1, 2, 3, 5 days")
print(f"Costs: {ROUND_TRIP_COST_BPS:.0f} bps round-trip")
print("NOTE: optimistic/conservative bracket same-day TP+SL ambiguity.\n")

print("SL | Hold | Scenario     | TP%   | SL%   | Time% | Ambig% | Win% N | Mean N | Med N | Final Net | MDD")
print("-"*116)
for r in result_rows:
    print(
        f"{r['stop_pct']:>2.0f}% | "
        f"{r['max_hold_days']:>4} | "
        f"{r['scenario']:<12} | "
        f"{r['tp_rate_pct']:>5.1f} | "
        f"{r['sl_rate_pct']:>5.1f} | "
        f"{r['time_exit_rate_pct']:>5.1f} | "
        f"{r['ambiguous_same_day_pct']:>6.1f} | "
        f"{r['net_win_rate_pct']:>6.1f} | "
        f"{r['net_mean_return_pct']:>6.2f}% | "
        f"{r['net_median_return_pct']:>6.2f}% | "
        f"${r['net_final_equity']:>10,.2f} | "
        f"{r['net_max_drawdown_pct']:>7.2f}%"
    )

print("\nCreated:")
print(f" {results_out}")
print(f" {trades_out}")
print(f" {summary_out}")
