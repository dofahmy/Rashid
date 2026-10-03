#!/usr/bin/env python3
"""
Check whether the database actually contains enough US 15m candle history
for a true third 30-day holdout before 2026-08-03.

Run:
    python check_candle_history_coverage.py
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import select, func

from core import database
from monitor.models import Stock, Candle

NY = ZoneInfo("America/New_York")


def fmt(ts):
    if ts is None:
        return "NONE"
    return datetime.fromtimestamp(int(ts), NY).isoformat(timespec="minutes")


def main():
    DB = database()

    holdout_start = int(datetime(2026, 7, 4, 16, 0, tzinfo=NY).timestamp())
    holdout_end   = int(datetime(2026, 8, 3, 16, 0, tzinfo=NY).timestamp())

    with DB() as s:
        us_symbols = list(
            s.execute(
                select(Stock.symbol)
                .where(Stock.market == "US")
                .order_by(Stock.symbol)
            ).scalars().all()
        )

        global_min, global_max = s.execute(
            select(func.min(Candle.ts), func.max(Candle.ts))
        ).one()

        aapl_min, aapl_max, aapl_count = s.execute(
            select(func.min(Candle.ts), func.max(Candle.ts), func.count())
            .where(Candle.symbol == "AAPL")
        ).one()

        symbols_with_any_holdout = s.execute(
            select(func.count(func.distinct(Candle.symbol)))
            .where(
                Candle.symbol.in_(us_symbols),
                Candle.ts >= holdout_start,
                Candle.ts < holdout_end,
            )
        ).scalar_one()

        holdout_candles = s.execute(
            select(func.count())
            .select_from(Candle)
            .where(
                Candle.symbol.in_(us_symbols),
                Candle.ts >= holdout_start,
                Candle.ts < holdout_end,
            )
        ).scalar_one()

        # Enough warmup: at least 200 candles before holdout start.
        # Count symbols with >=200 historical candles before the holdout.
        warm_counts = dict(
            s.execute(
                select(Candle.symbol, func.count())
                .where(
                    Candle.symbol.in_(us_symbols),
                    Candle.ts < holdout_start,
                )
                .group_by(Candle.symbol)
            ).all()
        )
        enough_warmup = sum(1 for sym in us_symbols if warm_counts.get(sym, 0) >= 200)

    print("\n=== CANDLE HISTORY COVERAGE CHECK ===")
    print(f"US symbols: {len(us_symbols)}")
    print(f"Global candle range: {fmt(global_min)} -> {fmt(global_max)}")
    print(f"AAPL candle range:   {fmt(aapl_min)} -> {fmt(aapl_max)} | count={aapl_count}")
    print()
    print(f"Requested third holdout: {fmt(holdout_start)} -> {fmt(holdout_end)}")
    print(f"Symbols with ANY candle in holdout: {symbols_with_any_holdout}/{len(us_symbols)}")
    print(f"Total US candles in holdout: {holdout_candles}")
    print(f"Symbols with >=200 warmup candles before holdout: {enough_warmup}/{len(us_symbols)}")

    if holdout_candles == 0:
        print("\nRESULT: No candle data exists for the requested holdout period.")
        print("The 0-trade backtest is therefore a DATA-COVERAGE result, not a strategy result.")
    elif enough_warmup == 0:
        print("\nRESULT: Holdout candles exist, but there is not enough pre-history to reconstruct the strategy.")
    else:
        print("\nRESULT: Some holdout data exists. If backtest still gives 0 trades, inspect strategy reconstruction.")


if __name__ == "__main__":
    main()
