#!/usr/bin/env python3
from core import database
from monitor.gann_analysis import get_egx_market_pivots

DB=database()
pivots=get_egx_market_pivots(DB,cache_seconds=0)
if not pivots:
    print("NO MAJOR EGX MARKET PIVOTS FOUND")
    print("Run/update Egypt data first so ^CASE30 and .CA stocks exist in market_candles_1d.")
else:
    print(f"Major EGX market pivots: {len(pivots)}")
    for p in pivots:
        typ="LOW" if p["kind"]=="L" else "HIGH"
        print(
            f'{p["date"]} {typ:<4} index={p["price"]:.2f} '
            f'score={p["score"]:.1f} breadth={p["breadth_pct"]:.1f}% '
            f'leaders={p["leaders_pct"]:.1f}% banks={p["banks_pct"]:.1f}% '
            f'swing={p["index_swing_pct"]:.1f}% available={p["available_date"]}'
        )
