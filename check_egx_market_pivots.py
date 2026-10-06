#!/usr/bin/env python3
import time
from core import database
from monitor.gann_analysis import get_egx_market_pivots

DB=database()
print("Calculating principal EGX market pivots...",flush=True)
started=time.time()
pivots=get_egx_market_pivots(DB,cache_seconds=0)
elapsed=time.time()-started

print(f"Calculation time: {elapsed:.2f} sec")
if not pivots:
    print("NO PRINCIPAL EGX MARKET PIVOTS FOUND")
    print("Current hard filters: score>=90, follow-through>=10%, major swing>=15%.")
else:
    source=pivots[-1].get("index_source","UNKNOWN")
    print("Index source:",source)
    if source=="SYNTHETIC_TOP30_LIQUID":
        print("NOTE: using synthetic equal-weight Top-30 liquid EGX proxy.")
    print(f"Principal EGX market pivots: {len(pivots)}")
    for p in pivots:
        typ="LOW" if p["kind"]=="L" else "HIGH"
        print(
            f'{p["date"]} {typ:<4} index={p["price"]:.2f} '
            f'score={p["score"]:.1f} breadth={p["breadth_pct"]:.1f}% '
            f'leaders={p["leaders_pct"]:.1f}% banks={p["banks_pct"]:.1f}% '
            f'prior={p["prior_swing_pct"]:.1f}% follow={p["follow_pct"]:.1f}% '
            f'available={p["available_date"]}'
        )
