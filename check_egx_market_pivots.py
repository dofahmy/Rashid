#!/usr/bin/env python3
import time
from core import database
from monitor.gann_analysis import _major_market_pivots_uncached

DB=database()
print("Calculating principal EGX market pivots...",flush=True)
started=time.time()
pivots,rejected=_major_market_pivots_uncached(DB,include_diagnostics=True)
elapsed=time.time()-started
print(f"Calculation time: {elapsed:.2f} sec")

if pivots:
    source=pivots[-1].get("index_source","UNKNOWN")
    print("Index source:",source)
    if source=="SYNTHETIC_TOP30_LIQUID":
        print("NOTE: using synthetic equal-weight Top-30 liquid EGX proxy; pivots are CLOSE-based.")
    print(f"\nPrincipal EGX market pivots: {len(pivots)}")
    for p in pivots:
        typ="LOW" if p["kind"]=="L" else "HIGH"
        print(
            f'{p["date"]} {typ:<4} close={p["price"]:.2f} '
            f'score={p["score"]:.1f} breadth={p["breadth_pct"]:.1f}% '
            f'leaders={p["leaders_pct"]:.1f}% banks={p["banks_pct"]:.1f}% '
            f'prior={p["prior_swing_pct"]:.1f}% follow={p["follow_pct"]:.1f}% '
            f'accepted_swing={p.get("swing_from_prev_pct","-")} '
            f'available={p["available_date"]}'
        )
else:
    print("NO PRINCIPAL EGX MARKET PIVOTS FOUND")

print("\nRejected/filtered candidates from 2025 onward:")
recent=[x for x in rejected if str(x.get("date",""))>="2025-01-01"]
if not recent:
    print("None")
else:
    for p in recent:
        typ="LOW" if p["kind"]=="L" else "HIGH"
        print(
            f'{p["date"]} {typ:<4} close={p["price"]:.2f} '
            f'score={p.get("score","-")} breadth={p.get("breadth_pct","-")}% '
            f'follow={p.get("follow_pct","-")}% '
            f'reason={p.get("reject_reason","")}'
        )
