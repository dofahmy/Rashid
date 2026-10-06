#!/usr/bin/env python3
import time
from core import database
from monitor.gann_analysis import _major_market_pivots_uncached

DB=database()
print("FINAL EGX MARKET-ANCHOR DECISION TEST")
print("Calculating close-based pivots + market breadth + leaders + banks + 5-model consensus...",flush=True)
started=time.time()
pivots,all_candidates=_major_market_pivots_uncached(DB,include_diagnostics=True)
elapsed=time.time()-started
print(f"Calculation time: {elapsed:.2f} sec")

if not pivots:
    print("\nNO FINAL PRINCIPAL PIVOTS")
else:
    source=pivots[-1].get("index_source","UNKNOWN")
    print("\nIndex source:",source)
    if source=="SYNTHETIC_TOP30_LIQUID":
        print("NOTE: official ^CASE30 unavailable; using close-based equal-weight Top-30 liquid EGX proxy.")
    print(f"\nFINAL PRINCIPAL EGX PIVOTS: {len(pivots)}")
    for p in pivots:
        typ="LOW" if p["kind"]=="L" else "HIGH"
        print(
            f'{p["date"]} {typ:<4} close={p["price"]:.2f} '
            f'quality={p["quality_balanced"]:.1f} stability={p["stability_pct"]:.0f}% '
            f'votes={p["consensus_votes"]}/5 final={p["final_score"]:.1f} '
            f'breadth={p["breadth_pct"]:.1f}% leaders={p["leaders_pct"]:.1f}% '
            f'banks={p["banks_pct"]:.1f}% prior={p["prior_swing_pct"]:.1f}% '
            f'follow={p["follow_pct"]:.1f}% '
            f'swing={p.get("swing_from_prev_pct","-")} '
            f'available={p["available_date"]}'
        )

print("\n2025+ CANDIDATE DECISION AUDIT:")
recent=[x for x in all_candidates if str(x.get("date",""))>="2025-01-01"]
for p in recent:
    typ="LOW" if p["kind"]=="L" else "HIGH"
    print(
        f'{p["date"]} {typ:<4} q={p["quality_balanced"]:.1f} '
        f'stability={p["stability_pct"]:.0f}% votes={p["consensus_votes"]}/5 '
        f'breadth={p["breadth_pct"]:.1f}% leaders={p["leaders_pct"]:.1f}% '
        f'banks={p["banks_pct"]:.1f}% follow={p["follow_pct"]:.1f}% '
        f'=> {p["decision"]}: {p["decision_reason"]}'
    )

# Decision-readiness checks
print("\nDECISION READINESS:")
if len(pivots)<4:
    print("WARNING: fewer than 4 historical principal pivots.")
else:
    print("PASS: enough historical principal pivots.")

alts=all(pivots[i]["kind"]!=pivots[i-1]["kind"] for i in range(1,len(pivots)))
print("PASS: alternating LOW/HIGH." if alts else "WARNING: non-alternating sequence exists.")

stable=sum(1 for p in pivots if p.get("consensus_votes",0)>=4)
print(f"High-stability anchors (>=4/5 models): {stable}/{len(pivots)}")

if len(pivots)>=2:
    latest_low=next((p for p in reversed(pivots) if p["kind"]=="L"),None)
    latest_high=next((p for p in reversed(pivots) if p["kind"]=="H"),None)
    print("Latest major LOW:", latest_low["date"] if latest_low else "NONE")
    print("Latest major HIGH:", latest_high["date"] if latest_high else "NONE")
    if latest_low and latest_high:
        print("PASS: Gann has both a latest principal LOW and HIGH anchor.")
    else:
        print("WARNING: Gann is missing one side of the current major range.")
