#!/usr/bin/env python3
from core import database
from monitor.gann_analysis import get_egx_market_pivots

DB=database()
p=get_egx_market_pivots(DB,cache_seconds=0)
print("DATE        TYPE  STABILITY FINAL  BREADTH LEADERS BANKS")
print("-"*67)
for x in p:
    typ="LOW" if x["kind"]=="L" else "HIGH"
    print(
        f'{x["date"]}  {typ:<4}  {x["stability_pct"]:>5.0f}%    '
        f'{x["final_score"]:>5.1f}   {x["breadth_pct"]:>5.1f}%  '
        f'{x["leaders_pct"]:>5.1f}%  {x["banks_pct"]:>5.1f}%'
    )
