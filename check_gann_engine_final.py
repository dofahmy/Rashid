#!/usr/bin/env python3
import sys,time
from core import database
from monitor.gann_analysis import analyze_symbol,GANN_ENGINE_VERSION

symbol=(sys.argv[1] if len(sys.argv)>1 else "COMI.CA").upper()
print("Engine:",GANN_ENGINE_VERSION)
print("Symbol:",symbol)
started=time.time()
a=analyze_symbol(database(),symbol)
print(f"Calculation time: {time.time()-started:.2f} sec")
if not a:
    raise SystemExit("NO ANALYSIS")
print("Anchor source:",a["anchor_source"])
print("Latest LOW:",a["latest_low"]["date"],a["latest_low"]["price"])
print("Latest HIGH:",a["latest_high"]["date"],a["latest_high"]["price"])
print("Anchor strength:",a["anchor_strength"])
print("Backtest:",a["backtest"])
print("Decision zone:",a["decision_summary"]["zone"])
print("TOPS:")
for x in a["tops"]:print(x)
print("LOWS:")
for x in a["lows"]:print(x)
print("NEXT TIMES:")
for x in a["next_times"]:print(x)
print("FINAL GANN ENGINE CHECK PASSED")
