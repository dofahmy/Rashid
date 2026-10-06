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
if not a:raise SystemExit("NO ANALYSIS")
print("Methodology:",a["methodology"])
print("Anchor source:",a["anchor_source"])
print("Latest LOW:",a["latest_low"]["date"],a["latest_low"]["price"])
print("Latest HIGH:",a["latest_high"]["date"],a["latest_high"]["price"])
print("Brown Composite Index:",a["brown_oscillator"])
print("Backtest:",{k:a["backtest"].get(k) for k in ("samples","hits","partials","misses","useful_rate","gate_reliability","gate_useful_rate","direction_ready","gate_reason")})
print("Decision regime:",a["decision_summary"]["regime"])
print("Message:",a["decision_summary"]["message"])
print("\nHORIZONTAL PRICE CONFLUENCE:")
for x in [a["nearest_support"],a["nearest_resistance"]]:
    if x:print(x)
print("\nVERTICAL TIME CONFLUENCE:")
for x in a["watch_windows"]:print(x)
print("\nPRICE-TIME-DIAGONAL CANDIDATES:")
for x in a["brown_candidates"]:
    print({"state":x["state"],"candidate_type":x["candidate_type"],"price":x["price"],
           "window":(x["window_start"],x["window_end"]),"score":x["decision_score"],
           "three_axis":x["three_axis_confluence"],"oscillator_confirmed":x["oscillator_confirmed"],
           "diagonal":x["diagonal"],"reason":x["gate_reason"]})
print("\nPUBLIC-METHOD LIMITATIONS:")
for x in a["brown_public_limitations"]:print("-",x)
print("\nCONSTANCE BROWN ENGINE CHECK PASSED")
