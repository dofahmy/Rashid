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

b=a["backtest"]
print("BACKTEST FULL:",{k:b[k] for k in ("samples","hits","partials","misses","hit_rate","useful_rate","reliability_score")})
print("BACKTEST TRAIN:",b["train"])
print("BACKTEST VALIDATION:",b["validation"])
print("DIRECTION READY:",b["direction_ready"],"|",b["gate_reason"])
print("GATE RELIABILITY:",b["gate_reliability"],"GATE USEFUL:",b["gate_useful_rate"])

print("\nMETHOD VALIDATION:")
for x in a["method_validation_rows"]:
    print(x)

print("\nDECISION REGIME:",a["decision_summary"]["regime"])
print("MESSAGE:",a["decision_summary"]["message"])
print("ZONE:",a["decision_summary"]["zone"])

print("\nFUTURE CANDIDATES:")
for x in a["tops"]+a["lows"]:
    print({
        "state":x["state"],"candidate_type":x["candidate_type"],"price":x["price"],
        "window":(x["window_start"],x["window_end"]),"score":x["decision_score"],
        "method_evidence":x["method_evidence"],"validated_methods":x["validated_methods"],
        "gate_reason":x["gate_reason"]
    })

print("\nWATCH WINDOWS:")
for x in a["watch_windows"]:
    print({
        "window":(x["window_start"],x["window_end"]),
        "watch_score":x["watch_score"],"method_evidence":x["method_evidence"],
        "validated_methods":x["validated_methods"],"methods":x["methods"]
    })

if not b["direction_ready"] and any(x["state"]=="DIRECTIONAL" for x in a["tops"]+a["lows"]):
    raise SystemExit("FAIL: directional signal escaped reliability gate")

print("\nFINAL VALIDATED GANN ENGINE CHECK PASSED")
