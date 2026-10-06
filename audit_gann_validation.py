#!/usr/bin/env python3
import sys,time
from core import database
from monitor.gann_analysis import analyze_symbol

symbols=sys.argv[1:] or ["COMI.CA","ADIB.CA","QNBA.CA","ABUK.CA","SWDY.CA"]
DB=database()
print("symbol | samples | validation_rel | validation_useful | regime | directional_count")
for s in symbols:
    try:
        a=analyze_symbol(DB,s)
        b=a["backtest"]
        print(
            f'{s} | {b["samples"]} | {b["gate_reliability"]} | {b["gate_useful_rate"]} | '
            f'{a["decision_summary"]["regime"]} | {len(a["directional_signals"])}'
        )
    except Exception as e:
        print(f"{s} | ERROR | {type(e).__name__}: {e}")
