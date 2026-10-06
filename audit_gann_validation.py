#!/usr/bin/env python3
import sys
from core import database
from monitor.gann_analysis import analyze_symbol,load_daily

symbols=sys.argv[1:] or ["COMI.CA","ADIB.CA","QNBA.CA","ABUK.CA","SWDY.CA"]
DB=database()
print("symbol | rows | samples | gate_rel | gate_useful | CI_state | regime | directional")
for s in symbols:
    try:
        df=load_daily(DB,s,2200)
        a=analyze_symbol(DB,s)
        if a is None:
            print(f"{s} | {len(df)} | NO ANALYSIS")
            continue
        b=a["backtest"]
        print(f'{s} | {len(df)} | {b["samples"]} | {b["gate_reliability"]} | {b["gate_useful_rate"]} | '
              f'{a["brown_oscillator"]["state"]} | {a["decision_summary"]["regime"]} | {len(a["directional_signals"])}')
    except Exception as e:
        try:
            n=len(load_daily(DB,s,2200))
        except Exception:
            n="?"
        print(f"{s} | {n} | ERROR | {type(e).__name__}: {e}")
