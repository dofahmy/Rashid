#!/usr/bin/env python3
import sys
import time
from sqlalchemy import select
from core import database
from monitor.gann_analysis import analyze_symbol, _daily_table

DB = database()

def list_egx_symbols():
    with DB() as s:
        t = _daily_table(s)
        rows = s.execute(
            select(t.c.symbol)
            .where(t.c.symbol.ilike("%.CA"))
            .distinct()
            .order_by(t.c.symbol)
        ).all()
    return [r[0] for r in rows if r and r[0]]

symbols = list_egx_symbols()
print(f"EGX symbols found: {len(symbols)}", flush=True)

three_axis = []
three_axis_ci = []
directional = []
errors = []

started = time.time()

for i, sym in enumerate(symbols, 1):
    try:
        a = analyze_symbol(DB, sym)
        if not a:
            errors.append((sym, "NO_ANALYSIS"))
            print(f"[{i}/{len(symbols)}] {sym}: NO_ANALYSIS", flush=True)
            continue

        candidates = a.get("brown_candidates") or []
        geom = [x for x in candidates if x.get("three_axis_confluence")]
        ci_ok = [x for x in geom if x.get("oscillator_confirmed")]
        dir_ok = [x for x in candidates if x.get("state") == "DIRECTIONAL"]

        if geom:
            three_axis.append((sym, a, geom))
        if ci_ok:
            three_axis_ci.append((sym, a, ci_ok))
        if dir_ok:
            directional.append((sym, a, dir_ok))

        tag = []
        if geom: tag.append(f"3AXIS={len(geom)}")
        if ci_ok: tag.append(f"CI={len(ci_ok)}")
        if dir_ok: tag.append(f"DIRECTIONAL={len(dir_ok)}")
        if not tag: tag.append("none")

        print(
            f"[{i}/{len(symbols)}] {sym}: " + " | ".join(tag),
            flush=True
        )
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        break
    except Exception as e:
        errors.append((sym, f"{type(e).__name__}: {e}"))
        print(f"[{i}/{len(symbols)}] {sym}: ERROR {type(e).__name__}: {e}", flush=True)

elapsed = time.time() - started

print("\n" + "="*88)
print("BROWN THREE-AXIS MARKET SCAN")
print("="*88)
print(f"Scanned symbols: {len(symbols)}")
print(f"Elapsed: {elapsed:.1f} sec")
print(f"Three-axis geometric confluence: {len(three_axis)} symbols")
print(f"Three-axis + Composite Index confirmation: {len(three_axis_ci)} symbols")
print(f"Fully DIRECTIONAL after historical gate: {len(directional)} symbols")
print(f"Errors / no analysis: {len(errors)}")

def print_group(title, rows):
    print("\n" + title)
    print("-"*88)
    if not rows:
        print("NONE")
        return
    for sym, a, candidates in rows:
        b = a.get("backtest") or {}
        osc = a.get("brown_oscillator") or {}
        print(
            f"\n{sym} | CI={osc.get('state')} | "
            f"gate_rel={b.get('gate_reliability')} | "
            f"gate_useful={b.get('gate_useful_rate')} | "
            f"regime={a.get('decision_summary',{}).get('regime')}"
        )
        for x in candidates:
            print(
                f"  {x.get('candidate_type')} "
                f"price={x.get('price')} "
                f"window={x.get('window_start')}..{x.get('window_end')} "
                f"score={x.get('decision_score')} "
                f"diag_ATR={x.get('diagonal',{}).get('distance_atr')} "
                f"CI_confirmed={x.get('oscillator_confirmed')} "
                f"state={x.get('state')}"
            )

print_group("1) THREE-AXIS GEOMETRIC CONFLUENCE", three_axis)
print_group("2) THREE-AXIS + COMPOSITE INDEX CONFIRMATION", three_axis_ci)
print_group("3) FULLY DIRECTIONAL (3-axis + CI + historical validation)", directional)

if errors:
    print("\nERRORS / NO ANALYSIS")
    print("-"*88)
    for sym, err in errors:
        print(sym, "|", err)
