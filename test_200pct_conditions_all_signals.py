#!/usr/bin/env python3
"""
Rajih — test discovered +200% conditions on ALL first-stage signals.

Input:
  /data/daily_200pct_stage2_enriched_signals.csv

Rules:
BASE:
  t-1_bb_width_pct > 100
  t-1_dollarvol_ratio_5_20 > 1.5

A:
  BASE + t-20_ret10_pct <= -15.3917

B:
  BASE + t-20_ret5_pct <= -6.7504

C:
  BASE + t-20_macd_signal_atr <= -0.0736252

D:
  BASE + t-20_rsi14 <= 41.4579

Reports:
- overall N and +20/+50/+100/+200 rates
- +200 lift vs all-signal baseline
- yearly 2024/2025/2026
- median max gain / drawdown / day63 close / first +200 day

This is descriptive confirmation, not a fresh untouched holdout.
"""

from __future__ import annotations
import csv, json, math, os
from pathlib import Path
from statistics import median

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_200pct_stage2_enriched_signals.csv"

OUT_RESULTS = DATA_DIR / "daily_200pct_all_signals_condition_results.csv"
OUT_YEARLY = DATA_DIR / "daily_200pct_all_signals_condition_yearly.csv"
OUT_MATCHES = DATA_DIR / "daily_200pct_all_signals_condition_matches.csv"
OUT_JSON = DATA_DIR / "daily_200pct_all_signals_condition_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_all_signals_condition_report.txt"

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}")

RULES = [
    ("BASE_SWEET_SPOT", [
        ("t-1_bb_width_pct", ">", 100.0),
        ("t-1_dollarvol_ratio_5_20", ">", 1.5),
    ]),
    ("CANDIDATE_A_RET10", [
        ("t-1_bb_width_pct", ">", 100.0),
        ("t-1_dollarvol_ratio_5_20", ">", 1.5),
        ("t-20_ret10_pct", "<=", -15.3917),
    ]),
    ("CANDIDATE_B_RET5", [
        ("t-1_bb_width_pct", ">", 100.0),
        ("t-1_dollarvol_ratio_5_20", ">", 1.5),
        ("t-20_ret5_pct", "<=", -6.7504),
    ]),
    ("CANDIDATE_C_MACD", [
        ("t-1_bb_width_pct", ">", 100.0),
        ("t-1_dollarvol_ratio_5_20", ">", 1.5),
        ("t-20_macd_signal_atr", "<=", -0.0736252),
    ]),
    ("CANDIDATE_D_RSI", [
        ("t-1_bb_width_pct", ">", 100.0),
        ("t-1_dollarvol_ratio_5_20", ">", 1.5),
        ("t-20_rsi14", "<=", 41.4579),
    ]),
]


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def passes_cond(row, cond):
    field, op, cut = cond
    x = fnum(row.get(field))
    if x is None:
        return False
    if op == ">": return x > cut
    if op == ">=": return x >= cut
    if op == "<": return x < cut
    if op == "<=": return x <= cut
    raise ValueError(op)


def passes_rule(row, conds):
    return all(passes_cond(row, c) for c in conds)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {
            "n":0,"symbols":0,"dates":0,
            "hit20_n":0,"hit20_pct":0.0,
            "hit50_n":0,"hit50_pct":0.0,
            "hit100_n":0,"hit100_pct":0.0,
            "hit200_n":0,"hit200_pct":0.0,
            "median_max_gain_63d_pct":None,
            "median_max_drawdown_63d_pct":None,
            "median_close_return_day63_pct":None,
            "median_first200_day":None,
        }

    def h(k):
        c = sum(int(float(r[k])) for r in rows)
        return c, 100*c/n

    h20,h50,h100,h200 = h("hit20"),h("hit50"),h("hit100"),h("hit200")
    mg = [fnum(r.get("max_gain_63d_pct")) for r in rows]
    dd = [fnum(r.get("max_drawdown_63d_pct")) for r in rows]
    cr = [fnum(r.get("close_return_day63_pct")) for r in rows]
    d2 = [fnum(r.get("first200_day")) for r in rows]
    mg = [x for x in mg if x is not None]
    dd = [x for x in dd if x is not None]
    cr = [x for x in cr if x is not None]
    d2 = [x for x in d2 if x is not None]

    return {
        "n":n,
        "symbols":len({r["symbol"] for r in rows}),
        "dates":len({r["setup_date"] for r in rows}),
        "hit20_n":h20[0],"hit20_pct":round(h20[1],4),
        "hit50_n":h50[0],"hit50_pct":round(h50[1],4),
        "hit100_n":h100[0],"hit100_pct":round(h100[1],4),
        "hit200_n":h200[0],"hit200_pct":round(h200[1],4),
        "median_max_gain_63d_pct":round(median(mg),4) if mg else None,
        "median_max_drawdown_63d_pct":round(median(dd),4) if dd else None,
        "median_close_return_day63_pct":round(median(cr),4) if cr else None,
        "median_first200_day":round(median(d2),2) if d2 else None,
    }


with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

for r in rows:
    r["year"] = r["setup_date"][:4]

baseline = summarize(rows)
base200 = baseline["hit200_pct"]

print("\nRajih — TEST +200% CONDITIONS ON ALL SIGNALS")
print(f"All signals: {baseline['n']}")
print(
    f"Baseline +20={baseline['hit20_pct']}% | +50={baseline['hit50_pct']}% | "
    f"+100={baseline['hit100_pct']}% | +200={baseline['hit200_pct']}%"
)

results = []
yearly = []
matches = []
summary = {"baseline": baseline, "rules": {}}

for name, conds in RULES:
    selected = [r for r in rows if passes_rule(r, conds)]
    s = summarize(selected)
    lift = s["hit200_pct"]/base200 if base200 else None

    rule_text = " AND ".join(f"{f} {op} {cut}" for f,op,cut in conds)
    overall = {
        "rule_name":name,
        "rule":rule_text,
        **s,
        "hit200_lift_vs_baseline":round(lift,4) if lift is not None else None,
    }
    results.append(overall)

    yr_obj = {}
    for y in ("2024","2025","2026"):
        yr_all = [r for r in rows if r["year"] == y]
        yr_sel = [r for r in selected if r["year"] == y]
        sy = summarize(yr_sel)
        by = summarize(yr_all)
        yl = sy["hit200_pct"]/by["hit200_pct"] if by["hit200_pct"] else None
        row = {
            "rule_name":name,
            "year":y,
            **sy,
            "baseline_hit200_pct":by["hit200_pct"],
            "hit200_lift_vs_year_baseline":round(yl,4) if yl is not None else None,
        }
        yearly.append(row)
        yr_obj[y] = row

    for r in selected:
        x = dict(r)
        x["matched_rule"] = name
        matches.append(x)

    summary["rules"][name] = {
        "definition": rule_text,
        "overall": overall,
        "yearly": yr_obj,
    }

write_csv(OUT_RESULTS, results)
write_csv(OUT_YEARLY, yearly)
write_csv(OUT_MATCHES, matches)
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== OVERALL RESULTS ===")
for r in results:
    print(
        f"{r['rule_name']}: N={r['n']} | +20={r['hit20_pct']}% | "
        f"+50={r['hit50_pct']}% | +100={r['hit100_pct']}% | "
        f"+200={r['hit200_pct']}% | lift={r['hit200_lift_vs_baseline']}x | "
        f"MedDD={r['median_max_drawdown_63d_pct']}%"
    )

print("\n=== YEARLY +200 RESULTS ===")
for name,_ in RULES:
    print(f"\n{name}")
    for r in yearly:
        if r["rule_name"] != name:
            continue
        print(
            f" {r['year']}: N={r['n']} | W200={r['hit200_n']} | "
            f"+200={r['hit200_pct']}% | baseline={r['baseline_hit200_pct']}% | "
            f"lift={r['hit200_lift_vs_year_baseline']}x"
        )

print("\n=== RISK / PATH SUMMARY ===")
for r in results:
    print(
        f"{r['rule_name']}: MedMax={r['median_max_gain_63d_pct']}% | "
        f"MedDD={r['median_max_drawdown_63d_pct']}% | "
        f"MedClose63={r['median_close_return_day63_pct']}% | "
        f"Median first +200 day={r['median_first200_day']}"
    )

report = []
report.append("RAJIH — +200% CONDITIONS ON ALL SIGNALS")
report.append("="*100)
report.append(
    f"Baseline N={baseline['n']} | +20={baseline['hit20_pct']}% | "
    f"+50={baseline['hit50_pct']}% | +100={baseline['hit100_pct']}% | "
    f"+200={baseline['hit200_pct']}%"
)
report.append("")
for r in results:
    report.append(
        f"{r['rule_name']} | N={r['n']} | +20={r['hit20_pct']}% | "
        f"+50={r['hit50_pct']}% | +100={r['hit100_pct']}% | "
        f"+200={r['hit200_pct']}% | lift={r['hit200_lift_vs_baseline']}x | "
        f"MedDD={r['median_max_drawdown_63d_pct']}%"
    )
report.append("")
report.append("YEARLY +200")
for name,_ in RULES:
    report.append(name)
    for r in yearly:
        if r["rule_name"] == name:
            report.append(
                f"  {r['year']}: N={r['n']} W={r['hit200_n']} "
                f"rate={r['hit200_pct']}% baseline={r['baseline_hit200_pct']}% "
                f"lift={r['hit200_lift_vs_year_baseline']}x"
            )

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_RESULTS, OUT_YEARLY, OUT_MATCHES, OUT_JSON, OUT_REPORT]:
    print(" ", p)
