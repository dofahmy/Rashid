#!/usr/bin/env python3
"""
Compare successful (+50% within 21 sessions) vs failed setups
inside daily_rule_validation30_diverse_dates.csv.

Reads the existing validation CSV only.
No database changes.

Outputs:
- daily_validation_success_vs_fail.csv
- daily_validation_best_split_rules.csv
- daily_validation_success_vs_fail_report.txt
"""

from __future__ import annotations

import os
import csv, math
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
from statistics import mean, median

INPUT = DATA_DIR / "daily_rule_validation30_diverse_dates.csv"
OUT_COMPARE = DATA_DIR / "daily_validation_success_vs_fail.csv"
OUT_RULES = DATA_DIR / "daily_validation_best_split_rules.csv"
OUT_REPORT = DATA_DIR / "daily_validation_success_vs_fail_report.txt"

NON_PREDICTIVE = {
    "rank","symbol","signal_date",
    "max_high_5d_pct","max_high_10d_pct","max_high_21d_pct",
    "mae_21d_pct","days_to_20","days_to_30","days_to_50",
    "days_to_100","days_to_peak"
}

def num(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except:
        return None

def fmt(x,n=4):
    return None if x is None else round(float(x),n)

def auc(w,c):
    w=[x for x in w if x is not None]
    c=[x for x in c if x is not None]
    if not w or not c: return None
    score=0.0
    for a in w:
        for b in c:
            if a>b: score+=1
            elif a==b: score+=0.5
    return score/(len(w)*len(c))

with open(INPUT,encoding="utf-8-sig",newline="") as fh:
    rows=list(csv.DictReader(fh))

if not rows:
    raise SystemExit(f"{INPUT} is empty")

for r in rows:
    r["_success"]=(num(r.get("max_high_21d_pct")) or -999)>=50

wins=[r for r in rows if r["_success"]]
fails=[r for r in rows if not r["_success"]]

numeric=[]
for col in rows[0]:
    if col in NON_PREDICTIVE:
        continue
    vals=[num(r.get(col)) for r in rows]
    if sum(v is not None for v in vals)>=20:
        numeric.append(col)

comparison=[]
for col in numeric:
    w=[num(r.get(col)) for r in wins]
    f=[num(r.get(col)) for r in fails]
    w=[x for x in w if x is not None]
    f=[x for x in f if x is not None]
    if len(w)<4 or len(f)<10:
        continue
    a=auc(w,f)
    separation=max(a,1-a)
    comparison.append({
        "feature":col,
        "winner_n":len(w),
        "fail_n":len(f),
        "winner_mean":fmt(mean(w)),
        "winner_median":fmt(median(w)),
        "fail_mean":fmt(mean(f)),
        "fail_median":fmt(median(f)),
        "auc_winner_gt_fail":fmt(a),
        "separation":fmt(separation),
        "winner_like_direction":"HIGHER" if a>=0.5 else "LOWER",
        "median_difference":fmt(median(w)-median(f)),
    })

comparison.sort(key=lambda x:x["separation"],reverse=True)

with open(OUT_COMPARE,"w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(comparison[0].keys()))
    w.writeheader(); w.writerows(comparison)

# Scan transparent single-variable thresholds.
rules=[]
for col in numeric:
    values=sorted(set(num(r.get(col)) for r in rows if num(r.get(col)) is not None))
    if len(values)<4:
        continue
    # midpoints + actual values; compact enough for 30 observations
    cuts=[]
    for i in range(len(values)-1):
        cuts.append((values[i]+values[i+1])/2)
    for cut in cuts:
        for op in (">=","<="):
            def passed(r):
                x=num(r.get(col))
                if x is None: return False
                return x>=cut if op==">=" else x<=cut
            wp=sum(passed(r) for r in wins)
            fp=sum(passed(r) for r in fails)
            n=wp+fp
            if wp<2 or n<4:
                continue
            precision=wp/n
            recall=wp/len(wins)
            fail_rate=fp/len(fails)
            base=len(wins)/len(rows)
            lift=precision/base if base else None
            rules.append({
                "feature":col,"operator":op,"cut":fmt(cut,6),
                "winners_captured":wp,"fails_captured":fp,"selected_n":n,
                "precision_pct":fmt(100*precision),
                "recall_pct":fmt(100*recall),
                "fail_pass_pct":fmt(100*fail_rate),
                "lift_vs_20pct_base":fmt(lift),
            })

rules.sort(key=lambda x:(x["precision_pct"],x["recall_pct"],-x["selected_n"]),reverse=True)

with open(OUT_RULES,"w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rules[0].keys()))
    w.writeheader(); w.writerows(rules)

lines=[]
lines.append("SUCCESS (+50%) VS FAIL COMPARISON")
lines.append("="*72)
lines.append(f"Total: {len(rows)} | Winners: {len(wins)} | Fails: {len(fails)}")
lines.append("Winners: " + ", ".join(r["symbol"] for r in wins))
lines.append("")
lines.append("TOP PRE-SIGNAL DIFFERENCES")
lines.append("-"*72)
for x in comparison[:15]:
    lines.append(
        f"{x['feature']}: winner median={x['winner_median']} | "
        f"fail median={x['fail_median']} | separation={x['separation']} | "
        f"winner-like={x['winner_like_direction']}"
    )
lines.append("")
lines.append("BEST SIMPLE THRESHOLDS")
lines.append("-"*72)
for x in rules[:20]:
    lines.append(
        f"{x['feature']} {x['operator']} {x['cut']} | "
        f"selected={x['selected_n']} | winners={x['winners_captured']} | "
        f"fails={x['fails_captured']} | precision={x['precision_pct']}% | "
        f"recall={x['recall_pct']}% | lift={x['lift_vs_20pct_base']}x"
    )

Path(OUT_REPORT).write_text("\n".join(lines),encoding="utf-8")

print("\n=== SUCCESS VS FAIL ===")
print(f"Winners: {len(wins)}/{len(rows)} = {100*len(wins)/len(rows):.2f}%")
print("Winner symbols:", ", ".join(r["symbol"] for r in wins))

print("\nTOP DIFFERENCES:")
for x in comparison[:10]:
    print(
        f"{x['feature']}: W med={x['winner_median']} | "
        f"F med={x['fail_median']} | sep={x['separation']} | "
        f"{x['winner_like_direction']}"
    )

print("\nBEST SIMPLE RULES:")
for x in rules[:10]:
    print(
        f"{x['feature']} {x['operator']} {x['cut']} | "
        f"W={x['winners_captured']} F={x['fails_captured']} | "
        f"precision={x['precision_pct']}% recall={x['recall_pct']}% "
        f"lift={x['lift_vs_20pct_base']}x"
    )

print("\nCreated:")
print(" ",OUT_COMPARE)
print(" ",OUT_RULES)
print(" ",OUT_REPORT)
