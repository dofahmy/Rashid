#!/usr/bin/env python3
"""
Rajih — third-feature analysis INSIDE the strongest sweet spot only.

Sweet spot fixed from prior analysis:
    T-1 Bollinger Width > 100
    T-1 Dollar-Volume Ratio 5/20 > 1.5

Expected pool from prior run:
    35 total Top-1 cases
    12 +200 winners
    23 non-winners

Goal:
Find a third feature that separates winners from non-winners
CONSISTENTLY across 2024, 2025, 2026.

IMPORTANT:
This is descriptive / exploratory.
2026 is included, so this is NOT an untouched validation.

Inputs:
  /data/daily_200pct_ranking_discovery_top1.csv
  /data/daily_200pct_ranking_holdout_2026_top1.csv

Outputs:
  /data/daily_200pct_sweetspot35_cases.csv
  /data/daily_200pct_sweetspot35_feature_stats.csv
  /data/daily_200pct_sweetspot35_third_feature_rules.csv
  /data/daily_200pct_sweetspot35_triples.csv
  /data/daily_200pct_sweetspot35_summary.json
  /data/daily_200pct_sweetspot35_report.txt

Run:
  python analyze_200pct_sweetspot35_third_feature.py
"""

from __future__ import annotations

import csv, json, math, os
from pathlib import Path
from statistics import mean, median

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

DISC = DATA_DIR / "daily_200pct_ranking_discovery_top1.csv"
HOLD = DATA_DIR / "daily_200pct_ranking_holdout_2026_top1.csv"

OUT_CASES = DATA_DIR / "daily_200pct_sweetspot35_cases.csv"
OUT_STATS = DATA_DIR / "daily_200pct_sweetspot35_feature_stats.csv"
OUT_RULES = DATA_DIR / "daily_200pct_sweetspot35_third_feature_rules.csv"
OUT_TRIPLES = DATA_DIR / "daily_200pct_sweetspot35_triples.csv"
OUT_JSON = DATA_DIR / "daily_200pct_sweetspot35_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_sweetspot35_report.txt"

YEARS = ("2024","2025","2026")

# Fixed sweet spot from prior analysis.
BB_MIN = 100.0
DV_MIN = 1.5

if not DISC.exists():
    raise SystemExit(f"Missing {DISC}")
if not HOLD.exists():
    raise SystemExit(f"Missing {HOLD}")


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def qtile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs)-1)*q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    a = pos-lo
    return xs[lo]*(1-a)+xs[hi]*a


def auc(w, l):
    w = [x for x in w if x is not None]
    l = [x for x in l if x is not None]
    if not w or not l:
        return None
    s = 0.0
    for a in w:
        for b in l:
            if a > b:
                s += 1
            elif a == b:
                s += 0.5
    return s/(len(w)*len(l))


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        x = dict(r)
        x["year"] = x["setup_date"][:4]
        x["hit200"] = int(float(x["hit200"]))
        out.append(x)
    return out


rows = load(DISC) + load(HOLD)
rows = [r for r in rows if r["year"] in YEARS]

sweet = []
for r in rows:
    bb = fnum(r.get("t-1_bb_width_pct"))
    dv = fnum(r.get("t-1_dollarvol_ratio_5_20"))
    if bb is None or dv is None:
        continue
    if bb > BB_MIN and dv > DV_MIN:
        sweet.append(r)

winners = [r for r in sweet if r["hit200"] == 1]
losers = [r for r in sweet if r["hit200"] == 0]

write_csv(OUT_CASES, sweet)

print("\nRajih — THIRD FEATURE INSIDE THE STRONGEST SWEET SPOT")
print(f"Sweet spot: T-1 BB Width > {BB_MIN} AND T-1 DollarVol Ratio > {DV_MIN}")
print(f"Total cases: {len(sweet)} | +200 winners: {len(winners)} | non-winners: {len(losers)}")

by_year = {
    y: {
        "all": [r for r in sweet if r["year"] == y],
        "w": [r for r in sweet if r["year"] == y and r["hit200"] == 1],
        "l": [r for r in sweet if r["year"] == y and r["hit200"] == 0],
    }
    for y in YEARS
}
for y in YEARS:
    d = by_year[y]
    print(f"{y}: N={len(d['all'])} W={len(d['w'])} L={len(d['l'])}")

# Candidate features: all numeric pre-signal features except the two sweet-spot defining ones.
exclude = {"t-1_bb_width_pct","t-1_dollarvol_ratio_5_20"}
feature_cols = [k for k in sweet[0] if k.startswith("t-") and k not in exclude]

# 1) Feature stats across all 3 years.
stats = []
for col in feature_cols:
    yearly = {}
    dirs = []
    ok = True

    for y in YEARS:
        wv = [fnum(r.get(col)) for r in by_year[y]["w"]]
        lv = [fnum(r.get(col)) for r in by_year[y]["l"]]
        wv = [x for x in wv if x is not None]
        lv = [x for x in lv if x is not None]

        # Tiny sample here, so require at least 2 winners and 3 losers/year.
        if len(wv) < 2 or len(lv) < 3:
            ok = False
            break

        a = auc(wv, lv)
        if a is None:
            ok = False
            break

        direction = 1 if a >= 0.5 else -1
        dirs.append(direction)
        yearly[y] = {
            "w_med": median(wv),
            "l_med": median(lv),
            "sep": max(a,1-a),
            "dir": direction,
            "w_n": len(wv),
            "l_n": len(lv),
        }

    if not ok:
        continue

    same = len(set(dirs)) == 1
    stats.append({
        "feature": col,
        "same_direction_all_3_years": int(same),
        "direction": "HIGHER" if dirs[0] == 1 else "LOWER",
        "min_sep": min(yearly[y]["sep"] for y in YEARS),
        "avg_sep": mean(yearly[y]["sep"] for y in YEARS),
        "winner_median_2024": yearly["2024"]["w_med"],
        "nonwinner_median_2024": yearly["2024"]["l_med"],
        "winner_median_2025": yearly["2025"]["w_med"],
        "nonwinner_median_2025": yearly["2025"]["l_med"],
        "winner_median_2026": yearly["2026"]["w_med"],
        "nonwinner_median_2026": yearly["2026"]["l_med"],
    })

stats.sort(
    key=lambda r:(r["same_direction_all_3_years"],r["min_sep"],r["avg_sep"]),
    reverse=True
)
write_csv(OUT_STATS, stats)

# 2) Candidate third-feature thresholds.
rules = []

for st in stats:
    if not st["same_direction_all_3_years"]:
        continue

    col = st["feature"]
    direction = 1 if st["direction"] == "HIGHER" else -1
    wvals = [fnum(r.get(col)) for r in winners]
    wvals = [x for x in wvals if x is not None]
    if len(wvals) < 8:
        continue

    # thresholds that preserve roughly 75%, 60%, 50% of winners overall.
    if direction == 1:
        candidates = [
            (qtile(wvals,0.25), ">="),
            (qtile(wvals,0.40), ">="),
            (qtile(wvals,0.50), ">="),
        ]
    else:
        candidates = [
            (qtile(wvals,0.75), "<="),
            (qtile(wvals,0.60), "<="),
            (qtile(wvals,0.50), "<="),
        ]

    for cut, op in candidates:
        if cut is None:
            continue

        def passed(r):
            x = fnum(r.get(col))
            if x is None:
                return False
            return x >= cut if op == ">=" else x <= cut

        yr = {}
        valid = True
        for y in YEARS:
            w = by_year[y]["w"]
            l = by_year[y]["l"]
            wp = sum(passed(r) for r in w)
            lp = sum(passed(r) for r in l)
            if not w or not l:
                valid = False
                break
            yr[y] = {
                "wp": wp,
                "lp": lp,
                "wr": wp/len(w),
                "lr": lp/len(l),
            }

        if not valid:
            continue

        total_wp = sum(passed(r) for r in winners)
        total_lp = sum(passed(r) for r in losers)
        if total_wp < 4:
            continue

        precision = total_wp/(total_wp+total_lp) if total_wp+total_lp else 0
        min_cov = min(yr[y]["wr"] for y in YEARS)
        min_gap = min(yr[y]["wr"]-yr[y]["lr"] for y in YEARS)

        # Require at least one winner retained in each year.
        if any(yr[y]["wp"] < 1 for y in YEARS):
            continue

        rules.append({
            "feature": col,
            "operator": op,
            "cut": cut,
            "winner_pass_total": total_wp,
            "nonwinner_pass_total": total_lp,
            "precision_pct": 100*precision,
            "winner_coverage_pct": 100*total_wp/len(winners),
            "min_year_winner_coverage_pct": 100*min_cov,
            "min_year_gap_pp": 100*min_gap,
            "winners_2024": yr["2024"]["wp"],
            "nonwinners_2024": yr["2024"]["lp"],
            "winner_cov_2024_pct": 100*yr["2024"]["wr"],
            "nonwinner_rate_2024_pct": 100*yr["2024"]["lr"],
            "winners_2025": yr["2025"]["wp"],
            "nonwinners_2025": yr["2025"]["lp"],
            "winner_cov_2025_pct": 100*yr["2025"]["wr"],
            "nonwinner_rate_2025_pct": 100*yr["2025"]["lr"],
            "winners_2026": yr["2026"]["wp"],
            "nonwinners_2026": yr["2026"]["lp"],
            "winner_cov_2026_pct": 100*yr["2026"]["wr"],
            "nonwinner_rate_2026_pct": 100*yr["2026"]["lr"],
        })

rules.sort(
    key=lambda r:(r["precision_pct"],r["min_year_gap_pp"],r["winner_pass_total"]),
    reverse=True
)
write_csv(OUT_RULES, rules)

# 3) Triples = sweet spot + third-feature rule.
# Since BB>100 and DV>1.5 are fixed, each row effectively represents a 3-condition rule.
triples = []
for r in rules:
    # Favor practical support and positive separation every year.
    if r["winner_pass_total"] < 5:
        continue
    triples.append({
        "triple_rule": (
            f"t-1_bb_width_pct > {BB_MIN} AND "
            f"t-1_dollarvol_ratio_5_20 > {DV_MIN} AND "
            f"{r['feature']} {r['operator']} {r['cut']:.6g}"
        ),
        **r,
    })

triples.sort(
    key=lambda r:(r["precision_pct"],r["min_year_gap_pp"],r["winner_pass_total"]),
    reverse=True
)
write_csv(OUT_TRIPLES, triples)

summary = {
    "sweet_spot": {
        "bb_width_tminus1_gt": BB_MIN,
        "dollarvol_ratio_tminus1_gt": DV_MIN,
        "total": len(sweet),
        "winners": len(winners),
        "nonwinners": len(losers),
        "base_precision_pct": 100*len(winners)/len(sweet) if sweet else None,
    },
    "counts_by_year": {
        y: {
            "n": len(by_year[y]["all"]),
            "winners": len(by_year[y]["w"]),
            "nonwinners": len(by_year[y]["l"]),
        } for y in YEARS
    },
    "top_feature_stats": stats[:20],
    "top_third_feature_rules": rules[:20],
    "top_triples": triples[:20],
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== THIRD-FEATURE SAME-DIRECTION STATS ===")
shown = 0
for r in stats:
    if not r["same_direction_all_3_years"]:
        continue
    print(
        f"{r['feature']}: {r['direction']} | min sep={r['min_sep']:.3f} | "
        f"2024 W/L={r['winner_median_2024']:.3f}/{r['nonwinner_median_2024']:.3f} | "
        f"2025={r['winner_median_2025']:.3f}/{r['nonwinner_median_2025']:.3f} | "
        f"2026={r['winner_median_2026']:.3f}/{r['nonwinner_median_2026']:.3f}"
    )
    shown += 1
    if shown >= 15:
        break

print("\n=== BEST THIRD-FEATURE RULES INSIDE SWEET SPOT ===")
for r in rules[:15]:
    print(
        f"{r['feature']} {r['operator']} {r['cut']:.4f} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | "
        f"winner coverage={r['winner_coverage_pct']:.1f}% | "
        f"min year gap={r['min_year_gap_pp']:.1f}pp"
    )

print("\n=== BEST 3-CONDITION RULES ===")
for r in triples[:15]:
    print(
        f"{r['triple_rule']} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | "
        f"coverage={r['winner_coverage_pct']:.1f}%"
    )

# Exact zero-nonwinner third features, if any.
exact = [r for r in triples if r["nonwinner_pass_total"] == 0]
print("\n=== EXACT ZERO-NONWINNER TRIPLES ===")
if exact:
    for r in exact[:15]:
        print(
            f"{r['triple_rule']} | W={r['winner_pass_total']} L=0 | "
            f"2024W={r['winners_2024']} 2025W={r['winners_2025']} 2026W={r['winners_2026']}"
        )
else:
    print("None.")

report = []
report.append("RAJIH — THIRD FEATURE INSIDE BB>100 + DV>1.5 SWEET SPOT")
report.append("="*100)
report.append(
    f"Sweet spot total={len(sweet)} | winners={len(winners)} | non-winners={len(losers)} | "
    f"base precision={100*len(winners)/len(sweet):.2f}%"
)
for y in YEARS:
    report.append(
        f"{y}: N={len(by_year[y]['all'])} W={len(by_year[y]['w'])} L={len(by_year[y]['l'])}"
    )
report.append("")

report.append("TOP THIRD-FEATURE STATS")
report.append("-"*100)
for r in stats[:15]:
    if r["same_direction_all_3_years"]:
        report.append(
            f"{r['feature']} | {r['direction']} | min_sep={r['min_sep']:.3f} | "
            f"2024={r['winner_median_2024']:.3f}/{r['nonwinner_median_2024']:.3f} | "
            f"2025={r['winner_median_2025']:.3f}/{r['nonwinner_median_2025']:.3f} | "
            f"2026={r['winner_median_2026']:.3f}/{r['nonwinner_median_2026']:.3f}"
        )

report.append("")
report.append("TOP 3-CONDITION RULES")
report.append("-"*100)
for r in triples[:15]:
    report.append(
        f"{r['triple_rule']} | W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | coverage={r['winner_coverage_pct']:.1f}% | "
        f"minYearGap={r['min_year_gap_pp']:.1f}pp"
    )

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_CASES,OUT_STATS,OUT_RULES,OUT_TRIPLES,OUT_JSON,OUT_REPORT]:
    print(" ", p)
