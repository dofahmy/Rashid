#!/usr/bin/env python3
"""
Rajih — find what is COMMON across the +200% Top-1 winners in 2024/2025/2026
and absent / rare among Top-1 stocks that did NOT reach +200%.

DESCRIPTIVE analysis only:
- 2026 is no longer treated as untouched here.
- We intentionally use all 3 years to search for common traits.
- This script does NOT claim predictive validation.

Inputs:
  /data/daily_200pct_ranking_discovery_top1.csv   # 2024 + 2025
  /data/daily_200pct_ranking_holdout_2026_top1.csv

Groups expected from prior run:
  2024 winners ~13
  2025 winners ~33
  2026 winners 17
  (actual counts are recomputed from files)

What it searches:
1) Numeric features where +200 winners differ from non-winners
   in the SAME direction in 2024, 2025, and 2026.
2) "Common winner zones":
   thresholds covering a large share of winners in every year while
   appearing much less often among non-winners.
3) Pair combinations that are strongly enriched in winners.
4) EXACT-EXCLUSIVE patterns:
   combinations seen in winners across all 3 years but in ZERO non-winners,
   if any exist. These are reported separately because small support can overfit.
5) Boolean/common states such as above/below SMA.

Outputs:
  /data/daily_200pct_common_3years_feature_stats.csv
  /data/daily_200pct_common_3years_single_rules.csv
  /data/daily_200pct_common_3years_pair_rules.csv
  /data/daily_200pct_common_3years_exclusive_patterns.csv
  /data/daily_200pct_common_3years_winners.csv
  /data/daily_200pct_common_3years_nonwinners.csv
  /data/daily_200pct_common_3years_summary.json
  /data/daily_200pct_common_3years_report.txt

Run:
  python analyze_200pct_common_17_vs_prior_years.py
"""

from __future__ import annotations

import csv, json, math, os
from pathlib import Path
from statistics import mean, median

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

DISC_FILE = DATA_DIR / "daily_200pct_ranking_discovery_top1.csv"
HOLD_FILE = DATA_DIR / "daily_200pct_ranking_holdout_2026_top1.csv"

OUT_STATS = DATA_DIR / "daily_200pct_common_3years_feature_stats.csv"
OUT_SINGLE = DATA_DIR / "daily_200pct_common_3years_single_rules.csv"
OUT_PAIR = DATA_DIR / "daily_200pct_common_3years_pair_rules.csv"
OUT_EXCL = DATA_DIR / "daily_200pct_common_3years_exclusive_patterns.csv"
OUT_WIN = DATA_DIR / "daily_200pct_common_3years_winners.csv"
OUT_LOSE = DATA_DIR / "daily_200pct_common_3years_nonwinners.csv"
OUT_JSON = DATA_DIR / "daily_200pct_common_3years_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_common_3years_report.txt"

YEARS = ("2024","2025","2026")

if not DISC_FILE.exists():
    raise SystemExit(f"Missing {DISC_FILE}")
if not HOLD_FILE.exists():
    raise SystemExit(f"Missing {HOLD_FILE}")


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
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["year"] = r["setup_date"][:4]
        r["hit200"] = int(float(r["hit200"]))
    return rows


rows = load(DISC_FILE) + load(HOLD_FILE)
rows = [r for r in rows if r["year"] in YEARS]

winners = [r for r in rows if r["hit200"] == 1]
losers = [r for r in rows if r["hit200"] == 0]

write_csv(OUT_WIN, winners)
write_csv(OUT_LOSE, losers)

by_year = {
    y: {
        "all": [r for r in rows if r["year"] == y],
        "w": [r for r in rows if r["year"] == y and r["hit200"] == 1],
        "l": [r for r in rows if r["year"] == y and r["hit200"] == 0],
    }
    for y in YEARS
}

print("\nRajih — COMMON TRAITS OF +200% TOP-1 WINNERS")
print("Descriptive cross-year analysis: 2024 + 2025 + 2026\n")
for y in YEARS:
    d = by_year[y]
    print(f"{y}: total={len(d['all'])} | +200 winners={len(d['w'])} | non-winners={len(d['l'])}")

print(f"\nTotal winners: {len(winners)}")
print(f"Total non-winners: {len(losers)}")

# Only pre-signal numeric features.
feature_cols = [k for k in rows[0] if k.startswith("t-")]

# ------------------------------------------------------------------
# 1) Cross-year feature direction stats
# ------------------------------------------------------------------
stats = []

for col in feature_cols:
    yearly = {}
    ok = True
    dirs = []

    for y in YEARS:
        wv = [fnum(r.get(col)) for r in by_year[y]["w"]]
        lv = [fnum(r.get(col)) for r in by_year[y]["l"]]
        wv = [x for x in wv if x is not None]
        lv = [x for x in lv if x is not None]

        if len(wv) < 5 or len(lv) < 20:
            ok = False
            break

        a = auc(wv, lv)
        if a is None:
            ok = False
            break

        direction = 1 if a >= 0.5 else -1
        dirs.append(direction)

        yearly[y] = {
            "w_n": len(wv),
            "l_n": len(lv),
            "w_med": median(wv),
            "l_med": median(lv),
            "auc": a,
            "sep": max(a,1-a),
            "direction": direction,
        }

    if not ok:
        continue

    same_direction = len(set(dirs)) == 1
    min_sep = min(yearly[y]["sep"] for y in YEARS)
    avg_sep = mean(yearly[y]["sep"] for y in YEARS)

    stats.append({
        "feature": col,
        "same_direction_all_3_years": int(same_direction),
        "direction": "HIGHER" if dirs[0] == 1 else "LOWER",
        "min_separation_3years": min_sep,
        "avg_separation_3years": avg_sep,
        "winner_median_2024": yearly["2024"]["w_med"],
        "nonwinner_median_2024": yearly["2024"]["l_med"],
        "sep_2024": yearly["2024"]["sep"],
        "winner_median_2025": yearly["2025"]["w_med"],
        "nonwinner_median_2025": yearly["2025"]["l_med"],
        "sep_2025": yearly["2025"]["sep"],
        "winner_median_2026": yearly["2026"]["w_med"],
        "nonwinner_median_2026": yearly["2026"]["l_med"],
        "sep_2026": yearly["2026"]["sep"],
    })

stats.sort(
    key=lambda r: (
        r["same_direction_all_3_years"],
        r["min_separation_3years"],
        r["avg_separation_3years"],
    ),
    reverse=True
)
write_csv(OUT_STATS, stats)

# ------------------------------------------------------------------
# 2) Common winner-zone single thresholds
#
# We create thresholds from pooled WINNER quantiles, then evaluate
# prevalence separately in each year. This is descriptive, not holdout.
# ------------------------------------------------------------------
single_rules = []

for st in stats:
    if not st["same_direction_all_3_years"]:
        continue

    col = st["feature"]
    direction = 1 if st["direction"] == "HIGHER" else -1

    all_wv = [fnum(r.get(col)) for r in winners]
    all_wv = [x for x in all_wv if x is not None]
    if len(all_wv) < 20:
        continue

    # Coverage-oriented winner thresholds.
    # HIGHER: 20th/30th/40th percentile means ~80/70/60% winner coverage.
    # LOWER: 80th/70th/60th percentile means ~80/70/60% winner coverage.
    if direction == 1:
        candidates = [
            (qtile(all_wv,0.20), ">=", 80),
            (qtile(all_wv,0.30), ">=", 70),
            (qtile(all_wv,0.40), ">=", 60),
        ]
    else:
        candidates = [
            (qtile(all_wv,0.80), "<=", 80),
            (qtile(all_wv,0.70), "<=", 70),
            (qtile(all_wv,0.60), "<=", 60),
        ]

    for cut, op, nominal_cov in candidates:
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
            if not w or not l:
                valid = False
                break
            wp = sum(passed(r) for r in w)
            lp = sum(passed(r) for r in l)
            yr[y] = {
                "w_pass": wp,
                "w_rate": wp/len(w),
                "l_pass": lp,
                "l_rate": lp/len(l),
            }

        if not valid:
            continue

        min_winner_cov = min(yr[y]["w_rate"] for y in YEARS)
        max_loser_rate = max(yr[y]["l_rate"] for y in YEARS)
        min_gap = min(yr[y]["w_rate"] - yr[y]["l_rate"] for y in YEARS)

        total_wp = sum(passed(r) for r in winners)
        total_lp = sum(passed(r) for r in losers)
        precision = total_wp / (total_wp + total_lp) if (total_wp+total_lp) else 0

        single_rules.append({
            "feature": col,
            "operator": op,
            "cut": cut,
            "nominal_winner_coverage_pct": nominal_cov,
            "min_winner_coverage_3years_pct": 100*min_winner_cov,
            "max_nonwinner_rate_3years_pct": 100*max_loser_rate,
            "min_prevalence_gap_3years_pp": 100*min_gap,
            "winner_pass_total": total_wp,
            "nonwinner_pass_total": total_lp,
            "precision_among_passers_pct": 100*precision,
            "winner_coverage_2024_pct": 100*yr["2024"]["w_rate"],
            "nonwinner_rate_2024_pct": 100*yr["2024"]["l_rate"],
            "winner_coverage_2025_pct": 100*yr["2025"]["w_rate"],
            "nonwinner_rate_2025_pct": 100*yr["2025"]["l_rate"],
            "winner_coverage_2026_pct": 100*yr["2026"]["w_rate"],
            "nonwinner_rate_2026_pct": 100*yr["2026"]["l_rate"],
        })

single_rules.sort(
    key=lambda r: (
        r["min_prevalence_gap_3years_pp"],
        r["min_winner_coverage_3years_pct"],
        -r["max_nonwinner_rate_3years_pct"],
    ),
    reverse=True
)
write_csv(OUT_SINGLE, single_rules)

# ------------------------------------------------------------------
# 3) Pair rules
# Use top single rules with robust positive gap in ALL years.
# ------------------------------------------------------------------
candidate_singles = [
    r for r in single_rules
    if r["min_winner_coverage_3years_pct"] >= 50
    and r["min_prevalence_gap_3years_pp"] >= 8
][:30]

pair_rules = []
exclusive = []

def single_pass(r, rule):
    x = fnum(r.get(rule["feature"]))
    if x is None:
        return False
    return x >= rule["cut"] if rule["operator"] == ">=" else x <= rule["cut"]

for i in range(len(candidate_singles)):
    for j in range(i+1, len(candidate_singles)):
        a = candidate_singles[i]
        b = candidate_singles[j]

        # Avoid same feature twice.
        if a["feature"] == b["feature"]:
            continue

        def passed(r):
            return single_pass(r,a) and single_pass(r,b)

        yr = {}
        okay = True
        for y in YEARS:
            w = by_year[y]["w"]
            l = by_year[y]["l"]
            wp = sum(passed(r) for r in w)
            lp = sum(passed(r) for r in l)
            if len(w) == 0 or len(l) == 0:
                okay = False
                break
            yr[y] = {
                "wp": wp, "wr": wp/len(w),
                "lp": lp, "lr": lp/len(l),
            }

        if not okay:
            continue

        total_wp = sum(passed(r) for r in winners)
        total_lp = sum(passed(r) for r in losers)
        if total_wp < 8:
            continue

        min_wcov = min(yr[y]["wr"] for y in YEARS)
        max_lrate = max(yr[y]["lr"] for y in YEARS)
        min_gap = min(yr[y]["wr"]-yr[y]["lr"] for y in YEARS)
        precision = total_wp/(total_wp+total_lp) if total_wp+total_lp else 0

        row = {
            "rule1": f"{a['feature']} {a['operator']} {a['cut']:.6g}",
            "rule2": f"{b['feature']} {b['operator']} {b['cut']:.6g}",
            "winner_pass_total": total_wp,
            "nonwinner_pass_total": total_lp,
            "precision_pct": 100*precision,
            "min_winner_coverage_3years_pct": 100*min_wcov,
            "max_nonwinner_rate_3years_pct": 100*max_lrate,
            "min_prevalence_gap_3years_pp": 100*min_gap,
            "winners_2024": yr["2024"]["wp"],
            "nonwinners_2024": yr["2024"]["lp"],
            "winner_coverage_2024_pct": 100*yr["2024"]["wr"],
            "nonwinner_rate_2024_pct": 100*yr["2024"]["lr"],
            "winners_2025": yr["2025"]["wp"],
            "nonwinners_2025": yr["2025"]["lp"],
            "winner_coverage_2025_pct": 100*yr["2025"]["wr"],
            "nonwinner_rate_2025_pct": 100*yr["2025"]["lr"],
            "winners_2026": yr["2026"]["wp"],
            "nonwinners_2026": yr["2026"]["lp"],
            "winner_coverage_2026_pct": 100*yr["2026"]["wr"],
            "nonwinner_rate_2026_pct": 100*yr["2026"]["lr"],
        }
        pair_rules.append(row)

        # Literal "not present in non-winners":
        # Require zero non-winners overall AND representation in winners every year.
        if (
            total_lp == 0
            and yr["2024"]["wp"] >= 2
            and yr["2025"]["wp"] >= 2
            and yr["2026"]["wp"] >= 2
        ):
            exclusive.append(row)

pair_rules.sort(
    key=lambda r: (
        r["min_prevalence_gap_3years_pp"],
        r["precision_pct"],
        r["winner_pass_total"],
    ),
    reverse=True
)
exclusive.sort(
    key=lambda r: (
        r["winner_pass_total"],
        r["min_winner_coverage_3years_pct"],
    ),
    reverse=True
)

write_csv(OUT_PAIR, pair_rules)
write_csv(OUT_EXCL, exclusive)

# ------------------------------------------------------------------
# 4) Also scan simple Boolean SMA states for cross-year commonality.
# ------------------------------------------------------------------
boolean_cols = [c for c in feature_cols if "above_sma" in c]
boolean_findings = []

for col in boolean_cols:
    for val in (0,1):
        yr = {}
        for y in YEARS:
            w = by_year[y]["w"]
            l = by_year[y]["l"]
            wp = sum(int(float(r.get(col,0) or 0)) == val for r in w)
            lp = sum(int(float(r.get(col,0) or 0)) == val for r in l)
            yr[y] = {
                "wr": wp/len(w),
                "lr": lp/len(l),
                "wp": wp,
                "lp": lp,
            }
        min_gap = min(yr[y]["wr"]-yr[y]["lr"] for y in YEARS)
        min_wcov = min(yr[y]["wr"] for y in YEARS)
        if min_gap > 0:
            boolean_findings.append({
                "feature": col,
                "state": val,
                "meaning": "ABOVE" if val == 1 else "BELOW",
                "min_winner_coverage_3years_pct": 100*min_wcov,
                "min_prevalence_gap_3years_pp": 100*min_gap,
                "winner_coverage_2024_pct":100*yr["2024"]["wr"],
                "nonwinner_rate_2024_pct":100*yr["2024"]["lr"],
                "winner_coverage_2025_pct":100*yr["2025"]["wr"],
                "nonwinner_rate_2025_pct":100*yr["2025"]["lr"],
                "winner_coverage_2026_pct":100*yr["2026"]["wr"],
                "nonwinner_rate_2026_pct":100*yr["2026"]["lr"],
            })

boolean_findings.sort(
    key=lambda r:(r["min_prevalence_gap_3years_pp"],r["min_winner_coverage_3years_pct"]),
    reverse=True
)

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
summary = {
    "counts": {
        y: {
            "top1_total": len(by_year[y]["all"]),
            "hit200_winners": len(by_year[y]["w"]),
            "nonwinners": len(by_year[y]["l"]),
        } for y in YEARS
    },
    "total_winners": len(winners),
    "total_nonwinners": len(losers),
    "top_same_direction_features": stats[:20],
    "top_single_common_rules": single_rules[:20],
    "top_pair_common_rules": pair_rules[:20],
    "exclusive_zero_nonwinner_patterns": exclusive[:20],
    "top_boolean_findings": boolean_findings[:20],
}
OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

lines = []
lines.append("RAJIH — WHAT THE +200% WINNERS SHARE ACROSS 2024 / 2025 / 2026")
lines.append("="*100)
lines.append("DESCRIPTIVE analysis. 2026 is included intentionally; this is no longer an untouched validation.")
lines.append("")
for y in YEARS:
    d = by_year[y]
    lines.append(f"{y}: Top1={len(d['all'])} | +200 winners={len(d['w'])} | non-winners={len(d['l'])}")

lines.append("")
lines.append("TOP FEATURES WITH SAME WINNER DIRECTION IN ALL 3 YEARS")
lines.append("-"*100)
for r in stats[:15]:
    if not r["same_direction_all_3_years"]:
        continue
    lines.append(
        f"{r['feature']} | {r['direction']} | min-sep={r['min_separation_3years']:.3f} | "
        f"2024 W/L med={r['winner_median_2024']:.3f}/{r['nonwinner_median_2024']:.3f} | "
        f"2025={r['winner_median_2025']:.3f}/{r['nonwinner_median_2025']:.3f} | "
        f"2026={r['winner_median_2026']:.3f}/{r['nonwinner_median_2026']:.3f}"
    )

lines.append("")
lines.append("TOP COMMON WINNER-ZONE SINGLE RULES")
lines.append("-"*100)
for r in single_rules[:15]:
    lines.append(
        f"{r['feature']} {r['operator']} {r['cut']:.5g} | "
        f"min winner coverage={r['min_winner_coverage_3years_pct']:.1f}% | "
        f"max non-winner rate={r['max_nonwinner_rate_3years_pct']:.1f}% | "
        f"min gap={r['min_prevalence_gap_3years_pp']:.1f}pp"
    )

lines.append("")
lines.append("TOP COMMON PAIRS")
lines.append("-"*100)
for r in pair_rules[:15]:
    lines.append(
        f"{r['rule1']} AND {r['rule2']} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | "
        f"min winner coverage={r['min_winner_coverage_3years_pct']:.1f}% | "
        f"min gap={r['min_prevalence_gap_3years_pp']:.1f}pp"
    )

lines.append("")
lines.append("EXACT EXCLUSIVE PATTERNS — ZERO NON-WINNERS")
lines.append("-"*100)
if exclusive:
    for r in exclusive[:15]:
        lines.append(
            f"{r['rule1']} AND {r['rule2']} | "
            f"W total={r['winner_pass_total']} | "
            f"2024 W={r['winners_2024']} | 2025 W={r['winners_2025']} | 2026 W={r['winners_2026']} | "
            f"NON-WINNERS=0"
        )
else:
    lines.append("None found with >=2 winners represented in EACH of 2024, 2025, and 2026.")

lines.append("")
lines.append("BOOLEAN / SMA STATES")
lines.append("-"*100)
for r in boolean_findings[:10]:
    lines.append(
        f"{r['feature']} = {r['state']} ({r['meaning']}) | "
        f"min winner coverage={r['min_winner_coverage_3years_pct']:.1f}% | "
        f"min gap={r['min_prevalence_gap_3years_pp']:.1f}pp"
    )

OUT_REPORT.write_text("\n".join(lines),encoding="utf-8")

print("\n=== SAME-DIRECTION FEATURES ACROSS ALL 3 YEARS ===")
shown = 0
for r in stats:
    if not r["same_direction_all_3_years"]:
        continue
    print(
        f"{r['feature']}: {r['direction']} | min sep={r['min_separation_3years']:.3f} | "
        f"2024 med W/L={r['winner_median_2024']:.3f}/{r['nonwinner_median_2024']:.3f} | "
        f"2025={r['winner_median_2025']:.3f}/{r['nonwinner_median_2025']:.3f} | "
        f"2026={r['winner_median_2026']:.3f}/{r['nonwinner_median_2026']:.3f}"
    )
    shown += 1
    if shown >= 15:
        break

print("\n=== STRONG COMMON WINNER ZONES ===")
for r in single_rules[:12]:
    print(
        f"{r['feature']} {r['operator']} {r['cut']:.4f} | "
        f"winner coverage min={r['min_winner_coverage_3years_pct']:.1f}% | "
        f"nonwinner max={r['max_nonwinner_rate_3years_pct']:.1f}% | "
        f"gap min={r['min_prevalence_gap_3years_pp']:.1f}pp"
    )

print("\n=== BEST COMMON PAIRS ===")
for r in pair_rules[:12]:
    print(
        f"{r['rule1']} AND {r['rule2']} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | "
        f"winner coverage min={r['min_winner_coverage_3years_pct']:.1f}% | "
        f"gap min={r['min_prevalence_gap_3years_pp']:.1f}pp"
    )

print("\n=== EXACT PATTERNS NOT SEEN IN ANY NON-WINNER ===")
if exclusive:
    for r in exclusive[:12]:
        print(
            f"{r['rule1']} AND {r['rule2']} | "
            f"W total={r['winner_pass_total']} | "
            f"2024={r['winners_2024']} 2025={r['winners_2025']} 2026={r['winners_2026']} | "
            f"NON-WINNERS=0"
        )
else:
    print("None with >=2 winners in EACH year and zero non-winners.")

print("\n=== BOOLEAN / SMA COMMON STATES ===")
for r in boolean_findings[:10]:
    print(
        f"{r['feature']} = {r['state']} ({r['meaning']}) | "
        f"winner coverage min={r['min_winner_coverage_3years_pct']:.1f}% | "
        f"gap min={r['min_prevalence_gap_3years_pp']:.1f}pp"
    )

print("\nCreated:")
for p in [OUT_STATS,OUT_SINGLE,OUT_PAIR,OUT_EXCL,OUT_WIN,OUT_LOSE,OUT_JSON,OUT_REPORT]:
    print(" ",p)
