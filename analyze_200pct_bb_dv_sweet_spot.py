#!/usr/bin/env python3
"""
Rajih — 2D sweet-spot analysis for +200% runners.

Purpose:
Find stable regions where +200% winners concentrate across 2024, 2025, 2026
using two recurring features:

  1) T-1 Bollinger Width
  2) T-1 Dollar Volume Ratio 5/20

Input:
  /data/daily_200pct_ranking_all_scored.csv

Population:
  Top-1/day ranked sample only is reconstructed from:
    /data/daily_200pct_ranking_discovery_top1.csv
    /data/daily_200pct_ranking_holdout_2026_top1.csv

Bins:

Bollinger Width:
  <40
  40-55
  55-70
  70-100
  >100

Dollar-volume ratio 5/20:
  <0.5
  0.5-0.7
  0.7-1.0
  1.0-1.5
  >1.5

For every 2D cell, calculate:
- N total
- +200 winners
- +200 rate
- yearly N/winners/rate for 2024, 2025, 2026
- minimum yearly +200 rate
- maximum yearly +200 rate
- rate spread across years
- minimum yearly sample size
- all-years lift vs overall Top-1 baseline
- minimum yearly lift vs each year's baseline

Also finds "stable sweet spots":
- at least 5 observations in EACH year
- at least 1 +200 winner in EACH year
- positive lift in EACH year
- ranked by minimum yearly lift, then overall +200 rate

Outputs:
  /data/daily_200pct_bb_dv_grid.csv
  /data/daily_200pct_bb_dv_sweet_spots.csv
  /data/daily_200pct_bb_dv_bbwidth_summary.csv
  /data/daily_200pct_bb_dv_dvratio_summary.csv
  /data/daily_200pct_bb_dv_summary.json
  /data/daily_200pct_bb_dv_report.txt

Run:
  python analyze_200pct_bb_dv_sweet_spot.py
"""

from __future__ import annotations

import csv, json, math, os
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

DISC = DATA_DIR / "daily_200pct_ranking_discovery_top1.csv"
HOLD = DATA_DIR / "daily_200pct_ranking_holdout_2026_top1.csv"

OUT_GRID = DATA_DIR / "daily_200pct_bb_dv_grid.csv"
OUT_SWEET = DATA_DIR / "daily_200pct_bb_dv_sweet_spots.csv"
OUT_BB = DATA_DIR / "daily_200pct_bb_dv_bbwidth_summary.csv"
OUT_DV = DATA_DIR / "daily_200pct_bb_dv_dvratio_summary.csv"
OUT_JSON = DATA_DIR / "daily_200pct_bb_dv_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_bb_dv_report.txt"

YEARS = ("2024","2025","2026")

if not DISC.exists():
    raise SystemExit(f"Missing {DISC}")
if not HOLD.exists():
    raise SystemExit(f"Missing {HOLD}")

BB_BINS = [
    ("<40", -math.inf, 40.0),
    ("40-55", 40.0, 55.0),
    ("55-70", 55.0, 70.0),
    ("70-100", 70.0, 100.0),
    (">100", 100.0, math.inf),
]

DV_BINS = [
    ("<0.5", -math.inf, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    ("0.7-1.0", 0.7, 1.0),
    ("1.0-1.5", 1.0, 1.5),
    (">1.5", 1.5, math.inf),
]


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        bb = fnum(r.get("t-1_bb_width_pct"))
        dv = fnum(r.get("t-1_dollarvol_ratio_5_20"))
        if bb is None or dv is None:
            continue
        x = dict(r)
        x["year"] = x["setup_date"][:4]
        x["hit200"] = int(float(x["hit200"]))
        x["_bb"] = bb
        x["_dv"] = dv
        out.append(x)
    return out


def assign_bin(x, bins):
    for label, lo, hi in bins:
        if lo <= x < hi:
            return label
    return None


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


rows = load(DISC) + load(HOLD)
rows = [r for r in rows if r["year"] in YEARS]

for r in rows:
    r["bb_bin"] = assign_bin(r["_bb"], BB_BINS)
    r["dv_bin"] = assign_bin(r["_dv"], DV_BINS)

print("\nRajih — BB WIDTH × DOLLAR-VOLUME RATIO SWEET-SPOT")
print(f"Top-1 observations: {len(rows)}")

# Overall/year baselines.
baseline = {}
for y in YEARS:
    yr = [r for r in rows if r["year"] == y]
    wins = sum(r["hit200"] for r in yr)
    baseline[y] = wins/len(yr) if yr else 0
    print(f"{y} baseline: {wins}/{len(yr)} = {100*baseline[y]:.2f}%")

all_wins = sum(r["hit200"] for r in rows)
all_base = all_wins/len(rows)
print(f"All years baseline: {all_wins}/{len(rows)} = {100*all_base:.2f}%")

# 2D grid.
grid = []
for bb_label, _, _ in BB_BINS:
    for dv_label, _, _ in DV_BINS:
        cell = [r for r in rows if r["bb_bin"] == bb_label and r["dv_bin"] == dv_label]
        if not cell:
            continue

        total_n = len(cell)
        total_w = sum(r["hit200"] for r in cell)
        rate = total_w/total_n

        row = {
            "bb_width_bin": bb_label,
            "dollarvol_ratio_bin": dv_label,
            "n_total": total_n,
            "winners_total": total_w,
            "hit200_rate_pct": 100*rate,
            "overall_lift_vs_baseline": rate/all_base if all_base else None,
        }

        yearly_rates = []
        yearly_lifts = []
        yearly_ns = []
        all_years_have_winner = True

        for y in YEARS:
            yr = [r for r in cell if r["year"] == y]
            n = len(yr)
            w = sum(r["hit200"] for r in yr)
            rr = w/n if n else None
            lift = rr/baseline[y] if rr is not None and baseline[y] else None

            row[f"n_{y}"] = n
            row[f"winners_{y}"] = w
            row[f"rate_{y}_pct"] = 100*rr if rr is not None else None
            row[f"lift_{y}"] = lift

            yearly_ns.append(n)
            if rr is not None:
                yearly_rates.append(rr)
            if lift is not None:
                yearly_lifts.append(lift)
            if w < 1:
                all_years_have_winner = False

        row["min_year_n"] = min(yearly_ns)
        row["all_3_years_have_winner"] = int(all_years_have_winner)
        row["min_year_rate_pct"] = 100*min(yearly_rates) if yearly_rates else None
        row["max_year_rate_pct"] = 100*max(yearly_rates) if yearly_rates else None
        row["year_rate_spread_pp"] = (
            100*(max(yearly_rates)-min(yearly_rates)) if yearly_rates else None
        )
        row["min_year_lift"] = min(yearly_lifts) if yearly_lifts else None
        row["max_year_lift"] = max(yearly_lifts) if yearly_lifts else None

        grid.append(row)

grid.sort(
    key=lambda r: (
        r["min_year_lift"] if r["min_year_lift"] is not None else -1,
        r["hit200_rate_pct"],
        r["n_total"],
    ),
    reverse=True
)
write_csv(OUT_GRID, grid)

# Stable sweet spots.
sweet = [
    r for r in grid
    if r["min_year_n"] >= 5
    and r["all_3_years_have_winner"] == 1
    and r["min_year_lift"] is not None
    and r["min_year_lift"] > 1.0
]

sweet.sort(
    key=lambda r: (
        r["min_year_lift"],
        r["hit200_rate_pct"],
        -r["year_rate_spread_pp"],
        r["n_total"],
    ),
    reverse=True
)
write_csv(OUT_SWEET, sweet)

# 1D summaries for BB width.
bb_summary = []
for bb_label, _, _ in BB_BINS:
    cell = [r for r in rows if r["bb_bin"] == bb_label]
    if not cell:
        continue
    row = {
        "bb_width_bin": bb_label,
        "n_total": len(cell),
        "winners_total": sum(r["hit200"] for r in cell),
    }
    row["hit200_rate_pct"] = 100*row["winners_total"]/row["n_total"]
    row["lift_vs_all_baseline"] = (
        (row["winners_total"]/row["n_total"])/all_base if all_base else None
    )
    for y in YEARS:
        yr = [r for r in cell if r["year"] == y]
        n = len(yr)
        w = sum(r["hit200"] for r in yr)
        row[f"n_{y}"] = n
        row[f"winners_{y}"] = w
        row[f"rate_{y}_pct"] = 100*w/n if n else None
    bb_summary.append(row)

write_csv(OUT_BB, bb_summary)

# 1D summaries for dollar-volume ratio.
dv_summary = []
for dv_label, _, _ in DV_BINS:
    cell = [r for r in rows if r["dv_bin"] == dv_label]
    if not cell:
        continue
    row = {
        "dollarvol_ratio_bin": dv_label,
        "n_total": len(cell),
        "winners_total": sum(r["hit200"] for r in cell),
    }
    row["hit200_rate_pct"] = 100*row["winners_total"]/row["n_total"]
    row["lift_vs_all_baseline"] = (
        (row["winners_total"]/row["n_total"])/all_base if all_base else None
    )
    for y in YEARS:
        yr = [r for r in cell if r["year"] == y]
        n = len(yr)
        w = sum(r["hit200"] for r in yr)
        row[f"n_{y}"] = n
        row[f"winners_{y}"] = w
        row[f"rate_{y}_pct"] = 100*w/n if n else None
    dv_summary.append(row)

write_csv(OUT_DV, dv_summary)

summary = {
    "baseline": {
        "all_years_pct": 100*all_base,
        **{f"{y}_pct":100*baseline[y] for y in YEARS},
    },
    "top_sweet_spots": sweet[:10],
    "top_grid_cells": grid[:15],
    "bb_width_summary": bb_summary,
    "dollarvol_ratio_summary": dv_summary,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== TOP STABLE SWEET SPOTS ===")
if not sweet:
    print("No cell met: >=5 observations in each year + >=1 winner each year + >1x lift each year.")
else:
    for i, r in enumerate(sweet[:12], 1):
        print(
            f"{i}. BB {r['bb_width_bin']} | DV {r['dollarvol_ratio_bin']} | "
            f"N={r['n_total']} W={r['winners_total']} "
            f"rate={r['hit200_rate_pct']:.2f}% | "
            f"min lift={r['min_year_lift']:.2f}x | "
            f"2024={r['rate_2024_pct']:.2f}% "
            f"2025={r['rate_2025_pct']:.2f}% "
            f"2026={r['rate_2026_pct']:.2f}%"
        )

print("\n=== ALL GRID CELLS — TOP BY MIN YEAR LIFT ===")
for i, r in enumerate(grid[:15], 1):
    print(
        f"{i}. BB {r['bb_width_bin']} | DV {r['dollarvol_ratio_bin']} | "
        f"N={r['n_total']} W={r['winners_total']} "
        f"rate={r['hit200_rate_pct']:.2f}% | "
        f"minN={r['min_year_n']} | "
        f"minLift={r['min_year_lift'] if r['min_year_lift'] is not None else 'NA'}"
    )

print("\n=== BB WIDTH ONLY ===")
for r in bb_summary:
    print(
        f"BB {r['bb_width_bin']}: N={r['n_total']} W={r['winners_total']} "
        f"rate={r['hit200_rate_pct']:.2f}% | "
        f"2024={r['rate_2024_pct']} 2025={r['rate_2025_pct']} 2026={r['rate_2026_pct']}"
    )

print("\n=== DOLLAR-VOLUME RATIO ONLY ===")
for r in dv_summary:
    print(
        f"DV {r['dollarvol_ratio_bin']}: N={r['n_total']} W={r['winners_total']} "
        f"rate={r['hit200_rate_pct']:.2f}% | "
        f"2024={r['rate_2024_pct']} 2025={r['rate_2025_pct']} 2026={r['rate_2026_pct']}"
    )

report = []
report.append("RAJIH — BB WIDTH × DOLLAR-VOLUME RATIO SWEET-SPOT")
report.append("="*95)
report.append(
    f"Overall Top-1 baseline +200 rate: {100*all_base:.2f}% "
    f"({all_wins}/{len(rows)})"
)
for y in YEARS:
    report.append(f"{y} baseline: {100*baseline[y]:.2f}%")
report.append("")

report.append("TOP STABLE SWEET SPOTS")
report.append("-"*95)
if sweet:
    for r in sweet[:15]:
        report.append(
            f"BB {r['bb_width_bin']} + DV {r['dollarvol_ratio_bin']} | "
            f"N={r['n_total']} W={r['winners_total']} rate={r['hit200_rate_pct']:.2f}% | "
            f"2024={r['rate_2024_pct']:.2f}% 2025={r['rate_2025_pct']:.2f}% "
            f"2026={r['rate_2026_pct']:.2f}% | minLift={r['min_year_lift']:.2f}x"
        )
else:
    report.append("No stable cell met all minimum-support conditions.")

report.append("")
report.append("BB WIDTH ONLY")
report.append("-"*95)
for r in bb_summary:
    report.append(
        f"{r['bb_width_bin']}: N={r['n_total']} W={r['winners_total']} "
        f"rate={r['hit200_rate_pct']:.2f}%"
    )

report.append("")
report.append("DOLLAR-VOLUME RATIO ONLY")
report.append("-"*95)
for r in dv_summary:
    report.append(
        f"{r['dollarvol_ratio_bin']}: N={r['n_total']} W={r['winners_total']} "
        f"rate={r['hit200_rate_pct']:.2f}%"
    )

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_GRID, OUT_SWEET, OUT_BB, OUT_DV, OUT_JSON, OUT_REPORT]:
    print(" ", p)
