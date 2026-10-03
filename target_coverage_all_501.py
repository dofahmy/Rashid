#!/usr/bin/env python3
"""
Rajih — target coverage across all 501 distinct-date setups.

Reads:
  /data/daily_rule_all_dates.csv

Uses the already-computed peak metric:
  max_high_21d_pct

Calculates the maximum target that was reached by at least:
  90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%
of all setups within 21 trading sessions.

Also prints how many setups reached common targets:
  +5, +10, +15, +20, +25, +30, +40, +50

Outputs:
  /data/daily_501_target_coverage.csv
  /data/daily_501_target_coverage_summary.json
"""

import os, csv, math, json
from pathlib import Path

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_rule_all_dates.csv"
OUT_CSV = DATA_DIR / "daily_501_target_coverage.csv"
OUT_JSON = DATA_DIR / "daily_501_target_coverage_summary.json"

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run test_daily_rule_all_501_dates.py first.")

rows = []
with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        try:
            x = float(r["max_high_21d_pct"])
            if math.isfinite(x):
                rows.append((r["symbol"], r["signal_date"], x))
        except:
            pass

if not rows:
    raise SystemExit("No usable max_high_21d_pct values.")

vals = sorted(rows, key=lambda x: x[2], reverse=True)
n = len(vals)

coverage_levels = [90,80,70,60,50,40,30,20,10]
coverage_rows = []

print("\n=== TARGET COVERAGE — ALL 501 SETUPS ===")
print(f"Usable setups: {n}")
print("Meaning: at least this share of setups reached the target at some point within 21 sessions.\n")

for pct in coverage_levels:
    need = math.ceil(n * pct / 100)
    target = vals[need-1][2]
    cutoff_symbol = vals[need-1][0]
    cutoff_date = vals[need-1][1]
    coverage_rows.append({
        "coverage_pct": pct,
        "stocks_required": need,
        "target_pct": round(target, 4),
        "cutoff_symbol": cutoff_symbol,
        "cutoff_date": cutoff_date,
    })
    print(f"{pct:>3}% coverage ({need}/{n}): target <= {target:.2f}%")

common_targets = [5,10,15,20,25,30,40,50]
common = []
print("\n=== COMMON TARGET HIT RATES ===")
for th in common_targets:
    count = sum(x[2] >= th for x in vals)
    rate = 100 * count / n
    common.append({
        "target_pct": th,
        "hit_n": count,
        "hit_rate_pct": round(rate, 4),
    })
    print(f"+{th:>2}%: {count}/{n} = {rate:.2f}%")

with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(coverage_rows[0].keys()))
    w.writeheader()
    w.writerows(coverage_rows)

summary = {
    "usable_setups": n,
    "coverage": coverage_rows,
    "common_targets": common,
}

OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nCreated:")
print(f" {OUT_CSV}")
print(f" {OUT_JSON}")
