#!/usr/bin/env python3
"""
Rajih — validate the two fixed early-confirmation rules across ALL available years.

Input:
  /data/daily_200pct_candidateA_early_features.csv

Candidate A pre-signal baseline:
  50 signals total, 14 reached +200% within 63 sessions.

Fixed confirmation rules:

RULE_14:
  Day-5 close above signal-day high
  AND first-5-day max drawdown >= -14.3149%

RULE_16:
  Day-5 close above signal-day high
  AND first-5-day max drawdown >= -16.2879%

No threshold search or tuning is performed here.

Reports for overall + each year 2024/2025/2026:
  N
  +200 winners
  +200 precision
  coverage/recall of Candidate-A +200 winners
  lift vs Candidate-A baseline
  +50 / +100 / +200
  median max gain 63d
  median day of first +200
  list of confirmed tickers/dates

Outputs:
  /data/daily_candidateA_confirmation_fixed_results.csv
  /data/daily_candidateA_confirmation_fixed_yearly.csv
  /data/daily_candidateA_confirmation_fixed_matches.csv
  /data/daily_candidateA_confirmation_fixed_summary.json
  /data/daily_candidateA_confirmation_fixed_report.txt

Run:
  python test_candidateA_fixed_confirmations_all_years.py
"""

from __future__ import annotations

import csv, json, math, os
from pathlib import Path
from statistics import median

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_200pct_candidateA_early_features.csv"

OUT_RESULTS = DATA_DIR / "daily_candidateA_confirmation_fixed_results.csv"
OUT_YEARLY = DATA_DIR / "daily_candidateA_confirmation_fixed_yearly.csv"
OUT_MATCHES = DATA_DIR / "daily_candidateA_confirmation_fixed_matches.csv"
OUT_JSON = DATA_DIR / "daily_candidateA_confirmation_fixed_summary.json"
OUT_REPORT = DATA_DIR / "daily_candidateA_confirmation_fixed_report.txt"

YEARS = ("2024","2025","2026")

RULES = [
    {
        "name": "CONFIRM_D5_HIGH_DD14",
        "dd_cut": -14.3149,
        "description": "d5_close_above_signal_high == 1 AND first5_max_drawdown_pct >= -14.3149",
    },
    {
        "name": "CONFIRM_D5_HIGH_DD16",
        "dd_cut": -16.2879,
        "description": "d5_close_above_signal_high == 1 AND first5_max_drawdown_pct >= -16.2879",
    },
]

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run analyze_candidateA_early_confirmation.py first.")


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def confirmed(row, rule):
    above = fnum(row.get("d5_close_above_signal_high"))
    dd = fnum(row.get("first5_max_drawdown_pct"))
    return above is not None and dd is not None and above >= 1 and dd >= rule["dd_cut"]


def summarize(sample, baseline_sample):
    n = len(sample)
    base_n = len(baseline_sample)
    base_w200 = sum(int(float(r["hit200"])) for r in baseline_sample)
    base_rate = base_w200/base_n if base_n else 0

    if n == 0:
        return {
            "n":0, "w200":0, "hit200_pct":0.0,
            "winner_recall_pct":0.0, "lift_vs_candidateA":None,
            "hit50_pct":None, "hit100_pct":None,
            "median_max_gain_63d_pct":None,
            "median_first200_day":None,
        }

    w200 = sum(int(float(r["hit200"])) for r in sample)

    def outcome_rate(th):
        key = f"hit{th}"
        if key in sample[0]:
            cnt = sum(int(float(r[key])) for r in sample)
            return 100*cnt/n
        # derive from max gain if hit50/hit100 weren't retained
        vals = [fnum(r.get("max_gain_63d_pct")) for r in sample]
        vals = [x for x in vals if x is not None]
        if not vals:
            return None
        return 100*sum(x >= th for x in vals)/len(vals)

    maxg = [fnum(r.get("max_gain_63d_pct")) for r in sample]
    maxg = [x for x in maxg if x is not None]
    d200 = [fnum(r.get("first200_day")) for r in sample]
    d200 = [x for x in d200 if x is not None]

    rate = w200/n
    return {
        "n": n,
        "w200": w200,
        "hit200_pct": round(100*rate,4),
        "winner_recall_pct": round(100*w200/base_w200,4) if base_w200 else None,
        "lift_vs_candidateA": round(rate/base_rate,4) if base_rate else None,
        "hit50_pct": round(outcome_rate(50),4) if outcome_rate(50) is not None else None,
        "hit100_pct": round(outcome_rate(100),4) if outcome_rate(100) is not None else None,
        "median_max_gain_63d_pct": round(median(maxg),4) if maxg else None,
        "median_first200_day": round(median(d200),2) if d200 else None,
    }


with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

for r in rows:
    if not r.get("year"):
        r["year"] = r["setup_date"][:4]

baseline_all = rows
baseline_year = {y:[r for r in rows if r["year"] == y] for y in YEARS}

base_w = sum(int(float(r["hit200"])) for r in rows)

print("\nRajih — FIXED CANDIDATE-A EARLY CONFIRMATIONS / ALL YEARS")
print(f"Candidate A baseline: N={len(rows)} | +200 winners={base_w} | rate={100*base_w/len(rows):.2f}%")

for y in YEARS:
    yy = baseline_year[y]
    w = sum(int(float(r["hit200"])) for r in yy)
    print(f"{y} baseline: N={len(yy)} | W200={w} | rate={100*w/len(yy):.2f}%")

results = []
yearly = []
matches = []
summary = {
    "candidateA_baseline": {
        "n": len(rows),
        "w200": base_w,
        "hit200_pct": 100*base_w/len(rows),
    },
    "rules": {},
}

for rule in RULES:
    selected = [r for r in rows if confirmed(r, rule)]
    overall = summarize(selected, rows)
    result = {
        "rule_name": rule["name"],
        "rule": rule["description"],
        **overall,
    }
    results.append(result)

    yrs = {}
    for y in YEARS:
        base = baseline_year[y]
        sel = [r for r in selected if r["year"] == y]
        sy = summarize(sel, base)
        row = {
            "rule_name": rule["name"],
            "year": y,
            **sy,
        }
        yearly.append(row)
        yrs[y] = row

    for r in selected:
        x = dict(r)
        x["confirmed_rule"] = rule["name"]
        matches.append(x)

    summary["rules"][rule["name"]] = {
        "definition": rule["description"],
        "overall": result,
        "yearly": yrs,
    }

write_csv(OUT_RESULTS, results)
write_csv(OUT_YEARLY, yearly)
write_csv(OUT_MATCHES, matches)
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== OVERALL FIXED CONFIRMATION RESULTS ===")
for r in results:
    print(
        f"{r['rule_name']}: N={r['n']} | W200={r['w200']} | "
        f"+200={r['hit200_pct']}% | recall={r['winner_recall_pct']}% | "
        f"lift={r['lift_vs_candidateA']}x | "
        f"+50={r['hit50_pct']}% | +100={r['hit100_pct']}%"
    )

print("\n=== YEARLY RESULTS ===")
for rule in RULES:
    print(f"\n{rule['name']}")
    for r in yearly:
        if r["rule_name"] != rule["name"]:
            continue
        print(
            f" {r['year']}: N={r['n']} | W200={r['w200']} | "
            f"+200={r['hit200_pct']}% | recall={r['winner_recall_pct']}% | "
            f"lift={r['lift_vs_candidateA']}x"
        )

print("\n=== CONFIRMED CASES ===")
for rule in RULES:
    print(f"\n{rule['name']}")
    selected = [r for r in matches if r["confirmed_rule"] == rule["name"]]
    for r in sorted(selected, key=lambda z:(z["year"],z["setup_date"],z["symbol"])):
        print(
            f" {r['year']} | {r['setup_date']} | {r['symbol']:<7} | "
            f"hit200={r['hit200']} | "
            f"D5_above_high={r['d5_close_above_signal_high']} | "
            f"DD5={float(r['first5_max_drawdown_pct']):.2f}% | "
            f"Max63={float(r['max_gain_63d_pct']):.1f}%"
        )

report = []
report.append("RAJIH — FIXED CANDIDATE-A EARLY CONFIRMATIONS / ALL YEARS")
report.append("="*100)
report.append(
    f"Candidate A baseline: N={len(rows)} W200={base_w} rate={100*base_w/len(rows):.2f}%"
)
report.append("")

for r in results:
    report.append(
        f"{r['rule_name']} | N={r['n']} W200={r['w200']} | "
        f"+200={r['hit200_pct']}% | recall={r['winner_recall_pct']}% | "
        f"lift={r['lift_vs_candidateA']}x | +50={r['hit50_pct']}% | +100={r['hit100_pct']}%"
    )
    for yr in yearly:
        if yr["rule_name"] == r["rule_name"]:
            report.append(
                f"  {yr['year']}: N={yr['n']} W200={yr['w200']} "
                f"rate={yr['hit200_pct']}% recall={yr['winner_recall_pct']}% "
                f"lift={yr['lift_vs_candidateA']}x"
            )
    report.append("")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_RESULTS, OUT_YEARLY, OUT_MATCHES, OUT_JSON, OUT_REPORT]:
    print(" ", p)
