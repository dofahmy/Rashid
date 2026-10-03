#!/usr/bin/env python3
"""
Rajih — locked stage-2 rule tested on ONE-PER-DATE samples.

NO threshold tuning in this script.

Inputs:
  /data/daily_200pct_stage2_enriched_signals.csv
  /data/daily_200pct_stage2_locked_rule.json

Locked stage-2 rule is loaded from JSON exactly as previously selected
using 2024–2025 discovery only.

This script reports TWO clean one-per-date views:

A) RULE-FIRST ONE-PER-DATE
   1. Apply the already-locked stage-2 rule to all first-stage signals.
   2. For each setup date, keep the passing signal with highest
      pre-signal 20-day dollar volume.

   This represents: "if several passing candidates appear today,
   take only the most liquid one."

B) DATE-FIRST ONE-PER-DATE
   1. For each setup date, first keep the highest-liquidity first-stage signal.
   2. Then check whether that one signal passes the locked stage-2 rule.

   This is a stricter regime-control test.

For each view, reports:
  +20 / +50 / +100 / +200 within 63 sessions
  median max gain
  median max drawdown
  median close return at day 63
  median first +200 day
  yearly 2024 / 2025 / 2026

Outputs:
  /data/daily_200pct_locked_rule_one_per_date_rule_first.csv
  /data/daily_200pct_locked_rule_one_per_date_date_first.csv
  /data/daily_200pct_locked_rule_one_per_date_yearly.csv
  /data/daily_200pct_locked_rule_one_per_date_summary.json
  /data/daily_200pct_locked_rule_one_per_date_report.txt

Run:
  python validate_200pct_locked_rule_one_per_date.py
"""

from __future__ import annotations

import csv, json, math, os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

INPUT = DATA_DIR / "daily_200pct_stage2_enriched_signals.csv"
LOCK_FILE = DATA_DIR / "daily_200pct_stage2_locked_rule.json"

OUT_RULE_FIRST = DATA_DIR / "daily_200pct_locked_rule_one_per_date_rule_first.csv"
OUT_DATE_FIRST = DATA_DIR / "daily_200pct_locked_rule_one_per_date_date_first.csv"
OUT_YEARLY = DATA_DIR / "daily_200pct_locked_rule_one_per_date_yearly.csv"
OUT_JSON = DATA_DIR / "daily_200pct_locked_rule_one_per_date_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_locked_rule_one_per_date_report.txt"

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}")
if not LOCK_FILE.exists():
    raise SystemExit(f"Missing {LOCK_FILE}")


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def write_csv(path, rows):
    if not rows:
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


def parse_rule(lock_json):
    rule = lock_json.get("locked_rule")
    if not rule:
        raise SystemExit("locked_rule not found in locked_rule.json")
    return rule


def one_pass(row, rr):
    x = fnum(row.get(rr["feature"]))
    if x is None:
        return False
    cut = float(rr["cut"])
    if rr["operator"] == ">=":
        return x >= cut
    if rr["operator"] == "<=":
        return x <= cut
    raise ValueError(f"Unknown operator {rr['operator']}")


def rule_pass(row, rule):
    if not one_pass(row, rule["r1"]):
        return False
    if rule.get("type") == "pair":
        return one_pass(row, rule["r2"])
    return True


def liq(row):
    # Prefer the original first-stage pre-signal liquidity field.
    for k in (
        "avg_dollar_volume20_tminus1",
        "t-1_avg_dollar_volume20",
        "avg_dollar_volume20",
    ):
        x = fnum(row.get(k))
        if x is not None:
            return x
    return -1.0


def enrich_types(row):
    x = dict(row)
    for k in ("hit20","hit50","hit100","hit200","future_discontinuity_flag"):
        if k in x and x[k] not in ("",None):
            x[k] = int(float(x[k]))
    for k in (
        "max_gain_63d_pct",
        "max_drawdown_63d_pct",
        "close_return_day63_pct",
        "first200_day",
    ):
        if k in x and x[k] not in ("",None):
            x[k] = fnum(x[k])
    x["year"] = x["setup_date"][:4]
    return x


def summarize(rows):
    if not rows:
        return {"n":0}

    n = len(rows)

    def hit(k):
        c = sum(int(r[k]) for r in rows)
        return c, 100*c/n

    h20 = hit("hit20")
    h50 = hit("hit50")
    h100 = hit("hit100")
    h200 = hit("hit200")

    mg = [fnum(r["max_gain_63d_pct"]) for r in rows]
    dd = [fnum(r["max_drawdown_63d_pct"]) for r in rows]
    cr = [fnum(r["close_return_day63_pct"]) for r in rows]
    d200 = [fnum(r.get("first200_day")) for r in rows if fnum(r.get("first200_day")) is not None]

    mg = [x for x in mg if x is not None]
    dd = [x for x in dd if x is not None]
    cr = [x for x in cr if x is not None]

    return {
        "n": n,
        "hit20_n": h20[0], "hit20_pct": round(h20[1],4),
        "hit50_n": h50[0], "hit50_pct": round(h50[1],4),
        "hit100_n": h100[0], "hit100_pct": round(h100[1],4),
        "hit200_n": h200[0], "hit200_pct": round(h200[1],4),
        "median_max_gain_63d_pct": round(median(mg),4) if mg else None,
        "mean_max_gain_63d_pct": round(mean(mg),4) if mg else None,
        "median_max_drawdown_63d_pct": round(median(dd),4) if dd else None,
        "median_close_return_day63_pct": round(median(cr),4) if cr else None,
        "median_first200_day": round(median(d200),2) if d200 else None,
    }


with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    rows = [enrich_types(r) for r in csv.DictReader(fh)]

lock_json = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
rule = parse_rule(lock_json)
rule_desc = lock_json.get("locked_rule_description") or str(rule)

print("\nRajih — LOCKED RULE / ONE-PER-DATE VALIDATION")
print(f"Input signals: {len(rows)}")
print(f"Locked rule: {rule_desc}")
print("No thresholds are changed in this script.\n")

# Baseline one-per-date from ALL first-stage signals.
by_date_all = defaultdict(list)
for r in rows:
    by_date_all[r["setup_date"]].append(r)

baseline_one_per_date = []
for d in sorted(by_date_all):
    baseline_one_per_date.append(max(by_date_all[d], key=lambda r:(liq(r), r["symbol"])))

# A) Rule first, then choose one highest-liquidity passing candidate/date.
passing = [r for r in rows if rule_pass(r, rule)]
by_date_pass = defaultdict(list)
for r in passing:
    by_date_pass[r["setup_date"]].append(r)

rule_first = []
for d in sorted(by_date_pass):
    rule_first.append(max(by_date_pass[d], key=lambda r:(liq(r), r["symbol"])))

# B) One per date first, then locked rule.
date_first = [r for r in baseline_one_per_date if rule_pass(r, rule)]

write_csv(OUT_RULE_FIRST, rule_first)
write_csv(OUT_DATE_FIRST, date_first)

base_summary = summarize(baseline_one_per_date)
rf_summary = summarize(rule_first)
df_summary = summarize(date_first)

# Yearly tables.
year_rows = []
for label, sample in (
    ("baseline_one_per_date", baseline_one_per_date),
    ("rule_first_one_per_date", rule_first),
    ("date_first_then_rule", date_first),
):
    by_year = defaultdict(list)
    for r in sample:
        by_year[r["year"]].append(r)
    for year in sorted(by_year):
        s = summarize(by_year[year])
        year_rows.append({"sample":label, "year":year, **s})

write_csv(OUT_YEARLY, year_rows)

summary = {
    "locked_rule_description": rule_desc,
    "locked_rule": rule,
    "baseline_one_per_date": base_summary,
    "rule_first_one_per_date": rf_summary,
    "date_first_then_rule": df_summary,
    "yearly": year_rows,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

def p(label, s):
    print(f"\n=== {label} ===")
    print(f"N: {s.get('n',0)}")
    if not s.get("n"):
        return
    print(f"+20%:  {s['hit20_n']}/{s['n']} = {s['hit20_pct']}%")
    print(f"+50%:  {s['hit50_n']}/{s['n']} = {s['hit50_pct']}%")
    print(f"+100%: {s['hit100_n']}/{s['n']} = {s['hit100_pct']}%")
    print(f"+200%: {s['hit200_n']}/{s['n']} = {s['hit200_pct']}%")
    print(f"Median max gain 63d: {s['median_max_gain_63d_pct']}%")
    print(f"Median max drawdown 63d: {s['median_max_drawdown_63d_pct']}%")
    print(f"Median close return day63: {s['median_close_return_day63_pct']}%")
    print(f"Median first +200 day: {s['median_first200_day']}")

p("BASELINE — ONE PER DATE, BEFORE STAGE-2 RULE", base_summary)
p("RULE FIRST -> ONE PER DATE", rf_summary)
p("ONE PER DATE FIRST -> LOCKED RULE", df_summary)

print("\n=== YEARLY ===")
for r in year_rows:
    print(
        f"{r['sample']:<27} {r['year']} | N={r['n']:<4} | "
        f"+50={r['hit50_pct']:>7}% | +100={r['hit100_pct']:>7}% | "
        f"+200={r['hit200_pct']:>7}% | "
        f"MedDD={r['median_max_drawdown_63d_pct']:>8}%"
    )

# Lift versus same-sample one-per-date baseline, overall and per year.
print("\n=== LIFT VS ONE-PER-DATE BASELINE ===")
if base_summary.get("hit200_pct"):
    print(
        f"Rule-first overall +200 lift: "
        f"{rf_summary['hit200_pct']/base_summary['hit200_pct']:.2f}x"
    )
    print(
        f"Date-first overall +200 lift: "
        f"{df_summary['hit200_pct']/base_summary['hit200_pct']:.2f}x"
    )

base_year = {r["year"]:r for r in year_rows if r["sample"]=="baseline_one_per_date"}
rf_year = {r["year"]:r for r in year_rows if r["sample"]=="rule_first_one_per_date"}
df_year = {r["year"]:r for r in year_rows if r["sample"]=="date_first_then_rule"}

for year in sorted(base_year):
    b = base_year[year]["hit200_pct"]
    rfv = rf_year.get(year,{}).get("hit200_pct")
    dfv = df_year.get(year,{}).get("hit200_pct")
    if b:
        print(
            f"{year}: baseline={b:.4f}% | "
            f"rule-first={rfv:.4f}% ({rfv/b:.2f}x) | "
            f"date-first={dfv:.4f}% ({dfv/b:.2f}x)"
        )

report = []
report.append("RAJIH — LOCKED RULE / ONE-PER-DATE VALIDATION")
report.append("="*90)
report.append(f"Locked rule: {rule_desc}")
report.append("")
for label,s in (
    ("BASELINE ONE-PER-DATE",base_summary),
    ("RULE-FIRST ONE-PER-DATE",rf_summary),
    ("DATE-FIRST THEN RULE",df_summary),
):
    report.append(label)
    report.append("-"*90)
    report.append(
        f"N={s.get('n',0)} | +20={s.get('hit20_pct')}% | +50={s.get('hit50_pct')}% | "
        f"+100={s.get('hit100_pct')}% | +200={s.get('hit200_pct')}%"
    )
    report.append(
        f"Median max63={s.get('median_max_gain_63d_pct')}% | "
        f"Median DD63={s.get('median_max_drawdown_63d_pct')}% | "
        f"Median close63={s.get('median_close_return_day63_pct')}% | "
        f"Median first +200 day={s.get('median_first200_day')}"
    )
    report.append("")

report.append("YEARLY")
report.append("-"*90)
for r in year_rows:
    report.append(
        f"{r['sample']} {r['year']} | N={r['n']} | "
        f"+50={r['hit50_pct']}% | +100={r['hit100_pct']}% | "
        f"+200={r['hit200_pct']}% | MedDD={r['median_max_drawdown_63d_pct']}%"
    )

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for pth in [OUT_RULE_FIRST,OUT_DATE_FIRST,OUT_YEARLY,OUT_JSON,OUT_REPORT]:
    print(" ",pth)
