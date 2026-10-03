#!/usr/bin/env python3
"""
Rajih — DAILY +200% candidate ranking
Train ranking score on 2024–2025 ONLY, then test Top-1 per date on untouched 2026.

Population:
- Starts from /data/daily_200pct_stage2_enriched_signals.csv
- Applies the already-locked stage-2 filter from:
  /data/daily_200pct_stage2_locked_rule.json
- Ranking is learned ONLY among signals that pass that locked filter.

Goal:
When several candidates appear on the same day, rank them and choose Top-1.

Anti-overfit discipline:
1) 2026 is NOT used for feature choice, direction, scaling, or weights.
2) Features must show the SAME winner direction in both 2024 and 2025.
3) Features must have enough non-null observations.
4) Use a simple additive robust z-score, not a high-capacity model.
5) Weights are determined only from the weaker of 2024/2025 univariate separations.
6) After the score is locked, evaluate Top-1/day on 2026 exactly once.

Discovery target:
- hit200 == 1 within 63 sessions.

Outputs:
  /data/daily_200pct_ranking_selected_features.csv
  /data/daily_200pct_ranking_discovery_top1.csv
  /data/daily_200pct_ranking_holdout_2026_top1.csv
  /data/daily_200pct_ranking_all_scored.csv
  /data/daily_200pct_ranking_summary.json
  /data/daily_200pct_ranking_report.txt

Run:
  python build_200pct_daily_ranking_holdout_2026.py
"""

from __future__ import annotations

import csv, json, math, os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))

INPUT = DATA_DIR / "daily_200pct_stage2_enriched_signals.csv"
LOCK_FILE = DATA_DIR / "daily_200pct_stage2_locked_rule.json"

OUT_FEATURES = DATA_DIR / "daily_200pct_ranking_selected_features.csv"
OUT_DISC_TOP1 = DATA_DIR / "daily_200pct_ranking_discovery_top1.csv"
OUT_HOLD_TOP1 = DATA_DIR / "daily_200pct_ranking_holdout_2026_top1.csv"
OUT_SCORED = DATA_DIR / "daily_200pct_ranking_all_scored.csv"
OUT_JSON = DATA_DIR / "daily_200pct_ranking_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_ranking_report.txt"

DISCOVERY_YEARS = {"2024", "2025"}
HOLDOUT_YEAR = "2026"

# Keep the ranker deliberately simple.
MAX_FEATURES = 8
MIN_YEAR_WINNERS = 20
MIN_YEAR_LOSERS = 200
MIN_SEPARATION_EACH_YEAR = 0.54

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


def robust_stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 10:
        return None, None
    med = median(xs)
    absdev = sorted(abs(x-med) for x in xs)
    mad = median(absdev)
    # Convert MAD to normal-like robust sigma.
    scale = 1.4826 * mad
    if scale <= 1e-12:
        # fallback standard deviation
        m = mean(xs)
        scale = math.sqrt(sum((x-m)**2 for x in xs)/max(1, len(xs)-1))
    if scale <= 1e-12:
        scale = 1.0
    return med, scale


def one_pass(row, rr):
    x = fnum(row.get(rr["feature"]))
    if x is None:
        return False
    cut = float(rr["cut"])
    if rr["operator"] == ">=":
        return x >= cut
    if rr["operator"] == "<=":
        return x <= cut
    raise ValueError(rr["operator"])


def rule_pass(row, rule):
    if not one_pass(row, rule["r1"]):
        return False
    if rule.get("type") == "pair":
        return one_pass(row, rule["r2"])
    return True


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

    maxg = [fnum(r.get("max_gain_63d_pct")) for r in rows]
    dd = [fnum(r.get("max_drawdown_63d_pct")) for r in rows]
    close63 = [fnum(r.get("close_return_day63_pct")) for r in rows]
    d200 = [fnum(r.get("first200_day")) for r in rows]

    maxg = [x for x in maxg if x is not None]
    dd = [x for x in dd if x is not None]
    close63 = [x for x in close63 if x is not None]
    d200 = [x for x in d200 if x is not None]

    return {
        "n": n,
        "hit20_n": h20[0], "hit20_pct": round(h20[1],4),
        "hit50_n": h50[0], "hit50_pct": round(h50[1],4),
        "hit100_n": h100[0], "hit100_pct": round(h100[1],4),
        "hit200_n": h200[0], "hit200_pct": round(h200[1],4),
        "median_max_gain_63d_pct": round(median(maxg),4) if maxg else None,
        "median_max_drawdown_63d_pct": round(median(dd),4) if dd else None,
        "median_close_return_day63_pct": round(median(close63),4) if close63 else None,
        "median_first200_day": round(median(d200),2) if d200 else None,
    }


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


# Load data.
with INPUT.open(encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

lock_json = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
locked_rule = lock_json["locked_rule"]
locked_desc = lock_json.get("locked_rule_description", str(locked_rule))

for r in rows:
    r["year"] = r["setup_date"][:4]
    for k in ("hit20","hit50","hit100","hit200"):
        r[k] = int(float(r[k]))

# Apply stage-2 locked filter first.
population = [r for r in rows if rule_pass(r, locked_rule)]

discovery = [r for r in population if r["year"] in DISCOVERY_YEARS]
d24 = [r for r in discovery if r["year"] == "2024"]
d25 = [r for r in discovery if r["year"] == "2025"]
holdout = [r for r in population if r["year"] == HOLDOUT_YEAR]

print("\nRajih — +200% DAILY RANKING")
print(f"Locked stage-2 filter: {locked_desc}")
print(f"Population after stage-2 filter: {len(population)}")
print(f"Discovery 2024–2025: {len(discovery)}")
print(f"2026 holdout: {len(holdout)}")
print("\nSelecting ranking features using 2024–2025 ONLY...\n")

# Candidate numeric pre-signal features.
feature_cols = [
    k for k in rows[0].keys()
    if k.startswith("t-") and k not in {
        # Exclude booleans because robust z-scoring them is not useful here.
        "t-20_above_sma20","t-20_above_sma50","t-20_above_sma200",
        "t-10_above_sma20","t-10_above_sma50","t-10_above_sma200",
        "t-5_above_sma20","t-5_above_sma50","t-5_above_sma200",
        "t-3_above_sma20","t-3_above_sma50","t-3_above_sma200",
        "t-1_above_sma20","t-1_above_sma50","t-1_above_sma200",
    }
]

feature_rows = []

for col in feature_cols:
    yearly = {}
    valid = True

    for year, sample in (("2024", d24), ("2025", d25)):
        w = [fnum(r.get(col)) for r in sample if r["hit200"] == 1]
        l = [fnum(r.get(col)) for r in sample if r["hit200"] == 0]
        w = [x for x in w if x is not None]
        l = [x for x in l if x is not None]

        if len(w) < MIN_YEAR_WINNERS or len(l) < MIN_YEAR_LOSERS:
            valid = False
            break

        a = auc(w, l)
        if a is None:
            valid = False
            break

        direction = 1 if a >= 0.5 else -1
        sep = max(a, 1-a)

        yearly[year] = {
            "auc": a,
            "direction": direction,
            "separation": sep,
            "winner_median": median(w),
            "loser_median": median(l),
            "winner_n": len(w),
            "loser_n": len(l),
        }

    if not valid:
        continue

    # Must point the same direction in both years.
    if yearly["2024"]["direction"] != yearly["2025"]["direction"]:
        continue

    if min(yearly["2024"]["separation"], yearly["2025"]["separation"]) < MIN_SEPARATION_EACH_YEAR:
        continue

    vals = [fnum(r.get(col)) for r in discovery]
    vals = [x for x in vals if x is not None]
    med, scale = robust_stats(vals)
    if med is None:
        continue

    direction = yearly["2024"]["direction"]
    min_sep = min(yearly["2024"]["separation"], yearly["2025"]["separation"])
    avg_sep = (yearly["2024"]["separation"] + yearly["2025"]["separation"]) / 2

    # Conservative weight based on the weaker discovery year.
    # 0.50 = no information.
    weight = max(0.0, min_sep - 0.50)

    feature_rows.append({
        "feature": col,
        "direction": "HIGHER" if direction == 1 else "LOWER",
        "direction_sign": direction,
        "weight": weight,
        "robust_median_2024_2025": med,
        "robust_scale_2024_2025": scale,
        "sep_2024": yearly["2024"]["separation"],
        "sep_2025": yearly["2025"]["separation"],
        "min_sep": min_sep,
        "avg_sep": avg_sep,
        "winner_median_2024": yearly["2024"]["winner_median"],
        "loser_median_2024": yearly["2024"]["loser_median"],
        "winner_median_2025": yearly["2025"]["winner_median"],
        "loser_median_2025": yearly["2025"]["loser_median"],
    })

feature_rows.sort(key=lambda r:(r["min_sep"], r["avg_sep"]), reverse=True)

# Reduce redundant features from same metric family/checkpoint.
selected = []
used_families = set()

def family(name):
    # e.g. t-20_distance_to_252d_high_pct -> distance_to_252d_high_pct
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 else name

for r in feature_rows:
    fam = family(r["feature"])
    if fam in used_families:
        continue
    selected.append(r)
    used_families.add(fam)
    if len(selected) >= MAX_FEATURES:
        break

if not selected:
    raise SystemExit("No stable discovery features passed selection criteria.")

write_csv(OUT_FEATURES, selected)

print("LOCKED RANKING FEATURES:")
for i, f in enumerate(selected, 1):
    print(
        f"{i}. {f['feature']} | {f['direction']} | "
        f"sep24={f['sep_2024']:.3f} sep25={f['sep_2025']:.3f} "
        f"weight={f['weight']:.4f}"
    )

# Score ALL stage-2 population using discovery-only parameters.
def score_row(r):
    total = 0.0
    used = 0
    parts = []

    for f in selected:
        x = fnum(r.get(f["feature"]))
        if x is None:
            continue
        z = (x - f["robust_median_2024_2025"]) / f["robust_scale_2024_2025"]
        # clip so one extreme feature cannot dominate.
        z = max(-3.0, min(3.0, z))
        contrib = f["direction_sign"] * f["weight"] * z
        total += contrib
        used += 1
        parts.append(contrib)

    if used == 0:
        return None, 0

    # Normalize by sum of weights actually available.
    denom = sum(f["weight"] for f in selected if fnum(r.get(f["feature"])) is not None)
    if denom <= 0:
        return None, used

    return total / denom, used


scored = []
for r in population:
    x = dict(r)
    s, used = score_row(r)
    x["rank_score"] = s
    x["rank_features_used"] = used
    scored.append(x)

# Keep only rows with a valid score.
scored_valid = [r for r in scored if fnum(r.get("rank_score")) is not None]

# Top-1 per date.
def top1_by_date(sample):
    by = defaultdict(list)
    for r in sample:
        by[r["setup_date"]].append(r)
    out = []
    for d in sorted(by):
        pool = sorted(
            by[d],
            key=lambda r: (
                fnum(r.get("rank_score")) if fnum(r.get("rank_score")) is not None else -1e99,
                fnum(r.get("avg_dollar_volume20_tminus1")) if fnum(r.get("avg_dollar_volume20_tminus1")) is not None else -1,
                r["symbol"],
            ),
            reverse=True
        )
        best = dict(pool[0])
        best["candidates_same_date"] = len(pool)
        out.append(best)
    return out

disc_scored = [r for r in scored_valid if r["year"] in DISCOVERY_YEARS]
hold_scored = [r for r in scored_valid if r["year"] == HOLDOUT_YEAR]

disc_top1 = top1_by_date(disc_scored)

# IMPORTANT: holdout is only evaluated AFTER feature selection and weights are locked.
hold_top1 = top1_by_date(hold_scored)

# Liquidity-only Top-1 baseline on SAME stage-2 passing population.
def liquidity_top1(sample):
    by = defaultdict(list)
    for r in sample:
        by[r["setup_date"]].append(r)
    out = []
    for d in sorted(by):
        pool = sorted(
            by[d],
            key=lambda r: (
                fnum(r.get("avg_dollar_volume20_tminus1")) if fnum(r.get("avg_dollar_volume20_tminus1")) is not None else -1,
                r["symbol"],
            ),
            reverse=True
        )
        out.append(pool[0])
    return out

disc_liq = liquidity_top1([r for r in population if r["year"] in DISCOVERY_YEARS])
hold_liq = liquidity_top1([r for r in population if r["year"] == HOLDOUT_YEAR])

write_csv(OUT_DISC_TOP1, disc_top1)
write_csv(OUT_HOLD_TOP1, hold_top1)
write_csv(OUT_SCORED, scored_valid)

disc_rank_sum = summarize(disc_top1)
disc_liq_sum = summarize(disc_liq)
hold_rank_sum = summarize(hold_top1)
hold_liq_sum = summarize(hold_liq)

# Year-by-year discovery diagnostics.
yearly = {}
for year in ("2024","2025","2026"):
    rtop = [r for r in (disc_top1 + hold_top1) if r["year"] == year]
    ltop = [r for r in (disc_liq + hold_liq) if r["year"] == year]
    yearly[year] = {
        "rank_top1": summarize(rtop),
        "liquidity_top1": summarize(ltop),
    }

summary = {
    "stage2_locked_rule": locked_desc,
    "ranking_training_years": ["2024","2025"],
    "holdout_year": "2026",
    "selected_features": selected,
    "discovery_2024_2025": {
        "rank_top1": disc_rank_sum,
        "liquidity_top1": disc_liq_sum,
    },
    "holdout_2026": {
        "rank_top1": hold_rank_sum,
        "liquidity_top1": hold_liq_sum,
    },
    "yearly": yearly,
}

OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== DISCOVERY 2024–2025 — TOP1/DAY ===")
print(
    f"Liquidity baseline: N={disc_liq_sum['n']} "
    f"+50={disc_liq_sum['hit50_pct']}% "
    f"+100={disc_liq_sum['hit100_pct']}% "
    f"+200={disc_liq_sum['hit200_pct']}%"
)
print(
    f"Rank score Top1:    N={disc_rank_sum['n']} "
    f"+50={disc_rank_sum['hit50_pct']}% "
    f"+100={disc_rank_sum['hit100_pct']}% "
    f"+200={disc_rank_sum['hit200_pct']}%"
)

print("\n=== 2026 UNTOUCHED HOLDOUT — TOP1/DAY ===")
print(
    f"Liquidity baseline: N={hold_liq_sum['n']} "
    f"+20={hold_liq_sum['hit20_pct']}% "
    f"+50={hold_liq_sum['hit50_pct']}% "
    f"+100={hold_liq_sum['hit100_pct']}% "
    f"+200={hold_liq_sum['hit200_pct']}% "
    f"MedDD={hold_liq_sum['median_max_drawdown_63d_pct']}%"
)
print(
    f"Rank score Top1:    N={hold_rank_sum['n']} "
    f"+20={hold_rank_sum['hit20_pct']}% "
    f"+50={hold_rank_sum['hit50_pct']}% "
    f"+100={hold_rank_sum['hit100_pct']}% "
    f"+200={hold_rank_sum['hit200_pct']}% "
    f"MedDD={hold_rank_sum['median_max_drawdown_63d_pct']}%"
)

if hold_liq_sum.get("hit200_pct"):
    print(
        f"+200 lift vs liquidity baseline: "
        f"{hold_rank_sum['hit200_pct']/hold_liq_sum['hit200_pct']:.2f}x"
    )

print("\n=== YEARLY TOP1 COMPARISON ===")
for year in ("2024","2025","2026"):
    a = yearly[year]["liquidity_top1"]
    b = yearly[year]["rank_top1"]
    print(
        f"{year} | liquidity: N={a.get('n',0)} +200={a.get('hit200_pct')}% "
        f"| rank: N={b.get('n',0)} +200={b.get('hit200_pct')}%"
    )

report = []
report.append("RAJIH — +200% DAILY RANKING / 2026 HOLDOUT")
report.append("="*88)
report.append(f"Stage-2 locked rule: {locked_desc}")
report.append("")
report.append("RANKING FEATURES — LOCKED FROM 2024–2025 ONLY")
report.append("-"*88)
for i,f in enumerate(selected,1):
    report.append(
        f"{i}. {f['feature']} | {f['direction']} | "
        f"sep24={f['sep_2024']:.4f} | sep25={f['sep_2025']:.4f} | "
        f"weight={f['weight']:.4f}"
    )
report.append("")
report.append("DISCOVERY 2024–2025 TOP1/DAY")
report.append("-"*88)
report.append(
    f"Liquidity: N={disc_liq_sum['n']} +50={disc_liq_sum['hit50_pct']}% "
    f"+100={disc_liq_sum['hit100_pct']}% +200={disc_liq_sum['hit200_pct']}%"
)
report.append(
    f"Rank:      N={disc_rank_sum['n']} +50={disc_rank_sum['hit50_pct']}% "
    f"+100={disc_rank_sum['hit100_pct']}% +200={disc_rank_sum['hit200_pct']}%"
)
report.append("")
report.append("2026 UNTOUCHED HOLDOUT TOP1/DAY")
report.append("-"*88)
report.append(
    f"Liquidity: N={hold_liq_sum['n']} +20={hold_liq_sum['hit20_pct']}% "
    f"+50={hold_liq_sum['hit50_pct']}% +100={hold_liq_sum['hit100_pct']}% "
    f"+200={hold_liq_sum['hit200_pct']}% MedDD={hold_liq_sum['median_max_drawdown_63d_pct']}%"
)
report.append(
    f"Rank:      N={hold_rank_sum['n']} +20={hold_rank_sum['hit20_pct']}% "
    f"+50={hold_rank_sum['hit50_pct']}% +100={hold_rank_sum['hit100_pct']}% "
    f"+200={hold_rank_sum['hit200_pct']}% MedDD={hold_rank_sum['median_max_drawdown_63d_pct']}%"
)
if hold_liq_sum.get("hit200_pct"):
    report.append(
        f"2026 +200 lift vs liquidity baseline: "
        f"{hold_rank_sum['hit200_pct']/hold_liq_sum['hit200_pct']:.2f}x"
    )

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_FEATURES, OUT_DISC_TOP1, OUT_HOLD_TOP1, OUT_SCORED, OUT_JSON, OUT_REPORT]:
    print(" ", p)
