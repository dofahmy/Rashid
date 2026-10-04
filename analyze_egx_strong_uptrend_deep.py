#!/usr/bin/env python3
"""
EGX STRONG_UPTREND deep dive

Input:
  /data/egx_strong_preshape_all.csv

Goal:
  Study ONLY historical mature signals whose pre-signal regime was STRONG_UPTREND,
  find which pre-signal shapes were associated with better +50% / +100% outcomes,
  and rank recent STRONG_UPTREND signals using only those historical relationships.

Method:
  1) Analyze the 52 mature STRONG_UPTREND cases.
  2) Compare winners (+50% within 6M) vs non-winners.
  3) Build quintile / threshold tables for the most relevant pre-signal features.
  4) Search simple 1-feature and 2-feature rules with minimum sample size to reduce
     obvious overfitting.
  5) Rank recent STRONG_UPTREND signals by how many historically favorable rules
     they satisfy.

Outputs:
  /data/egx_strong_uptrend_cases.csv
  /data/egx_strong_uptrend_feature_compare.csv
  /data/egx_strong_uptrend_bins.csv
  /data/egx_strong_uptrend_rules.csv
  /data/egx_strong_uptrend_recent_ranked.csv
  /data/egx_strong_uptrend_report.txt
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

DATA = Path("/data")
INFILE = DATA / "egx_strong_preshape_all.csv"

OUT_CASES = DATA / "egx_strong_uptrend_cases.csv"
OUT_FEATURES = DATA / "egx_strong_uptrend_feature_compare.csv"
OUT_BINS = DATA / "egx_strong_uptrend_bins.csv"
OUT_RULES = DATA / "egx_strong_uptrend_rules.csv"
OUT_RECENT = DATA / "egx_strong_uptrend_recent_ranked.csv"
OUT_REPORT = DATA / "egx_strong_uptrend_report.txt"

df = pd.read_csv(INFILE, dtype={"symbol": str})
df["signal_date"] = pd.to_datetime(df["signal_date"])

su = df[(df["pre_regime"] == "STRONG_UPTREND") & (df["mature_6m"] == 1)].copy()
if su.empty:
    raise SystemExit("No mature STRONG_UPTREND cases found.")

features = [
    "pre_6m_ret_pct",
    "pre_3m_ret_pct",
    "pre_1m_ret_pct",
    "pre_6m_range_pct",
    "pre_realized_vol_pct",
    "pre_atr14_pct",
    "pre_dist_6m_high_pct",
    "pre_dist_6m_low_pct",
    "pre_max_drawdown_pct",
    "pre_trend_slope_pct",
    "pre_trend_r2",
    "pre_middle50_frac",
    "pre_vol_ratio20_126",
    "pre_adv20",
    "pre_ret20_std_pct",
]

for c in features + ["m6_hit20","m6_hit50","m6_hit100","m6_close_ret_pct","m6_max_gain_pct","m6_max_drawdown_pct"]:
    su[c] = pd.to_numeric(su[c], errors="coerce")

su["success50"] = (su["m6_hit50"] == 1).astype(int)
su["success100"] = (su["m6_hit100"] == 1).astype(int)
su.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

base50 = 100 * su["success50"].mean()
base100 = 100 * su["success100"].mean()

def rank_sep(w, l):
    w = pd.Series(w).dropna().astype(float).to_numpy()
    l = pd.Series(l).dropna().astype(float).to_numpy()
    if not len(w) or not len(l):
        return np.nan, ""
    comb = pd.Series(np.concatenate([w,l]))
    ranks = comb.rank(method="average").to_numpy()
    rw = ranks[:len(w)].sum()
    U = rw - len(w)*(len(w)+1)/2
    auc = U/(len(w)*len(l))
    if auc >= .5:
        return auc, "HIGHER"
    return 1-auc, "LOWER"

# Feature comparison
feat_rows = []
w = su[su["success50"] == 1]
l = su[su["success50"] == 0]

for c in features:
    sep, direction = rank_sep(w[c], l[c])
    feat_rows.append({
        "feature": c,
        "winner50_median": w[c].median(),
        "loser50_median": l[c].median(),
        "winner50_mean": w[c].mean(),
        "loser50_mean": l[c].mean(),
        "direction_for_success50": direction,
        "separation": sep,
    })

feat = pd.DataFrame(feat_rows).sort_values("separation", ascending=False)
feat.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

# Quantile bins
bin_rows = []
for c in features:
    x = su[[c,"success50","success100","m6_close_ret_pct","m6_max_gain_pct","m6_max_drawdown_pct"]].dropna().copy()
    if x[c].nunique() < 4:
        continue
    try:
        x["bin"] = pd.qcut(x[c], q=min(4, x[c].nunique()), duplicates="drop")
    except Exception:
        continue
    for b, g in x.groupby("bin", observed=True):
        bin_rows.append({
            "feature": c,
            "bin": str(b),
            "n": len(g),
            "hit50_pct": 100*g["success50"].mean(),
            "hit100_pct": 100*g["success100"].mean(),
            "median_close_6m_pct": g["m6_close_ret_pct"].median(),
            "median_max_gain_6m_pct": g["m6_max_gain_pct"].median(),
            "median_max_dd_6m_pct": g["m6_max_drawdown_pct"].median(),
        })

bins = pd.DataFrame(bin_rows)
bins.to_csv(OUT_BINS, index=False, encoding="utf-8-sig")

# Candidate simple rules
# Use quartile/median cutpoints derived ONLY from the 52-case mature sample.
cutpoints = {}
for c in features:
    s = su[c].dropna()
    if len(s) < 20:
        continue
    vals = sorted(set([
        float(s.quantile(.25)),
        float(s.quantile(.40)),
        float(s.quantile(.50)),
        float(s.quantile(.60)),
        float(s.quantile(.75)),
    ]))
    cutpoints[c] = vals

rules = []

def add_rule(desc, mask, complexity):
    g = su[mask].copy()
    n = len(g)
    if n < 10:
        return
    hit50 = 100*g["success50"].mean()
    hit100 = 100*g["success100"].mean()
    rules.append({
        "rule": desc,
        "complexity": complexity,
        "n": n,
        "hit50_pct": hit50,
        "lift50_vs_base": hit50 - base50,
        "hit100_pct": hit100,
        "lift100_vs_base": hit100 - base100,
        "median_close_6m_pct": g["m6_close_ret_pct"].median(),
        "median_max_gain_6m_pct": g["m6_max_gain_pct"].median(),
        "median_max_dd_6m_pct": g["m6_max_drawdown_pct"].median(),
    })

# 1-feature rules
for c, vals in cutpoints.items():
    for t in vals:
        add_rule(f"{c} <= {t:.6g}", su[c] <= t, 1)
        add_rule(f"{c} >= {t:.6g}", su[c] >= t, 1)

# 2-feature rules: only top 8 features by separation, to control search size
top_feats = feat.head(8)["feature"].tolist()
single_conditions = {}
for c in top_feats:
    vals = cutpoints.get(c, [])
    conds = []
    for t in vals:
        conds.append((f"{c} <= {t:.6g}", su[c] <= t))
        conds.append((f"{c} >= {t:.6g}", su[c] >= t))
    single_conditions[c] = conds

for c1, c2 in itertools.combinations(top_feats, 2):
    for d1, m1 in single_conditions.get(c1, []):
        for d2, m2 in single_conditions.get(c2, []):
            add_rule(f"{d1} AND {d2}", m1 & m2, 2)

rules_df = pd.DataFrame(rules)
if not rules_df.empty:
    rules_df = rules_df.sort_values(
        ["lift50_vs_base","n","lift100_vs_base"],
        ascending=[False,False,False]
    ).drop_duplicates("rule")
rules_df.to_csv(OUT_RULES, index=False, encoding="utf-8-sig")

# Pick "robust" rules for recent ranking:
# min n=15, lift +10pp or better, complexity <=2.
robust = rules_df[
    (rules_df["n"] >= 15) &
    (rules_df["lift50_vs_base"] >= 10) &
    (rules_df["complexity"] <= 2)
].copy()

# Keep at most 10, preferring simpler and larger samples.
robust = robust.sort_values(
    ["complexity","lift50_vs_base","n"],
    ascending=[True,False,False]
).head(10)

# Evaluate recent STRONG_UPTREND signals, including immature ones.
recent = df[df["pre_regime"] == "STRONG_UPTREND"].copy()
recent = recent[recent["signal_date"].dt.year == 2026].copy()

def eval_rule_string(row, rule):
    parts = rule.split(" AND ")
    ok = True
    for p in parts:
        toks = p.split()
        col, op, val = toks[0], toks[1], float(toks[2])
        rv = row.get(col, np.nan)
        if pd.isna(rv):
            return False
        if op == "<=":
            ok = ok and (rv <= val)
        elif op == ">=":
            ok = ok and (rv >= val)
    return bool(ok)

robust_rules = robust["rule"].tolist()

scores = []
matched_text = []
for _, r in recent.iterrows():
    matched = []
    weighted = 0.0
    totalw = 0.0
    for _, rr in robust.iterrows():
        wgt = rr["n"]
        totalw += wgt
        if eval_rule_string(r, rr["rule"]):
            matched.append(rr["rule"])
            weighted += wgt * rr["hit50_pct"]
    if matched:
        score = weighted / sum(
            rr["n"] for _, rr in robust.iterrows()
            if eval_rule_string(r, rr["rule"])
        )
    else:
        score = base50
    scores.append(score)
    matched_text.append(" | ".join(matched))

recent["matched_robust_rules"] = [len(x.split(" | ")) if x else 0 for x in matched_text]
recent["historical_hit50_score"] = scores
recent["matched_rules_text"] = matched_text

recent = recent.sort_values(
    ["historical_hit50_score","matched_robust_rules","signal_date"],
    ascending=[False,False,False]
)

recent_cols = [
    "signal_date","symbol","signal_price",
    "historical_hit50_score","matched_robust_rules",
    "pre_6m_ret_pct","pre_3m_ret_pct","pre_1m_ret_pct",
    "pre_6m_range_pct","pre_realized_vol_pct","pre_atr14_pct",
    "pre_dist_6m_high_pct","pre_dist_6m_low_pct",
    "pre_trend_slope_pct","pre_trend_r2","pre_vol_ratio20_126",
    "matched_rules_text",
]
recent[recent_cols].to_csv(OUT_RECENT, index=False, encoding="utf-8-sig")

print("\n=== STRONG_UPTREND HISTORICAL SAMPLE ===")
print("Mature cases:", len(su))
print("Hit +20%:", f"{100*su['m6_hit20'].mean():.2f}%")
print("Hit +50%:", f"{base50:.2f}%")
print("Hit +100%:", f"{base100:.2f}%")
print("Positive close 6M:", f"{100*(su['m6_close_ret_pct']>0).mean():.2f}%")
print("Median 6M close return:", f"{su['m6_close_ret_pct'].median():.2f}%")
print("Median max gain:", f"{su['m6_max_gain_pct'].median():.2f}%")
print("Median max drawdown:", f"{su['m6_max_drawdown_pct'].median():.2f}%")

print("\n=== TOP FEATURES INSIDE STRONG_UPTREND ===")
print(feat.head(10).to_string(index=False))

print("\n=== BEST SIMPLE RULES ===")
if len(rules_df):
    print(rules_df.head(20).to_string(index=False))
else:
    print("No rules.")

print("\n=== ROBUST RULES USED FOR RECENT RANKING ===")
if len(robust):
    print(robust.to_string(index=False))
else:
    print("No rule met robust criteria; recent ranking falls back to base rate.")

print("\n=== 2026 STRONG_UPTREND SIGNALS RANKED ===")
print(recent[recent_cols].to_string(index=False))

report = []
report.append("EGX STRONG_UPTREND DEEP DIVE")
report.append("="*120)
report.append(f"Mature cases: {len(su)}")
report.append(f"Hit +20%: {100*su['m6_hit20'].mean():.2f}%")
report.append(f"Hit +50%: {base50:.2f}%")
report.append(f"Hit +100%: {base100:.2f}%")
report.append(f"Positive close 6M: {100*(su['m6_close_ret_pct']>0).mean():.2f}%")
report.append(f"Median 6M close return: {su['m6_close_ret_pct'].median():.2f}%")
report.append(f"Median max gain: {su['m6_max_gain_pct'].median():.2f}%")
report.append(f"Median max drawdown: {su['m6_max_drawdown_pct'].median():.2f}%")
report.append("")
report.append("TOP FEATURES")
report.append(feat.head(15).to_string(index=False))
report.append("")
report.append("BEST SIMPLE RULES")
report.append(rules_df.head(30).to_string(index=False) if len(rules_df) else "No rules")
report.append("")
report.append("ROBUST RULES")
report.append(robust.to_string(index=False) if len(robust) else "No robust rules")
report.append("")
report.append("2026 STRONG_UPTREND SIGNALS")
report.append(recent[recent_cols].to_string(index=False))
OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_CASES, OUT_FEATURES, OUT_BINS, OUT_RULES, OUT_RECENT, OUT_REPORT]:
    print(" ", p)
