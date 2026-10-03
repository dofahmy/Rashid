#!/usr/bin/env python3
"""
Rajih — second-stage +200% runner discovery on 2024–2025,
with 2026 kept untouched until ONE rule is locked.

Input:
  /data/daily_200pct_locked_rule_all_signals.csv
  market_candles_1d

Population:
  Signals already produced by the first-stage locked rule:
      ATR14% T-1 >= 10.96%
      RET5 T-1 <= -9.22%

Outcome:
  winner = hit200 == 1 within 63 trading sessions

Split:
  Discovery/learning: setup year 2024 or 2025
  Untouched holdout: setup year 2026

Feature checkpoints:
  T-20, T-10, T-5, T-3, T-1

Selection discipline:
  1) Analyze ONLY 2024–2025.
  2) Build candidate single-threshold rules from strongest discovery features.
  3) Build pair rules from the best single rules.
  4) A candidate must work in BOTH 2024 and 2025:
       - minimum 25 selected signals in each year
       - minimum 4 +200 winners in each year
  5) Lock exactly ONE rule using discovery data only.
     Ranking score rewards:
       - the weaker of the 2024/2025 lifts
       - discovery recall
       - sample size
  6) Only after the rule is locked, evaluate that ONE rule on 2026.

This avoids tuning thresholds after seeing 2026.

Outputs:
  /data/daily_200pct_stage2_enriched_signals.csv
  /data/daily_200pct_stage2_feature_comparison.csv
  /data/daily_200pct_stage2_discovery_candidates.csv
  /data/daily_200pct_stage2_locked_rule.json
  /data/daily_200pct_stage2_holdout_2026.csv
  /data/daily_200pct_stage2_report.txt

Run:
  python discover_200pct_2024_2025_holdout_2026.py
"""

from __future__ import annotations

import os, csv, json, math
from pathlib import Path
from statistics import mean, median

from sqlalchemy import MetaData, Table, select
from core import database


DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_200pct_locked_rule_all_signals.csv"

OUT_ENRICH = DATA_DIR / "daily_200pct_stage2_enriched_signals.csv"
OUT_COMP = DATA_DIR / "daily_200pct_stage2_feature_comparison.csv"
OUT_CAND = DATA_DIR / "daily_200pct_stage2_discovery_candidates.csv"
OUT_LOCK = DATA_DIR / "daily_200pct_stage2_locked_rule.json"
OUT_HOLD = DATA_DIR / "daily_200pct_stage2_holdout_2026.csv"
OUT_REPORT = DATA_DIR / "daily_200pct_stage2_report.txt"

CHECKPOINTS = [-20, -10, -5, -3, -1]
DISCOVERY_YEARS = {"2024", "2025"}
HOLDOUT_YEAR = "2026"

MIN_SELECTED_PER_DISCOVERY_YEAR = 25
MIN_WINNERS_PER_DISCOVERY_YEAR = 4
TOP_FEATURES_FOR_RULES = 24
TOP_SINGLE_RULES_FOR_PAIRS = 20

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run validate_daily_200pct_locked_rule.py first.")


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def stdev(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))


def ema(vals, n):
    out = [None]*len(vals)
    if not vals:
        return out
    a = 2/(n+1)
    e = float(vals[0])
    out[0] = e
    for i in range(1, len(vals)):
        e = a*float(vals[i]) + (1-a)*e
        out[i] = e
    return out


def sma(vals, n):
    out = [None]*len(vals)
    s = 0.0
    for i, x in enumerate(vals):
        s += x
        if i >= n:
            s -= vals[i-n]
        if i >= n-1:
            out[i] = s/n
    return out


def atr(h, l, c, n=14):
    out = [None]*len(c)
    if len(c) <= n:
        return out
    tr = [None]*len(c)
    for i in range(1, len(c)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = sum(tr[1:n+1])/n
    out[n] = a
    for i in range(n+1, len(c)):
        a = ((n-1)*a + tr[i])/n
        out[i] = a
    return out


def rsi(c, n=14):
    out = [None]*len(c)
    if len(c) <= n:
        return out
    gains, losses = [], []
    for i in range(1, len(c)):
        d = c[i]-c[i-1]
        gains.append(max(d,0))
        losses.append(max(-d,0))
    ag = sum(gains[:n])/n
    al = sum(losses[:n])/n
    out[n] = 100 if al == 0 and ag > 0 else (50 if al == 0 else 100-100/(1+ag/al))
    for i in range(n+1, len(c)):
        ag = ((n-1)*ag + gains[i-1])/n
        al = ((n-1)*al + losses[i-1])/n
        out[i] = 100 if al == 0 and ag > 0 else (50 if al == 0 else 100-100/(1+ag/al))
    return out


def adx(h, l, c, n=14):
    m = len(c)
    out = [None]*m
    if m < 2*n+1:
        return out

    tr = [0.0]*m
    pdm = [0.0]*m
    mdm = [0.0]*m

    for i in range(1, m):
        up = h[i]-h[i-1]
        dn = l[i-1]-l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        mdm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))

    atrs = sum(tr[1:n+1])
    ps = sum(pdm[1:n+1])
    ms = sum(mdm[1:n+1])
    dx = [None]*m

    for i in range(n, m):
        if i > n:
            atrs = atrs-atrs/n+tr[i]
            ps = ps-ps/n+pdm[i]
            ms = ms-ms/n+mdm[i]
        if atrs <= 0:
            continue
        pdi = 100*ps/atrs
        mdi = 100*ms/atrs
        den = pdi+mdi
        dx[i] = 0 if den == 0 else 100*abs(pdi-mdi)/den

    seed = [x for x in dx[n:2*n] if x is not None]
    if len(seed) < n:
        return out

    a = sum(seed)/n
    out[2*n-1] = a
    for i in range(2*n, m):
        if dx[i] is not None:
            a = ((n-1)*a + dx[i])/n
            out[i] = a
    return out


def load_symbol(DB, daily, sym):
    with DB() as s:
        raw = list(s.execute(
            select(
                daily.c.session_date, daily.c.o, daily.c.h, daily.c.l,
                daily.c.c, daily.c.v, daily.c.adj_c
            )
            .where(daily.c.symbol == sym)
            .order_by(daily.c.session_date)
        ).all())

    rows = []
    for d,o,h,l,c,v,ac in raw:
        ro,rh,rl,rc,adj = map(finite,(o,h,l,c,ac))
        if None in (ro,rh,rl,rc,adj) or min(ro,rh,rl,rc,adj) <= 0:
            continue
        fac = adj/rc
        rows.append({
            "date": str(d),
            "raw_close": rc,
            "open": ro*fac,
            "high": rh*fac,
            "low": rl*fac,
            "close": adj,
            "volume": float(v or 0),
        })
    return rows


def build(rows):
    c = [r["close"] for r in rows]
    o = [r["open"] for r in rows]
    h = [r["high"] for r in rows]
    l = [r["low"] for r in rows]
    v = [r["volume"] for r in rows]
    rc = [r["raw_close"] for r in rows]

    s20 = sma(c,20)
    s50 = sma(c,50)
    s200 = sma(c,200)
    e20 = ema(c,20)
    e50 = ema(c,50)
    a14 = atr(h,l,c,14)
    r14 = rsi(c,14)
    ax = adx(h,l,c,14)
    e12 = ema(c,12)
    e26 = ema(c,26)
    macd = [e12[i]-e26[i] if e12[i] is not None and e26[i] is not None else None for i in range(len(c))]
    macd0 = [0 if x is None else x for x in macd]
    msig = ema(macd0,9)

    rets = [None]*len(c)
    gaps = [None]*len(c)
    dollars = [rc[i]*v[i] for i in range(len(c))]
    for i in range(1,len(c)):
        rets[i] = c[i]/c[i-1]-1
        gaps[i] = 100*(o[i]/c[i-1]-1)

    return {
        "c":c, "o":o, "h":h, "l":l, "v":v, "rc":rc,
        "s20":s20, "s50":s50, "s200":s200,
        "e20":e20, "e50":e50, "a14":a14, "r14":r14, "adx":ax,
        "macd":macd, "msig":msig, "rets":rets, "gaps":gaps,
        "dollars":dollars,
    }


def feat(F, i):
    if i < 20:
        return None

    c,h,l,v,rc = F["c"],F["h"],F["l"],F["v"],F["rc"]

    def maxh(n):
        return max(h[max(0,i-n+1):i+1])

    def minl(n):
        return min(l[max(0,i-n+1):i+1])

    def avg_range(n):
        a = max(0,i-n+1)
        return mean(100*(h[j]-l[j])/c[j] for j in range(a,i+1))

    rv20 = stdev(F["rets"][max(1,i-19):i+1])
    vol20 = mean(v[i-19:i+1])
    vol5 = mean(v[i-4:i+1]) if i >= 4 else None
    dol20 = mean(F["dollars"][i-19:i+1])
    dol5 = mean(F["dollars"][i-4:i+1]) if i >= 4 else None
    bbmid = F["s20"][i]
    bbsd = stdev(c[i-19:i+1])
    gaps20 = [x for x in F["gaps"][i-19:i+1] if x is not None]
    ups = sum(1 for j in range(i-19,i+1) if j>0 and c[j]>c[j-1])/20

    return {
        "price": rc[i],
        "atr_pct": 100*F["a14"][i]/c[i] if F["a14"][i] else None,
        "rsi14": F["r14"][i],
        "adx14": F["adx"][i],
        "ema20_vs_ema50_pct": 100*(F["e20"][i]/F["e50"][i]-1) if F["e20"][i] and F["e50"][i] else None,
        "above_sma20": 1 if F["s20"][i] and c[i] > F["s20"][i] else 0,
        "above_sma50": 1 if F["s50"][i] and c[i] > F["s50"][i] else 0,
        "above_sma200": 1 if F["s200"][i] and c[i] > F["s200"][i] else 0,
        "macd_atr": F["macd"][i]/F["a14"][i] if F["macd"][i] is not None and F["a14"][i] else None,
        "macd_signal_atr": (F["macd"][i]-F["msig"][i])/F["a14"][i] if F["macd"][i] is not None and F["msig"][i] is not None and F["a14"][i] else None,
        "avg_range5_pct": avg_range(5),
        "avg_range20_pct": avg_range(20),
        "realized_vol20_pct": 100*rv20 if rv20 is not None else None,
        "vol_ratio_5_20": vol5/vol20 if vol5 is not None and vol20 else None,
        "dollarvol_ratio_5_20": dol5/dol20 if dol5 is not None and dol20 else None,
        "avg_dollar_volume20": dol20,
        "distance_to_20d_high_pct": 100*(c[i]/maxh(20)-1),
        "distance_to_126d_high_pct": 100*(c[i]/maxh(126)-1) if i>=125 else None,
        "distance_to_252d_high_pct": 100*(c[i]/maxh(252)-1) if i>=251 else None,
        "distance_to_20d_low_pct": 100*(c[i]/minl(20)-1),
        "distance_to_126d_low_pct": 100*(c[i]/minl(126)-1) if i>=125 else None,
        "distance_to_252d_low_pct": 100*(c[i]/minl(252)-1) if i>=251 else None,
        "min_gap20_pct": min(gaps20) if gaps20 else None,
        "max_gap20_pct": max(gaps20) if gaps20 else None,
        "up_day_ratio20": ups,
        "ret5_pct": 100*(c[i]/c[i-5]-1) if i>=5 else None,
        "ret10_pct": 100*(c[i]/c[i-10]-1) if i>=10 else None,
        "ret20_pct": 100*(c[i]/c[i-20]-1) if i>=20 else None,
        "bb_width_pct": 100*(4*bbsd/bbmid) if bbsd is not None and bbmid else None,
        "bb_z": (c[i]-bbmid)/bbsd if bbsd not in (None,0) and bbmid is not None else None,
    }


def auc(w, c):
    w = [x for x in w if x is not None]
    c = [x for x in c if x is not None]
    if not w or not c:
        return None
    s = 0.0
    for a in w:
        for b in c:
            if a>b: s += 1
            elif a==b: s += 0.5
    return s/(len(w)*len(c))


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w",encoding="utf-8-sig",newline="") as fh:
        w = csv.DictWriter(fh,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def rule_pass(row, rule):
    def one(rr):
        x = finite(row.get(rr["feature"]))
        if x is None:
            return False
        return x >= rr["cut"] if rr["operator"] == ">=" else x <= rr["cut"]

    if rule["type"] == "single":
        return one(rule["r1"])
    return one(rule["r1"]) and one(rule["r2"])


def perf(rows, rule):
    selected = [r for r in rows if rule_pass(r,rule)]
    if not selected:
        return {"n":0,"wins":0,"precision":0.0,"recall":0.0}
    wins = sum(int(r["hit200"]) for r in selected)
    total_wins = sum(int(r["hit200"]) for r in rows)
    return {
        "n": len(selected),
        "wins": wins,
        "precision": wins/len(selected),
        "recall": wins/total_wins if total_wins else 0.0,
    }


# Load first-stage signals.
with INPUT.open(encoding="utf-8-sig",newline="") as fh:
    signals = list(csv.DictReader(fh))

for r in signals:
    r["year"] = r["setup_date"][:4]
    r["hit200"] = int(float(r["hit200"]))

DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d",md,autoload_with=s.get_bind())

cache = {}
def get(sym):
    if sym not in cache:
        rows = load_symbol(DB,daily,sym)
        cache[sym] = (rows,build(rows),{r["date"]:i for i,r in enumerate(rows)})
    return cache[sym]

print("\nRajih — 2024–2025 DISCOVERY / 2026 UNTOUCHED HOLDOUT")
print(f"Input first-stage signals: {len(signals)}")
print("Enriching pre-signal features...\n")

enriched = []
for idx,r in enumerate(signals,1):
    rows,F,dmap = get(r["symbol"])
    i = dmap.get(r["setup_date"])
    if i is None:
        continue

    x = dict(r)
    for cp in CHECKPOINTS:
        j = i+cp
        if j < 20:
            continue
        ft = feat(F,j)
        if ft:
            for k,v in ft.items():
                x[f"t{cp}_{k}"] = v
    enriched.append(x)

    if idx%500==0 or idx==len(signals):
        print(f"Enriched {idx}/{len(signals)}")

write_csv(OUT_ENRICH,enriched)

discovery = [r for r in enriched if r["year"] in DISCOVERY_YEARS]
holdout = [r for r in enriched if r["year"] == HOLDOUT_YEAR]
d2024 = [r for r in discovery if r["year"]=="2024"]
d2025 = [r for r in discovery if r["year"]=="2025"]

base_all = sum(r["hit200"] for r in discovery)/len(discovery)
base24 = sum(r["hit200"] for r in d2024)/len(d2024)
base25 = sum(r["hit200"] for r in d2025)/len(d2025)
base26 = sum(r["hit200"] for r in holdout)/len(holdout)

print("\nDiscovery counts:")
print(f"2024: N={len(d2024)} winners={sum(r['hit200'] for r in d2024)} base={100*base24:.2f}%")
print(f"2025: N={len(d2025)} winners={sum(r['hit200'] for r in d2025)} base={100*base25:.2f}%")
print(f"2026 HOLDOUT (still untouched for model selection): N={len(holdout)} winners={sum(r['hit200'] for r in holdout)} base={100*base26:.2f}%")

# Discovery feature separation.
feature_cols = [k for k in enriched[0] if k.startswith("t-")]
winners = [r for r in discovery if r["hit200"]==1]
losers = [r for r in discovery if r["hit200"]==0]

comp = []
for col in feature_cols:
    wv = [finite(r.get(col)) for r in winners]
    lv = [finite(r.get(col)) for r in losers]
    wv = [x for x in wv if x is not None]
    lv = [x for x in lv if x is not None]
    if len(wv)<30 or len(lv)<300:
        continue
    a = auc(wv,lv)
    if a is None:
        continue
    comp.append({
        "feature":col,
        "winner_n":len(wv),
        "loser_n":len(lv),
        "winner_median":median(wv),
        "loser_median":median(lv),
        "winner_mean":mean(wv),
        "loser_mean":mean(lv),
        "auc_winner_gt_loser":a,
        "separation":max(a,1-a),
        "winner_like_direction":"HIGHER" if a>=0.5 else "LOWER",
    })

comp.sort(key=lambda r:r["separation"],reverse=True)
write_csv(OUT_COMP,comp)

# Candidate single rules from discovery only.
single_candidates = []
for c in comp[:TOP_FEATURES_FOR_RULES]:
    col = c["feature"]
    vals = sorted(set(finite(r.get(col)) for r in discovery if finite(r.get(col)) is not None))
    if len(vals)<20:
        continue

    # fixed quantile grid in discovery.
    cuts = []
    for q in [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]:
        pos = round((len(vals)-1)*q)
        cuts.append(vals[pos])

    for cut in sorted(set(cuts)):
        for op in (">=","<="):
            rule = {
                "type":"single",
                "r1":{"feature":col,"operator":op,"cut":cut},
                "r2":None,
            }
            p24 = perf(d2024,rule)
            p25 = perf(d2025,rule)
            pall = perf(discovery,rule)

            if p24["n"] < MIN_SELECTED_PER_DISCOVERY_YEAR or p25["n"] < MIN_SELECTED_PER_DISCOVERY_YEAR:
                continue
            if p24["wins"] < MIN_WINNERS_PER_DISCOVERY_YEAR or p25["wins"] < MIN_WINNERS_PER_DISCOVERY_YEAR:
                continue

            lift24 = p24["precision"]/base24 if base24 else 0
            lift25 = p25["precision"]/base25 if base25 else 0
            liftall = pall["precision"]/base_all if base_all else 0
            minlift = min(lift24,lift25)

            # Discovery-only score: rewards cross-year robustness, recall, and support.
            score = minlift * math.sqrt(max(pall["recall"],1e-9)) * math.log1p(pall["n"])

            single_candidates.append({
                "rule":rule,
                "score":score,
                "p24":p24,"p25":p25,"pall":pall,
                "lift24":lift24,"lift25":lift25,"liftall":liftall,
                "minlift":minlift,
            })

single_candidates.sort(key=lambda x:(x["score"],x["minlift"],x["pall"]["precision"]),reverse=True)

# Pair candidates from top discovery singles only.
top_singles = single_candidates[:TOP_SINGLE_RULES_FOR_PAIRS]
pair_candidates = []

for a in range(len(top_singles)):
    for b in range(a+1,len(top_singles)):
        r1 = top_singles[a]["rule"]["r1"]
        r2 = top_singles[b]["rule"]["r1"]
        if r1["feature"] == r2["feature"]:
            continue

        rule = {"type":"pair","r1":r1,"r2":r2}
        p24 = perf(d2024,rule)
        p25 = perf(d2025,rule)
        pall = perf(discovery,rule)

        if p24["n"] < MIN_SELECTED_PER_DISCOVERY_YEAR or p25["n"] < MIN_SELECTED_PER_DISCOVERY_YEAR:
            continue
        if p24["wins"] < MIN_WINNERS_PER_DISCOVERY_YEAR or p25["wins"] < MIN_WINNERS_PER_DISCOVERY_YEAR:
            continue

        lift24 = p24["precision"]/base24 if base24 else 0
        lift25 = p25["precision"]/base25 if base25 else 0
        liftall = pall["precision"]/base_all if base_all else 0
        minlift = min(lift24,lift25)
        score = minlift * math.sqrt(max(pall["recall"],1e-9)) * math.log1p(pall["n"])

        pair_candidates.append({
            "rule":rule,
            "score":score,
            "p24":p24,"p25":p25,"pall":pall,
            "lift24":lift24,"lift25":lift25,"liftall":liftall,
            "minlift":minlift,
        })

all_candidates = single_candidates + pair_candidates
if not all_candidates:
    raise SystemExit("No discovery candidate passed cross-year support requirements.")

all_candidates.sort(key=lambda x:(x["score"],x["minlift"],x["pall"]["precision"]),reverse=True)
locked = all_candidates[0]

# Save discovery candidates WITHOUT holdout results.
cand_rows = []
for rank,c in enumerate(all_candidates[:100],1):
    rr = c["rule"]
    desc = f"{rr['r1']['feature']} {rr['r1']['operator']} {rr['r1']['cut']:.6g}"
    if rr["type"]=="pair":
        desc += f" AND {rr['r2']['feature']} {rr['r2']['operator']} {rr['r2']['cut']:.6g}"
    cand_rows.append({
        "rank":rank,
        "type":rr["type"],
        "rule":desc,
        "discovery_score":c["score"],
        "2024_n":c["p24"]["n"],
        "2024_winners":c["p24"]["wins"],
        "2024_precision_pct":100*c["p24"]["precision"],
        "2024_lift":c["lift24"],
        "2025_n":c["p25"]["n"],
        "2025_winners":c["p25"]["wins"],
        "2025_precision_pct":100*c["p25"]["precision"],
        "2025_lift":c["lift25"],
        "discovery_n":c["pall"]["n"],
        "discovery_winners":c["pall"]["wins"],
        "discovery_precision_pct":100*c["pall"]["precision"],
        "discovery_recall_pct":100*c["pall"]["recall"],
        "discovery_lift":c["liftall"],
        "min_year_lift":c["minlift"],
    })
write_csv(OUT_CAND,cand_rows)

# NOW unlock 2026 exactly once for the selected rule.
hold_perf = perf(holdout,locked["rule"])
hold_selected = [r for r in holdout if rule_pass(r,locked["rule"])]
write_csv(OUT_HOLD,hold_selected)

locked_desc = f"{locked['rule']['r1']['feature']} {locked['rule']['r1']['operator']} {locked['rule']['r1']['cut']:.6g}"
if locked["rule"]["type"]=="pair":
    locked_desc += f" AND {locked['rule']['r2']['feature']} {locked['rule']['r2']['operator']} {locked['rule']['r2']['cut']:.6g}"

hold_lift = hold_perf["precision"]/base26 if base26 else 0

lock_json = {
    "split": {
        "discovery_years":["2024","2025"],
        "holdout_year":"2026",
    },
    "discovery_base_rates": {
        "2024_hit200_pct":100*base24,
        "2025_hit200_pct":100*base25,
        "combined_hit200_pct":100*base_all,
    },
    "locked_rule_description":locked_desc,
    "locked_rule":locked["rule"],
    "discovery_performance":{
        "2024":locked["p24"],
        "2025":locked["p25"],
        "combined":locked["pall"],
        "2024_lift":locked["lift24"],
        "2025_lift":locked["lift25"],
        "combined_lift":locked["liftall"],
        "selection_score":locked["score"],
    },
    "holdout_2026":{
        "base_n":len(holdout),
        "base_winners":sum(r["hit200"] for r in holdout),
        "base_hit200_pct":100*base26,
        "selected_n":hold_perf["n"],
        "selected_winners":hold_perf["wins"],
        "precision_pct":100*hold_perf["precision"],
        "recall_pct":100*hold_perf["recall"],
        "lift_vs_2026_base":hold_lift,
    },
}
OUT_LOCK.write_text(json.dumps(lock_json,ensure_ascii=False,indent=2),encoding="utf-8")

report = []
report.append("RAJIH — 2024–2025 DISCOVERY / 2026 UNTOUCHED HOLDOUT")
report.append("="*88)
report.append(f"Discovery 2024: N={len(d2024)}, +200 winners={sum(r['hit200'] for r in d2024)}, base={100*base24:.2f}%")
report.append(f"Discovery 2025: N={len(d2025)}, +200 winners={sum(r['hit200'] for r in d2025)}, base={100*base25:.2f}%")
report.append(f"2026 holdout: N={len(holdout)}, +200 winners={sum(r['hit200'] for r in holdout)}, base={100*base26:.2f}%")
report.append("")
report.append("TOP DISCOVERY FEATURES")
report.append("-"*88)
for r in comp[:15]:
    report.append(
        f"{r['feature']}: winner med={r['winner_median']:.4f} | "
        f"loser med={r['loser_median']:.4f} | sep={r['separation']:.4f} | "
        f"{r['winner_like_direction']}"
    )
report.append("")
report.append("LOCKED RULE — SELECTED WITHOUT USING 2026")
report.append("-"*88)
report.append(locked_desc)
report.append(
    f"2024: N={locked['p24']['n']} winners={locked['p24']['wins']} "
    f"precision={100*locked['p24']['precision']:.2f}% lift={locked['lift24']:.2f}x"
)
report.append(
    f"2025: N={locked['p25']['n']} winners={locked['p25']['wins']} "
    f"precision={100*locked['p25']['precision']:.2f}% lift={locked['lift25']:.2f}x"
)
report.append(
    f"Combined discovery: N={locked['pall']['n']} winners={locked['pall']['wins']} "
    f"precision={100*locked['pall']['precision']:.2f}% "
    f"recall={100*locked['pall']['recall']:.2f}% lift={locked['liftall']:.2f}x"
)
report.append("")
report.append("2026 HOLDOUT — REVEALED AFTER RULE LOCK")
report.append("-"*88)
report.append(
    f"Base: {sum(r['hit200'] for r in holdout)}/{len(holdout)} = {100*base26:.2f}%"
)
report.append(
    f"Locked rule: {hold_perf['wins']}/{hold_perf['n']} = "
    f"{100*hold_perf['precision']:.2f}% | recall={100*hold_perf['recall']:.2f}% | "
    f"lift={hold_lift:.2f}x"
)
OUT_REPORT.write_text("\n".join(report),encoding="utf-8")

print("\n=== DISCOVERY 2024–2025 ===")
print(f"2024 base +200: {100*base24:.2f}% ({sum(r['hit200'] for r in d2024)}/{len(d2024)})")
print(f"2025 base +200: {100*base25:.2f}% ({sum(r['hit200'] for r in d2025)}/{len(d2025)})")

print("\nTOP FEATURES — DISCOVERY ONLY:")
for r in comp[:12]:
    print(
        f"{r['feature']}: W med={r['winner_median']:.3f} | "
        f"L med={r['loser_median']:.3f} | sep={r['separation']:.3f} | "
        f"{r['winner_like_direction']}"
    )

print("\n=== LOCKED RULE — CHOSEN BEFORE LOOKING AT 2026 ===")
print(locked_desc)
print(
    f"2024: N={locked['p24']['n']} W={locked['p24']['wins']} "
    f"precision={100*locked['p24']['precision']:.2f}% lift={locked['lift24']:.2f}x"
)
print(
    f"2025: N={locked['p25']['n']} W={locked['p25']['wins']} "
    f"precision={100*locked['p25']['precision']:.2f}% lift={locked['lift25']:.2f}x"
)
print(
    f"Discovery combined: N={locked['pall']['n']} W={locked['pall']['wins']} "
    f"precision={100*locked['pall']['precision']:.2f}% "
    f"recall={100*locked['pall']['recall']:.2f}% lift={locked['liftall']:.2f}x"
)

print("\n=== 2026 UNTOUCHED HOLDOUT ===")
print(f"2026 baseline: {sum(r['hit200'] for r in holdout)}/{len(holdout)} = {100*base26:.2f}%")
print(
    f"Locked rule: {hold_perf['wins']}/{hold_perf['n']} = "
    f"{100*hold_perf['precision']:.2f}%"
)
print(f"Recall of 2026 +200 winners: {100*hold_perf['recall']:.2f}%")
print(f"Lift vs 2026 baseline: {hold_lift:.2f}x")

print("\nCreated:")
for p in [OUT_ENRICH,OUT_COMP,OUT_CAND,OUT_LOCK,OUT_HOLD,OUT_REPORT]:
    print(" ",p)
