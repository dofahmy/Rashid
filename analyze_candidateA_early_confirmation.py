#!/usr/bin/env python3
"""
Rajih — early confirmation analysis AFTER Candidate A signal.

Candidate A pre-signal rule is already fixed:
  T-1 BB Width > 100
  T-1 Dollar Volume Ratio 5/20 > 1.5
  T-20 Ret10 <= -15.3917%

Goal:
Among Candidate A matches, compare +200 winners vs non-winners using ONLY the
first 1–5 trading days AFTER the signal, to find early confirmation features.

Input:
  /data/daily_200pct_all_signals_condition_matches.csv
  market_candles_1d

Candidate A rows are selected from matched_rule == CANDIDATE_A_RET10.

Post-signal features measured at day 1..5:
- close return from signal close
- high gain from signal close
- low drawdown from signal close
- close location within day's range
- gap from prior close
- volume ratio vs prior 20d avg
- dollar-volume ratio vs prior 20d avg
- close above signal-day high
- close above prior 5d high
- no new low vs signal low
- consecutive green closes
- cumulative 5d return
- max high gain in first 3/5 days
- max drawdown in first 3/5 days
- whether +10%, +20% touched by day 1/2/3/5

Searches for:
1) same-direction winner/non-winner differences across 2024, 2025, 2026
2) simple confirmation rules
3) two-condition confirmation pairs

Outputs:
  /data/daily_200pct_candidateA_early_features.csv
  /data/daily_200pct_candidateA_early_feature_stats.csv
  /data/daily_200pct_candidateA_early_rules.csv
  /data/daily_200pct_candidateA_early_pairs.csv
  /data/daily_200pct_candidateA_early_summary.json
  /data/daily_200pct_candidateA_early_report.txt

Run:
  python analyze_candidateA_early_confirmation.py
"""

from __future__ import annotations

import csv, json, math, os
from pathlib import Path
from statistics import mean, median
from sqlalchemy import MetaData, Table, select
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
MATCHES = DATA_DIR / "daily_200pct_all_signals_condition_matches.csv"

OUT_FEAT = DATA_DIR / "daily_200pct_candidateA_early_features.csv"
OUT_STATS = DATA_DIR / "daily_200pct_candidateA_early_feature_stats.csv"
OUT_RULES = DATA_DIR / "daily_200pct_candidateA_early_rules.csv"
OUT_PAIRS = DATA_DIR / "daily_200pct_candidateA_early_pairs.csv"
OUT_JSON = DATA_DIR / "daily_200pct_candidateA_early_summary.json"
OUT_REPORT = DATA_DIR / "daily_200pct_candidateA_early_report.txt"

YEARS = ("2024","2025","2026")

if not MATCHES.exists():
    raise SystemExit(f"Missing {MATCHES}")


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def pct(a,b):
    return 100*(a/b-1)


def auc(w,l):
    w=[x for x in w if x is not None]
    l=[x for x in l if x is not None]
    if not w or not l:
        return None
    s=0.0
    for a in w:
        for b in l:
            if a>b: s+=1
            elif a==b: s+=0.5
    return s/(len(w)*len(l))


def qtile(xs,q):
    xs=sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs)==1:
        return xs[0]
    pos=(len(xs)-1)*q
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi: return xs[lo]
    a=pos-lo
    return xs[lo]*(1-a)+xs[hi]*a


def write_csv(path,rows):
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w",encoding="utf-8-sig",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def load_symbol(DB,daily,sym):
    with DB() as s:
        raw=list(s.execute(
            select(
                daily.c.session_date,daily.c.o,daily.c.h,daily.c.l,
                daily.c.c,daily.c.v,daily.c.adj_c
            )
            .where(daily.c.symbol==sym)
            .order_by(daily.c.session_date)
        ).all())
    rows=[]
    for d,o,h,l,c,v,ac in raw:
        ro,rh,rl,rc,adj=map(fnum,(o,h,l,c,ac))
        if None in (ro,rh,rl,rc,adj) or min(ro,rh,rl,rc,adj)<=0:
            continue
        fac=adj/rc
        rows.append({
            "date":str(d),
            "open":ro*fac,
            "high":rh*fac,
            "low":rl*fac,
            "close":adj,
            "raw_close":rc,
            "volume":float(v or 0),
        })
    return rows


# Load Candidate A matches only.
with MATCHES.open(encoding="utf-8-sig",newline="") as fh:
    raw=list(csv.DictReader(fh))

signals=[r for r in raw if r.get("matched_rule")=="CANDIDATE_A_RET10"]
for r in signals:
    r["year"]=r["setup_date"][:4]
    r["hit200"]=int(float(r["hit200"]))

print("\nRajih — CANDIDATE A EARLY CONFIRMATION")
print(f"Candidate A signals: {len(signals)}")
print(f"+200 winners: {sum(r['hit200'] for r in signals)}")
print(f"Non-winners: {sum(1-r['hit200'] for r in signals)}")

DB=database()
with DB() as s:
    md=MetaData()
    daily=Table("market_candles_1d",md,autoload_with=s.get_bind())

cache={}
def get(sym):
    if sym not in cache:
        rows=load_symbol(DB,daily,sym)
        cache[sym]=(rows,{r["date"]:i for i,r in enumerate(rows)})
    return cache[sym]

feat_rows=[]

for r in signals:
    rows,dmap=get(r["symbol"])
    i=dmap.get(r["setup_date"])
    if i is None or i<20 or i+5>=len(rows):
        continue

    sig=rows[i]
    sig_close=sig["close"]
    sig_high=sig["high"]
    sig_low=sig["low"]

    prev20_vol=mean(x["volume"] for x in rows[i-19:i+1])
    prev20_dol=mean(x["raw_close"]*x["volume"] for x in rows[i-19:i+1])
    prior5_high=max(x["high"] for x in rows[i-4:i+1])

    x={
        "symbol":r["symbol"],
        "setup_date":r["setup_date"],
        "year":r["year"],
        "hit200":r["hit200"],
        "max_gain_63d_pct":r.get("max_gain_63d_pct"),
        "first200_day":r.get("first200_day"),
    }

    green_streak=0
    maxh3=-1e99; maxh5=-1e99
    maxdd3=1e99; maxdd5=1e99

    for d in range(1,6):
        bar=rows[i+d]
        prev=rows[i+d-1]

        cret=pct(bar["close"],sig_close)
        hret=pct(bar["high"],sig_close)
        lret=pct(bar["low"],sig_close)
        gap=pct(bar["open"],prev["close"])
        day_range=bar["high"]-bar["low"]
        close_loc=(bar["close"]-bar["low"])/day_range if day_range>0 else 0.5
        vol_ratio=bar["volume"]/prev20_vol if prev20_vol else None
        dol=bar["raw_close"]*bar["volume"]
        dol_ratio=dol/prev20_dol if prev20_dol else None

        x[f"d{d}_close_ret_pct"]=cret
        x[f"d{d}_high_ret_pct"]=hret
        x[f"d{d}_low_ret_pct"]=lret
        x[f"d{d}_gap_pct"]=gap
        x[f"d{d}_close_location"]=close_loc
        x[f"d{d}_vol_ratio20"]=vol_ratio
        x[f"d{d}_dollarvol_ratio20"]=dol_ratio
        x[f"d{d}_close_above_signal_high"]=1 if bar["close"]>sig_high else 0
        x[f"d{d}_close_above_prior5_high"]=1 if bar["close"]>prior5_high else 0
        x[f"d{d}_no_new_low_vs_signal"]=1 if bar["low"]>=sig_low else 0
        x[f"d{d}_green"]=1 if bar["close"]>prev["close"] else 0

        if bar["close"]>prev["close"]:
            green_streak += 1
        else:
            green_streak = 0
        x[f"d{d}_green_streak"]=green_streak

        maxh5=max(maxh5,hret)
        maxdd5=min(maxdd5,lret)
        if d<=3:
            maxh3=max(maxh3,hret)
            maxdd3=min(maxdd3,lret)

    x["first3_max_high_pct"]=maxh3
    x["first3_max_drawdown_pct"]=maxdd3
    x["first5_max_high_pct"]=maxh5
    x["first5_max_drawdown_pct"]=maxdd5
    x["first5_close_ret_pct"]=pct(rows[i+5]["close"],sig_close)

    for th in (10,20):
        for horizon in (1,2,3,5):
            touched=any(pct(rows[i+d]["high"],sig_close)>=th for d in range(1,horizon+1))
            x[f"hit{th}_by_d{horizon}"]=1 if touched else 0

    feat_rows.append(x)

write_csv(OUT_FEAT,feat_rows)

by_year={
    y:{
        "w":[r for r in feat_rows if r["year"]==y and r["hit200"]==1],
        "l":[r for r in feat_rows if r["year"]==y and r["hit200"]==0],
    } for y in YEARS
}

# Numeric feature comparison.
ignore={"symbol","setup_date","year","hit200","max_gain_63d_pct","first200_day"}
feature_cols=[k for k in feat_rows[0] if k not in ignore]

stats=[]
for col in feature_cols:
    yearly={}
    dirs=[]
    ok=True
    for y in YEARS:
        w=[fnum(r.get(col)) for r in by_year[y]["w"]]
        l=[fnum(r.get(col)) for r in by_year[y]["l"]]
        w=[v for v in w if v is not None]
        l=[v for v in l if v is not None]
        if len(w)<2 or len(l)<3:
            ok=False; break
        a=auc(w,l)
        if a is None:
            ok=False; break
        dr=1 if a>=0.5 else -1
        dirs.append(dr)
        yearly[y]={
            "w_med":median(w),"l_med":median(l),
            "sep":max(a,1-a),"dir":dr
        }
    if not ok:
        continue
    same=len(set(dirs))==1
    stats.append({
        "feature":col,
        "same_direction_all_3_years":int(same),
        "direction":"HIGHER" if dirs[0]==1 else "LOWER",
        "min_sep":min(yearly[y]["sep"] for y in YEARS),
        "avg_sep":mean(yearly[y]["sep"] for y in YEARS),
        "winner_median_2024":yearly["2024"]["w_med"],
        "nonwinner_median_2024":yearly["2024"]["l_med"],
        "winner_median_2025":yearly["2025"]["w_med"],
        "nonwinner_median_2025":yearly["2025"]["l_med"],
        "winner_median_2026":yearly["2026"]["w_med"],
        "nonwinner_median_2026":yearly["2026"]["l_med"],
    })

stats.sort(key=lambda r:(r["same_direction_all_3_years"],r["min_sep"],r["avg_sep"]),reverse=True)
write_csv(OUT_STATS,stats)

# Simple rules from same-direction features.
winners=[r for r in feat_rows if r["hit200"]==1]
losers=[r for r in feat_rows if r["hit200"]==0]
rules=[]

for st in stats:
    if not st["same_direction_all_3_years"]:
        continue
    col=st["feature"]
    direction=1 if st["direction"]=="HIGHER" else -1
    wvals=[fnum(r.get(col)) for r in winners]
    wvals=[v for v in wvals if v is not None]
    if len(wvals)<8:
        continue

    if direction==1:
        cuts=[(qtile(wvals,.25),">="),(qtile(wvals,.40),">="),(qtile(wvals,.50),">=")]
    else:
        cuts=[(qtile(wvals,.75),"<="),(qtile(wvals,.60),"<="),(qtile(wvals,.50),"<=")]

    for cut,op in cuts:
        if cut is None:
            continue

        def passed(row):
            x=fnum(row.get(col))
            if x is None:
                return False
            return x>=cut if op==">=" else x<=cut

        yr={}
        valid=True
        for y in YEARS:
            w=by_year[y]["w"]; l=by_year[y]["l"]
            wp=sum(passed(z) for z in w); lp=sum(passed(z) for z in l)
            if not w or not l:
                valid=False; break
            yr[y]={"wp":wp,"lp":lp,"wr":wp/len(w),"lr":lp/len(l)}
        if not valid or any(yr[y]["wp"]<1 for y in YEARS):
            continue

        tw=sum(passed(z) for z in winners)
        tl=sum(passed(z) for z in losers)
        if tw<4:
            continue
        precision=tw/(tw+tl) if tw+tl else 0
        min_gap=min(yr[y]["wr"]-yr[y]["lr"] for y in YEARS)

        rules.append({
            "feature":col,"operator":op,"cut":cut,
            "winner_pass_total":tw,"nonwinner_pass_total":tl,
            "precision_pct":100*precision,
            "winner_coverage_pct":100*tw/len(winners),
            "min_year_gap_pp":100*min_gap,
            "winners_2024":yr["2024"]["wp"],"nonwinners_2024":yr["2024"]["lp"],
            "winners_2025":yr["2025"]["wp"],"nonwinners_2025":yr["2025"]["lp"],
            "winners_2026":yr["2026"]["wp"],"nonwinners_2026":yr["2026"]["lp"],
        })

rules.sort(key=lambda r:(r["precision_pct"],r["min_year_gap_pp"],r["winner_pass_total"]),reverse=True)
write_csv(OUT_RULES,rules)

# Pair rules from top simple rules.
top=rules[:20]
pairs=[]

def pass_rule(row,rr):
    x=fnum(row.get(rr["feature"]))
    if x is None:
        return False
    return x>=rr["cut"] if rr["operator"]==">=" else x<=rr["cut"]

for a in range(len(top)):
    for b in range(a+1,len(top)):
        r1=top[a]; r2=top[b]
        if r1["feature"]==r2["feature"]:
            continue

        def passed(row):
            return pass_rule(row,r1) and pass_rule(row,r2)

        yr={}
        valid=True
        for y in YEARS:
            w=by_year[y]["w"]; l=by_year[y]["l"]
            wp=sum(passed(z) for z in w); lp=sum(passed(z) for z in l)
            if not w or not l:
                valid=False; break
            yr[y]={"wp":wp,"lp":lp,"wr":wp/len(w),"lr":lp/len(l)}
        if not valid or any(yr[y]["wp"]<1 for y in YEARS):
            continue

        tw=sum(passed(z) for z in winners)
        tl=sum(passed(z) for z in losers)
        if tw<4:
            continue
        precision=tw/(tw+tl) if tw+tl else 0
        min_gap=min(yr[y]["wr"]-yr[y]["lr"] for y in YEARS)

        pairs.append({
            "rule1":f"{r1['feature']} {r1['operator']} {r1['cut']:.6g}",
            "rule2":f"{r2['feature']} {r2['operator']} {r2['cut']:.6g}",
            "winner_pass_total":tw,"nonwinner_pass_total":tl,
            "precision_pct":100*precision,
            "winner_coverage_pct":100*tw/len(winners),
            "min_year_gap_pp":100*min_gap,
            "winners_2024":yr["2024"]["wp"],"nonwinners_2024":yr["2024"]["lp"],
            "winners_2025":yr["2025"]["wp"],"nonwinners_2025":yr["2025"]["lp"],
            "winners_2026":yr["2026"]["wp"],"nonwinners_2026":yr["2026"]["lp"],
        })

pairs.sort(key=lambda r:(r["precision_pct"],r["min_year_gap_pp"],r["winner_pass_total"]),reverse=True)
write_csv(OUT_PAIRS,pairs)

summary={
    "candidateA_total":len(feat_rows),
    "winners":len(winners),
    "nonwinners":len(losers),
    "top_feature_stats":stats[:20],
    "top_rules":rules[:20],
    "top_pairs":pairs[:20],
}
OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

print("\n=== EARLY FEATURES — SAME DIRECTION ALL 3 YEARS ===")
shown=0
for r in stats:
    if not r["same_direction_all_3_years"]:
        continue
    print(
        f"{r['feature']}: {r['direction']} | min sep={r['min_sep']:.3f} | "
        f"2024 W/L={r['winner_median_2024']:.3f}/{r['nonwinner_median_2024']:.3f} | "
        f"2025={r['winner_median_2025']:.3f}/{r['nonwinner_median_2025']:.3f} | "
        f"2026={r['winner_median_2026']:.3f}/{r['nonwinner_median_2026']:.3f}"
    )
    shown+=1
    if shown>=15:
        break

print("\n=== BEST EARLY CONFIRMATION RULES ===")
for r in rules[:15]:
    print(
        f"{r['feature']} {r['operator']} {r['cut']:.4f} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | "
        f"coverage={r['winner_coverage_pct']:.1f}% | "
        f"min year gap={r['min_year_gap_pp']:.1f}pp"
    )

print("\n=== BEST EARLY CONFIRMATION PAIRS ===")
for r in pairs[:15]:
    print(
        f"{r['rule1']} AND {r['rule2']} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% | "
        f"coverage={r['winner_coverage_pct']:.1f}%"
    )

report=[]
report.append("RAJIH — CANDIDATE A EARLY CONFIRMATION")
report.append("="*100)
report.append(
    f"Candidate A total={len(feat_rows)} | winners={len(winners)} | non-winners={len(losers)}"
)
report.append("")
report.append("TOP SAME-DIRECTION FEATURES")
for r in stats[:15]:
    if r["same_direction_all_3_years"]:
        report.append(
            f"{r['feature']} | {r['direction']} | min_sep={r['min_sep']:.3f}"
        )
report.append("")
report.append("TOP EARLY RULES")
for r in rules[:15]:
    report.append(
        f"{r['feature']} {r['operator']} {r['cut']:.4f} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% coverage={r['winner_coverage_pct']:.1f}%"
    )
report.append("")
report.append("TOP EARLY PAIRS")
for r in pairs[:15]:
    report.append(
        f"{r['rule1']} AND {r['rule2']} | "
        f"W={r['winner_pass_total']} L={r['nonwinner_pass_total']} | "
        f"precision={r['precision_pct']:.1f}% coverage={r['winner_coverage_pct']:.1f}%"
    )

OUT_REPORT.write_text("\n".join(report),encoding="utf-8")

print("\nCreated:")
for p in [OUT_FEAT,OUT_STATS,OUT_RULES,OUT_PAIRS,OUT_JSON,OUT_REPORT]:
    print(" ",p)
