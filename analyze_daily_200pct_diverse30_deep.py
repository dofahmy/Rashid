#!/usr/bin/env python3
"""
Rajih — deep research on 30 diverse +200%/3-month winners vs matched controls.

Input:
  /data/daily_200pct_3months_clean.csv
  market_candles_1d

Selection:
- From the clean +200% pool, select 30 winners across approximately evenly spaced start dates.
- Within each selected date, choose the highest 20d dollar-volume candidate.
- Prefer unique symbols.

Checkpoints:
  T-20, T-10, T-5, T-3, T-1

Features:
  price, ATR%, RSI14, ADX14, EMA20/50 spread,
  SMA20/50/200 relations, MACD normalized by ATR,
  avg range 5/20, realized vol 20,
  volume ratio 5/20, dollar-volume ratio 5/20,
  distance to 20d/126d/252d highs and lows,
  min gap20, max gap20, up-day ratio20,
  return 5/10/20, Bollinger width and z-score.

Matched controls:
- Same start date
- Start price in roughly same bucket
- Similar 20d dollar liquidity
- Enough history
- Did NOT reach +200% within next 63 sessions
- No >100% adjusted one-day discontinuity in prior 20 or next 63 sessions
- Up to 10 controls per winner

Outputs:
  /data/daily_200pct_diverse30_winners.csv
  /data/daily_200pct_diverse30_controls.csv
  /data/daily_200pct_winner_vs_control.csv
  /data/daily_200pct_threshold_rules.csv
  /data/daily_200pct_pair_rules.csv
  /data/daily_200pct_deep_report.txt
  /data/daily_200pct_deep_summary.json

Run:
  python analyze_daily_200pct_diverse30_deep.py
"""

from __future__ import annotations

import os, csv, json, math
from pathlib import Path
from statistics import mean, median

from sqlalchemy import MetaData, Table, select, func
from core import database

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
INPUT = DATA_DIR / "daily_200pct_3months_clean.csv"

OUT_WIN = DATA_DIR / "daily_200pct_diverse30_winners.csv"
OUT_CTRL = DATA_DIR / "daily_200pct_diverse30_controls.csv"
OUT_COMP = DATA_DIR / "daily_200pct_winner_vs_control.csv"
OUT_RULES = DATA_DIR / "daily_200pct_threshold_rules.csv"
OUT_PAIRS = DATA_DIR / "daily_200pct_pair_rules.csv"
OUT_REPORT = DATA_DIR / "daily_200pct_deep_report.txt"
OUT_JSON = DATA_DIR / "daily_200pct_deep_summary.json"

CHECKPOINTS = [-20,-10,-5,-3,-1]
TARGET = 200.0
LOOKAHEAD = 63
MAX_CONTROLS_PER_WINNER = 10

if not INPUT.exists():
    raise SystemExit(f"Missing {INPUT}. Run find_daily_200pct_3months_clean.py first.")

def fnum(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except:
        return None

def pct(a,b):
    return 100*(a/b-1)

def stdev(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs)<2:
        return None
    m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def ema(vals,n):
    out=[None]*len(vals)
    if not vals: return out
    a=2/(n+1)
    e=float(vals[0]); out[0]=e
    for i in range(1,len(vals)):
        e=a*float(vals[i])+(1-a)*e
        out[i]=e
    return out

def sma(vals,n):
    out=[None]*len(vals)
    s=0.0
    for i,x in enumerate(vals):
        s+=x
        if i>=n: s-=vals[i-n]
        if i>=n-1: out[i]=s/n
    return out

def atr(h,l,c,n=14):
    out=[None]*len(c)
    if len(c)<=n: return out
    tr=[None]*len(c)
    for i in range(1,len(c)):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=sum(tr[1:n+1])/n
    out[n]=a
    for i in range(n+1,len(c)):
        a=((n-1)*a+tr[i])/n
        out[i]=a
    return out

def rsi(c,n=14):
    out=[None]*len(c)
    if len(c)<=n: return out
    gains=[]; losses=[]
    for i in range(1,len(c)):
        d=c[i]-c[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains[:n])/n; al=sum(losses[:n])/n
    out[n]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    for i in range(n+1,len(c)):
        ag=((n-1)*ag+gains[i-1])/n
        al=((n-1)*al+losses[i-1])/n
        out[i]=100 if al==0 and ag>0 else (50 if al==0 else 100-100/(1+ag/al))
    return out

def adx(h,l,c,n=14):
    m=len(c); out=[None]*m
    if m<2*n+1: return out
    tr=[0.0]*m; pdm=[0.0]*m; mdm=[0.0]*m
    for i in range(1,m):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0.0
        mdm[i]=dn if dn>up and dn>0 else 0.0
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    atrs=sum(tr[1:n+1]); ps=sum(pdm[1:n+1]); ms=sum(mdm[1:n+1])
    dx=[None]*m
    for i in range(n,m):
        if i>n:
            atrs=atrs-atrs/n+tr[i]
            ps=ps-ps/n+pdm[i]
            ms=ms-ms/n+mdm[i]
        if atrs<=0: continue
        pdi=100*ps/atrs; mdi=100*ms/atrs
        den=pdi+mdi
        dx[i]=0 if den==0 else 100*abs(pdi-mdi)/den
    seed=[x for x in dx[n:2*n] if x is not None]
    if len(seed)<n: return out
    a=sum(seed)/n; out[2*n-1]=a
    for i in range(2*n,m):
        if dx[i] is not None:
            a=((n-1)*a+dx[i])/n
            out[i]=a
    return out

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
            "date":str(d),"raw_close":rc,"close":adj,
            "open":ro*fac,"high":rh*fac,"low":rl*fac,
            "volume":float(v or 0)
        })
    return rows

def build(rows):
    c=[r["close"] for r in rows]
    o=[r["open"] for r in rows]
    h=[r["high"] for r in rows]
    l=[r["low"] for r in rows]
    v=[r["volume"] for r in rows]
    rc=[r["raw_close"] for r in rows]
    e20=ema(c,20); e50=ema(c,50)
    s20=sma(c,20); s50=sma(c,50); s200=sma(c,200)
    a14=atr(h,l,c,14); r14=rsi(c,14); aX=adx(h,l,c,14)
    e12=ema(c,12); e26=ema(c,26)
    macd=[None if e12[i] is None or e26[i] is None else e12[i]-e26[i] for i in range(len(c))]
    sig=ema([0 if x is None else x for x in macd],9)

    ret=[None]*len(c); gap=[None]*len(c)
    for i in range(1,len(c)):
        ret[i]=c[i]/c[i-1]-1
        gap[i]=100*(o[i]/c[i-1]-1)

    return locals()

def feat(F,i):
    c=F["c"]; h=F["h"]; l=F["l"]; v=F["v"]; rc=F["rc"]
    if i<20: return None
    def avg_range(a,b):
        return mean(100*(h[j]-l[j])/c[j] for j in range(a,b+1))
    def maxh(n):
        return max(h[max(0,i-n+1):i+1])
    def minl(n):
        return min(l[max(0,i-n+1):i+1])

    rv=stdev(F["ret"][max(1,i-19):i+1])
    vol20=mean(v[i-19:i+1])
    vol5=mean(v[i-4:i+1]) if i>=4 else None
    dol=[rc[j]*v[j] for j in range(len(c))]
    dol20=mean(dol[i-19:i+1])
    dol5=mean(dol[i-4:i+1]) if i>=4 else None
    bb_mid=F["s20"][i]
    bb_sd=stdev(c[i-19:i+1]) if i>=19 else None

    ups=sum(1 for j in range(i-19,i+1) if j>0 and c[j]>c[j-1]) / 20

    out={
        "price":rc[i],
        "atr_pct":100*F["a14"][i]/c[i] if F["a14"][i] else None,
        "rsi14":F["r14"][i],
        "adx14":F["aX"][i],
        "ema20_vs_ema50_pct":100*(F["e20"][i]/F["e50"][i]-1) if F["e20"][i] and F["e50"][i] else None,
        "above_sma20":1 if F["s20"][i] and c[i]>F["s20"][i] else 0,
        "above_sma50":1 if F["s50"][i] and c[i]>F["s50"][i] else 0,
        "above_sma200":1 if F["s200"][i] and c[i]>F["s200"][i] else 0,
        "macd_atr":(F["macd"][i]/F["a14"][i]) if F["macd"][i] is not None and F["a14"][i] else None,
        "macd_signal_atr":((F["macd"][i]-F["sig"][i])/F["a14"][i]) if F["macd"][i] is not None and F["sig"][i] is not None and F["a14"][i] else None,
        "avg_range5_pct":avg_range(i-4,i) if i>=4 else None,
        "avg_range20_pct":avg_range(i-19,i),
        "realized_vol20_pct":100*rv if rv is not None else None,
        "vol_ratio_5_20":vol5/vol20 if vol5 is not None and vol20 else None,
        "dollarvol_ratio_5_20":dol5/dol20 if dol5 is not None and dol20 else None,
        "avg_dollar_volume20":dol20,
        "distance_to_20d_high_pct":100*(c[i]/maxh(20)-1),
        "distance_to_126d_high_pct":100*(c[i]/maxh(126)-1) if i>=125 else None,
        "distance_to_252d_high_pct":100*(c[i]/maxh(252)-1) if i>=251 else None,
        "distance_to_20d_low_pct":100*(c[i]/minl(20)-1),
        "distance_to_126d_low_pct":100*(c[i]/minl(126)-1) if i>=125 else None,
        "distance_to_252d_low_pct":100*(c[i]/minl(252)-1) if i>=251 else None,
        "min_gap20_pct":min(x for x in F["gap"][i-19:i+1] if x is not None),
        "max_gap20_pct":max(x for x in F["gap"][i-19:i+1] if x is not None),
        "up_day_ratio20":ups,
        "ret5_pct":100*(c[i]/c[i-5]-1) if i>=5 else None,
        "ret10_pct":100*(c[i]/c[i-10]-1) if i>=10 else None,
        "ret20_pct":100*(c[i]/c[i-20]-1) if i>=20 else None,
        "bb_width_pct":100*(4*bb_sd/bb_mid) if bb_sd is not None and bb_mid else None,
        "bb_z":(c[i]-bb_mid)/bb_sd if bb_sd not in (None,0) and bb_mid is not None else None,
    }
    return out

def clean_forward(rows,i):
    for j in range(max(1,i-19),min(len(rows)-1,i+LOOKAHEAD)+1):
        if abs(pct(rows[j]["close"],rows[j-1]["close"]))>100:
            return False
    return True

def hits_200(rows,i):
    start=rows[i]["close"]
    for j in range(i+1,min(len(rows),i+LOOKAHEAD+1)):
        if pct(rows[j]["high"],start)>=200:
            return True
    return False

def auc(w,c):
    w=[x for x in w if x is not None]; c=[x for x in c if x is not None]
    if not w or not c: return None
    s=0.0
    for a in w:
        for b in c:
            if a>b: s+=1
            elif a==b: s+=0.5
    return s/(len(w)*len(c))

# Load winner pool.
with INPUT.open(encoding="utf-8-sig",newline="") as fh:
    pool=list(csv.DictReader(fh))

# Pick 30 approximately evenly spaced start dates.
by_date={}
for r in pool:
    by_date.setdefault(r["start_date"],[]).append(r)
dates=sorted(by_date)
if len(dates)<30:
    raise SystemExit("Not enough distinct winner dates.")

idxs=sorted(set(round(i*(len(dates)-1)/29) for i in range(30)))
chosen_dates=[dates[i] for i in idxs]

winners=[]
used=set()
for d in chosen_dates:
    cand=sorted(by_date[d],key=lambda r:float(r["avg_dollar_volume20"]),reverse=True)
    pick=None
    for r in cand:
        if r["symbol"] not in used:
            pick=r; break
    if pick is None:
        pick=cand[0]
    used.add(pick["symbol"])
    winners.append(pick)

DB=database()
with DB() as s:
    md=MetaData()
    daily=Table("market_candles_1d",md,autoload_with=s.get_bind())

# Universe metadata.
with DB() as s:
    symbols=list(s.execute(
        select(daily.c.symbol)
        .group_by(daily.c.symbol)
        .having(func.count()>=300)
        .order_by(daily.c.symbol)
    ).scalars().all())

cache={}
def get(sym):
    if sym not in cache:
        rows=load_symbol(DB,daily,sym)
        cache[sym]=(rows,build(rows))
    return cache[sym]

# Winner features.
winner_rows=[]
for w in winners:
    rows,F=get(w["symbol"])
    dates2=[r["date"] for r in rows]
    try: i=dates2.index(w["start_date"])
    except ValueError: continue
    base={
        "symbol":w["symbol"],"start_date":w["start_date"],
        "start_raw_close":float(w["start_raw_close"]),
        "avg_dollar_volume20":float(w["avg_dollar_volume20"]),
        "month1_max_gain_pct":float(w["month1_max_gain_pct"]),
        "first_200_touch_day":int(float(w["first_200_touch_day"])),
        "max_gain_63d_pct":float(w["max_gain_63d_pct"]),
    }
    for cp in CHECKPOINTS:
        ft=feat(F,i+cp)
        if ft:
            for k,v in ft.items():
                base[f"t{cp}_{k}"]=v
    winner_rows.append(base)

# Matched controls.
control_rows=[]
winner_dates=set(w["start_date"] for w in winners)
winner_syms=set(w["symbol"] for w in winners)

for wi,w in enumerate(winners,1):
    target_date=w["start_date"]
    wp=float(w["start_raw_close"])
    wliq=float(w["avg_dollar_volume20"])
    candidates=[]

    for sym in symbols:
        if sym in winner_syms:
            continue
        rows,F=get(sym)
        dmap={r["date"]:idx for idx,r in enumerate(rows)}
        i=dmap.get(target_date)
        if i is None or i<252 or i+LOOKAHEAD>=len(rows):
            continue
        price=rows[i]["raw_close"]
        if price<=0:
            continue
        # broad same-price neighborhood: 0.5x to 2x, but at least $1
        if price < max(1.0, wp*0.5) or price > wp*2.0:
            continue
        ft=feat(F,i)
        if not ft or ft["avg_dollar_volume20"] is None:
            continue
        liq=ft["avg_dollar_volume20"]
        if liq<=0:
            continue
        # liquidity within ~0.25x..4x
        if liq < wliq/4 or liq > wliq*4:
            continue
        if not clean_forward(rows,i):
            continue
        if hits_200(rows,i):
            continue

        # matching score in log price/liquidity space
        score=abs(math.log(price/wp))+abs(math.log(liq/wliq))
        candidates.append((score,sym,i,rows,F))

    candidates.sort(key=lambda x:x[0])
    for score,sym,i,rows,F in candidates[:MAX_CONTROLS_PER_WINNER]:
        base={
            "matched_winner_symbol":w["symbol"],
            "symbol":sym,
            "start_date":target_date,
            "start_raw_close":rows[i]["raw_close"],
            "match_score":score,
        }
        for cp in CHECKPOINTS:
            ft=feat(F,i+cp)
            if ft:
                for k,v in ft.items():
                    base[f"t{cp}_{k}"]=v
        control_rows.append(base)

    print(f"Matched {wi}/{len(winners)} winners | controls so far={len(control_rows)}")

def write_csv(path,rows):
    if not rows: return
    keys=[]
    seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w",encoding="utf-8-sig",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=keys)
        w.writeheader(); w.writerows(rows)

write_csv(OUT_WIN,winner_rows)
write_csv(OUT_CTRL,control_rows)

# Compare numeric features.
feature_cols=[k for k in winner_rows[0] if k.startswith("t-")]
comparison=[]
for col in feature_cols:
    wv=[fnum(r.get(col)) for r in winner_rows]
    cv=[fnum(r.get(col)) for r in control_rows]
    wv=[x for x in wv if x is not None]; cv=[x for x in cv if x is not None]
    if len(wv)<15 or len(cv)<50:
        continue
    a=auc(wv,cv)
    if a is None: continue
    comparison.append({
        "feature":col,
        "winner_n":len(wv),
        "control_n":len(cv),
        "winner_median":median(wv),
        "control_median":median(cv),
        "winner_mean":mean(wv),
        "control_mean":mean(cv),
        "auc_winner_gt_control":a,
        "separation":max(a,1-a),
        "winner_like_direction":"HIGHER" if a>=0.5 else "LOWER",
    })
comparison.sort(key=lambda r:r["separation"],reverse=True)
write_csv(OUT_COMP,comparison)

# Threshold scan on top features.
rules=[]
base_rate=len(winner_rows)/(len(winner_rows)+len(control_rows))
all_rows=[(1,r) for r in winner_rows]+[(0,r) for r in control_rows]

for comp in comparison[:30]:
    col=comp["feature"]
    vals=sorted(set(fnum(r.get(col)) for _,r in all_rows if fnum(r.get(col)) is not None))
    if len(vals)<6: continue
    # decile-ish candidate cuts.
    cuts=[]
    for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        pos=min(len(vals)-1,max(0,round((len(vals)-1)*q)))
        cuts.append(vals[pos])
    for cut in sorted(set(cuts)):
        for op in (">=","<="):
            selected=[(y,r) for y,r in all_rows if fnum(r.get(col)) is not None and ((fnum(r.get(col))>=cut) if op==">=" else (fnum(r.get(col))<=cut))]
            if len(selected)<20: continue
            wins=sum(y for y,_ in selected)
            ctrls=len(selected)-wins
            if wins<3: continue
            precision=wins/len(selected)
            recall=wins/len(winner_rows)
            rules.append({
                "feature":col,"operator":op,"cut":cut,
                "selected_n":len(selected),"winners":wins,"controls":ctrls,
                "precision_pct":100*precision,
                "recall_pct":100*recall,
                "lift_vs_base":precision/base_rate if base_rate else None,
            })

rules.sort(key=lambda r:(r["lift_vs_base"],r["precision_pct"],r["recall_pct"]),reverse=True)
write_csv(OUT_RULES,rules)

# Pair scan on top 12 simple rules with distinct features.
top_rules=[]
usedkeys=set()
for r in rules:
    key=(r["feature"],r["operator"],round(float(r["cut"]),8))
    if key in usedkeys: continue
    top_rules.append(r); usedkeys.add(key)
    if len(top_rules)>=12: break

pairs=[]
for a in range(len(top_rules)):
    for b in range(a+1,len(top_rules)):
        r1=top_rules[a]; r2=top_rules[b]
        if r1["feature"]==r2["feature"]:
            continue
        def passr(r,rule):
            x=fnum(r.get(rule["feature"]))
            if x is None: return False
            return x>=rule["cut"] if rule["operator"]==">=" else x<=rule["cut"]
        sel=[(y,r) for y,r in all_rows if passr(r,r1) and passr(r,r2)]
        if len(sel)<15: continue
        wins=sum(y for y,_ in sel)
        if wins<3: continue
        precision=wins/len(sel)
        pairs.append({
            "rule1":f"{r1['feature']} {r1['operator']} {r1['cut']}",
            "rule2":f"{r2['feature']} {r2['operator']} {r2['cut']}",
            "selected_n":len(sel),
            "winners":wins,
            "controls":len(sel)-wins,
            "precision_pct":100*precision,
            "recall_pct":100*wins/len(winner_rows),
            "lift_vs_base":precision/base_rate if base_rate else None,
        })
pairs.sort(key=lambda r:(r["lift_vs_base"],r["precision_pct"],r["recall_pct"]),reverse=True)
write_csv(OUT_PAIRS,pairs)

report=[]
report.append("RAJIH — +200% / 3-MONTH DEEP RESEARCH")
report.append("="*80)
report.append(f"Winners: {len(winner_rows)} | Controls: {len(control_rows)} | Base winner share={100*base_rate:.2f}%")
report.append("Winner dates: " + ", ".join(r["start_date"] for r in winner_rows))
report.append("")
report.append("TOP SEPARATING FEATURES")
report.append("-"*80)
for r in comparison[:20]:
    report.append(
        f"{r['feature']}: W med={r['winner_median']:.4f} | "
        f"C med={r['control_median']:.4f} | sep={r['separation']:.4f} | "
        f"{r['winner_like_direction']}"
    )
report.append("")
report.append("TOP SINGLE RULES")
report.append("-"*80)
for r in rules[:20]:
    report.append(
        f"{r['feature']} {r['operator']} {r['cut']:.6g} | "
        f"N={r['selected_n']} | W={r['winners']} C={r['controls']} | "
        f"precision={r['precision_pct']:.2f}% | recall={r['recall_pct']:.2f}% | "
        f"lift={r['lift_vs_base']:.2f}x"
    )
report.append("")
report.append("TOP PAIRS")
report.append("-"*80)
for r in pairs[:20]:
    report.append(
        f"{r['rule1']} AND {r['rule2']} | N={r['selected_n']} | "
        f"W={r['winners']} C={r['controls']} | precision={r['precision_pct']:.2f}% | "
        f"recall={r['recall_pct']:.2f}% | lift={r['lift_vs_base']:.2f}x"
    )
OUT_REPORT.write_text("\n".join(report),encoding="utf-8")

summary={
    "winners":len(winner_rows),
    "controls":len(control_rows),
    "winner_dates":[r["start_date"] for r in winner_rows],
    "winner_symbols":[r["symbol"] for r in winner_rows],
    "top_features":comparison[:20],
    "top_rules":rules[:20],
    "top_pairs":pairs[:20],
}
OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

print("\n=== +200% DEEP RESEARCH ===")
print(f"Winners: {len(winner_rows)}")
print(f"Matched controls: {len(control_rows)}")
print("\nWinner sample:")
for i,r in enumerate(winner_rows,1):
    print(
        f"{i:>2}. {r['symbol']:<6} {r['start_date']} "
        f"${r['start_raw_close']:.2f} | M1={r['month1_max_gain_pct']:.1f}% | "
        f"+200 day={r['first_200_touch_day']} | 63d={r['max_gain_63d_pct']:.1f}%"
    )

print("\nTOP DIFFERENCES:")
for r in comparison[:12]:
    print(
        f"{r['feature']}: W med={r['winner_median']:.3f} | "
        f"C med={r['control_median']:.3f} | sep={r['separation']:.3f} | "
        f"{r['winner_like_direction']}"
    )

print("\nBEST SINGLE RULES:")
for r in rules[:10]:
    print(
        f"{r['feature']} {r['operator']} {r['cut']:.4f} | "
        f"N={r['selected_n']} W={r['winners']} C={r['controls']} | "
        f"precision={r['precision_pct']:.2f}% recall={r['recall_pct']:.2f}% "
        f"lift={r['lift_vs_base']:.2f}x"
    )

print("\nBEST PAIRS:")
for r in pairs[:10]:
    print(
        f"{r['rule1']} AND {r['rule2']} | N={r['selected_n']} "
        f"W={r['winners']} C={r['controls']} | "
        f"precision={r['precision_pct']:.2f}% recall={r['recall_pct']:.2f}% "
        f"lift={r['lift_vs_base']:.2f}x"
    )

print("\nCreated:")
for p in [OUT_WIN,OUT_CTRL,OUT_COMP,OUT_RULES,OUT_PAIRS,OUT_REPORT,OUT_JSON]:
    print(" ",p)
