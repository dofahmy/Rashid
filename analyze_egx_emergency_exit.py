#!/usr/bin/env python3
import sys, subprocess, importlib.util

REQUIRED = ["numpy", "pandas", "requests"]
missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing])

import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path

"""
EGX EMERGENCY EXIT RESEARCH
===========================

FIXED ENTRY
-----------
Standalone R2 + Slope:
    R2 >= 0.791694
    Slope >= 67.5062

FIXED FAILURE EXIT
------------------
Only during first 60 trading sessions:
    R2 <= entry_R2 - 0.05
    Slope <= entry_Slope * 45%
    Close < prior 10-day low

FIXED PEAK EXIT
---------------
    Slope falls 57.5% from post-entry peak
    R2 falls 0.04 from post-entry peak
    Close < prior 5-day low

THIS SCRIPT SEARCHES ONLY FOR:
------------------------------
An EMERGENCY / WORST-LOSER exit that stops catastrophic losses earlier.

Candidate families:
1) HARD_STOP:
      close loss from entry <= threshold
2) HARD_STOP_2CLOSES:
      two consecutive closes below threshold
3) LOSS_PLUS_10D_BREAK:
      loss from entry <= threshold AND close below prior 10-day low
4) PEAK_DRAWDOWN:
      drawdown from highest close since entry >= threshold AND current return <= cap
5) FAST_DAMAGE:
      by day 10/15/20/30 the trade is below a loss threshold and never gained much

Emergency exit is tested only in first 90 sessions.

Ranking is done only on mature 1-year trades.
We heavily penalize false exits from trades that later reach +50%.

Outputs:
  /data/egx_emergency_exit_grid.csv
  /data/egx_emergency_exit_best_trades.csv
  /data/egx_emergency_exit_2026.csv
  /data/egx_emergency_exit_report.txt
"""

DATA = Path("/data")
SIGNALS = DATA / "egx_r2slope_standalone_all.csv"

OUT_GRID = DATA / "egx_emergency_exit_grid.csv"
OUT_BEST = DATA / "egx_emergency_exit_best_trades.csv"
OUT_2026 = DATA / "egx_emergency_exit_2026.csv"
OUT_REPORT = DATA / "egx_emergency_exit_report.txt"

LOOKBACK = 126
MATURITY = 252
MAX_HOLD = 504

# Fixed failure exit
FAIL_MAX_AGE = 60
FAIL_R2_DROP = 0.05
FAIL_SLOPE_DROP = 0.55      # current <= 45% of entry slope
FAIL_BREAK_DAYS = 10

# Fixed peak exit
PEAK_R2_DROP = 0.04
PEAK_SLOPE_DROP = 0.575
PEAK_BREAK_DAYS = 5
PEAK_MAX_ARM_AGE = 30

# Emergency
EMERGENCY_MAX_AGE = 90

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})

def download(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.CA"
    params = {
        "period1": int(pd.Timestamp("2019-01-01", tz="UTC").timestamp()),
        "period2": int(pd.Timestamp.now("UTC").timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None

    obj = r.json()
    res = ((obj.get("chart") or {}).get("result") or [])
    if not res:
        return None

    z = res[0]
    ts = z.get("timestamp") or []
    q = (((z.get("indicators") or {}).get("quote") or [{}])[0])
    adj = ((z.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not ts:
        return None

    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "o": q.get("open"), "h": q.get("high"), "l": q.get("low"),
        "c": q.get("close"), "v": q.get("volume"),
    })
    df["adj_c"] = adj if adj and len(adj) == len(df) else df["c"]

    for c in ["o","h","l","c","v","adj_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["o","h","l","c","adj_c"])
    df = df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    factor = df["adj_c"] / df["c"]
    df["ah"] = df["h"] * factor
    df["al"] = df["l"] * factor
    df["ac"] = df["adj_c"]
    return df

def add_metrics(df):
    n = len(df)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    x = np.arange(LOOKBACK, dtype=float)
    xd = x - x.mean()
    xss = np.sum(xd*xd)
    ac = df["ac"].to_numpy(float)

    for i in range(LOOKBACK, n):
        y = ac[i-LOOKBACK:i]
        if np.any(~np.isfinite(y)):
            continue
        ym = y.mean()
        yd = y - ym
        b = np.sum(xd*yd)/xss
        pred = ym + b*xd
        ssr = np.sum((y-pred)**2)
        sst = np.sum(yd*yd)
        r2[i] = 1-ssr/sst if sst > 0 else 0.0
        slope[i] = 100*(b*(LOOKBACK-1))/ym if ym else np.nan

    out = df.copy()
    out["trend_slope"] = slope
    out["trend_r2"] = r2
    return out

signals = pd.read_csv(SIGNALS, dtype={"symbol":str})
signals["signal_date"] = pd.to_datetime(signals["signal_date"])
symbols = sorted(signals["symbol"].unique())

histories = {}

print("\nEGX EMERGENCY EXIT RESEARCH")
print("Signals:", len(signals), "Symbols:", len(symbols))
print("Entry / Failure Exit / Peak Exit are FIXED.")
print("Searching only emergency protection for worst losers.")

for n,s in enumerate(symbols,1):
    df = download(s)
    if df is not None and len(df) > LOOKBACK:
        histories[s] = add_metrics(df)
    if n % 25 == 0 or n == len(symbols):
        print(f"[{n}/{len(symbols)}] usable={len(histories)}")
    time.sleep(0.02)

# Prepare trade metadata.
base = []
for sig in signals.itertuples(index=False):
    df = histories.get(sig.symbol)
    if df is None:
        continue

    d = pd.Timestamp(sig.signal_date).normalize()
    ids = df.index[df["date"] == d].tolist()
    if not ids:
        continue

    i = ids[-1]
    if i+1 >= len(df):
        continue

    entry = float(sig.signal_price)
    entry_slope = float(df.at[i,"trend_slope"]) if np.isfinite(df.at[i,"trend_slope"]) else float(sig.pre_trend_slope_pct)
    entry_r2 = float(df.at[i,"trend_r2"]) if np.isfinite(df.at[i,"trend_r2"]) else float(sig.pre_trend_r2)

    end = min(len(df)-1, i+MAX_HOLD)
    mature = (i+MATURITY < len(df))
    horizon_end = min(len(df)-1, i+MATURITY)
    hseg = df.loc[i+1:horizon_end]

    max_gain_1y = 100*(float(hseg["ah"].max())/entry - 1) if len(hseg) else np.nan
    min_dd_1y = 100*(float(hseg["al"].min())/entry - 1) if len(hseg) else np.nan
    final_1y = 100*(float(df.at[horizon_end,"ac"])/entry - 1) if len(hseg) else np.nan

    tp50_hits = hseg.index[hseg["ah"] >= entry*1.50].tolist()
    tp50_idx = tp50_hits[0] if tp50_hits else None

    # "catastrophic loser" label for research only:
    # did NOT reach +50% in 1Y and suffered <= -30% drawdown.
    catastrophic = int((tp50_idx is None) and np.isfinite(min_dd_1y) and min_dd_1y <= -30)

    base.append({
        "symbol":sig.symbol,
        "signal_date":d,
        "year":int(d.year),
        "i":i,
        "end":end,
        "entry":entry,
        "entry_slope":entry_slope,
        "entry_r2":entry_r2,
        "mature_1y":mature,
        "winner50_1y":int(tp50_idx is not None),
        "tp50_idx":tp50_idx,
        "max_gain_1y":max_gain_1y,
        "min_dd_1y":min_dd_1y,
        "final_1y":final_1y,
        "catastrophic_loser":catastrophic,
    })

print("Prepared trades:", len(base))
print("Mature 1Y:", sum(x["mature_1y"] for x in base))
print("Catastrophic losers:", sum(x["catastrophic_loser"] for x in base if x["mature_1y"]))

def fixed_failure_idx(df,t):
    i=t["i"]
    end=min(t["end"], i+FAIL_MAX_AGE)
    for j in range(i+5,end+1):
        r=df.at[j,"trend_r2"]
        s=df.at[j,"trend_slope"]
        c=df.at[j,"ac"]

        rbad=np.isfinite(r) and r <= t["entry_r2"]-FAIL_R2_DROP
        sbad=np.isfinite(s) and s <= t["entry_slope"]*(1-FAIL_SLOPE_DROP)

        if j >= FAIL_BREAK_DAYS:
            low10=df.loc[j-FAIL_BREAK_DAYS:j-1,"al"].min()
            pbad=np.isfinite(low10) and c < low10
            if rbad and sbad and pbad:
                return j
    return None

def fixed_peak_idx(df,t):
    i=t["i"]
    end=t["end"]
    peak_price=-np.inf
    peak_s=-np.inf
    peak_r=-np.inf
    armed=False
    armed_idx=None

    for j in range(i+1,end+1):
        h=df.at[j,"ah"]
        s=df.at[j,"trend_slope"]
        r=df.at[j,"trend_r2"]
        c=df.at[j,"ac"]

        if np.isfinite(h): peak_price=max(peak_price,h)
        if np.isfinite(s): peak_s=max(peak_s,s)
        if np.isfinite(r): peak_r=max(peak_r,r)

        if j-i < 10:
            continue

        near_high=np.isfinite(peak_price) and h >= 0.98*peak_price
        sroll=np.isfinite(s) and np.isfinite(peak_s) and peak_s>0 and s <= peak_s*(1-PEAK_SLOPE_DROP)
        rroll=np.isfinite(r) and np.isfinite(peak_r) and r <= peak_r-PEAK_R2_DROP

        if (not armed) and near_high and sroll and rroll:
            armed=True
            armed_idx=j

        if armed and j-armed_idx > PEAK_MAX_ARM_AGE:
            armed=False
            armed_idx=None

        if armed and j >= PEAK_BREAK_DAYS:
            low5=df.loc[j-PEAK_BREAK_DAYS:j-1,"al"].min()
            if np.isfinite(low5) and c < low5:
                return j
    return None

for t in base:
    df=histories[t["symbol"]]
    t["fixed_failure_idx"]=fixed_failure_idx(df,t)
    t["fixed_peak_idx"]=fixed_peak_idx(df,t)

# Emergency candidates.
candidates=[]

# hard close loss
for loss in [0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30]:
    candidates.append(("HARD_STOP",loss,None,None,None))

# 2 closes under threshold
for loss in [0.10,0.12,0.15,0.18,0.20,0.22,0.25]:
    candidates.append(("HARD_STOP_2CLOSES",loss,None,None,None))

# loss + 10d low break
for loss in [0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25]:
    candidates.append(("LOSS_PLUS_10D_BREAK",loss,None,None,None))

# drawdown from best close after entry + current return cap
for dd in [0.15,0.20,0.25,0.30,0.35]:
    for cap in [0.00,-0.05,-0.10]:
        candidates.append(("PEAK_DRAWDOWN",None,dd,cap,None))

# fast damage at fixed day:
# day, current loss threshold, max gain ceiling before then
for day in [10,15,20,30]:
    for loss in [0.08,0.10,0.12,0.15,0.18,0.20]:
        for maxgain in [0.05,0.10,0.15]:
            candidates.append(("FAST_DAMAGE",loss,None,None,(day,maxgain)))

def emergency_idx(df,t,cand):
    fam,loss,dd,cap,extra=cand
    i=t["i"]
    end=min(t["end"], i+EMERGENCY_MAX_AGE)

    peak_close=t["entry"]
    below_count=0

    for j in range(i+1,end+1):
        c=float(df.at[j,"ac"])
        h=float(df.at[j,"ah"])
        ret=c/t["entry"]-1

        peak_close=max(peak_close,c)

        if fam=="HARD_STOP":
            if ret <= -loss:
                return j

        elif fam=="HARD_STOP_2CLOSES":
            if ret <= -loss:
                below_count += 1
            else:
                below_count = 0
            if below_count >= 2:
                return j

        elif fam=="LOSS_PLUS_10D_BREAK":
            if j >= 10:
                low10=df.loc[j-10:j-1,"al"].min()
                if ret <= -loss and np.isfinite(low10) and c < low10:
                    return j

        elif fam=="PEAK_DRAWDOWN":
            peakdd=c/peak_close-1
            if peakdd <= -dd and ret <= cap:
                return j

        elif fam=="FAST_DAMAGE":
            day,maxgain=extra
            held=j-i
            if held == day:
                seg=df.loc[i+1:j]
                gain=seg["ah"].max()/t["entry"]-1
                if ret <= -loss and gain < maxgain:
                    return j

    return None

rows=[]
trade_sets={}
total=len(candidates)

for ci,cand in enumerate(candidates,1):
    ots=[]

    for t in base:
        df=histories[t["symbol"]]
        eidx=emergency_idx(df,t,cand)
        fidx=t["fixed_failure_idx"]
        pidx=t["fixed_peak_idx"]

        options=[]
        if eidx is not None: options.append((eidx,"EMERGENCY_EXIT"))
        if fidx is not None: options.append((fidx,"FAILURE_EXIT"))
        if pidx is not None: options.append((pidx,"PEAK_EXIT"))

        if options:
            exit_idx,reason=min(options,key=lambda x:x[0])
        else:
            exit_idx=t["end"]
            reason="OPEN_OR_TIME"

        ret=100*(float(df.at[exit_idx,"ac"])/t["entry"]-1)
        seg=df.loc[t["i"]+1:exit_idx]
        maxdd=100*(float(seg["al"].min())/t["entry"]-1) if len(seg) else np.nan

        false_emergency_winner = (
            reason=="EMERGENCY_EXIT"
            and t["winner50_1y"]==1
            and t["tp50_idx"] is not None
            and exit_idx < t["tp50_idx"]
        )

        ots.append({
            "signal_date":t["signal_date"].date().isoformat(),
            "year":t["year"],
            "symbol":t["symbol"],
            "mature_1y":int(t["mature_1y"]),
            "winner50_1y":t["winner50_1y"],
            "catastrophic_loser":t["catastrophic_loser"],
            "max_gain_1y":t["max_gain_1y"],
            "min_dd_1y":t["min_dd_1y"],
            "exit_date":df.at[exit_idx,"date"].date().isoformat(),
            "sessions_held":exit_idx-t["i"],
            "exit_reason":reason,
            "return_pct":ret,
            "max_dd_before_exit_pct":maxdd,
            "false_emergency_winner50":int(false_emergency_winner),
        })

    ot=pd.DataFrame(ots)
    mature=ot[ot["mature_1y"]==1].copy()
    if mature.empty:
        continue

    winners=mature[mature["winner50_1y"]==1]
    cats=mature[mature["catastrophic_loser"]==1]

    emergency_n=int((mature["exit_reason"]=="EMERGENCY_EXIT").sum())
    false_w=int(mature["false_emergency_winner50"].sum())
    preserve=100*(1-false_w/len(winners)) if len(winners) else np.nan

    cats_caught=int((cats["exit_reason"]=="EMERGENCY_EXIT").sum())
    cats_caught_pct=100*cats_caught/len(cats) if len(cats) else np.nan

    # tail metrics after applying complete system
    p10=mature["return_pct"].quantile(0.10)
    p05=mature["return_pct"].quantile(0.05)
    worst=mature["return_pct"].min()

    fam,loss,dd,cap,extra=cand
    rows.append({
        "family":fam,
        "loss_threshold_pct":loss*100 if loss is not None else np.nan,
        "peak_drawdown_pct":dd*100 if dd is not None else np.nan,
        "current_return_cap_pct":cap*100 if cap is not None else np.nan,
        "fast_day":extra[0] if extra is not None else np.nan,
        "fast_max_gain_pct":extra[1]*100 if extra is not None else np.nan,
        "mature_n":len(mature),
        "emergency_exit_pct":100*emergency_n/len(mature),
        "avg_return_pct":mature["return_pct"].mean(),
        "median_return_pct":mature["return_pct"].median(),
        "positive_pct":100*(mature["return_pct"]>0).mean(),
        "p10_return_pct":p10,
        "p05_return_pct":p05,
        "worst_return_pct":worst,
        "median_max_dd_pct":mature["max_dd_before_exit_pct"].median(),
        "winner50_n":len(winners),
        "winner50_preserved_pct":preserve,
        "catastrophic_n":len(cats),
        "catastrophic_caught_pct":cats_caught_pct,
    })

    trade_sets[cand]=ot

    if ci%20==0 or ci==total:
        print(f"[{ci}/{total} emergency rules] {cand[0]}")

grid=pd.DataFrame(rows)

# We want strong winner protection.
eligible=grid[
    (grid["winner50_preserved_pct"]>=90) &
    (grid["catastrophic_caught_pct"]>=40)
].copy()

if eligible.empty:
    eligible=grid[
        (grid["winner50_preserved_pct"]>=85) &
        (grid["catastrophic_caught_pct"]>=30)
    ].copy()

if eligible.empty:
    eligible=grid.copy()

# Tail-protection-heavy score.
eligible["score"]=(
    2.0*eligible["winner50_preserved_pct"].rank(pct=True)
    + 2.0*eligible["catastrophic_caught_pct"].rank(pct=True)
    + eligible["p05_return_pct"].rank(pct=True)
    + eligible["p10_return_pct"].rank(pct=True)
    + eligible["worst_return_pct"].rank(pct=True)
    + eligible["median_return_pct"].rank(pct=True)
    + eligible["avg_return_pct"].rank(pct=True)
)

eligible=eligible.sort_values(
    ["score","winner50_preserved_pct","catastrophic_caught_pct","p05_return_pct"],
    ascending=False
)

# Save all grid with score where eligible.
merge_cols=["family","loss_threshold_pct","peak_drawdown_pct","current_return_cap_pct","fast_day","fast_max_gain_pct"]
grid=grid.merge(eligible[merge_cols+["score"]],on=merge_cols,how="left")
grid=grid.sort_values(["score","winner50_preserved_pct"],ascending=False)
grid.to_csv(OUT_GRID,index=False,encoding="utf-8-sig")

best=eligible.iloc[0]

# recover candidate
def same(a,b):
    if a is None and (b is None or pd.isna(b)): return True
    if a is None or b is None or pd.isna(b): return False
    return abs(float(a)-float(b))<1e-12

best_cand=None
for cand in candidates:
    fam,loss,dd,cap,extra=cand
    if fam!=best["family"]: continue
    if not same(loss*100 if loss is not None else None,best["loss_threshold_pct"]): continue
    if not same(dd*100 if dd is not None else None,best["peak_drawdown_pct"]): continue
    if not same(cap*100 if cap is not None else None,best["current_return_cap_pct"]): continue
    if extra is not None:
        if not same(extra[0],best["fast_day"]): continue
        if not same(extra[1]*100,best["fast_max_gain_pct"]): continue
    else:
        if not (pd.isna(best["fast_day"]) and pd.isna(best["fast_max_gain_pct"])): continue
    best_cand=cand
    break

best_t=trade_sets[best_cand].copy()
best_t.to_csv(OUT_BEST,index=False,encoding="utf-8-sig")

best_2026=best_t[best_t["year"]==2026].copy()
best_2026.to_csv(OUT_2026,index=False,encoding="utf-8-sig")

show=[
    "family","loss_threshold_pct","peak_drawdown_pct","current_return_cap_pct",
    "fast_day","fast_max_gain_pct","mature_n","emergency_exit_pct",
    "avg_return_pct","median_return_pct","positive_pct",
    "p10_return_pct","p05_return_pct","worst_return_pct","median_max_dd_pct",
    "winner50_preserved_pct","catastrophic_caught_pct","score"
]

print("\n=== TOP 25 EMERGENCY EXIT RULES ===")
print(eligible.head(25)[show].to_string(index=False))

print("\n=== BEST EMERGENCY EXIT RULE ===")
print(best[show].to_string())

print("\n=== BEST EMERGENCY RULE 2026 ===")
print("2026 trades:",len(best_2026))
print("Emergency exits:",int((best_2026["exit_reason"]=="EMERGENCY_EXIT").sum()))
print("Failure exits:",int((best_2026["exit_reason"]=="FAILURE_EXIT").sum()))
print("Peak exits:",int((best_2026["exit_reason"]=="PEAK_EXIT").sum()))
print("Still open/time-end:",int((best_2026["exit_reason"]=="OPEN_OR_TIME").sum()))

if len(best_2026):
    print("Average return:",f"{best_2026['return_pct'].mean():.2f}%")
    print("Median return:",f"{best_2026['return_pct'].median():.2f}%")
    print("Positive:",f"{100*(best_2026['return_pct']>0).mean():.2f}%")
    print(best_2026[
        ["signal_date","symbol","exit_date","sessions_held","exit_reason",
         "return_pct","max_gain_1y","min_dd_1y",
         "catastrophic_loser","winner50_1y","false_emergency_winner50"]
    ].to_string(index=False))

report=[]
report.append("EGX EMERGENCY EXIT RESEARCH")
report.append("="*130)
report.append("ENTRY FIXED: R2+Slope standalone")
report.append("FAILURE FIXED: R2 -0.05 from entry + Slope -55% from entry + 10D low break, first 60 sessions")
report.append("PEAK FIXED: R2 -0.04 from peak + Slope -57.5% from peak + 5D low break")
report.append("")
report.append("TOP 25")
report.append(eligible.head(25)[show].to_string(index=False))
report.append("")
report.append("BEST RULE")
report.append(best[show].to_string())
report.append("")
report.append("BEST RULE 2026")
report.append(best_2026[
    ["signal_date","symbol","exit_date","sessions_held","exit_reason",
     "return_pct","max_gain_1y","min_dd_1y",
     "catastrophic_loser","winner50_1y","false_emergency_winner50"]
].to_string(index=False))

OUT_REPORT.write_text("\n".join(report),encoding="utf-8")

print("\nCreated:")
for p in [OUT_GRID,OUT_BEST,OUT_2026,OUT_REPORT]:
    print(" ",p)
