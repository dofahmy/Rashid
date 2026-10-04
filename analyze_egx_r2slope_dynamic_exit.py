#!/usr/bin/env python3
"""
EGX dynamic R2 + Slope exit research
====================================
Goal:
Do NOT force TP +50%.
Enter on the standalone R2+Slope signal, then trail the trend itself and exit
when the SHAPE of R2/Slope rolls over.

Entry:
    R2 >= 0.791694
    Slope >= 67.5062
Source signals:
    /data/egx_r2slope_standalone_all.csv

For every open trade, R2 and Slope are recalculated DAILY from the previous
126 sessions only (no future data).

Candidate dynamic exit families tested:
A) Slope drawdown from its post-entry peak:
       current_slope <= peak_slope * (1 - drop)
B) R2 drawdown from its post-entry peak:
       current_r2 <= peak_r2 - drop
C) Combined rollover:
       BOTH A and B
D) Peak-turn confirmation:
       slope has fallen from peak AND current slope < its 5-day SMA
       optionally with R2 breakdown

Protection against exiting immediately:
    minimum hold = 10 sessions

The script compares exits on:
- average/median realized return
- positive %
- average peak gain while in trade
- profit capture ratio = realized return / available peak gain
- drawdown before exit
- time held

It also prints the best candidate rules and detailed 2026 exits.

Outputs:
 /data/egx_r2slope_dynamic_exit_grid.csv
 /data/egx_r2slope_dynamic_exit_best_trades.csv
 /data/egx_r2slope_dynamic_exit_2026.csv
 /data/egx_r2slope_dynamic_exit_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time

DATA = Path("/data")
SIGNALS = DATA / "egx_r2slope_standalone_all.csv"

OUT_GRID = DATA / "egx_r2slope_dynamic_exit_grid.csv"
OUT_BEST = DATA / "egx_r2slope_dynamic_exit_best_trades.csv"
OUT_2026 = DATA / "egx_r2slope_dynamic_exit_2026.csv"
OUT_REPORT = DATA / "egx_r2slope_dynamic_exit_report.txt"

LOOKBACK = 126
MIN_HOLD = 10
MAX_HOLD = 504  # about 2 trading years, enough to compare old signals

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

def calc_roll_metrics(df):
    n = len(df)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    x = np.arange(LOOKBACK, dtype=float)
    xm = x.mean()
    xd = x-xm
    xss = np.sum(xd*xd)

    ac = df["ac"].to_numpy(float)
    # metric for day i uses PREVIOUS 126 sessions, same convention as entry study
    for i in range(LOOKBACK, n):
        y = ac[i-LOOKBACK:i]
        if np.any(~np.isfinite(y)):
            continue
        ym = y.mean()
        yd = y-ym
        b = np.sum(xd*yd)/xss
        pred = ym+b*xd
        ssr = np.sum((y-pred)**2)
        sst = np.sum(yd*yd)
        r2[i] = 1-ssr/sst if sst>0 else 0
        slope[i] = 100*(b*(LOOKBACK-1))/ym if ym else np.nan

    df = df.copy()
    df["trend_slope"] = slope
    df["trend_r2"] = r2
    df["slope_sma5"] = pd.Series(slope).rolling(5).mean().to_numpy()
    return df

signals = pd.read_csv(SIGNALS, dtype={"symbol":str})
signals["signal_date"] = pd.to_datetime(signals["signal_date"])
symbols = sorted(signals["symbol"].unique())

histories = {}
print("\nEGX DYNAMIC R2+SLOPE EXIT RESEARCH")
print("Signals:", len(signals), "Symbols:", len(symbols))

for n,s in enumerate(symbols,1):
    df = download(s)
    if df is not None and len(df) > LOOKBACK:
        histories[s] = calc_roll_metrics(df)
    if n % 25 == 0 or n == len(symbols):
        print(f"[{n}/{len(symbols)}] usable={len(histories)}")
    time.sleep(0.02)

# Candidate parameter grid.
slope_drops = [0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60]
r2_drops = [0.03,0.05,0.08,0.10,0.15,0.20]

rules = []
for sd in slope_drops:
    rules.append(("SLOPE_PEAK_DROP",sd,None))
    rules.append(("SLOPE_PEAK_DROP_TURN",sd,None))
for rd in r2_drops:
    rules.append(("R2_PEAK_DROP",None,rd))
for sd in slope_drops:
    for rd in r2_drops:
        rules.append(("BOTH_PEAK_DROP",sd,rd))
        rules.append(("BOTH_PEAK_DROP_TURN",sd,rd))

def first_exit(seg, rule, sd, rd):
    peak_s = -np.inf
    peak_r = -np.inf

    for k,(idx,row) in enumerate(seg.iterrows(), start=1):
        s = row["trend_slope"]
        r = row["trend_r2"]

        if np.isfinite(s):
            peak_s = max(peak_s,s)
        if np.isfinite(r):
            peak_r = max(peak_r,r)

        if k < MIN_HOLD or not np.isfinite(s) or not np.isfinite(r):
            continue

        slope_break = (np.isfinite(peak_s) and peak_s > 0 and s <= peak_s*(1-sd)) if sd is not None else False
        r2_break = (np.isfinite(peak_r) and r <= peak_r-rd) if rd is not None else False
        turn = np.isfinite(row["slope_sma5"]) and s < row["slope_sma5"]

        hit = False
        if rule == "SLOPE_PEAK_DROP":
            hit = slope_break
        elif rule == "SLOPE_PEAK_DROP_TURN":
            hit = slope_break and turn
        elif rule == "R2_PEAK_DROP":
            hit = r2_break
        elif rule == "BOTH_PEAK_DROP":
            hit = slope_break and r2_break
        elif rule == "BOTH_PEAK_DROP_TURN":
            hit = slope_break and r2_break and turn

        if hit:
            return idx, peak_s, peak_r
    return None, peak_s, peak_r

grid_rows = []
trade_cache = {}

for rule,sd,rd in rules:
    trades = []

    for sig in signals.itertuples(index=False):
        symbol = sig.symbol
        df = histories.get(symbol)
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
        end = min(len(df)-1, i+MAX_HOLD)
        seg = df.loc[i+1:end].copy()
        if seg.empty:
            continue

        j, peak_s, peak_r = first_exit(seg,rule,sd,rd)
        if j is None:
            j = end
            reason = "OPEN_OR_TIME"
        else:
            reason = rule

        exit_close = float(df.at[j,"ac"])
        ret = 100*(exit_close/entry-1)
        peak_gain = 100*(float(df.loc[i+1:j,"ah"].max())/entry-1)
        dd = 100*(float(df.loc[i+1:j,"al"].min())/entry-1)
        capture = ret/peak_gain if peak_gain > 0 else np.nan

        trades.append({
            "signal_date": d.date().isoformat(),
            "year": d.year,
            "symbol": symbol,
            "entry": entry,
            "exit_date": df.at[j,"date"].date().isoformat(),
            "sessions_held": int(j-i),
            "exit_reason": reason,
            "return_pct": ret,
            "peak_gain_before_exit_pct": peak_gain,
            "max_dd_before_exit_pct": dd,
            "capture_ratio": capture,
            "exit_slope": float(df.at[j,"trend_slope"]) if np.isfinite(df.at[j,"trend_slope"]) else np.nan,
            "exit_r2": float(df.at[j,"trend_r2"]) if np.isfinite(df.at[j,"trend_r2"]) else np.nan,
        })

    t = pd.DataFrame(trades)
    if t.empty:
        continue

    # Ranking emphasizes realized return, positive rate, and profit capture.
    mature = t[t["exit_reason"] != "OPEN_OR_TIME"]
    grid_rows.append({
        "rule": rule,
        "slope_peak_drop": sd,
        "r2_peak_drop": rd,
        "n": len(t),
        "triggered_n": len(mature),
        "triggered_pct": 100*len(mature)/len(t),
        "avg_return_pct": t["return_pct"].mean(),
        "median_return_pct": t["return_pct"].median(),
        "positive_pct": 100*(t["return_pct"]>0).mean(),
        "avg_peak_gain_pct": t["peak_gain_before_exit_pct"].mean(),
        "median_peak_gain_pct": t["peak_gain_before_exit_pct"].median(),
        "median_capture_ratio": t.loc[t["capture_ratio"].replace([np.inf,-np.inf],np.nan).notna(),"capture_ratio"].median(),
        "median_max_dd_pct": t["max_dd_before_exit_pct"].median(),
        "median_sessions_held": t["sessions_held"].median(),
    })
    trade_cache[(rule,sd,rd)] = t

grid = pd.DataFrame(grid_rows)

# Robust score; avoid rules that almost never trigger.
eligible = grid[grid["triggered_pct"] >= 70].copy()
if eligible.empty:
    eligible = grid.copy()

# rank multiple desirable features, no future leakage inside individual exits
eligible["score"] = (
    eligible["median_return_pct"].rank(pct=True) +
    eligible["avg_return_pct"].rank(pct=True) +
    eligible["positive_pct"].rank(pct=True) +
    eligible["median_capture_ratio"].fillna(-999).rank(pct=True) +
    eligible["median_max_dd_pct"].rank(pct=True)
)

eligible = eligible.sort_values(
    ["score","median_return_pct","avg_return_pct"],
    ascending=False
)

grid = grid.merge(
    eligible[["rule","slope_peak_drop","r2_peak_drop","score"]],
    on=["rule","slope_peak_drop","r2_peak_drop"],
    how="left"
).sort_values(["score","median_return_pct"], ascending=False)

grid.to_csv(OUT_GRID,index=False,encoding="utf-8-sig")

best = eligible.iloc[0]
best_key = (
    best["rule"],
    None if pd.isna(best["slope_peak_drop"]) else float(best["slope_peak_drop"]),
    None if pd.isna(best["r2_peak_drop"]) else float(best["r2_peak_drop"])
)

# Handle NaN-key mismatch robustly by find.
best_t = None
for (rule,sd,rd),t in trade_cache.items():
    if rule != best["rule"]:
        continue
    sd_ok = (sd is None and pd.isna(best["slope_peak_drop"])) or (sd is not None and abs(sd-float(best["slope_peak_drop"]))<1e-12)
    rd_ok = (rd is None and pd.isna(best["r2_peak_drop"])) or (rd is not None and abs(rd-float(best["r2_peak_drop"]))<1e-12)
    if sd_ok and rd_ok:
        best_t = t.copy()
        break

best_t.to_csv(OUT_BEST,index=False,encoding="utf-8-sig")
best_2026 = best_t[best_t["year"]==2026].copy()
best_2026.to_csv(OUT_2026,index=False,encoding="utf-8-sig")

print("\n=== TOP 15 DYNAMIC EXIT RULES ===")
show = [
    "rule","slope_peak_drop","r2_peak_drop","n","triggered_pct",
    "avg_return_pct","median_return_pct","positive_pct",
    "median_capture_ratio","median_max_dd_pct","median_sessions_held","score"
]
print(eligible.head(15)[show].to_string(index=False))

print("\n=== BEST RULE ===")
print(best[show].to_string())

print("\n=== BEST RULE 2026 ===")
print("2026 trades:",len(best_2026))
print("Exited:",int((best_2026["exit_reason"]!="OPEN_OR_TIME").sum()))
print("Still open/time-end:",int((best_2026["exit_reason"]=="OPEN_OR_TIME").sum()))
if len(best_2026):
    print("Average return:",f"{best_2026['return_pct'].mean():.2f}%")
    print("Median return:",f"{best_2026['return_pct'].median():.2f}%")
    print("Positive:",f"{100*(best_2026['return_pct']>0).mean():.2f}%")
    print(best_2026[
        ["signal_date","symbol","exit_date","sessions_held","exit_reason",
         "return_pct","peak_gain_before_exit_pct","max_dd_before_exit_pct",
         "exit_slope","exit_r2"]
    ].to_string(index=False))

report = []
report.append("EGX DYNAMIC R2 + SLOPE EXIT RESEARCH")
report.append("="*120)
report.append("Entry: standalone R2>=0.791694 AND Slope>=67.5062")
report.append("Exit: dynamic rollover in R2/Slope, not fixed TP50")
report.append("")
report.append("TOP 15 RULES")
report.append(eligible.head(15)[show].to_string(index=False))
report.append("")
report.append("BEST RULE")
report.append(best[show].to_string())
report.append("")
report.append("BEST RULE 2026")
report.append(best_2026[
    ["signal_date","symbol","exit_date","sessions_held","exit_reason",
     "return_pct","peak_gain_before_exit_pct","max_dd_before_exit_pct",
     "exit_slope","exit_r2"]
].to_string(index=False))

OUT_REPORT.write_text("\n".join(report),encoding="utf-8")

print("\nCreated:")
for p in [OUT_GRID,OUT_BEST,OUT_2026,OUT_REPORT]:
    print(" ",p)
