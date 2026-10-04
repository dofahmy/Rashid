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
EGX FAILURE EXIT RESEARCH
=========================

ENTRY is FIXED:
    Standalone R2 + Slope signal from:
    /data/egx_r2slope_standalone_all.csv

PEAK EXIT is FIXED:
    - post-entry Slope falls 57.5% from its peak
    - R2 falls at least 0.04 from its post-entry peak
    - then close breaks below prior 5-day low

This script DOES NOT change entry or peak exit.
It searches ONLY for an EARLY FAILURE EXIT to protect trades that fail after entry.

Important design:
- Failure exit is allowed only during the first 60 trading sessions.
- Minimum hold before failure exit: 5 sessions.
- Ranking uses MATURE 1-year trades (252 sessions available) so recent 2026 trades
  do not bias rule selection.
- We explicitly measure how often a failure rule wrongly exits a trade that
  would later reach +50%.

Candidate failure families:
1) PRICE_STOP
2) PRICE + R2 deterioration
3) PRICE + SLOPE deterioration
4) PRICE + R2 + SLOPE deterioration
5) NO_TRACTION after 20/30/40 sessions
6) STRUCTURE_FAIL = price below prior 10d low + R2/Slope deterioration

Outputs:
  /data/egx_failure_exit_grid.csv
  /data/egx_failure_exit_best_trades.csv
  /data/egx_failure_exit_2026.csv
  /data/egx_failure_exit_report.txt
"""

DATA = Path("/data")
SIGNALS = DATA / "egx_r2slope_standalone_all.csv"

OUT_GRID = DATA / "egx_failure_exit_grid.csv"
OUT_BEST = DATA / "egx_failure_exit_best_trades.csv"
OUT_2026 = DATA / "egx_failure_exit_2026.csv"
OUT_REPORT = DATA / "egx_failure_exit_report.txt"

LOOKBACK = 126
MATURITY = 252
MAX_HOLD = 504

# Fixed peak exit
PEAK_R2_DROP = 0.04
PEAK_SLOPE_DROP = 0.575
PEAK_BREAK_DAYS = 5
PEAK_MAX_ARM_AGE = 30

# Failure research
FAIL_MIN_HOLD = 5
FAIL_MAX_AGE = 60

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
        "o": q.get("open"),
        "h": q.get("high"),
        "l": q.get("low"),
        "c": q.get("close"),
        "v": q.get("volume"),
    })
    df["adj_c"] = adj if adj and len(adj) == len(df) else df["c"]

    for c in ["o","h","l","c","v","adj_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["o","h","l","c","adj_c"])
    df = df[(df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) & (df["c"] > 0)]
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
    xm = x.mean()
    xd = x - xm
    xss = np.sum(xd * xd)
    ac = df["ac"].to_numpy(float)

    for i in range(LOOKBACK, n):
        y = ac[i-LOOKBACK:i]  # prior data only
        if np.any(~np.isfinite(y)):
            continue
        ym = y.mean()
        yd = y - ym
        b = np.sum(xd * yd) / xss
        pred = ym + b * xd
        ssr = np.sum((y - pred) ** 2)
        sst = np.sum(yd * yd)

        r2[i] = 1 - ssr / sst if sst > 0 else 0.0
        slope[i] = 100 * (b * (LOOKBACK - 1)) / ym if ym else np.nan

    out = df.copy()
    out["trend_slope"] = slope
    out["trend_r2"] = r2
    return out

signals = pd.read_csv(SIGNALS, dtype={"symbol": str})
signals["signal_date"] = pd.to_datetime(signals["signal_date"])
symbols = sorted(signals["symbol"].unique())

histories = {}

print("\nEGX FAILURE EXIT RESEARCH")
print("Signals:", len(signals), "Symbols:", len(symbols))
print("Entry: FIXED standalone R2+Slope")
print("Peak exit: FIXED slope -57.5%, R2 -0.04, close below prior 5D low")
print("Searching only EARLY FAILURE EXIT.")

for n, symbol in enumerate(symbols, 1):
    df = download(symbol)
    if df is not None and len(df) > LOOKBACK:
        histories[symbol] = add_metrics(df)

    if n % 25 == 0 or n == len(symbols):
        print(f"[{n}/{len(symbols)}] usable={len(histories)}")

    time.sleep(0.02)

# ---------------------------------------------------------
# Prepare trade paths once.
# ---------------------------------------------------------
trades_base = []

for sig in signals.itertuples(index=False):
    df = histories.get(sig.symbol)
    if df is None:
        continue

    d = pd.Timestamp(sig.signal_date).normalize()
    ids = df.index[df["date"] == d].tolist()
    if not ids:
        continue

    i = ids[-1]
    if i + 1 >= len(df):
        continue

    entry = float(sig.signal_price)
    entry_slope = float(df.at[i, "trend_slope"]) if np.isfinite(df.at[i, "trend_slope"]) else float(sig.pre_trend_slope_pct)
    entry_r2 = float(df.at[i, "trend_r2"]) if np.isfinite(df.at[i, "trend_r2"]) else float(sig.pre_trend_r2)

    end = min(len(df) - 1, i + MAX_HOLD)
    mature_1y = (i + MATURITY < len(df))

    future_1y_end = min(len(df)-1, i+MATURITY)
    future_1y = df.loc[i+1:future_1y_end]

    max_gain_1y = 100 * (float(future_1y["ah"].max()) / entry - 1) if len(future_1y) else np.nan

    tp50_hits = future_1y.index[future_1y["ah"] >= entry * 1.50].tolist()
    tp50_idx = tp50_hits[0] if tp50_hits else None

    trades_base.append({
        "symbol": sig.symbol,
        "signal_date": d,
        "year": int(d.year),
        "i": i,
        "end": end,
        "entry": entry,
        "entry_slope": entry_slope,
        "entry_r2": entry_r2,
        "mature_1y": mature_1y,
        "max_gain_1y": max_gain_1y,
        "winner50_1y": int(tp50_idx is not None),
        "tp50_idx": tp50_idx,
    })

print("Prepared trades:", len(trades_base))
print("Mature 1Y trades:", sum(x["mature_1y"] for x in trades_base))

def peak_exit_index(df, i, end):
    peak_price = -np.inf
    peak_slope = -np.inf
    peak_r2 = -np.inf
    armed = False
    armed_idx = None

    for j in range(i+1, end+1):
        s = df.at[j, "trend_slope"]
        r = df.at[j, "trend_r2"]
        h = df.at[j, "ah"]
        c = df.at[j, "ac"]

        if np.isfinite(h):
            peak_price = max(peak_price, h)
        if np.isfinite(s):
            peak_slope = max(peak_slope, s)
        if np.isfinite(r):
            peak_r2 = max(peak_r2, r)

        if j-i < 10:
            continue

        near_high = np.isfinite(peak_price) and h >= 0.98 * peak_price
        slope_roll = (
            np.isfinite(s) and np.isfinite(peak_slope) and peak_slope > 0
            and s <= peak_slope * (1 - PEAK_SLOPE_DROP)
        )
        r2_roll = (
            np.isfinite(r) and np.isfinite(peak_r2)
            and r <= peak_r2 - PEAK_R2_DROP
        )

        if (not armed) and near_high and slope_roll and r2_roll:
            armed = True
            armed_idx = j

        if armed and j - armed_idx > PEAK_MAX_ARM_AGE:
            armed = False
            armed_idx = None

        if armed and j >= PEAK_BREAK_DAYS:
            prior_low = df.loc[j-PEAK_BREAK_DAYS:j-1, "al"].min()
            if np.isfinite(prior_low) and c < prior_low:
                return j

    return None

# Cache baseline peak exits.
for t in trades_base:
    df = histories[t["symbol"]]
    t["peak_exit_idx"] = peak_exit_index(df, t["i"], t["end"])

# ---------------------------------------------------------
# Failure candidate definitions
# ---------------------------------------------------------
candidates = []

# 1) simple price stop
for stop in [0.08,0.10,0.12,0.15,0.18,0.20,0.25]:
    candidates.append(("PRICE_STOP", stop, None, None, None, None))

# 2) price + R2 deterioration
for stop in [0.08,0.10,0.12,0.15,0.18,0.20]:
    for r2d in [0.03,0.05,0.08,0.10,0.15,0.20]:
        candidates.append(("PRICE_R2", stop, r2d, None, None, None))

# 3) price + Slope deterioration
for stop in [0.08,0.10,0.12,0.15,0.18,0.20]:
    for sd in [0.20,0.30,0.40,0.50,0.60]:
        candidates.append(("PRICE_SLOPE", stop, None, sd, None, None))

# 4) price + both
for stop in [0.08,0.10,0.12,0.15,0.18,0.20]:
    for r2d in [0.03,0.05,0.08,0.10,0.15]:
        for sd in [0.20,0.30,0.40,0.50]:
            candidates.append(("PRICE_BOTH", stop, r2d, sd, None, None))

# 5) no traction after N sessions:
# at N, if max gain so far is less than gain threshold and current return <= current ceiling
for day in [20,30,40]:
    for max_gain in [0.05,0.10,0.15,0.20]:
        for current_ceiling in [0.00,0.05]:
            candidates.append(("NO_TRACTION", None, None, None, day, (max_gain, current_ceiling)))

# 6) structure failure
for r2d in [0.05,0.08,0.10,0.15]:
    for sd in [0.25,0.35,0.45,0.55]:
        candidates.append(("STRUCTURE_FAIL", None, r2d, sd, None, None))

def failure_exit_index(df, t, cand):
    family, stop, r2d, sd, day, extra = cand
    i = t["i"]
    fail_end = min(t["end"], i + FAIL_MAX_AGE)

    running_high = -np.inf

    for j in range(i+1, fail_end+1):
        held = j-i
        if held < FAIL_MIN_HOLD:
            continue

        c = float(df.at[j, "ac"])
        h = float(df.at[j, "ah"])
        running_high = max(running_high, h)

        ret = c / t["entry"] - 1
        s = df.at[j, "trend_slope"]
        r = df.at[j, "trend_r2"]

        price_fail = (ret <= -stop) if stop is not None else False
        r2_fail = (
            np.isfinite(r) and np.isfinite(t["entry_r2"])
            and r <= t["entry_r2"] - r2d
        ) if r2d is not None else False
        slope_fail = (
            np.isfinite(s) and np.isfinite(t["entry_slope"]) and t["entry_slope"] > 0
            and s <= t["entry_slope"] * (1 - sd)
        ) if sd is not None else False

        if family == "PRICE_STOP":
            if price_fail:
                return j

        elif family == "PRICE_R2":
            if price_fail and r2_fail:
                return j

        elif family == "PRICE_SLOPE":
            if price_fail and slope_fail:
                return j

        elif family == "PRICE_BOTH":
            if price_fail and r2_fail and slope_fail:
                return j

        elif family == "NO_TRACTION":
            if held == day:
                max_gain_thr, current_ceiling = extra
                gain_so_far = running_high / t["entry"] - 1
                if gain_so_far < max_gain_thr and ret <= current_ceiling:
                    return j

        elif family == "STRUCTURE_FAIL":
            if j >= 10:
                prior_low = df.loc[j-10:j-1, "al"].min()
                price_structure_fail = np.isfinite(prior_low) and c < prior_low
                if price_structure_fail and r2_fail and slope_fail:
                    return j

    return None

# ---------------------------------------------------------
# Evaluate all candidate failure rules on top of fixed peak exit.
# ---------------------------------------------------------
rows = []
trade_sets = {}
total = len(candidates)

for ci, cand in enumerate(candidates, 1):
    family, stop, r2d, sd, day, extra = cand
    out_trades = []

    for t in trades_base:
        df = histories[t["symbol"]]

        fidx = failure_exit_index(df, t, cand)
        pidx = t["peak_exit_idx"]

        # First exit wins.
        if fidx is not None and (pidx is None or fidx <= pidx):
            exit_idx = fidx
            reason = "FAILURE_EXIT"
        elif pidx is not None:
            exit_idx = pidx
            reason = "PEAK_EXIT"
        else:
            exit_idx = t["end"]
            reason = "OPEN_OR_TIME"

        ret = 100 * (float(df.at[exit_idx, "ac"]) / t["entry"] - 1)
        seg = df.loc[t["i"]+1:exit_idx]
        max_dd = 100 * (float(seg["al"].min()) / t["entry"] - 1) if len(seg) else np.nan

        false_failure_winner50 = (
            reason == "FAILURE_EXIT"
            and t["winner50_1y"] == 1
            and t["tp50_idx"] is not None
            and exit_idx < t["tp50_idx"]
        )

        out_trades.append({
            "signal_date": t["signal_date"].date().isoformat(),
            "year": t["year"],
            "symbol": t["symbol"],
            "mature_1y": int(t["mature_1y"]),
            "winner50_1y": t["winner50_1y"],
            "max_gain_1y": t["max_gain_1y"],
            "exit_date": df.at[exit_idx, "date"].date().isoformat(),
            "sessions_held": exit_idx - t["i"],
            "exit_reason": reason,
            "return_pct": ret,
            "max_dd_before_exit_pct": max_dd,
            "false_failure_winner50": int(false_failure_winner50),
        })

    ot = pd.DataFrame(out_trades)
    mature = ot[ot["mature_1y"] == 1].copy()
    if mature.empty:
        continue

    winners = mature[mature["winner50_1y"] == 1]
    nonwinners = mature[mature["winner50_1y"] == 0]

    failure_n = int((mature["exit_reason"] == "FAILURE_EXIT").sum())
    false_winner_n = int(mature["false_failure_winner50"].sum())

    winner_preserve_pct = (
        100 * (1 - false_winner_n / len(winners))
        if len(winners) else np.nan
    )

    failure_nonwinner_n = int((nonwinners["exit_reason"] == "FAILURE_EXIT").sum())

    rows.append({
        "family": family,
        "price_stop_pct": stop*100 if stop is not None else np.nan,
        "r2_drop_from_entry": r2d,
        "slope_drop_from_entry_pct": sd*100 if sd is not None else np.nan,
        "no_traction_day": day if day is not None else np.nan,
        "no_traction_max_gain_pct": extra[0]*100 if extra is not None else np.nan,
        "no_traction_current_ceiling_pct": extra[1]*100 if extra is not None else np.nan,
        "mature_n": len(mature),
        "failure_exit_n": failure_n,
        "failure_exit_pct": 100*failure_n/len(mature),
        "avg_return_pct": mature["return_pct"].mean(),
        "median_return_pct": mature["return_pct"].median(),
        "positive_pct": 100*(mature["return_pct"] > 0).mean(),
        "median_max_dd_pct": mature["max_dd_before_exit_pct"].median(),
        "winner50_n": len(winners),
        "false_failure_winner50_n": false_winner_n,
        "winner50_preserved_pct": winner_preserve_pct,
        "nonwinner_n": len(nonwinners),
        "failure_nonwinner_n": failure_nonwinner_n,
        "failure_nonwinner_pct": 100*failure_nonwinner_n/len(nonwinners) if len(nonwinners) else np.nan,
    })

    trade_sets[cand] = ot

    if ci % 25 == 0 or ci == total:
        print(f"[{ci}/{total} failure rules] {family}")

grid = pd.DataFrame(rows)

# Safety-first eligibility:
# preserve >=90% of +50 winners and actually catch >=10% of non-winners.
eligible = grid[
    (grid["winner50_preserved_pct"] >= 90) &
    (grid["failure_nonwinner_pct"] >= 10)
].copy()

if eligible.empty:
    eligible = grid[
        (grid["winner50_preserved_pct"] >= 85) &
        (grid["failure_nonwinner_pct"] >= 5)
    ].copy()

if eligible.empty:
    eligible = grid.copy()

# Reward return + drawdown + catching failures, heavily reward preserving winners.
eligible["score"] = (
    eligible["avg_return_pct"].rank(pct=True)
    + eligible["median_return_pct"].rank(pct=True)
    + eligible["positive_pct"].rank(pct=True)
    + eligible["median_max_dd_pct"].rank(pct=True)
    + eligible["failure_nonwinner_pct"].rank(pct=True)
    + 2.0 * eligible["winner50_preserved_pct"].rank(pct=True)
)

eligible = eligible.sort_values(
    ["score", "winner50_preserved_pct", "median_return_pct", "avg_return_pct"],
    ascending=False
)

grid = grid.merge(
    eligible[[
        "family","price_stop_pct","r2_drop_from_entry","slope_drop_from_entry_pct",
        "no_traction_day","no_traction_max_gain_pct","no_traction_current_ceiling_pct","score"
    ]],
    on=[
        "family","price_stop_pct","r2_drop_from_entry","slope_drop_from_entry_pct",
        "no_traction_day","no_traction_max_gain_pct","no_traction_current_ceiling_pct"
    ],
    how="left"
).sort_values(["score","winner50_preserved_pct"], ascending=False)

grid.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")

best = eligible.iloc[0]

# Locate original candidate.
best_cand = None
for cand in candidates:
    fam, stop, r2d, sd, day, extra = cand
    def same(a,b):
        if a is None and (b is None or (isinstance(b,float) and np.isnan(b))):
            return True
        if a is None:
            return False
        return abs(float(a)-float(b)) < 1e-12

    if fam != best["family"]:
        continue
    if not same(stop*100 if stop is not None else None, best["price_stop_pct"]):
        continue
    if not same(r2d, best["r2_drop_from_entry"]):
        continue
    if not same(sd*100 if sd is not None else None, best["slope_drop_from_entry_pct"]):
        continue
    if not same(day, best["no_traction_day"]):
        continue
    if extra is not None:
        if not same(extra[0]*100, best["no_traction_max_gain_pct"]):
            continue
        if not same(extra[1]*100, best["no_traction_current_ceiling_pct"]):
            continue
    else:
        if not (pd.isna(best["no_traction_max_gain_pct"]) and pd.isna(best["no_traction_current_ceiling_pct"])):
            continue
    best_cand = cand
    break

best_t = trade_sets[best_cand].copy()
best_t.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

best_2026 = best_t[best_t["year"] == 2026].copy()
best_2026.to_csv(OUT_2026, index=False, encoding="utf-8-sig")

show = [
    "family","price_stop_pct","r2_drop_from_entry","slope_drop_from_entry_pct",
    "no_traction_day","no_traction_max_gain_pct","no_traction_current_ceiling_pct",
    "mature_n","failure_exit_pct","avg_return_pct","median_return_pct",
    "positive_pct","median_max_dd_pct","winner50_preserved_pct",
    "failure_nonwinner_pct","score"
]

print("\n=== TOP 25 FAILURE EXIT RULES ===")
print(eligible.head(25)[show].to_string(index=False))

print("\n=== BEST FAILURE EXIT RULE ===")
print(best[show].to_string())

print("\n=== BEST FAILURE RULE 2026 ===")
print("2026 trades:", len(best_2026))
print("Failure exits:", int((best_2026["exit_reason"]=="FAILURE_EXIT").sum()))
print("Peak exits:", int((best_2026["exit_reason"]=="PEAK_EXIT").sum()))
print("Still open/time-end:", int((best_2026["exit_reason"]=="OPEN_OR_TIME").sum()))

if len(best_2026):
    print("Average return:", f"{best_2026['return_pct'].mean():.2f}%")
    print("Median return:", f"{best_2026['return_pct'].median():.2f}%")
    print("Positive:", f"{100*(best_2026['return_pct']>0).mean():.2f}%")
    print(best_2026[
        ["signal_date","symbol","exit_date","sessions_held","exit_reason",
         "return_pct","max_gain_1y","max_dd_before_exit_pct",
         "winner50_1y","false_failure_winner50"]
    ].to_string(index=False))

report = []
report.append("EGX FAILURE EXIT RESEARCH")
report.append("="*130)
report.append("ENTRY FIXED: standalone R2+Slope")
report.append("PEAK EXIT FIXED: Slope -57.5%, R2 -0.04, close below prior 5D low")
report.append("Research target: early FAILURE EXIT only")
report.append("")
report.append("TOP 25 FAILURE RULES")
report.append(eligible.head(25)[show].to_string(index=False))
report.append("")
report.append("BEST FAILURE RULE")
report.append(best[show].to_string())
report.append("")
report.append("BEST RULE 2026")
report.append(best_2026[
    ["signal_date","symbol","exit_date","sessions_held","exit_reason",
     "return_pct","max_gain_1y","max_dd_before_exit_pct",
     "winner50_1y","false_failure_winner50"]
].to_string(index=False))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_GRID, OUT_BEST, OUT_2026, OUT_REPORT]:
    print(" ", p)
