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
EGX Extended Exit Fine-Tuning
=============================

Focus:
Expand the previous best family because the best Slope drop was at the top
edge of the old search range.

Family:
R2 deterioration + Slope rollover + close below prior 5-day low.

Extended grid:
- R2 drop:    0.02 to 0.15
- Slope drop: 55% to 75%
- Price confirmation fixed at prior 5-day low break

Entry source:
    /data/egx_r2slope_standalone_all.csv

Outputs:
    /data/egx_exit_extended_grid.csv
    /data/egx_exit_extended_best_trades.csv
    /data/egx_exit_extended_2026.csv
    /data/egx_exit_extended_report.txt
"""

DATA = Path("/data")
SIGNALS = DATA / "egx_r2slope_standalone_all.csv"

OUT_GRID = DATA / "egx_exit_extended_grid.csv"
OUT_BEST = DATA / "egx_exit_extended_best_trades.csv"
OUT_2026 = DATA / "egx_exit_extended_2026.csv"
OUT_REPORT = DATA / "egx_exit_extended_report.txt"

LOOKBACK = 126
MAX_HOLD = 504
MIN_HOLD = 10
MAX_ARM_AGE = 30
PRICE_BREAK = 5

R2_DROPS = [round(x, 2) for x in np.arange(0.02, 0.151, 0.01)]
SLOPE_DROPS = [round(x, 3) for x in np.arange(0.55, 0.751, 0.025)]

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
    xm = x.mean()
    xd = x-xm
    xss = np.sum(xd*xd)
    ac = df["ac"].to_numpy(float)

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

        r2[i] = 1-ssr/sst if sst > 0 else 0.0
        slope[i] = 100*(b*(LOOKBACK-1))/ym if ym else np.nan

    out = df.copy()
    out["trend_slope"] = slope
    out["trend_r2"] = r2
    return out

signals = pd.read_csv(SIGNALS, dtype={"symbol": str})
signals["signal_date"] = pd.to_datetime(signals["signal_date"])
symbols = sorted(signals["symbol"].unique())

histories = {}

print("\nEGX EXTENDED EXIT FINE-TUNING")
print("Signals:", len(signals), "Symbols:", len(symbols))
print("R2 drops:", R2_DROPS)
print("Slope drops:", [round(x*100,1) for x in SLOPE_DROPS])
print("Price confirmation: CLOSE BELOW PRIOR 5D LOW")

for n, symbol in enumerate(symbols,1):
    df = download(symbol)
    if df is not None and len(df) > LOOKBACK:
        histories[symbol] = add_metrics(df)

    if n % 25 == 0 or n == len(symbols):
        print(f"[{n}/{len(symbols)}] usable={len(histories)}")

    time.sleep(0.02)

def simulate(df, i, entry, r2_drop, slope_drop):
    end = min(len(df)-1, i+MAX_HOLD)
    if i+1 > end:
        return None

    peak_price = -np.inf
    peak_slope = -np.inf
    peak_r2 = -np.inf

    armed = False
    armed_idx = None
    exit_idx = None

    for j in range(i+1, end+1):
        held = j-i

        s = df.at[j,"trend_slope"]
        r = df.at[j,"trend_r2"]
        h = df.at[j,"ah"]
        c = df.at[j,"ac"]

        if np.isfinite(h):
            peak_price = max(peak_price,h)
        if np.isfinite(s):
            peak_slope = max(peak_slope,s)
        if np.isfinite(r):
            peak_r2 = max(peak_r2,r)

        if held < MIN_HOLD:
            continue

        price_near_high = np.isfinite(peak_price) and h >= 0.98*peak_price

        slope_roll = (
            np.isfinite(s) and np.isfinite(peak_slope) and peak_slope > 0
            and s <= peak_slope*(1-slope_drop)
        )

        r2_roll = (
            np.isfinite(r) and np.isfinite(peak_r2)
            and r <= peak_r2-r2_drop
        )

        if (not armed) and price_near_high and slope_roll and r2_roll:
            armed = True
            armed_idx = j

        if armed and (j-armed_idx > MAX_ARM_AGE):
            armed = False
            armed_idx = None

        if armed and j >= PRICE_BREAK:
            prior_low = df.loc[j-PRICE_BREAK:j-1,"al"].min()
            if np.isfinite(prior_low) and c < prior_low:
                exit_idx = j
                break

    if exit_idx is None:
        exit_idx = end
        reason = "OPEN_OR_TIME"
    else:
        reason = "R2_SLOPE_DIVERGENCE+CLOSE_BELOW_5D_LOW"

    seg = df.loc[i+1:exit_idx]

    ret = 100*(float(df.at[exit_idx,"ac"])/entry - 1)
    peak_gain = 100*(float(seg["ah"].max())/entry - 1)
    max_dd = 100*(float(seg["al"].min())/entry - 1)

    giveback = peak_gain-ret
    capture = ret/peak_gain if peak_gain > 0 else np.nan

    return {
        "exit_idx": exit_idx,
        "exit_reason": reason,
        "sessions_held": exit_idx-i,
        "return_pct": ret,
        "peak_gain_before_exit_pct": peak_gain,
        "giveback_from_peak_pct": giveback,
        "capture_ratio": capture,
        "max_dd_before_exit_pct": max_dd,
        "exit_slope": float(df.at[exit_idx,"trend_slope"]) if np.isfinite(df.at[exit_idx,"trend_slope"]) else np.nan,
        "exit_r2": float(df.at[exit_idx,"trend_r2"]) if np.isfinite(df.at[exit_idx,"trend_r2"]) else np.nan,
    }

grid_rows = []
trade_sets = {}

combos = [(r,s) for r in R2_DROPS for s in SLOPE_DROPS]
total = len(combos)

for combo_idx, (r2_drop, slope_drop) in enumerate(combos,1):
    trades = []

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

        res = simulate(df, i, float(sig.signal_price), r2_drop, slope_drop)
        if res is None:
            continue

        trades.append({
            "signal_date": d.date().isoformat(),
            "year": int(d.year),
            "symbol": sig.symbol,
            "entry": float(sig.signal_price),
            "r2_drop": r2_drop,
            "slope_drop_pct": slope_drop*100,
            "exit_date": df.at[res["exit_idx"],"date"].date().isoformat(),
            **{k:v for k,v in res.items() if k != "exit_idx"},
        })

    t = pd.DataFrame(trades)
    if t.empty:
        continue

    triggered = t[t["exit_reason"]!="OPEN_OR_TIME"]

    grid_rows.append({
        "r2_drop": r2_drop,
        "slope_drop_pct": slope_drop*100,
        "price_break_days": PRICE_BREAK,
        "n": len(t),
        "triggered_n": len(triggered),
        "triggered_pct": 100*len(triggered)/len(t),
        "avg_return_pct": t["return_pct"].mean(),
        "median_return_pct": t["return_pct"].median(),
        "positive_pct": 100*(t["return_pct"]>0).mean(),
        "median_capture_ratio": t["capture_ratio"].replace([np.inf,-np.inf],np.nan).median(),
        "median_giveback_from_peak_pct": t["giveback_from_peak_pct"].median(),
        "median_max_dd_pct": t["max_dd_before_exit_pct"].median(),
        "median_sessions_held": t["sessions_held"].median(),
    })

    trade_sets[(r2_drop,slope_drop)] = t

    if combo_idx % 15 == 0 or combo_idx == total:
        print(
            f"[{combo_idx}/{total} rules] "
            f"R2={r2_drop:.2f} slope_drop={slope_drop*100:.1f}%"
        )

grid = pd.DataFrame(grid_rows)

# Prefer rules that still trigger a meaningful portion of the sample.
eligible = grid[grid["triggered_pct"] >= 50].copy()
if eligible.empty:
    eligible = grid.copy()

eligible["score"] = (
    eligible["avg_return_pct"].rank(pct=True)
    + eligible["median_return_pct"].rank(pct=True)
    + eligible["positive_pct"].rank(pct=True)
    + eligible["median_capture_ratio"].fillna(-999).rank(pct=True)
    + (-eligible["median_giveback_from_peak_pct"]).rank(pct=True)
    + eligible["median_max_dd_pct"].rank(pct=True)
)

eligible = eligible.sort_values(
    ["score","median_return_pct","avg_return_pct"],
    ascending=False
)

grid = grid.merge(
    eligible[["r2_drop","slope_drop_pct","score"]],
    on=["r2_drop","slope_drop_pct"],
    how="left"
).sort_values(["score","median_return_pct"],ascending=False)

grid.to_csv(OUT_GRID,index=False,encoding="utf-8-sig")

best = eligible.iloc[0]
best_key = (float(best["r2_drop"]), float(best["slope_drop_pct"])/100.0)
best_t = trade_sets[best_key].copy()
best_t.to_csv(OUT_BEST,index=False,encoding="utf-8-sig")

best_2026 = best_t[best_t["year"]==2026].copy()
best_2026.to_csv(OUT_2026,index=False,encoding="utf-8-sig")

show = [
    "r2_drop","slope_drop_pct","price_break_days","n","triggered_pct",
    "avg_return_pct","median_return_pct","positive_pct",
    "median_capture_ratio","median_giveback_from_peak_pct",
    "median_max_dd_pct","median_sessions_held","score"
]

print("\n=== TOP 25 EXTENDED EXIT RULES ===")
print(eligible.head(25)[show].to_string(index=False))

print("\n=== BEST EXTENDED EXIT RULE ===")
print(best[show].to_string())

print("\n=== BEST EXTENDED RULE 2026 ===")
print("2026 trades:",len(best_2026))
print("Exited:",int((best_2026["exit_reason"]!="OPEN_OR_TIME").sum()))
print("Still open/time-end:",int((best_2026["exit_reason"]=="OPEN_OR_TIME").sum()))

if len(best_2026):
    print("Average return:",f"{best_2026['return_pct'].mean():.2f}%")
    print("Median return:",f"{best_2026['return_pct'].median():.2f}%")
    print("Positive:",f"{100*(best_2026['return_pct']>0).mean():.2f}%")
    print("Median peak capture:",f"{best_2026['capture_ratio'].median():.3f}")
    print("Median giveback:",f"{best_2026['giveback_from_peak_pct'].median():.2f}%")

    print(best_2026[
        ["signal_date","symbol","exit_date","sessions_held","exit_reason",
         "return_pct","peak_gain_before_exit_pct","giveback_from_peak_pct",
         "capture_ratio","max_dd_before_exit_pct","exit_slope","exit_r2"]
    ].to_string(index=False))

report = []
report.append("EGX EXTENDED EXIT FINE-TUNING")
report.append("="*120)
report.append("Family: R2 deterioration + Slope rollover + close below prior 5D low")
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
     "return_pct","peak_gain_before_exit_pct","giveback_from_peak_pct",
     "capture_ratio","max_dd_before_exit_pct","exit_slope","exit_r2"]
].to_string(index=False))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_GRID, OUT_BEST, OUT_2026, OUT_REPORT]:
    print(" ", p)
