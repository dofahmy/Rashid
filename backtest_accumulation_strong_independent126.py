#!/usr/bin/env python3
"""
Rajih — Accumulation STRONG independent-signal backtest

Input:
  /data/accum_strong_clean_signals.csv

Rule:
  Keep only the FIRST STRONG signal per symbol, then block that symbol
  for the next 126 trading sessions (~6 months). Any STRONG signal inside
  that cooldown is ignored.

This removes repeated counting of the same stock during one extended
STRONG episode / same broad move.

Outputs:
  /data/accum_strong_independent126_signals.csv
  /data/accum_strong_independent126_yearly.csv
  /data/accum_strong_independent126_summary.csv
  /data/accum_strong_independent126_report.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/data")
INFILE = DATA / "accum_strong_clean_signals.csv"

OUT_SIGNALS = DATA / "accum_strong_independent126_signals.csv"
OUT_YEARLY = DATA / "accum_strong_independent126_yearly.csv"
OUT_SUMMARY = DATA / "accum_strong_independent126_summary.csv"
OUT_REPORT = DATA / "accum_strong_independent126_report.txt"

df = pd.read_csv(INFILE, dtype={"symbol": str})
df["signal_date"] = pd.to_datetime(df["signal_date"])
df = df.sort_values(["symbol", "signal_date"]).reset_index(drop=True)

# We need trading-session cooldown, not calendar days.
# Use signal-date ordering within each symbol and the mature horizon itself:
# after keeping one signal, ignore later signals whose date falls before
# the kept signal's 126-session outcome window. Since the source rows do not
# store session index, use the stock's own signal sequence plus a conservative
# calendar approximation only if necessary.
#
# Better: use the fact that each row already has m6 metrics only when 126 sessions
# are available, but not the exact session index. To avoid approximation, we use
# 180 calendar days as an operational proxy (~126 trading sessions).
#
# If you want exact trading-session cooldown from DB, use the exact script below
# generated separately. This CSV-based script is fast and diagnostic.

COOLDOWN_CAL_DAYS = 180

kept = []
for symbol, g in df.groupby("symbol", sort=False):
    g = g.sort_values("signal_date")
    last_kept = None
    for _, r in g.iterrows():
        d = r["signal_date"]
        if last_kept is None or (d - last_kept).days >= COOLDOWN_CAL_DAYS:
            kept.append(r)
            last_kept = d

out = pd.DataFrame(kept).sort_values(["signal_date","symbol"]).reset_index(drop=True)
out.to_csv(OUT_SIGNALS, index=False, encoding="utf-8-sig")

def summarize_horizon(x, prefix, mature_col):
    y = x[x[mature_col] == 1].copy()
    if len(y) == 0:
        return {}
    return {
        "n": len(y),
        "positive_close_n": int((y[f"{prefix}_close_ret_pct"] > 0).sum()),
        "positive_close_pct": 100*(y[f"{prefix}_close_ret_pct"] > 0).mean(),
        "avg_close_ret_pct": y[f"{prefix}_close_ret_pct"].mean(),
        "median_close_ret_pct": y[f"{prefix}_close_ret_pct"].median(),
        "avg_max_gain_pct": y[f"{prefix}_max_gain_pct"].mean(),
        "median_max_gain_pct": y[f"{prefix}_max_gain_pct"].median(),
        "median_max_drawdown_pct": y[f"{prefix}_max_drawdown_pct"].median(),
        "hit20_n": int(y[f"{prefix}_hit20"].sum()),
        "hit20_pct": 100*y[f"{prefix}_hit20"].mean(),
        "hit50_n": int(y[f"{prefix}_hit50"].sum()),
        "hit50_pct": 100*y[f"{prefix}_hit50"].mean(),
        "hit100_n": int(y[f"{prefix}_hit100"].sum()),
        "hit100_pct": 100*y[f"{prefix}_hit100"].mean(),
        "hit200_n": int(y[f"{prefix}_hit200"].sum()),
        "hit200_pct": 100*y[f"{prefix}_hit200"].mean(),
    }

s3 = summarize_horizon(out, "m3", "mature_3m")
s6 = summarize_horizon(out, "m6", "mature_6m")

summary_rows = []
for label, s in [("3M", s3), ("6M", s6)]:
    row = {"horizon": label}
    row.update(s)
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

year_rows = []
for year in [2024, 2025, 2026]:
    y = out[out["year"] == year]
    row = {"year": year, "signals": len(y)}
    a = summarize_horizon(y, "m3", "mature_3m")
    b = summarize_horizon(y, "m6", "mature_6m")
    for k, v in a.items():
        row[f"m3_{k}"] = v
    for k, v in b.items():
        row[f"m6_{k}"] = v
    year_rows.append(row)

yearly = pd.DataFrame(year_rows)
yearly.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")

print("\n=== INDEPENDENT STRONG SIGNALS ===")
print("Original clean STRONG signals:", len(df))
print("Independent signals kept:", len(out))
print("Distinct symbols:", out["symbol"].nunique())
print("Distinct dates:", out["signal_date"].dt.date.nunique())

print("\n=== 3 MONTHS / 63 SESSIONS ===")
for k, v in s3.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== 6 MONTHS / 126 SESSIONS ===")
for k, v in s6.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== TOP 20 BY 6M MAX GAIN ===")
top = out[out["mature_6m"] == 1].sort_values("m6_max_gain_pct", ascending=False).head(20)
print(top[[
    "year","signal_date","symbol","raw_signal_close",
    "m6_close_ret_pct","m6_max_gain_pct","m6_max_drawdown_pct"
]].to_string(index=False))

print("\n=== WORST 20 BY 6M CLOSE RETURN ===")
worst = out[out["mature_6m"] == 1].sort_values("m6_close_ret_pct").head(20)
print(worst[[
    "year","signal_date","symbol","raw_signal_close",
    "m6_close_ret_pct","m6_max_gain_pct","m6_max_drawdown_pct"
]].to_string(index=False))

report = []
report.append("RAJIH — ACCUMULATION STRONG INDEPENDENT-SIGNAL TEST")
report.append("="*100)
report.append(f"Original clean STRONG signals: {len(df)}")
report.append(f"Independent signals kept: {len(out)}")
report.append(f"Distinct symbols: {out['symbol'].nunique()}")
report.append("")
report.append("3 MONTHS / 63 SESSIONS")
for k,v in s3.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("6 MONTHS / 126 SESSIONS")
for k,v in s6.items():
    report.append(f"{k}: {v}")
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("TOP 20 BY 6M MAX GAIN")
report.append(top[[
    "year","signal_date","symbol","raw_signal_close",
    "m6_close_ret_pct","m6_max_gain_pct","m6_max_drawdown_pct"
]].to_string(index=False))
report.append("")
report.append("WORST 20 BY 6M CLOSE RETURN")
report.append(worst[[
    "year","signal_date","symbol","raw_signal_close",
    "m6_close_ret_pct","m6_max_gain_pct","m6_max_drawdown_pct"
]].to_string(index=False))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_SIGNALS, OUT_YEARLY, OUT_SUMMARY, OUT_REPORT]:
    print(" ", p)
