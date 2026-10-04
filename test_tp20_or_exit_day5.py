#!/usr/bin/env python3
"""
Rajih — 5-session exit test

Rule:
- Enter at signal close.
- If +20% target is touched during Days 1..5 -> exit at +20%.
- If +20% is NOT touched by Day 5 -> exit at Day-5 adjusted close.
- No stop-loss in this test.

Uses:
/data/us_stop_before_tp20_cases.csv
and market_candles_1d from the database.

Outputs:
/data/us_tp20_or_day5_exit_cases.csv
/data/us_tp20_or_day5_exit_yearly.csv
/data/us_tp20_or_day5_exit_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select
from core import database

DATA_DIR = Path("/data")
INFILE = DATA_DIR / "us_stop_before_tp20_cases.csv"
OUT_CASES = DATA_DIR / "us_tp20_or_day5_exit_cases.csv"
OUT_YEAR = DATA_DIR / "us_tp20_or_day5_exit_yearly.csv"
OUT_REPORT = DATA_DIR / "us_tp20_or_day5_exit_report.txt"

TP = 20.0

signals = pd.read_csv(INFILE)
signals["signal_date"] = pd.to_datetime(signals["signal_date"])

DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

rows = []

for n, r in signals.iterrows():
    sym = r["symbol"]
    sig_date = pd.Timestamp(r["signal_date"]).date()

    with DB() as s:
        data = s.execute(
            select(
                daily.c.session_date,
                daily.c.o, daily.c.h, daily.c.l, daily.c.c,
                daily.c.v, daily.c.adj_c
            )
            .where(daily.c.symbol == sym)
            .order_by(daily.c.session_date)
        ).all()

    df = pd.DataFrame(
        data,
        columns=["session_date","o","h","l","c","v","adj_c"]
    )
    if df.empty:
        continue

    for c in ["o","h","l","c","v","adj_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["c","adj_c"]).reset_index(drop=True)

    hits = df.index[pd.to_datetime(df["session_date"]).dt.date == sig_date].tolist()
    if not hits:
        continue

    i = hits[0]
    if i + 5 >= len(df):
        continue

    base = float(df.at[i, "adj_c"])
    day5_close = float(df.at[i+5, "adj_c"])
    day5_ret = 100.0 * (day5_close / base - 1.0)

    tp20_5d = int(
        pd.notna(r.get("first_tp20_day")) and
        float(r["first_tp20_day"]) <= 5
    )

    trade_ret = TP if tp20_5d else day5_ret

    rows.append({
        "year": int(r["year"]),
        "signal_date": sig_date.isoformat(),
        "symbol": sym,
        "signal_close": base,
        "tp20_within_5d": tp20_5d,
        "first_tp20_day": r.get("first_tp20_day"),
        "day5_close": day5_close,
        "day5_close_return_pct": day5_ret,
        "trade_exit": "TP20" if tp20_5d else "DAY5_CLOSE",
        "trade_return_pct": trade_ret,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

def stats(x):
    return {
        "trades": len(x),
        "tp20_n": int(x["tp20_within_5d"].sum()),
        "tp20_pct": 100*x["tp20_within_5d"].mean() if len(x) else np.nan,
        "positive_trades_n": int((x["trade_return_pct"] > 0).sum()),
        "positive_trades_pct": 100*(x["trade_return_pct"] > 0).mean() if len(x) else np.nan,
        "avg_return_pct": x["trade_return_pct"].mean() if len(x) else np.nan,
        "median_return_pct": x["trade_return_pct"].median() if len(x) else np.nan,
        "worst_return_pct": x["trade_return_pct"].min() if len(x) else np.nan,
        "best_return_pct": x["trade_return_pct"].max() if len(x) else np.nan,
        "avg_non_tp_day5_return_pct": x.loc[x["tp20_within_5d"] == 0, "day5_close_return_pct"].mean()
            if (x["tp20_within_5d"] == 0).any() else np.nan,
        "median_non_tp_day5_return_pct": x.loc[x["tp20_within_5d"] == 0, "day5_close_return_pct"].median()
            if (x["tp20_within_5d"] == 0).any() else np.nan,
    }

overall = stats(out)

year_rows = []
for y in [2024, 2025, 2026]:
    s = stats(out[out["year"] == y])
    s["year"] = y
    year_rows.append(s)

yearly = pd.DataFrame(year_rows)
cols = ["year"] + [c for c in yearly.columns if c != "year"]
yearly = yearly[cols]
yearly.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

print("\n=== TP20 OR EXIT DAY 5 — OVERALL ===")
for k, v in overall.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

print("\n=== YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== WORST 20 DAY-5 EXITS ===")
print(
    out[out["tp20_within_5d"] == 0]
    .sort_values("trade_return_pct")
    .head(20)[
        ["year","signal_date","symbol","day5_close_return_pct","trade_return_pct"]
    ]
    .to_string(index=False)
)

report = []
report.append("RAJIH — TP20 OR EXIT AT DAY 5")
report.append("="*90)
for k, v in overall.items():
    if isinstance(v, float):
        report.append(f"{k}: {v:.2f}")
    else:
        report.append(f"{k}: {v}")
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("WORST 20 DAY-5 EXITS")
report.append(
    out[out["tp20_within_5d"] == 0]
    .sort_values("trade_return_pct")
    .head(20)[
        ["year","signal_date","symbol","day5_close_return_pct","trade_return_pct"]
    ]
    .to_string(index=False)
)
OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
print(OUT_CASES)
print(OUT_YEAR)
print(OUT_REPORT)
