#!/usr/bin/env python3
"""
EGX STRONG_UPTREND rule backtest with fixed TP +50% on all available EGX data.

Rule:
  pre_trend_r2 >= 0.791694
  AND
  pre_trend_slope_pct >= 67.5062

Entry:
  STRONG signal close.

Exit:
  Take profit at +50% intraday high touch after entry.
  If +50% is never touched, exit at the last available close for that signal,
  and separately report fixed-horizon exits at:
      63 sessions (3M)
      126 sessions (6M)
      252 sessions (1Y)
      504 sessions (2Y)

Important:
- Uses all historical signals already produced in:
    /data/egx_strong_preshape_all.csv
- Does NOT filter out corporate actions beyond what was already in the source study.
- TP is considered hit if adjusted HIGH reaches +50% from signal price.
- Because this is daily data, we know the target was touched that day, not the exact intraday time.

Outputs:
  /data/egx_rule_r2slope_tp50_all_cases.csv
  /data/egx_rule_r2slope_tp50_summary.csv
  /data/egx_rule_r2slope_tp50_yearly.csv
  /data/egx_rule_r2slope_tp50_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time

DATA = Path("/data")
INFILE = DATA / "egx_strong_preshape_all.csv"

OUT_CASES = DATA / "egx_rule_r2slope_tp50_all_cases.csv"
OUT_SUM = DATA / "egx_rule_r2slope_tp50_summary.csv"
OUT_YEAR = DATA / "egx_rule_r2slope_tp50_yearly.csv"
OUT_REPORT = DATA / "egx_rule_r2slope_tp50_report.txt"

R2_MIN = 0.791694
SLOPE_MIN = 67.5062
TP_PCT = 50.0

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36"
})

def yahoo_download(symbol, start="2019-01-01"):
    y = f"{symbol}.CA"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y}"
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
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
    df = df[(df["o"] > 0) & (df["h"] > 0) & (df["l"] > 0) & (df["c"] > 0)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # adjusted OHLC
    factor = df["adj_c"] / df["c"]
    df["ao"] = df["o"] * factor
    df["ah"] = df["h"] * factor
    df["al"] = df["l"] * factor
    df["ac"] = df["adj_c"]
    return df

src = pd.read_csv(INFILE, dtype={"symbol": str})
src["signal_date"] = pd.to_datetime(src["signal_date"])

sel = src[
    (src["pre_regime"] == "STRONG_UPTREND") &
    (pd.to_numeric(src["pre_trend_r2"], errors="coerce") >= R2_MIN) &
    (pd.to_numeric(src["pre_trend_slope_pct"], errors="coerce") >= SLOPE_MIN)
].copy()

sel = sel.sort_values(["signal_date","symbol"]).reset_index(drop=True)

print("\nEGX RULE BACKTEST")
print("Rule: pre_trend_r2 >= %.6f AND pre_trend_slope_pct >= %.4f" % (R2_MIN, SLOPE_MIN))
print("Matched signals:", len(sel))
print("Distinct symbols:", sel["symbol"].nunique())

histories = {}
rows = []

for n, r in enumerate(sel.itertuples(index=False), 1):
    symbol = str(r.symbol)
    if symbol not in histories:
        histories[symbol] = yahoo_download(symbol)
        time.sleep(0.03)

    hist = histories[symbol]
    if hist is None or hist.empty:
        continue

    d = pd.Timestamp(r.signal_date).normalize()
    # exact signal row
    idxs = hist.index[hist["date"] == d].tolist()
    if not idxs:
        # nearest previous trading day if exact date mismatch
        prev = hist.index[hist["date"] <= d].tolist()
        if not prev:
            continue
        i = prev[-1]
    else:
        i = idxs[-1]

    # Use original recorded signal price for consistency with the source study.
    entry = float(r.signal_price if hasattr(r, "signal_price") else r.raw_signal_close_egp)
    tp_price = entry * 1.50

    future = hist.iloc[i+1:].copy()
    if future.empty:
        continue

    tp_hits = future.index[future["ah"] >= tp_price].tolist()
    tp_hit = len(tp_hits) > 0

    if tp_hit:
        j = tp_hits[0]
        tp_row = hist.loc[j]
        sessions_to_tp = int(j - i)
        tp_date = tp_row["date"]
        realized_tp_ret = TP_PCT
        mfe_before_tp = 100 * (hist.loc[i+1:j, "ah"].max() / entry - 1)
        mae_before_tp = 100 * (hist.loc[i+1:j, "al"].min() / entry - 1)
    else:
        j = hist.index[-1]
        tp_date = pd.NaT
        sessions_to_tp = np.nan
        realized_tp_ret = np.nan
        mfe_before_tp = 100 * (future["ah"].max() / entry - 1)
        mae_before_tp = 100 * (future["al"].min() / entry - 1)

    row = {
        "signal_date": d.date().isoformat(),
        "year": int(d.year),
        "symbol": symbol,
        "signal_price": entry,
        "pre_trend_r2": float(r.pre_trend_r2),
        "pre_trend_slope_pct": float(r.pre_trend_slope_pct),
        "tp50_hit": int(tp_hit),
        "tp50_date": tp_date.date().isoformat() if tp_hit else "",
        "sessions_to_tp50": sessions_to_tp,
        "mae_before_tp_or_now_pct": float(mae_before_tp),
        "mfe_before_tp_or_now_pct": float(mfe_before_tp),
    }

    # fixed horizons
    for label, h in [("3m",63),("6m",126),("1y",252),("2y",504)]:
        if i + h < len(hist):
            close_ret = 100 * (float(hist.at[i+h,"ac"]) / entry - 1)
            max_gain = 100 * (float(hist.loc[i+1:i+h,"ah"].max()) / entry - 1)
            max_dd = 100 * (float(hist.loc[i+1:i+h,"al"].min()) / entry - 1)
            row[f"{label}_mature"] = 1
            row[f"{label}_close_ret_pct"] = close_ret
            row[f"{label}_max_gain_pct"] = max_gain
            row[f"{label}_max_dd_pct"] = max_dd
        else:
            row[f"{label}_mature"] = 0
            row[f"{label}_close_ret_pct"] = np.nan
            row[f"{label}_max_gain_pct"] = np.nan
            row[f"{label}_max_dd_pct"] = np.nan

    # "TP50 or horizon exit" system returns:
    for label in ["3m","6m","1y","2y"]:
        h = {"3m":63,"6m":126,"1y":252,"2y":504}[label]
        if row[f"{label}_mature"] == 1:
            # if TP touched on/before horizon => +50, else exit at horizon close
            if tp_hit and sessions_to_tp <= h:
                row[f"{label}_system_ret_pct"] = TP_PCT
                row[f"{label}_system_exit"] = "TP50"
            else:
                row[f"{label}_system_ret_pct"] = row[f"{label}_close_ret_pct"]
                row[f"{label}_system_exit"] = f"EXIT_{label.upper()}"
        else:
            row[f"{label}_system_ret_pct"] = np.nan
            row[f"{label}_system_exit"] = ""

    rows.append(row)

    print(
        f"[{n}/{len(sel)}] {symbol} {d.date()} | "
        f"TP50={'YES' if tp_hit else 'NO'}"
        + (f" in {sessions_to_tp} sessions" if tp_hit else "")
    )

out = pd.DataFrame(rows)
out.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

def horizon_summary(label):
    g = out[out[f"{label}_mature"] == 1].copy()
    if g.empty:
        return {}
    tp_by_h = ((g["tp50_hit"] == 1) & (g["sessions_to_tp50"] <= {"3m":63,"6m":126,"1y":252,"2y":504}[label]))
    return {
        "horizon": label,
        "n": len(g),
        "tp50_n": int(tp_by_h.sum()),
        "tp50_pct": 100 * tp_by_h.mean(),
        "avg_system_ret_pct": g[f"{label}_system_ret_pct"].mean(),
        "median_system_ret_pct": g[f"{label}_system_ret_pct"].median(),
        "positive_system_pct": 100 * (g[f"{label}_system_ret_pct"] > 0).mean(),
        "avg_buyhold_close_ret_pct": g[f"{label}_close_ret_pct"].mean(),
        "median_buyhold_close_ret_pct": g[f"{label}_close_ret_pct"].median(),
        "median_max_dd_pct": g[f"{label}_max_dd_pct"].median(),
        "worst_max_dd_pct": g[f"{label}_max_dd_pct"].min(),
    }

summaries = [horizon_summary(x) for x in ["3m","6m","1y","2y"]]
sumdf = pd.DataFrame(summaries)
sumdf.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

year_rows = []
for year, g in out.groupby("year"):
    row = {
        "year": int(year),
        "signals": len(g),
        "tp50_ever_n": int(g["tp50_hit"].sum()),
        "tp50_ever_pct": 100*g["tp50_hit"].mean(),
        "median_sessions_to_tp50": g.loc[g["tp50_hit"]==1,"sessions_to_tp50"].median(),
    }
    for label in ["3m","6m","1y","2y"]:
        gg = g[g[f"{label}_mature"] == 1]
        if len(gg):
            h = {"3m":63,"6m":126,"1y":252,"2y":504}[label]
            tp_by_h = ((gg["tp50_hit"]==1) & (gg["sessions_to_tp50"] <= h))
            row[f"{label}_n"] = len(gg)
            row[f"{label}_tp50_pct"] = 100*tp_by_h.mean()
            row[f"{label}_avg_system_ret_pct"] = gg[f"{label}_system_ret_pct"].mean()
            row[f"{label}_median_system_ret_pct"] = gg[f"{label}_system_ret_pct"].median()
    year_rows.append(row)

yearly = pd.DataFrame(year_rows).sort_values("year")
yearly.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

print("\n=== RULE SAMPLE ===")
print("Signals:", len(out))
print("TP50 ever:", int(out["tp50_hit"].sum()), f"({100*out['tp50_hit'].mean():.2f}%)")
if out["tp50_hit"].sum():
    print("Median sessions to TP50:", out.loc[out["tp50_hit"]==1, "sessions_to_tp50"].median())
    print("Median MAE before TP50:", f"{out.loc[out['tp50_hit']==1,'mae_before_tp_or_now_pct'].median():.2f}%")

print("\n=== TP50 SYSTEM BY HORIZON ===")
print(sumdf.to_string(index=False))

print("\n=== YEARLY ===")
print(yearly.to_string(index=False))

print("\n=== ALL MATCHED SIGNALS ===")
show_cols = [
    "year","signal_date","symbol","signal_price",
    "pre_trend_r2","pre_trend_slope_pct",
    "tp50_hit","tp50_date","sessions_to_tp50",
    "6m_system_ret_pct","1y_system_ret_pct","2y_system_ret_pct",
    "mae_before_tp_or_now_pct"
]
print(out[show_cols].to_string(index=False))

report = []
report.append("EGX RULE BACKTEST — R2 + SLOPE WITH TP +50%")
report.append("="*110)
report.append(f"Rule: pre_trend_r2 >= {R2_MIN} AND pre_trend_slope_pct >= {SLOPE_MIN}")
report.append(f"Matched signals: {len(out)}")
report.append(f"TP50 ever: {int(out['tp50_hit'].sum())}/{len(out)} = {100*out['tp50_hit'].mean():.2f}%")
if out["tp50_hit"].sum():
    report.append(f"Median sessions to TP50: {out.loc[out['tp50_hit']==1,'sessions_to_tp50'].median()}")
    report.append(f"Median MAE before TP50: {out.loc[out['tp50_hit']==1,'mae_before_tp_or_now_pct'].median():.2f}%")
report.append("")
report.append("TP50 SYSTEM BY HORIZON")
report.append(sumdf.to_string(index=False))
report.append("")
report.append("YEARLY")
report.append(yearly.to_string(index=False))
report.append("")
report.append("ALL MATCHED SIGNALS")
report.append(out[show_cols].to_string(index=False))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_CASES, OUT_SUM, OUT_YEAR, OUT_REPORT]:
    print(" ", p)
