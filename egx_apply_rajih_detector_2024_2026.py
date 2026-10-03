#!/usr/bin/env python3
"""
Rajih — Apply the US-discovered detector to the Egyptian Exchange (EGX), 2024-2026.

This is a cross-market experiment. It does NOT retune thresholds for Egypt.

Detector pipeline used:
FIRST-STAGE (same family used before Candidate A):
  1) T-1 ATR14% >= 10.96
  2) T-1 5-day return <= -9.22%

CANDIDATE A:
  3) T-1 Bollinger Width% > 100
  4) T-1 Dollar-Volume Ratio 5/20 > 1.5
  5) T-20 10-day return <= -15.3917%

EARLY CONFIRMATION after 5 sessions:
  6) Day-5 close > signal-day high
  7) First-5-day max drawdown from signal close >= -16.2879%

Outcome:
  +50%, +100%, +200% by adjusted HIGH within the next 63 trading sessions,
  measured from signal-day adjusted close.

Data:
  - EGX symbol universe discovered from TradingView scanner, with fallback to
    stockanalysis.com table if available.
  - Daily OHLCV downloaded via Yahoo Finance using <SYMBOL>.CA.

Notes:
  - Monetary USD liquidity filters used in the US research are deliberately NOT
    copied to Egypt because EGP turnover is not directly comparable.
  - Corporate-action anomalies are rejected if adjusted-close one-day jump >100%
    in prior 20 sessions or forward 63 sessions.
  - 2026 signals without 63 future sessions are shown as "unmatured" and are not
    counted in outcome precision.

Outputs:
  /data/egx_detector_all_signals.csv
  /data/egx_detector_confirmed.csv
  /data/egx_detector_yearly.csv
  /data/egx_detector_summary.json
  /data/egx_detector_report.txt
  /data/egx_detector_cache/*.csv

Install if needed:
  pip install -U yfinance pandas requests lxml

Run:
  cd /app
  python egx_apply_rajih_detector_2024_2026.py
"""

from __future__ import annotations

import json, math, os, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(os.getenv("RAJIH_DATA_DIR", "/data"))
CACHE_DIR = DATA_DIR / "egx_detector_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = DATA_DIR / "egx_detector_all_signals.csv"
OUT_CONF = DATA_DIR / "egx_detector_confirmed.csv"
OUT_YEAR = DATA_DIR / "egx_detector_yearly.csv"
OUT_JSON = DATA_DIR / "egx_detector_summary.json"
OUT_REPORT = DATA_DIR / "egx_detector_report.txt"

START = "2023-01-01"   # enough lookback for 2024 signals
END = "2026-10-05"     # exclusive-ish; current research cutoff around 2026-10-02

ATR_MIN = 10.96
RET5_MAX = -9.22
BB_WIDTH_MIN = 100.0
DV_RATIO_MIN = 1.5
T20_RET10_MAX = -15.3917
CONFIRM_DD5_MIN = -16.2879

YEARS = (2024, 2025, 2026)


def discover_egx_symbols():
    """Try TradingView scanner first; fallback to StockAnalysis HTML."""
    url = "https://scanner.tradingview.com/egypt/scan"
    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "EGX"},
        ],
        "options": {"lang": "en"},
        "markets": ["egypt"],
        "symbols": {"query": {"types": ["stock"]}, "tickers": []},
        "columns": ["name", "description", "type", "exchange"],
        "range": [0, 1000],
    }

    try:
        r = requests.post(url, json=payload, timeout=30, headers={
            "User-Agent": "Mozilla/5.0"
        })
        r.raise_for_status()
        js = r.json()
        syms = []
        for item in js.get("data", []):
            d = item.get("d") or []
            if d:
                s = str(d[0]).strip().upper()
                if s and s not in syms:
                    syms.append(s)
        if len(syms) >= 100:
            print(f"TradingView EGX universe: {len(syms)} symbols")
            return syms
    except Exception as e:
        print("TradingView universe failed:", e)

    try:
        tabs = pd.read_html("https://stockanalysis.com/list/egyptian-stock-exchange/")
        for tab in tabs:
            cols = {str(c).lower(): c for c in tab.columns}
            if "symbol" in cols and len(tab) >= 50:
                syms = (
                    tab[cols["symbol"]]
                    .astype(str).str.strip().str.upper()
                    .drop_duplicates().tolist()
                )
                print(f"StockAnalysis EGX universe: {len(syms)} symbols")
                return syms
    except Exception as e:
        print("StockAnalysis universe failed:", e)

    raise RuntimeError("Could not discover EGX symbols from TradingView or StockAnalysis.")


def download_symbol(sym):
    path = CACHE_DIR / f"{sym}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if len(df) >= 50:
                return df
        except Exception:
            pass

    ticker = sym if sym.endswith(".CA") else f"{sym}.CA"
    try:
        df = yf.download(
            ticker,
            start=START,
            end=END,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=30,
        )
    except TypeError:
        df = yf.download(
            ticker,
            start=START,
            end=END,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df = df[["Date","Open","High","Low","Close","Adj Close","Volume"]].copy()
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close","Adj Close"])
    df = df[(df["Close"] > 0) & (df["Adj Close"] > 0)]
    if len(df) < 80:
        return None

    df.to_csv(path, index=False)
    return df


def add_features(df):
    d = df.copy().sort_values("Date").reset_index(drop=True)

    # Adjust OHLC using Yahoo adjustment factor.
    fac = d["Adj Close"] / d["Close"]
    d["ao"] = d["Open"] * fac
    d["ah"] = d["High"] * fac
    d["al"] = d["Low"] * fac
    d["ac"] = d["Adj Close"]

    prev = d["ac"].shift(1)
    tr = pd.concat([
        d["ah"] - d["al"],
        (d["ah"] - prev).abs(),
        (d["al"] - prev).abs(),
    ], axis=1).max(axis=1)

    # Wilder ATR14 (common ATR definition).
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["atr14_pct"] = 100 * atr / d["ac"]

    d["ret5_pct"] = 100 * (d["ac"] / d["ac"].shift(5) - 1)
    d["ret10_pct"] = 100 * (d["ac"] / d["ac"].shift(10) - 1)

    sma20 = d["ac"].rolling(20).mean()
    std20 = d["ac"].rolling(20).std(ddof=0)
    upper = sma20 + 2*std20
    lower = sma20 - 2*std20
    d["bb_width_pct"] = 100 * (upper - lower) / sma20

    dollarvol = d["Close"] * d["Volume"]
    d["dollarvol_ratio_5_20"] = (
        dollarvol.rolling(5).mean() / dollarvol.rolling(20).mean()
    )

    d["adj_close_jump_pct"] = 100 * (d["ac"] / d["ac"].shift(1) - 1)

    return d


def passes_presignal(d, i):
    # Signal day = i. All detector state is evaluated through T-1.
    t1 = i - 1
    t20 = i - 20
    if t20 - 10 < 0 or t1 < 20:
        return False

    vals = {
        "atr": d.at[t1, "atr14_pct"],
        "r5": d.at[t1, "ret5_pct"],
        "bb": d.at[t1, "bb_width_pct"],
        "dv": d.at[t1, "dollarvol_ratio_5_20"],
        "t20r10": d.at[t20, "ret10_pct"],
    }
    if any(pd.isna(v) or not np.isfinite(v) for v in vals.values()):
        return False

    if vals["atr"] < ATR_MIN:
        return False
    if vals["r5"] > RET5_MAX:
        return False
    if vals["bb"] <= BB_WIDTH_MIN:
        return False
    if vals["dv"] <= DV_RATIO_MIN:
        return False
    if vals["t20r10"] > T20_RET10_MAX:
        return False

    # Reject suspicious corporate-action discontinuities in prior 20 sessions.
    prior_jump = d.loc[max(1, i-20):i, "adj_close_jump_pct"].abs()
    if (prior_jump > 100).any():
        return False

    return True


def analyze_symbol(sym, df):
    d = add_features(df)
    out = []

    # Need five sessions after signal for confirmation.
    for i in range(31, len(d)-5):
        date = pd.Timestamp(d.at[i, "Date"])
        if date.year not in YEARS:
            continue
        if not passes_presignal(d, i):
            continue

        sig_close = float(d.at[i, "ac"])
        sig_high = float(d.at[i, "ah"])
        if not (sig_close > 0 and sig_high > 0):
            continue

        # Early confirmation from sessions +1..+5.
        future5 = d.iloc[i+1:i+6]
        dd5 = 100 * (future5["al"].min() / sig_close - 1)
        d5_close = float(d.at[i+5, "ac"])
        confirmed = (d5_close > sig_high) and (dd5 >= CONFIRM_DD5_MIN)

        # Mature 63-session outcome only if available.
        mature = i + 63 < len(d)
        hit50 = hit100 = hit200 = None
        max_gain63 = None
        first200_day = None

        if mature:
            f63 = d.iloc[i+1:i+64]
            # Reject suspicious forward adjusted-close jump >100%.
            jumps = d.loc[i+1:i+63, "adj_close_jump_pct"].abs()
            clean_forward = not (jumps > 100).any()
            if clean_forward:
                gains = 100 * (f63["ah"].to_numpy() / sig_close - 1)
                max_gain63 = float(np.nanmax(gains))
                hit50 = int(np.any(gains >= 50))
                hit100 = int(np.any(gains >= 100))
                hit200 = int(np.any(gains >= 200))
                idx = np.where(gains >= 200)[0]
                first200_day = int(idx[0] + 1) if len(idx) else None
            else:
                mature = False

        out.append({
            "symbol": sym,
            "yahoo_symbol": f"{sym}.CA",
            "signal_date": date.date().isoformat(),
            "year": date.year,
            "signal_close": sig_close,
            "t1_atr14_pct": float(d.at[i-1, "atr14_pct"]),
            "t1_ret5_pct": float(d.at[i-1, "ret5_pct"]),
            "t1_bb_width_pct": float(d.at[i-1, "bb_width_pct"]),
            "t1_dollarvol_ratio_5_20": float(d.at[i-1, "dollarvol_ratio_5_20"]),
            "t20_ret10_pct": float(d.at[i-20, "ret10_pct"]),
            "d5_close": d5_close,
            "d5_close_above_signal_high": int(d5_close > sig_high),
            "first5_max_drawdown_pct": float(dd5),
            "confirmed": int(confirmed),
            "mature_63d": int(mature),
            "hit50": hit50,
            "hit100": hit100,
            "hit200": hit200,
            "max_gain63_pct": max_gain63,
            "first200_day": first200_day,
        })

    return out


def summarize(df, confirmed_only=False):
    x = df[df["confirmed"] == 1].copy() if confirmed_only else df.copy()
    mature = x[x["mature_63d"] == 1].copy()

    result = {
        "signals": int(len(x)),
        "mature_signals": int(len(mature)),
        "unmatured_signals": int(len(x) - len(mature)),
        "distinct_symbols": int(x["symbol"].nunique()) if len(x) else 0,
    }
    for th in (50,100,200):
        col = f"hit{th}"
        if len(mature):
            result[f"hit{th}_n"] = int(mature[col].fillna(0).sum())
            result[f"hit{th}_pct"] = float(100*mature[col].fillna(0).mean())
        else:
            result[f"hit{th}_n"] = 0
            result[f"hit{th}_pct"] = None
    return result


symbols = discover_egx_symbols()
print(f"\nScanning {len(symbols)} EGX symbols from {START} to {END} ...")

all_rows = []
ok = 0
failed = []

for n, sym in enumerate(symbols, 1):
    try:
        df = download_symbol(sym)
        if df is None or len(df) < 80:
            failed.append(sym)
            continue
        ok += 1
        rows = analyze_symbol(sym, df)
        if rows:
            all_rows.extend(rows)
        if n % 20 == 0 or rows:
            print(
                f"[{n}/{len(symbols)}] {sym}: rows={len(df)} "
                f"pre-signals={len(rows)} confirmed={sum(r['confirmed'] for r in rows)}"
            )
        time.sleep(0.05)
    except Exception as e:
        failed.append(sym)
        print(f"[{n}/{len(symbols)}] {sym} ERROR: {e}")

if not all_rows:
    raise SystemExit("No detector signals found. Check downloads/universe/data quality.")

all_df = pd.DataFrame(all_rows).sort_values(["signal_date","symbol"]).reset_index(drop=True)
conf_df = all_df[all_df["confirmed"] == 1].copy()

all_df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
conf_df.to_csv(OUT_CONF, index=False, encoding="utf-8-sig")

year_rows = []
for y in YEARS:
    yr = all_df[all_df["year"] == y]
    pre = summarize(yr, confirmed_only=False)
    con = summarize(yr, confirmed_only=True)
    year_rows.append({
        "year": y,
        "pre_signals": pre["signals"],
        "confirmed": con["signals"],
        "confirmed_mature": con["mature_signals"],
        "confirmed_unmatured": con["unmatured_signals"],
        "confirmed_symbols": con["distinct_symbols"],
        "hit50_n": con["hit50_n"],
        "hit50_pct": con["hit50_pct"],
        "hit100_n": con["hit100_n"],
        "hit100_pct": con["hit100_pct"],
        "hit200_n": con["hit200_n"],
        "hit200_pct": con["hit200_pct"],
    })

year_df = pd.DataFrame(year_rows)
year_df.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

overall_pre = summarize(all_df, confirmed_only=False)
overall_conf = summarize(all_df, confirmed_only=True)

summary = {
    "universe_discovered": len(symbols),
    "symbols_with_usable_daily_data": ok,
    "failed_or_insufficient_symbols": len(failed),
    "detector": {
        "first_stage": {
            "t1_atr14_pct_gte": ATR_MIN,
            "t1_ret5_pct_lte": RET5_MAX,
        },
        "candidate_A": {
            "t1_bb_width_pct_gt": BB_WIDTH_MIN,
            "t1_dollarvol_ratio_5_20_gt": DV_RATIO_MIN,
            "t20_ret10_pct_lte": T20_RET10_MAX,
        },
        "confirmation": {
            "d5_close_gt_signal_high": True,
            "first5_max_drawdown_pct_gte": CONFIRM_DD5_MIN,
        },
    },
    "overall_presignal": overall_pre,
    "overall_confirmed": overall_conf,
    "yearly": year_rows,
    "failed_symbols": failed,
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== EGX OVERALL ===")
print(f"Universe discovered: {len(symbols)}")
print(f"Usable daily history: {ok}")
print(f"Pre-signals: {overall_pre['signals']}")
print(f"Confirmed: {overall_conf['signals']}")
print(f"Mature confirmed: {overall_conf['mature_signals']}")
print(
    f"+50={overall_conf['hit50_n']}/{overall_conf['mature_signals']} "
    f"({overall_conf['hit50_pct']})"
)
print(
    f"+100={overall_conf['hit100_n']}/{overall_conf['mature_signals']} "
    f"({overall_conf['hit100_pct']})"
)
print(
    f"+200={overall_conf['hit200_n']}/{overall_conf['mature_signals']} "
    f"({overall_conf['hit200_pct']})"
)

print("\n=== EGX YEARLY ===")
print(year_df.to_string(index=False))

print("\n=== CONFIRMED EGX CASES ===")
cols = [
    "year","signal_date","symbol","signal_close",
    "t1_atr14_pct","t1_ret5_pct","t1_bb_width_pct",
    "t1_dollarvol_ratio_5_20","t20_ret10_pct",
    "first5_max_drawdown_pct","hit50","hit100","hit200","max_gain63_pct"
]
if len(conf_df):
    print(conf_df[cols].to_string(index=False))
else:
    print("None")

report = []
report.append("RAJIH — EGX DETECTOR 2024-2026")
report.append("="*100)
report.append(f"Universe discovered: {len(symbols)}")
report.append(f"Usable symbols: {ok}")
report.append(f"Pre-signals: {overall_pre['signals']}")
report.append(f"Confirmed: {overall_conf['signals']}")
report.append(f"Mature confirmed: {overall_conf['mature_signals']}")
report.append(
    f"+50: {overall_conf['hit50_n']} ({overall_conf['hit50_pct']}%) | "
    f"+100: {overall_conf['hit100_n']} ({overall_conf['hit100_pct']}%) | "
    f"+200: {overall_conf['hit200_n']} ({overall_conf['hit200_pct']}%)"
)
report.append("")
report.append("YEARLY")
report.append(year_df.to_string(index=False))
report.append("")
report.append("CONFIRMED CASES")
report.append(conf_df[cols].to_string(index=False) if len(conf_df) else "None")
OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nCreated:")
for p in [OUT_ALL, OUT_CONF, OUT_YEAR, OUT_JSON, OUT_REPORT]:
    print(" ", p)
