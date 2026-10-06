#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGX TURN-ALIGNMENT ALPHA TEST V1
================================

Answers the exact question:

At a major market LOW ("buy-side turn"), did stocks whose OWN major LOW was
aligned with the market LOW window outperform stocks that were not aligned
over the next 1/3/6/12 months?

At a major market HIGH ("sell-side turn"), did stocks whose OWN major HIGH was
aligned with the market HIGH window subsequently underperform more / suffer
worse drawdowns than non-aligned stocks?

This is DIFFERENT from the prior DNA-active test.

The script reuses the final experiment's exact adjusted-OHLC, pivot, window,
sector benchmark, and forward-return methodology.

Important interpretation:
- "Aligned" is a historical/outcome classification because the major pivot
  itself is confirmed with future bars.
- Therefore this test tells us whether alignment identifies stronger leaders /
  weaker names AFTER confirmed turns.
- It does NOT make same-day alignment a causal live signal by itself.

Run:
    python /app/egx_turn_alignment_alpha_v1.py \
      --base-script /app/egx_final_decision_experiment_v2.py \
      --output-dir /app/egx_turn_alignment_alpha_v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

SEED = 20261006
RNG = np.random.default_rng(SEED)
BOOTSTRAP_TRIALS = 5000
ALIGN_GAP_SESSIONS = 3


def load_base(path):
    spec = importlib.util.spec_from_file_location("egx_final_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bootstrap_median_diff(a, b, trials=BOOTSTRAP_TRIALS):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    sims = np.empty(trials, float)
    for i in range(trials):
        aa = RNG.choice(a, len(a), replace=True)
        bb = RNG.choice(b, len(b), replace=True)
        sims[i] = np.median(aa) - np.median(bb)
    return (
        float(np.median(a)-np.median(b)),
        float(np.percentile(sims, 2.5)),
        float(np.percentile(sims, 97.5)),
    )


def nearest_stock_index(df, d):
    dates = pd.to_datetime(df["date"]).to_numpy()
    i = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(d))))
    if i >= len(df):
        i = len(df)-1
    if i < 0:
        return None
    if abs((pd.Timestamp(df.iloc[i]["date"]) - pd.Timestamp(d)).days) > 10:
        return None
    return i


def stock_aligned(stock_pivots, stock_df, market_date, turn_type, market_cal_map):
    """
    Same turn type and within +/- ALIGN_GAP_SESSIONS on the common market calendar.
    """
    if stock_pivots is None or stock_pivots.empty:
        return False, None, None

    target_ci = market_cal_map.get(pd.Timestamp(market_date))
    if target_ci is None:
        return False, None, None

    p = stock_pivots[stock_pivots["type"] == turn_type].copy()
    if p.empty:
        return False, None, None

    cis = []
    for _, r in p.iterrows():
        ci = market_cal_map.get(pd.Timestamp(r["date"]))
        if ci is not None:
            cis.append((abs(ci-target_ci), r))

    if not cis:
        return False, None, None

    dist, r = min(cis, key=lambda x: x[0])
    return dist <= ALIGN_GAP_SESSIONS, pd.Timestamp(r["date"]), int(dist)


def benchmark_return(base, bench_df, start_date, end_date, col):
    return base.benchmark_return_on_dates(bench_df, start_date, end_date, col)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = load_base(args.base_script)

    print("[1/7] Loading adjusted EGX data...", flush=True)
    stocks = base.load_stocks(args.max_symbols)
    index_df, index_source = base.load_index()
    sectors = base.fetch_sector_map()

    print("[2/7] Detecting stock pivots and market turns...", flush=True)
    pframes = []
    stock_pivots = {}
    ranges = []

    for k, (sym, df) in enumerate(stocks.items(), 1):
        p = base.detect_pivots(df)
        stock_pivots[sym] = p
        sec = sectors.get(base.base_symbol(sym), "UNKNOWN")
        if len(p):
            q = p.copy()
            q["symbol"] = sym
            q["sector"] = sec
            pframes.append(q)

        ranges.append({
            "symbol": sym,
            "sector": sec,
            "start_date": df["date"].min(),
            "end_date": df["date"].max(),
            "rows": len(df),
            "major_pivots": len(p),
        })
        if k % 25 == 0 or k == len(stocks):
            print(f"  pivots {k}/{len(stocks)}", flush=True)

    pivots = pd.concat(pframes, ignore_index=True) if pframes else pd.DataFrame()
    ranges = pd.DataFrame(ranges)

    cal = base.market_calendar(stocks, index_df)
    pivots_ci = base.add_cal_index(pivots, cal)
    sector_windows, market_turns = base.build_market_turns(pivots_ci, ranges, cal)
    tier = market_turns[market_turns["tier_a"] == 1].copy()

    print(f"  Tier-A turns: {len(tier)}", flush=True)

    print("[3/7] Building sector benchmarks...", flush=True)
    sector_indices = base.sector_equal_weight_index(stocks, sectors, cal)

    cal_map = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(cal["date"]))}

    print("[4/7] Measuring aligned vs non-aligned forward performance...", flush=True)
    rows = []

    for wi, w in tier.iterrows():
        d = pd.Timestamp(w["date"])
        typ = str(w["dominant_turn"])

        for sym, df in stocks.items():
            i = nearest_stock_index(df, d)
            if i is None:
                continue

            aligned, pivot_date, dist = stock_aligned(
                stock_pivots.get(sym), df, d, typ, cal_map
            )

            sec = sectors.get(base.base_symbol(sym), "UNKNOWN")
            secidx = sector_indices.get(sec)

            for hname, h in base.FORWARD_HORIZONS.items():
                fm = base.forward_metrics(df, i, h)
                if fm is None:
                    continue

                entry_date = pd.Timestamp(df.iloc[i]["date"])
                mret = benchmark_return(
                    base, index_df, entry_date, fm["exit_date"], "close"
                )
                sret = benchmark_return(
                    base, secidx, entry_date, fm["exit_date"], "sector_index"
                ) if secidx is not None else np.nan

                rows.append({
                    "market_turn_date": d,
                    "turn_type": typ,
                    "symbol": sym,
                    "sector": sec,
                    "aligned_same_turn": int(aligned),
                    "aligned_stock_pivot_date": pivot_date,
                    "alignment_distance_sessions": dist,
                    "horizon": hname,
                    "entry_date": entry_date,
                    "exit_date": fm["exit_date"],
                    "return_pct": fm["return_pct"],
                    "market_relative_pct": fm["return_pct"]-mret if np.isfinite(mret) else np.nan,
                    "sector_relative_pct": fm["return_pct"]-sret if np.isfinite(sret) else np.nan,
                    "mfe_pct": fm["mfe_pct"],
                    "mae_pct": fm["mae_pct"],
                })

    perf = pd.DataFrame(rows)
    perf.to_csv(out/"turn_aligned_vs_non_aligned_forward_returns.csv", index=False)

    print("[5/7] Summarizing alpha...", flush=True)
    summary = []

    for (typ, horizon), g in perf.groupby(["turn_type", "horizon"]):
        a = g[g["aligned_same_turn"] == 1]
        b = g[g["aligned_same_turn"] == 0]

        for metric in [
            "return_pct", "market_relative_pct", "sector_relative_pct",
            "mfe_pct", "mae_pct"
        ]:
            av = a[metric].to_numpy(float)
            bv = b[metric].to_numpy(float)
            diff, lo, hi = bootstrap_median_diff(av, bv)

            summary.append({
                "turn_type": typ,
                "horizon": horizon,
                "metric": metric,
                "aligned_n": int(np.isfinite(av).sum()),
                "non_aligned_n": int(np.isfinite(bv).sum()),
                "aligned_mean": float(np.nanmean(av)) if np.isfinite(av).any() else np.nan,
                "non_aligned_mean": float(np.nanmean(bv)) if np.isfinite(bv).any() else np.nan,
                "aligned_median": float(np.nanmedian(av)) if np.isfinite(av).any() else np.nan,
                "non_aligned_median": float(np.nanmedian(bv)) if np.isfinite(bv).any() else np.nan,
                "median_diff_aligned_minus_non": diff,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "aligned_positive_rate": float(np.nanmean(av > 0)) if len(av) else np.nan,
                "non_aligned_positive_rate": float(np.nanmean(bv > 0)) if len(bv) else np.nan,
            })

    summary = pd.DataFrame(summary)
    summary.to_csv(out/"TURN_ALIGNMENT_ALPHA_SUMMARY.csv", index=False)

    print("[6/7] Building 12M decision table...", flush=True)
    q12 = summary[summary["horizon"] == "12M"].copy()

    # BUY interpretation at LOW:
    # aligned should have HIGHER forward/relative returns.
    # SELL interpretation at HIGH:
    # aligned should have LOWER forward returns / worse relative returns.
    decision_rows = []

    for typ in ("LOW", "HIGH"):
        for metric in ("return_pct", "market_relative_pct", "sector_relative_pct"):
            q = q12[(q12["turn_type"] == typ) & (q12["metric"] == metric)]
            if q.empty:
                continue
            r = q.iloc[0]
            diff = float(r["median_diff_aligned_minus_non"])
            lo = float(r["bootstrap_ci_low"])
            hi = float(r["bootstrap_ci_high"])

            if typ == "LOW":
                supports = np.isfinite(lo) and lo > 0
                direction = "aligned should outperform"
            else:
                supports = np.isfinite(hi) and hi < 0
                direction = "aligned should underperform after sell-side HIGH"

            decision_rows.append({
                "turn_type": typ,
                "metric": metric,
                "expected_direction": direction,
                "median_diff_aligned_minus_non": diff,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "supports_alignment_edge": int(supports),
            })

    decision = pd.DataFrame(decision_rows)
    decision.to_csv(out/"TURN_ALIGNMENT_12M_DECISION.csv", index=False)

    print("[7/7] Writing report...", flush=True)

    low_sector = decision[
        (decision["turn_type"] == "LOW") &
        (decision["metric"] == "sector_relative_pct")
    ]
    high_sector = decision[
        (decision["turn_type"] == "HIGH") &
        (decision["metric"] == "sector_relative_pct")
    ]

    report = {
        "index_source": index_source,
        "stocks_loaded": len(stocks),
        "tier_a_turns": len(tier),
        "performance_rows": len(perf),
        "low_12m_sector_alignment_edge": (
            bool(low_sector.iloc[0]["supports_alignment_edge"])
            if len(low_sector) else None
        ),
        "high_12m_sector_alignment_edge": (
            bool(high_sector.iloc[0]["supports_alignment_edge"])
            if len(high_sector) else None
        ),
    }

    (out/"run_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== EGX TURN ALIGNMENT ALPHA V1 ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\n=== 12M DECISION ===")
    if len(decision):
        print(decision.to_string(index=False))
    else:
        print("No 12M decision rows.")

    print("\n=== 12M LOW DETAILS ===")
    x = q12[q12["turn_type"] == "LOW"]
    print(x.to_string(index=False) if len(x) else "NONE")

    print("\n=== 12M HIGH DETAILS ===")
    x = q12[q12["turn_type"] == "HIGH"]
    print(x.to_string(index=False) if len(x) else "NONE")

    print("\nFiles:")
    print(out/"TURN_ALIGNMENT_ALPHA_SUMMARY.csv")
    print(out/"TURN_ALIGNMENT_12M_DECISION.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-script",
        default="/app/egx_final_decision_experiment_v2.py"
    )
    p.add_argument(
        "--output-dir",
        default="/app/egx_turn_alignment_alpha_v1"
    )
    p.add_argument("--max-symbols", type=int, default=None)
    main(p.parse_args())
