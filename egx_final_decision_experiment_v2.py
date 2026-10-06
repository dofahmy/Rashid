#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGX FINAL DECISION EXPERIMENT V2
================================

Goal
----
One comprehensive experiment that can produce a final evidence-based decision
on three separate hypotheses:

A) STOCK DNA ALPHA
   Do stocks that are "DNA-active" around major market turns outperform stocks
   that are NOT DNA-active over the next 1/3/6/12 months?

B) CAUSAL MARKET-TURN PREDICTION
   Even if stock DNA is weak, do major market highs/lows have a repeatable
   *formation pattern* that can be recognized BEFORE the turn using only
   information available at that date?

C) INCREMENTAL PLANETARY VALUE
   Does adding GEO/HELIO planetary information improve out-of-sample prediction
   over price/breadth/sector information alone?

This script deliberately separates:
  - descriptive anatomy of historical turns,
  - causal prediction of future turns,
  - stock-selection alpha after turns,
  - incremental planetary value.

No future information is used in predictive features. Future-confirmed pivots
are used ONLY as labels / historical outcomes.

Existing project assumptions
----------------------------
The /app project provides:
    from core import database
    from monitor.gann_analysis import _daily_table, _load_egx_panel

The daily table contains columns similar to:
    symbol, session_date, o, h, l, c, v, adj_c

Dependencies
------------
    pandas
    numpy
    sqlalchemy
    requests
    pyswisseph

No scikit-learn is required; logistic regression and AUC are implemented here.

Run
---
python /app/egx_final_decision_experiment_v2.py \
  --output-dir /app/egx_final_decision_experiment_v2

Quick test
----------
python /app/egx_final_decision_experiment_v2.py \
  --output-dir /app/egx_final_decision_test \
  --max-symbols 50

Main outputs
------------
FINAL_DECISION.json
FINAL_DECISION_SUMMARY.txt

DNA:
    stock_dna_rules.csv
    stock_dna_event_map.csv
    dna_active_vs_inactive_forward_returns.csv
    DNA_ALPHA_SUMMARY.csv

Turn anatomy / prediction:
    major_market_turn_windows.csv
    turn_anatomy_profiles.csv
    daily_causal_features.csv
    predictive_model_results.csv
    predictive_test_predictions.csv

Planetary incremental-value test:
    predictive_model_results.csv
      model_type = PRICE_BREADTH_ONLY
      model_type = PRICE_BREADTH_PLUS_PLANETARY

Methodological rules
--------------------
1) OHLC is adjusted with adj_c / c before pivot extraction.
2) HIGH and LOW are modeled separately.
3) Market-turn windows use fixed +/-3-session windows, not transitive chaining.
4) Synthetic index confirmation is diagnostic only.
5) Stock DNA is learned from an early training period, validated, then frozen.
6) Stock-DNA alpha is tested only AFTER each stock's freeze date.
7) Forward returns are measured at 21/63/126/252 stock sessions.
8) Active vs inactive is compared within the SAME market windows.
9) Performance includes raw return, market-relative, sector-relative, median,
   win rate, max adverse excursion, max favorable excursion, bootstrap CI.
10) Predictive model features are causal. Labels may be future-confirmed turns.
11) Predictive model uses chronological TRAIN / VALIDATION / TEST.
12) Planetary features are judged only by incremental TEST performance.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import swisseph as swe
except ImportError as exc:
    raise SystemExit(
        "Missing pyswisseph. Install gcc/g++/make then: pip install pyswisseph"
    ) from exc

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================

SEED = 20261006
RNG = np.random.default_rng(SEED)

PLANETS = {
    "MERCURY": swe.MERCURY,
    "VENUS": swe.VENUS,
    "MARS": swe.MARS,
    "JUPITER": swe.JUPITER,
    "SATURN": swe.SATURN,
    "URANUS": swe.URANUS,
    "NEPTUNE": swe.NEPTUNE,
    "PLUTO": swe.PLUTO,
}

HARMONICS = np.array(
    [7.5, 15, 22.5, 30, 45, 60, 72, 90, 120, 135, 144, 180],
    dtype=float
)

# Major stock pivot rules
PIVOT_LEFT = 20
PIVOT_RIGHT = 20
MIN_PIVOT_SPACING = 15
MIN_PIVOT_MOVE_PCT = 10.0

# Market turn windows
TURN_WINDOW_GAP = 3
MIN_MARKET_STOCKS = 12
MIN_MARKET_BREADTH = 0.15
MIN_SECTOR_SYMBOLS = 3
MIN_SECTOR_BREADTH = 0.25
MIN_STRONG_SECTORS = 3

# Stock DNA
DNA_TRAIN_FRAC = 0.45
DNA_VALID_FRAC = 0.20
DNA_MIN_ROWS = 900
DNA_MIN_TRAIN_PIVOT_HITS = 2
DNA_MIN_VALID_EVENTS = 2
DNA_MIN_VALID_HITS = 1
DNA_MIN_VALID_ENRICHMENT = 1.10
DNA_MAX_RULES = 5
DNA_ANGLE_TOL = 2.0
DNA_EVENT_WINDOW = 3

# Forward performance
FORWARD_HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}
BOOTSTRAP_TRIALS = 3000

# Predictive model
PRED_HORIZONS = (3, 5, 10)
TRAIN_FRAC = 0.60
VALID_FRAC = 0.20
RIDGE_L2 = 1.0
LOGIT_STEPS = 1200
LOGIT_LR = 0.05

# Decision thresholds
DNA_ALPHA_MIN_MEDIAN_DIFF_12M = 3.0      # percentage points
DNA_ALPHA_MIN_OUTPERFORM_RATE = 0.55
TURN_MODEL_MIN_AUC = 0.60
TURN_MODEL_MIN_LIFT = 1.25
PLANETARY_MIN_AUC_IMPROVEMENT = 0.02

# Global ephemeris / feature caches
GLOBAL_EPH = None
GLOBAL_DATE_TO_ROW = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def pct(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or a == 0:
        return np.nan
    return (b / a - 1.0) * 100.0

def adist(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)

def sdelta(a, b):
    return ((b - a + 180.0) % 360.0) - 180.0

def folded(x):
    x = x % 360.0
    return float(360.0 - x if x > 180.0 else x)

def base_symbol(s):
    s = str(s).strip().upper()
    return s[:-3] if s.endswith(".CA") else s

def pick_col(t, aliases, required=True):
    m = {c.name.lower(): c for c in t.c}
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    if required:
        raise KeyError(f"Missing {aliases}; available={list(m)}")
    return None

def bootstrap_ci(values, trials=BOOTSTRAP_TRIALS):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    sims = np.empty(trials, dtype=float)
    for i in range(trials):
        x = RNG.choice(a, size=len(a), replace=True)
        sims[i] = np.median(x)
    return float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))

def paired_bootstrap_diff(active, inactive, trials=BOOTSTRAP_TRIALS):
    a = np.asarray(active, dtype=float)
    b = np.asarray(inactive, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    sims = np.empty(trials, dtype=float)
    for i in range(trials):
        aa = RNG.choice(a, len(a), replace=True)
        bb = RNG.choice(b, len(b), replace=True)
        sims[i] = np.median(aa) - np.median(bb)
    return (
        float(np.median(a) - np.median(b)),
        float(np.percentile(sims, 2.5)),
        float(np.percentile(sims, 97.5)),
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_stocks(max_symbols=None):
    from sqlalchemy import select
    from core import database
    from monitor.gann_analysis import _daily_table

    DB = database()
    out = {}

    with DB() as s:
        t = _daily_table(s)
        cs = pick_col(t, ["symbol", "ticker", "sym"])
        cd = pick_col(t, ["session_date", "date", "d", "datetime", "timestamp", "ts"])
        cc = pick_col(t, ["c", "close"])
        cadj = pick_col(t, ["adj_c", "adj_close", "adjusted_close"], False)
        co = pick_col(t, ["o", "open"], False)
        ch = pick_col(t, ["h", "high"], False)
        cl = pick_col(t, ["l", "low"], False)
        cv = pick_col(t, ["v", "volume"], False)

        syms = [
            r[0] for r in s.execute(
                select(cs).where(cs.ilike("%.CA")).distinct().order_by(cs)
            ).all()
            if r and r[0]
        ]
        if max_symbols:
            syms = syms[:max_symbols]

        print(f"EGX symbols found: {len(syms)}", flush=True)

        for k, sym in enumerate(syms, 1):
            cols = [cd, cc]
            names = ["date", "close"]

            for c, n in ((cadj, "adj_close"), (co, "open"), (ch, "high"),
                         (cl, "low"), (cv, "volume")):
                if c is not None:
                    cols.append(c)
                    names.append(n)

            rows = s.execute(
                select(*cols).where(cs == sym).order_by(cd)
            ).all()
            if not rows:
                continue

            df = pd.DataFrame(rows, columns=names)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            for c in ["open", "high", "low", "close", "adj_close", "volume"]:
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if "open" not in df: df["open"] = df["close"]
            if "high" not in df: df["high"] = df["close"]
            if "low" not in df: df["low"] = df["close"]
            if "volume" not in df: df["volume"] = np.nan

            # Adjust whole OHLC history.
            if "adj_close" in df.columns:
                fac = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan)
                good = fac.notna() & (fac > 0)
                if good.any():
                    for c in ["open", "high", "low"]:
                        df.loc[good, c] = df.loc[good, c] * fac[good]
                    df.loc[good, "close"] = df.loc[good, "adj_close"]

            df = (
                df.dropna(subset=["date", "open", "high", "low", "close"])
                  .sort_values("date")
                  .drop_duplicates("date", keep="last")
                  .reset_index(drop=True)
            )
            df = df[df["close"] > 0].reset_index(drop=True)

            if len(df) >= 250:
                out[str(sym).upper()] = df

            if k % 25 == 0 or k == len(syms):
                print(f"  loaded {k}/{len(syms)}", flush=True)

    return out


def load_index():
    from core import database
    from monitor.gann_analysis import _load_egx_panel

    DB = database()
    idx, panel, source = _load_egx_panel(DB)
    x = idx[["d", "c"]].copy()
    x.columns = ["date", "close"]
    x["date"] = pd.to_datetime(x["date"])
    x["open"] = x["close"]
    x["high"] = x["close"]
    x["low"] = x["close"]
    return x[["date", "open", "high", "low", "close"]].reset_index(drop=True), str(source)


def fetch_sector_map():
    try:
        import requests
        payload = {
            "filter": [{"left": "exchange", "operation": "equal", "right": "EGX"}],
            "options": {"lang": "en"},
            "markets": ["egypt"],
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": ["name", "description", "sector", "industry"],
            "range": [0, 1000],
        }
        r = requests.post(
            "https://scanner.tradingview.com/egypt/scan",
            json=payload,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        ans = {}
        for item in r.json().get("data", []):
            d = item.get("d") or []
            if len(d) >= 3:
                ans[base_symbol(d[0])] = str(d[2]).strip() if d[2] else "UNKNOWN"
        print(f"Sector labels: {len(ans)}", flush=True)
        return ans
    except Exception as e:
        print(f"[sector warning] {e}", flush=True)
        return {}


# ============================================================
# EPHEMERIS
# ============================================================

def calc_ephemeris(dates):
    rows = []
    dates = pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values()

    for k, d in enumerate(dates, 1):
        j = swe.julday(d.year, d.month, d.day, 12.0)
        row = {"date": pd.Timestamp(d)}

        for system, helio in (("GEO", False), ("HELIO", True)):
            flags = swe.FLG_SWIEPH | swe.FLG_SPEED
            if helio:
                flags |= swe.FLG_HELCTR

            pos = {}
            spd = {}
            for name, pid in PLANETS.items():
                xx, _ = swe.calc_ut(j, pid, flags)
                pos[name] = float(xx[0]) % 360.0
                spd[name] = float(xx[3])
                row[f"{system}_{name}_lon"] = pos[name]
                row[f"{system}_{name}_speed"] = spd[name]
                row[f"{system}_{name}_retro"] = int(spd[name] < 0)

            names = list(PLANETS)
            for i in range(len(names)):
                for z in range(i + 1, len(names)):
                    a, b = names[i], names[z]
                    row[f"{system}_{a}_{b}_angle"] = adist(pos[a], pos[b])

        rows.append(row)

        if k % 500 == 0 or k == len(dates):
            print(f"  ephemeris {k}/{len(dates)}", flush=True)

    return pd.DataFrame(rows).reset_index(drop=True)


def prepare_ephemeris(calendar_dates):
    global GLOBAL_EPH, GLOBAL_DATE_TO_ROW
    GLOBAL_EPH = calc_ephemeris(calendar_dates)
    GLOBAL_DATE_TO_ROW = {
        pd.Timestamp(d): i
        for i, d in enumerate(pd.to_datetime(GLOBAL_EPH["date"]))
    }


def eph_rows_for_dates(dates):
    idx = []
    for d in pd.to_datetime(dates):
        j = GLOBAL_DATE_TO_ROW.get(pd.Timestamp(d))
        if j is not None:
            idx.append(j)
    return GLOBAL_EPH.iloc[idx].reset_index(drop=True)


# ============================================================
# PIVOTS / WINDOWS
# ============================================================

def detect_pivots(df):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    rows = []

    for i in range(PIVOT_LEFT, len(df) - PIVOT_RIGHT):
        if h[i] >= np.max(h[i-PIVOT_LEFT:i]) and h[i] > np.max(h[i+1:i+1+PIVOT_RIGHT]):
            rows.append({"index": i, "date": df.loc[i, "date"], "type": "HIGH", "price": h[i]})
        if l[i] <= np.min(l[i-PIVOT_LEFT:i]) and l[i] < np.min(l[i+1:i+1+PIVOT_RIGHT]):
            rows.append({"index": i, "date": df.loc[i, "date"], "type": "LOW", "price": l[i]})

    if not rows:
        return pd.DataFrame(columns=["index", "date", "type", "price"])

    p = pd.DataFrame(rows).sort_values(["index", "type"]).reset_index(drop=True)

    collapsed = []
    for _, r in p.iterrows():
        d = r.to_dict()
        if not collapsed or collapsed[-1]["type"] != d["type"]:
            collapsed.append(d)
        else:
            old = collapsed[-1]
            better = (
                (d["type"] == "HIGH" and d["price"] > old["price"]) or
                (d["type"] == "LOW" and d["price"] < old["price"])
            )
            if better:
                collapsed[-1] = d

    clean = []
    for r in collapsed:
        if not clean:
            clean.append(r)
            continue
        old = clean[-1]
        spacing = int(r["index"]) - int(old["index"])
        move = abs(pct(float(old["price"]), float(r["price"])))
        if spacing < MIN_PIVOT_SPACING:
            continue
        if not np.isfinite(move) or move < MIN_PIVOT_MOVE_PCT:
            continue
        clean.append(r)

    return pd.DataFrame(clean)


def market_calendar(stocks, index_df):
    dates = set(pd.to_datetime(index_df["date"]).tolist())
    for df in stocks.values():
        dates.update(pd.to_datetime(df["date"]).tolist())
    return pd.DataFrame({"date": sorted(dates)}).reset_index(drop=True)


def add_cal_index(df, cal):
    if df.empty:
        return df.copy()
    m = {pd.Timestamp(d): i for i, d in enumerate(cal["date"])}
    x = df.copy()
    x["cal_index"] = pd.to_datetime(x["date"]).map(m)
    return x.dropna(subset=["cal_index"]).assign(
        cal_index=lambda q: q["cal_index"].astype(int)
    )


def fixed_windows(df, cal, gap=TURN_WINDOW_GAP):
    if df.empty:
        return []

    out = []
    for typ in ("HIGH", "LOW"):
        x = df[df["type"] == typ]
        if x.empty:
            continue

        at = {
            int(ci): set(g["symbol"].astype(str))
            for ci, g in x.groupby("cal_index")
        }
        if not at:
            continue

        lo_ci, hi_ci = min(at), max(at)
        scored = []

        for center in range(lo_ci, hi_ci + 1):
            syms = set()
            for ci in range(center-gap, center+gap+1):
                syms.update(at.get(ci, set()))
            if syms:
                scored.append((center, len(syms), syms))

        cand = []
        for center, n, syms in scored:
            neigh = [nn for cc, nn, _ in scored if abs(cc-center) <= gap]
            if n >= max(neigh):
                cand.append((center, n, syms))

        cand.sort(key=lambda z: (-z[1], z[0]))
        kept = []
        for center, n, syms in cand:
            if any(abs(center-k[0]) <= 2*gap for k in kept):
                continue
            kept.append((center, n, syms))

        for center, n, syms in sorted(kept):
            lci = max(0, center-gap)
            hci = min(len(cal)-1, center+gap)
            gx = x[(x["cal_index"] >= lci) & (x["cal_index"] <= hci)].copy()
            out.append({
                "turn_type": typ,
                "center_ci": center,
                "date": pd.Timestamp(cal.iloc[center]["date"]),
                "window_start": pd.Timestamp(cal.iloc[lci]["date"]),
                "window_end": pd.Timestamp(cal.iloc[hci]["date"]),
                "symbols": set(gx["symbol"].astype(str)),
                "rows": gx,
            })
    return out


def eligible_count(ranges, date, sector=None):
    x = ranges if sector is None else ranges[ranges["sector"] == sector]
    return int(((x["start_date"] <= date) & (x["end_date"] >= date)).sum())


def build_market_turns(pivots, ranges, cal):
    # sector windows
    sector_rows = []
    for sec, sdf in pivots.groupby("sector"):
        for w in fixed_windows(sdf, cal):
            syms = sorted(w["symbols"])
            elig = eligible_count(ranges, w["date"], sec)
            br = len(syms)/elig if elig else 0.0
            if len(syms) < MIN_SECTOR_SYMBOLS or br < MIN_SECTOR_BREADTH:
                continue
            sector_rows.append({
                "sector": sec,
                "date": w["date"],
                "dominant_turn": w["turn_type"],
                "unique_stocks": len(syms),
                "eligible_sector_stocks": elig,
                "sector_breadth_pct": 100*br,
                "symbols": " ".join(syms),
            })
    sector_windows = pd.DataFrame(sector_rows)
    sec_ci = add_cal_index(sector_windows, cal) if len(sector_windows) else pd.DataFrame()

    # market windows
    market_rows = []
    for w in fixed_windows(pivots, cal):
        syms = sorted(w["symbols"])
        elig = eligible_count(ranges, w["date"])
        br = len(syms)/elig if elig else 0.0

        if len(syms) < MIN_MARKET_STOCKS or br < MIN_MARKET_BREADTH:
            continue

        ci = int(w["center_ci"])
        if len(sec_ci):
            near = sec_ci[
                (np.abs(sec_ci["cal_index"] - ci) <= TURN_WINDOW_GAP) &
                (sec_ci["dominant_turn"] == w["turn_type"])
            ]
            strong_secs = sorted(set(near["sector"]))
        else:
            strong_secs = []

        tier_a = int(
            (len(strong_secs) >= MIN_STRONG_SECTORS and br >= MIN_MARKET_BREADTH) or
            (len(strong_secs) >= 2 and br >= 0.25)
        )

        market_rows.append({
            "date": w["date"],
            "window_start": w["window_start"],
            "window_end": w["window_end"],
            "dominant_turn": w["turn_type"],
            "unique_stocks": len(syms),
            "eligible_market_stocks": elig,
            "market_breadth_pct": 100*br,
            "strong_sector_count": len(strong_secs),
            "strong_sectors": " | ".join(strong_secs),
            "tier_a": tier_a,
            "symbols": " ".join(syms),
        })

    market = pd.DataFrame(market_rows).sort_values("date").reset_index(drop=True)
    return sector_windows, market


# ============================================================
# STOCK DNA
# ============================================================

def stock_eph(df):
    e = eph_rows_for_dates(df["date"])
    if len(e) != len(df):
        # fallback date merge
        return df[["date"]].merge(GLOBAL_EPH, on="date", how="left")
    return e


def pair_sep(eph, i, system, p1, p2):
    return adist(
        float(eph.iloc[i][f"{system}_{p1}_lon"]),
        float(eph.iloc[i][f"{system}_{p2}_lon"]),
    )


def choose_anchor_pair(pivots, train_end):
    p = pivots[pivots["index"] <= train_end].copy()
    if len(p) < 2:
        return None

    cutoff = int(train_end * 0.70)
    early = p[p["index"] <= cutoff].copy()
    if len(early) < 2:
        early = p

    best = None
    best_move = -1.0

    for i in range(len(early)):
        for j in range(i+1, len(early)):
            a = early.iloc[i]
            b = early.iloc[j]
            if a["type"] == b["type"]:
                continue
            m = abs(pct(float(a["price"]), float(b["price"])))
            if np.isfinite(m) and m > best_move:
                best_move = m
                best = (a.to_dict(), b.to_dict())
    return best


def derive_dna_candidates(pivots, eph, anchors, learn_start, train_end):
    tp = pivots[(pivots["index"] >= learn_start) & (pivots["index"] <= train_end)]
    if tp.empty:
        return []

    bucket = {}
    planets = list(PLANETS)

    for anchor in anchors:
        ai = int(anchor["index"])

        for system in ("GEO", "HELIO"):
            for a in range(len(planets)):
                for b in range(a+1, len(planets)):
                    p1, p2 = planets[a], planets[b]
                    origin = pair_sep(eph, ai, system, p1, p2)

                    targets = {}
                    for h in HARMONICS:
                        for sg in (-1, 1):
                            t = folded(origin + sg*h)
                            targets[round(t, 3)] = (float(h), float(t))

                    for _, pr in tp.iterrows():
                        pi = int(pr["index"])
                        cur = pair_sep(eph, pi, system, p1, p2)

                        best = None
                        for _, (h, t) in targets.items():
                            err = abs(cur - t)
                            if err <= DNA_ANGLE_TOL and (best is None or err < best[0]):
                                best = (err, h, t)
                        if best is None:
                            continue

                        err, h, t = best
                        sig = (
                            ai, system, p1, p2, round(t, 3)
                        )
                        if sig not in bucket:
                            bucket[sig] = {
                                "anchor_index": ai,
                                "anchor_date": anchor["date"],
                                "anchor_type": anchor["type"],
                                "system": system,
                                "planet_1": p1,
                                "planet_2": p2,
                                "origin_value": origin,
                                "offset_deg": h,
                                "target_value": t,
                                "train_pivot_hits": set(),
                                "errors": [],
                            }
                        bucket[sig]["train_pivot_hits"].add(pi)
                        bucket[sig]["errors"].append(err)

    out = []
    for r in bucket.values():
        hits = len(r["train_pivot_hits"])
        if hits < DNA_MIN_TRAIN_PIVOT_HITS:
            continue
        z = dict(r)
        z["train_pivot_hits"] = hits
        z["train_mean_error_deg"] = float(np.mean(z["errors"]))
        z.pop("errors", None)
        out.append(z)

    return out


def rule_events(rule, eph, start, end):
    p1, p2 = rule["planet_1"], rule["planet_2"]
    system = rule["system"]
    target = float(rule["target_value"])

    ev = []
    prev = pair_sep(eph, max(start-1, 0), system, p1, p2)

    for i in range(start, end+1):
        cur = pair_sep(eph, i, system, p1, p2)
        lo, hi = sorted((prev, cur))
        if lo - DNA_ANGLE_TOL <= target <= hi + DNA_ANGLE_TOL and prev != cur:
            ev.append(i)
        prev = cur

    clean = []
    for i in ev:
        if not clean or i-clean[-1] > 1:
            clean.append(i)
    return clean


def pivot_indices(pivots, start, end):
    return pivots[
        (pivots["index"] >= start) &
        (pivots["index"] <= end)
    ]["index"].astype(int).to_numpy()


def is_pivot_hit(i, piv):
    return len(piv) and int(np.min(np.abs(piv-i))) <= DNA_EVENT_WINDOW


def validate_dna_candidates(cands, pivots, eph, start, end):
    piv = pivot_indices(pivots, start, end)
    if end < start:
        return pd.DataFrame()

    baseline = (
        sum(is_pivot_hit(i, piv) for i in range(start, end+1)) / (end-start+1)
        if end >= start else 0.0
    )

    rows = []
    for r in cands:
        ev = rule_events(r, eph, start, end)
        hits = sum(is_pivot_hit(i, piv) for i in ev)
        hr = hits/len(ev) if ev else 0.0
        enr = hr/baseline if baseline > 0 else 0.0

        z = dict(r)
        z.update({
            "validation_events": len(ev),
            "validation_hits": hits,
            "validation_hit_rate": hr,
            "validation_baseline": baseline,
            "validation_enrichment": enr,
            "score": (
                8*r["train_pivot_hits"]
                + 25*hr
                + 8*math.log1p(max(enr, 0))
                - r["train_mean_error_deg"]
            )
        })
        rows.append(z)

    return pd.DataFrame(rows)


def freeze_stock_dna(symbol, df, pivots, eph, sector):
    if len(df) < DNA_MIN_ROWS or len(pivots) < 6:
        return pd.DataFrame(), None

    train_end = max(300, int(len(df)*DNA_TRAIN_FRAC)) - 1
    valid_end = max(train_end+120, int(len(df)*(DNA_TRAIN_FRAC+DNA_VALID_FRAC))) - 1
    valid_end = min(valid_end, len(df)-253)

    if valid_end <= train_end+50:
        return pd.DataFrame(), None

    pair = choose_anchor_pair(pivots, train_end)
    if pair is None:
        return pd.DataFrame(), None

    a1, a2 = pair
    learn_start = max(int(a1["index"]), int(a2["index"])) + 1
    if train_end-learn_start < 80:
        return pd.DataFrame(), None

    cands = derive_dna_candidates(pivots, eph, [a1, a2], learn_start, train_end)
    if not cands:
        return pd.DataFrame(), None

    val = validate_dna_candidates(cands, pivots, eph, train_end+1, valid_end)
    if val.empty:
        return pd.DataFrame(), None

    q = val[
        (val["validation_events"] >= DNA_MIN_VALID_EVENTS) &
        (val["validation_hits"] >= DNA_MIN_VALID_HITS) &
        (val["validation_enrichment"] >= DNA_MIN_VALID_ENRICHMENT)
    ].copy()

    if q.empty:
        return pd.DataFrame(), None

    q = q.sort_values(
        ["score", "validation_enrichment", "train_pivot_hits"],
        ascending=False
    )

    chosen = []
    cap = defaultdict(int)
    for _, r in q.iterrows():
        key = (r["system"], r["planet_1"], r["planet_2"])
        if cap[key] >= 2:
            continue
        z = r.to_dict()
        z["symbol"] = symbol
        z["sector"] = sector
        z["freeze_date"] = df.iloc[valid_end]["date"]
        z["freeze_index"] = valid_end
        chosen.append(z)
        cap[key] += 1
        if len(chosen) >= DNA_MAX_RULES:
            break

    return pd.DataFrame(chosen), valid_end


def dna_active_on_date(rules, df, eph, date):
    if rules.empty:
        return False, 0

    d = pd.Timestamp(date)
    if d < pd.Timestamp(rules.iloc[0]["freeze_date"]):
        return False, 0

    dates = pd.to_datetime(df["date"])
    pos = np.searchsorted(dates.to_numpy(), np.datetime64(d))
    if pos >= len(df):
        pos = len(df)-1
    if pos < 1:
        return False, 0

    start = max(1, pos-DNA_EVENT_WINDOW)
    end = min(len(df)-1, pos+DNA_EVENT_WINDOW)

    hits = 0
    for _, r in rules.iterrows():
        ev = rule_events(r.to_dict(), eph, start, end)
        if ev:
            hits += 1

    return hits > 0, hits


# ============================================================
# FORWARD PERFORMANCE
# ============================================================

def forward_metrics(df, idx, horizon):
    if idx < 0 or idx >= len(df)-1:
        return None

    end = min(idx+horizon, len(df)-1)
    if end <= idx:
        return None

    entry = float(df.iloc[idx]["close"])
    future = df.iloc[idx+1:end+1]["close"].astype(float).to_numpy()
    exitp = float(df.iloc[end]["close"])

    if entry <= 0 or len(future) == 0:
        return None

    ret = pct(entry, exitp)
    path = (future/entry - 1.0)*100.0

    return {
        "return_pct": ret,
        "mfe_pct": float(np.max(path)),
        "mae_pct": float(np.min(path)),
        "exit_date": df.iloc[end]["date"],
        "sessions": end-idx,
    }


def sector_equal_weight_index(stocks, sector_map, cal):
    """
    Causal/simple equal-weight sector close index built from adjusted closes.
    Used only as a relative benchmark, not for prediction.
    """
    by_sector = defaultdict(list)

    for sym, df in stocks.items():
        sec = sector_map.get(base_symbol(sym), "UNKNOWN")
        x = df[["date", "close"]].copy().rename(columns={"close": sym})
        by_sector[sec].append(x)

    out = {}
    for sec, frames in by_sector.items():
        if not frames:
            continue
        z = frames[0]
        for f in frames[1:]:
            z = z.merge(f, on="date", how="outer")
        z = z.sort_values("date")
        cols = [c for c in z.columns if c != "date"]

        # Normalize each series at its first valid point so large-price stocks
        # don't dominate the equal-weight sector proxy.
        norm = []
        for c in cols:
            s = z[c].astype(float)
            fv = s.dropna()
            if len(fv):
                norm.append(s / float(fv.iloc[0]))
        if norm:
            z["sector_index"] = pd.concat(norm, axis=1).mean(axis=1, skipna=True)
            out[sec] = z[["date", "sector_index"]].dropna().reset_index(drop=True)

    return out


def benchmark_return_on_dates(bench_df, start_date, end_date, col):
    if bench_df is None or len(bench_df) == 0:
        return np.nan

    d = pd.to_datetime(bench_df["date"]).to_numpy()
    s = np.searchsorted(d, np.datetime64(pd.Timestamp(start_date)))
    e = np.searchsorted(d, np.datetime64(pd.Timestamp(end_date)))

    s = min(max(s, 0), len(bench_df)-1)
    e = min(max(e, 0), len(bench_df)-1)

    if e <= s:
        return np.nan

    return pct(float(bench_df.iloc[s][col]), float(bench_df.iloc[e][col]))


def evaluate_dna_alpha(stocks, stock_rules, market_turns, sector_map, index_df, sector_indices):
    rows = []

    rule_groups = {
        sym: g.copy()
        for sym, g in stock_rules.groupby("symbol")
    } if len(stock_rules) else {}

    # precompute stock eph lazily
    eph_cache = {}

    for wi, w in market_turns[market_turns["tier_a"] == 1].iterrows():
        d = pd.Timestamp(w["date"])

        for sym, df in stocks.items():
            rules = rule_groups.get(sym)
            if rules is None or rules.empty:
                continue

            freeze = pd.Timestamp(rules.iloc[0]["freeze_date"])
            if d <= freeze:
                continue

            dates = pd.to_datetime(df["date"]).to_numpy()
            idx = np.searchsorted(dates, np.datetime64(d))
            if idx >= len(df):
                idx = len(df)-1
            if idx < 0 or idx >= len(df):
                continue

            # avoid stale stock date: nearest available session should be close to market date
            actual_date = pd.Timestamp(df.iloc[idx]["date"])
            if abs((actual_date-d).days) > 10:
                continue

            if sym not in eph_cache:
                eph_cache[sym] = stock_eph(df)
            active, dna_hits = dna_active_on_date(rules, df, eph_cache[sym], d)

            sec = sector_map.get(base_symbol(sym), "UNKNOWN")
            secidx = sector_indices.get(sec)

            for hname, h in FORWARD_HORIZONS.items():
                fm = forward_metrics(df, idx, h)
                if fm is None:
                    continue

                market_ret = benchmark_return_on_dates(
                    index_df, actual_date, fm["exit_date"], "close"
                )
                sector_ret = benchmark_return_on_dates(
                    secidx, actual_date, fm["exit_date"], "sector_index"
                ) if secidx is not None else np.nan

                rows.append({
                    "market_turn_date": d,
                    "turn_type": w["dominant_turn"],
                    "symbol": sym,
                    "sector": sec,
                    "dna_active": int(active),
                    "dna_rule_hits": dna_hits,
                    "horizon": hname,
                    "entry_date": actual_date,
                    "exit_date": fm["exit_date"],
                    "return_pct": fm["return_pct"],
                    "market_return_pct": market_ret,
                    "market_relative_pct": fm["return_pct"] - market_ret if np.isfinite(market_ret) else np.nan,
                    "sector_return_pct": sector_ret,
                    "sector_relative_pct": fm["return_pct"] - sector_ret if np.isfinite(sector_ret) else np.nan,
                    "mfe_pct": fm["mfe_pct"],
                    "mae_pct": fm["mae_pct"],
                })

    return pd.DataFrame(rows)


def summarize_dna_alpha(perf):
    rows = []
    if perf.empty:
        return pd.DataFrame()

    for (turn, horizon), g in perf.groupby(["turn_type", "horizon"]):
        a = g[g["dna_active"] == 1]
        b = g[g["dna_active"] == 0]

        for metric in ["return_pct", "market_relative_pct", "sector_relative_pct", "mfe_pct", "mae_pct"]:
            av = a[metric].to_numpy(float)
            bv = b[metric].to_numpy(float)
            diff, lo, hi = paired_bootstrap_diff(av, bv)

            rows.append({
                "turn_type": turn,
                "horizon": horizon,
                "metric": metric,
                "active_n": int(np.isfinite(av).sum()),
                "inactive_n": int(np.isfinite(bv).sum()),
                "active_mean": float(np.nanmean(av)) if np.isfinite(av).any() else np.nan,
                "inactive_mean": float(np.nanmean(bv)) if np.isfinite(bv).any() else np.nan,
                "active_median": float(np.nanmedian(av)) if np.isfinite(av).any() else np.nan,
                "inactive_median": float(np.nanmedian(bv)) if np.isfinite(bv).any() else np.nan,
                "median_diff_active_minus_inactive": diff,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "active_win_rate": float(np.nanmean(av > 0)) if len(av) else np.nan,
                "inactive_win_rate": float(np.nanmean(bv > 0)) if len(bv) else np.nan,
                "active_outperform_inactive_median": int(np.isfinite(diff) and diff > 0),
            })

    return pd.DataFrame(rows)


# ============================================================
# DAILY CAUSAL MARKET FEATURES
# ============================================================

def build_daily_panel(stocks, cal, sector_map):
    """
    Daily cross-sectional features using ONLY current/past data.
    """
    cal_dates = pd.to_datetime(cal["date"])
    per_stock = []

    for k, (sym, df) in enumerate(stocks.items(), 1):
        x = df[["date", "close", "high", "low", "volume"]].copy()
        x = x.sort_values("date").reset_index(drop=True)

        c = x["close"].astype(float)
        h = x["high"].astype(float)
        l = x["low"].astype(float)

        x["ret1"] = c.pct_change()
        x["ret5"] = c.pct_change(5)
        x["ret20"] = c.pct_change(20)

        x["ma20"] = c.rolling(20).mean()
        x["ma50"] = c.rolling(50).mean()

        x["above_ma20"] = (c > x["ma20"]).astype(float)
        x["above_ma50"] = (c > x["ma50"]).astype(float)

        roll_low20 = l.rolling(20, min_periods=20).min()
        roll_high20 = h.rolling(20, min_periods=20).max()
        roll_low60 = l.rolling(60, min_periods=60).min()
        roll_high60 = h.rolling(60, min_periods=60).max()

        x["near_20d_low"] = ((c / roll_low20 - 1.0) <= 0.03).astype(float)
        x["near_20d_high"] = ((roll_high20 / c - 1.0) <= 0.03).astype(float)
        x["near_60d_low"] = ((c / roll_low60 - 1.0) <= 0.05).astype(float)
        x["near_60d_high"] = ((roll_high60 / c - 1.0) <= 0.05).astype(float)

        # causal higher-low / lower-high proxy
        x["higher_low_5"] = (l.rolling(5).min() > l.shift(5).rolling(5).min()).astype(float)
        x["lower_high_5"] = (h.rolling(5).max() < h.shift(5).rolling(5).max()).astype(float)

        x["vol20"] = x["ret1"].rolling(20).std()
        x["symbol"] = sym
        x["sector"] = sector_map.get(base_symbol(sym), "UNKNOWN")
        per_stock.append(x)

    panel = pd.concat(per_stock, ignore_index=True)

    rows = []
    for d, g in panel.groupby("date"):
        n = len(g)
        if n < 10:
            continue

        r = {
            "date": pd.Timestamp(d),
            "n_stocks": n,
            "pct_up_1d": 100*float((g["ret1"] > 0).mean()),
            "pct_down_1d": 100*float((g["ret1"] < 0).mean()),
            "median_ret1_pct": 100*float(g["ret1"].median()),
            "median_ret5_pct": 100*float(g["ret5"].median()),
            "median_ret20_pct": 100*float(g["ret20"].median()),
            "dispersion_ret1_pct": 100*float(g["ret1"].std()),
            "pct_above_ma20": 100*float(g["above_ma20"].mean()),
            "pct_above_ma50": 100*float(g["above_ma50"].mean()),
            "pct_near_20d_low": 100*float(g["near_20d_low"].mean()),
            "pct_near_20d_high": 100*float(g["near_20d_high"].mean()),
            "pct_near_60d_low": 100*float(g["near_60d_low"].mean()),
            "pct_near_60d_high": 100*float(g["near_60d_high"].mean()),
            "pct_higher_low_5": 100*float(g["higher_low_5"].mean()),
            "pct_lower_high_5": 100*float(g["lower_high_5"].mean()),
            "median_vol20_pct": 100*float(g["vol20"].median()),
        }

        # Sector breadth dispersion / consensus
        secstats = []
        for sec, sg in g.groupby("sector"):
            if len(sg) < 3:
                continue
            secstats.append({
                "sector": sec,
                "above20": float(sg["above_ma20"].mean()),
                "low20": float(sg["near_20d_low"].mean()),
                "high20": float(sg["near_20d_high"].mean()),
                "ret5": float(sg["ret5"].median()),
            })

        if secstats:
            ss = pd.DataFrame(secstats)
            r["sector_pct_weak"] = 100*float((ss["above20"] < 0.35).mean())
            r["sector_pct_strong"] = 100*float((ss["above20"] > 0.65).mean())
            r["sector_low_consensus"] = 100*float((ss["low20"] > 0.40).mean())
            r["sector_high_consensus"] = 100*float((ss["high20"] > 0.40).mean())
            r["sector_ret5_dispersion_pct"] = 100*float(ss["ret5"].std())
        else:
            r["sector_pct_weak"] = np.nan
            r["sector_pct_strong"] = np.nan
            r["sector_low_consensus"] = np.nan
            r["sector_high_consensus"] = np.nan
            r["sector_ret5_dispersion_pct"] = np.nan

        rows.append(r)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # causal slopes / acceleration
    for c in [
        "pct_above_ma20", "pct_near_20d_low", "pct_near_20d_high",
        "pct_higher_low_5", "pct_lower_high_5", "median_ret5_pct",
        "median_vol20_pct"
    ]:
        daily[f"{c}_chg3"] = daily[c] - daily[c].shift(3)
        daily[f"{c}_chg5"] = daily[c] - daily[c].shift(5)

    return daily


def add_planetary_daily_features(daily):
    e = daily[["date"]].merge(GLOBAL_EPH, on="date", how="left")

    feats = pd.DataFrame({"date": daily["date"]})

    # compact, pre-specified planetary feature set to limit overfitting:
    # nearest harmonic distance for each pair + retro states.
    for system in ("GEO", "HELIO"):
        for p in PLANETS:
            feats[f"{system}_{p}_retro"] = e[f"{system}_{p}_retro"].astype(float)

        names = list(PLANETS)
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                ang = e[f"{system}_{a}_{b}_angle"].astype(float)
                dist = np.min(np.abs(ang.to_numpy()[:, None] - HARMONICS[None, :]), axis=1)
                feats[f"{system}_{a}_{b}_harmonic_dist"] = dist
                feats[f"{system}_{a}_{b}_near2"] = (dist <= 2.0).astype(float)

    return daily.merge(feats, on="date", how="left")


# ============================================================
# TURN ANATOMY
# ============================================================

def anatomy_profiles(daily, turns):
    offsets = [-20, -10, -5, -3, -1, 0]
    rows = []

    dmap = {pd.Timestamp(d): i for i, d in enumerate(daily["date"])}

    feature_cols = [
        c for c in daily.columns
        if c != "date" and pd.api.types.is_numeric_dtype(daily[c])
    ]

    for typ in ("LOW", "HIGH"):
        tw = turns[
            (turns["tier_a"] == 1) &
            (turns["dominant_turn"] == typ)
        ]

        for off in offsets:
            snapshots = []
            for d in tw["date"]:
                i = dmap.get(pd.Timestamp(d))
                if i is None:
                    continue
                j = i + off
                if j < 0 or j >= len(daily):
                    continue
                snapshots.append(daily.iloc[j][feature_cols].astype(float))

            if not snapshots:
                continue

            z = pd.DataFrame(snapshots)
            for c in feature_cols:
                vals = z[c].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                rows.append({
                    "turn_type": typ,
                    "offset_sessions": off,
                    "feature": c,
                    "n_windows": len(vals),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                })

    return pd.DataFrame(rows)


# ============================================================
# PREDICTIVE LABELS
# ============================================================

def build_future_turn_labels(daily, turns):
    dmap = {pd.Timestamp(d): i for i, d in enumerate(daily["date"])}
    low_idx = []
    high_idx = []

    for _, r in turns[turns["tier_a"] == 1].iterrows():
        i = dmap.get(pd.Timestamp(r["date"]))
        if i is None:
            continue
        (low_idx if r["dominant_turn"] == "LOW" else high_idx).append(i)

    out = daily[["date"]].copy()

    for h in PRED_HORIZONS:
        for typ, arr in (("LOW", low_idx), ("HIGH", high_idx)):
            y = np.zeros(len(daily), dtype=int)
            for i in range(len(daily)):
                if any(i < j <= i+h for j in arr):
                    y[i] = 1
            out[f"future_{typ}_{h}"] = y

    return out


# ============================================================
# LOGISTIC REGRESSION / METRICS
# ============================================================

def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0/(1.0 + np.exp(-x))

def fit_logit(X, y, l2=RIDGE_L2, steps=LOGIT_STEPS, lr=LOGIT_LR):
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    w = np.zeros(X.shape[1], dtype=float)

    # class weighting for rare events
    pos = max(y.sum(), 1.0)
    neg = max(len(y)-pos, 1.0)
    pos_w = neg/pos
    weights = np.where(y == 1, pos_w, 1.0)

    for _ in range(steps):
        p = sigmoid(X @ w)
        grad = X.T @ ((p-y)*weights) / len(y)
        grad += l2*w/len(y)
        w -= lr*grad

    return w

def auc_score(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    # Mann-Whitney
    ranks = pd.Series(p).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def confusion_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    y = np.asarray(y, int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    specificity = tn/(tn+fp) if tn+fp else 0.0
    balacc = (recall+specificity)/2
    base = y.mean() if len(y) else np.nan
    lift = precision/base if base and base > 0 else np.nan

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balacc,
        "base_rate": base,
        "precision_lift": lift,
        "alert_rate": float(pred.mean()),
    }

def choose_threshold(y, p):
    best = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 37):
        m = confusion_metrics(y, p, t)
        score = m["balanced_accuracy"]
        if score > best[1]:
            best = (float(t), float(score))
    return best[0]


def prepare_model_matrix(df, feature_cols, label_col):
    z = df[["date"] + feature_cols + [label_col]].copy()
    z = z.replace([np.inf, -np.inf], np.nan)

    # strictly past-known median imputation by global TRAIN medians later.
    return z


def run_predictive_models(daily_with_planet, labels):
    data = daily_with_planet.merge(labels, on="date", how="inner").sort_values("date").reset_index(drop=True)

    price_cols = [
        c for c in data.columns
        if c not in ["date"] and
        not c.startswith("GEO_") and
        not c.startswith("HELIO_") and
        not c.startswith("future_")
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    planet_cols = [
        c for c in data.columns
        if (c.startswith("GEO_") or c.startswith("HELIO_"))
        and pd.api.types.is_numeric_dtype(data[c])
    ]

    # remove raw n_stocks as it mostly reflects listing history
    price_cols = [c for c in price_cols if c != "n_stocks"]

    rows = []
    preds = []

    for typ in ("LOW", "HIGH"):
        for h in PRED_HORIZONS:
            label = f"future_{typ}_{h}"

            for model_type, cols in (
                ("PRICE_BREADTH_ONLY", price_cols),
                ("PRICE_BREADTH_PLUS_PLANETARY", price_cols + planet_cols),
            ):
                z = prepare_model_matrix(data, cols, label)

                # drop very early rows with almost no features
                z = z.iloc[80:].reset_index(drop=True)
                n = len(z)
                if n < 300:
                    continue

                a = int(n*TRAIN_FRAC)
                b = int(n*(TRAIN_FRAC+VALID_FRAC))
                a = max(a, 150)
                b = max(b, a+80)
                b = min(b, n-80)

                tr = z.iloc[:a].copy()
                va = z.iloc[a:b].copy()
                te = z.iloc[b:].copy()

                # TRAIN-derived imputation/scaling only
                med = tr[cols].median()
                trX = tr[cols].fillna(med).to_numpy(float)
                vaX = va[cols].fillna(med).to_numpy(float)
                teX = te[cols].fillna(med).to_numpy(float)

                mu = np.nanmean(trX, axis=0)
                sd = np.nanstd(trX, axis=0)
                sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
                mu[~np.isfinite(mu)] = 0.0

                trX = (trX-mu)/sd
                vaX = (vaX-mu)/sd
                teX = (teX-mu)/sd

                # intercept
                trX = np.column_stack([np.ones(len(trX)), trX])
                vaX = np.column_stack([np.ones(len(vaX)), vaX])
                teX = np.column_stack([np.ones(len(teX)), teX])

                ytr = tr[label].to_numpy(int)
                yva = va[label].to_numpy(int)
                yte = te[label].to_numpy(int)

                w = fit_logit(trX, ytr)
                pva = sigmoid(vaX @ w)
                pte = sigmoid(teX @ w)

                threshold = choose_threshold(yva, pva)
                vm = confusion_metrics(yva, pva, threshold)
                tm = confusion_metrics(yte, pte, threshold)

                row = {
                    "turn_type": typ,
                    "horizon_sessions": h,
                    "model_type": model_type,
                    "n_features": len(cols),
                    "train_start": tr["date"].min(),
                    "train_end": tr["date"].max(),
                    "validation_start": va["date"].min(),
                    "validation_end": va["date"].max(),
                    "test_start": te["date"].min(),
                    "test_end": te["date"].max(),
                    "threshold": threshold,
                    "validation_auc": auc_score(yva, pva),
                    "test_auc": auc_score(yte, pte),
                    "validation_precision": vm["precision"],
                    "validation_recall": vm["recall"],
                    "validation_lift": vm["precision_lift"],
                    "test_precision": tm["precision"],
                    "test_recall": tm["recall"],
                    "test_specificity": tm["specificity"],
                    "test_balanced_accuracy": tm["balanced_accuracy"],
                    "test_base_rate": tm["base_rate"],
                    "test_precision_lift": tm["precision_lift"],
                    "test_alert_rate": tm["alert_rate"],
                    "test_tp": tm["tp"],
                    "test_fp": tm["fp"],
                    "test_tn": tm["tn"],
                    "test_fn": tm["fn"],
                }
                rows.append(row)

                for d, yy, pp in zip(te["date"], yte, pte):
                    preds.append({
                        "date": d,
                        "turn_type": typ,
                        "horizon_sessions": h,
                        "model_type": model_type,
                        "actual": int(yy),
                        "probability": float(pp),
                        "threshold": threshold,
                        "alert": int(pp >= threshold),
                    })

                print(
                    f"    {typ} h={h:2d} {model_type:<28} "
                    f"TEST AUC={row['test_auc']:.3f} "
                    f"lift={row['test_precision_lift']:.2f} "
                    f"recall={row['test_recall']:.2f}",
                    flush=True
                )

    return pd.DataFrame(rows), pd.DataFrame(preds)


# ============================================================
# FINAL DECISION
# ============================================================

def make_final_decision(dna_summary, model_results):
    decision = {}

    # ---- DNA alpha decision ----
    dna_pass = False
    dna_detail = "No usable 12M DNA-active vs inactive comparison."

    if len(dna_summary):
        q = dna_summary[
            (dna_summary["horizon"] == "12M") &
            (dna_summary["metric"] == "sector_relative_pct") &
            (dna_summary["turn_type"] == "LOW")
        ]

        if len(q):
            r = q.iloc[0]
            diff = float(r["median_diff_active_minus_inactive"])
            ci_lo = float(r["bootstrap_ci_low"])
            active_wr = float(r["active_win_rate"])
            dna_pass = (
                np.isfinite(diff) and diff >= DNA_ALPHA_MIN_MEDIAN_DIFF_12M and
                np.isfinite(ci_lo) and ci_lo > 0 and
                np.isfinite(active_wr) and active_wr >= DNA_ALPHA_MIN_OUTPERFORM_RATE
            )
            dna_detail = (
                f"LOW 12M sector-relative median DNA-active minus inactive = {diff:.2f}pp; "
                f"95% bootstrap CI [{ci_lo:.2f}, {float(r['bootstrap_ci_high']):.2f}]; "
                f"DNA-active positive-rate={active_wr:.1%}."
            )

    decision["stock_dna_alpha_pass"] = bool(dna_pass)
    decision["stock_dna_alpha_detail"] = dna_detail

    # ---- causal turn model ----
    turn_pass_rows = []
    for _, r in model_results[
        model_results["model_type"] == "PRICE_BREADTH_ONLY"
    ].iterrows():
        if (
            np.isfinite(r["test_auc"]) and r["test_auc"] >= TURN_MODEL_MIN_AUC and
            np.isfinite(r["test_precision_lift"]) and r["test_precision_lift"] >= TURN_MODEL_MIN_LIFT
        ):
            turn_pass_rows.append(r.to_dict())

    decision["causal_turn_model_pass"] = len(turn_pass_rows) >= 2
    decision["causal_turn_model_pass_count"] = len(turn_pass_rows)
    decision["causal_turn_model_best"] = (
        sorted(turn_pass_rows, key=lambda x: (x["test_auc"], x["test_precision_lift"]), reverse=True)[:5]
        if turn_pass_rows else []
    )

    # ---- planetary incremental value ----
    improvements = []
    if len(model_results):
        p = model_results.pivot_table(
            index=["turn_type", "horizon_sessions"],
            columns="model_type",
            values="test_auc",
            aggfunc="first"
        ).reset_index()

        if (
            "PRICE_BREADTH_ONLY" in p.columns and
            "PRICE_BREADTH_PLUS_PLANETARY" in p.columns
        ):
            p["auc_improvement"] = (
                p["PRICE_BREADTH_PLUS_PLANETARY"] -
                p["PRICE_BREADTH_ONLY"]
            )
            improvements = p.to_dict("records")

    planet_pass_count = sum(
        1 for x in improvements
        if np.isfinite(x.get("auc_improvement", np.nan))
        and x["auc_improvement"] >= PLANETARY_MIN_AUC_IMPROVEMENT
    )

    decision["planetary_incremental_value_pass"] = planet_pass_count >= 2
    decision["planetary_auc_improvements"] = improvements

    # ---- final interpretation ----
    if decision["stock_dna_alpha_pass"] and decision["causal_turn_model_pass"]:
        verdict = "STRONG_COMBINED_EDGE"
        meaning = (
            "Market-turn structure predicts future turns out-of-sample AND DNA-active stocks "
            "show superior 12M sector-relative performance after LOW windows."
        )
    elif decision["causal_turn_model_pass"]:
        verdict = "MARKET_TURN_EDGE_ONLY"
        meaning = (
            "The repeatable edge is in causal market-turn formation. Stock-specific planetary DNA "
            "does not add enough verified stock-selection alpha."
        )
    elif decision["stock_dna_alpha_pass"]:
        verdict = "STOCK_DNA_ALPHA_ONLY"
        meaning = (
            "Stock DNA shows selection alpha, but the general market-turn formation model is not "
            "strong enough for standalone predictive timing."
        )
    else:
        verdict = "NO_VERIFIED_PREDICTIVE_EDGE"
        meaning = (
            "Historical turn anatomy can still be described, but neither causal turn prediction "
            "nor stock-DNA alpha passed the pre-defined out-of-sample thresholds."
        )

    decision["final_verdict"] = verdict
    decision["final_meaning"] = meaning

    return decision


# ============================================================
# MAIN
# ============================================================

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/12] Loading adjusted EGX stock history...", flush=True)
    stocks = load_stocks(args.max_symbols)
    index_df, index_source = load_index()
    sectors = fetch_sector_map()

    print("\n[2/12] Detecting major pivots for every stock...", flush=True)
    pframes = []
    ranges = []
    stock_pivots = {}

    for k, (sym, df) in enumerate(stocks.items(), 1):
        p = detect_pivots(df)
        stock_pivots[sym] = p
        sec = sectors.get(base_symbol(sym), "UNKNOWN")

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

    print("\n[3/12] Building fixed HIGH/LOW market and sector windows...", flush=True)
    cal = market_calendar(stocks, index_df)
    pivots = add_cal_index(pivots, cal)
    sector_windows, market_turns = build_market_turns(pivots, ranges, cal)

    print(
        f"  market turns={len(market_turns)} | Tier-A={int(market_turns['tier_a'].sum())}",
        flush=True
    )

    print("\n[4/12] Computing GEO + HELIO ephemeris ONCE for whole market calendar...", flush=True)
    prepare_ephemeris(cal["date"])

    print("\n[5/12] Learning, validating and freezing STOCK-SPECIFIC DNA...", flush=True)
    rule_frames = []
    stock_eph_cache = {}
    dna_eligible = 0

    for k, (sym, df) in enumerate(stocks.items(), 1):
        p = stock_pivots[sym]
        e = stock_eph(df)
        stock_eph_cache[sym] = e
        sec = sectors.get(base_symbol(sym), "UNKNOWN")
        rules, freeze_idx = freeze_stock_dna(sym, df, p, e, sec)

        if len(rules):
            rule_frames.append(rules)
            dna_eligible += 1

        if k % 20 == 0 or k == len(stocks):
            print(
                f"  DNA {k}/{len(stocks)} | stocks with frozen DNA={dna_eligible}",
                flush=True
            )

    stock_rules = pd.concat(rule_frames, ignore_index=True) if rule_frames else pd.DataFrame()

    print("\n[6/12] Building equal-weight sector benchmarks...", flush=True)
    sector_indices = sector_equal_weight_index(stocks, sectors, cal)

    print("\n[7/12] Testing DNA-ACTIVE vs DNA-INACTIVE future performance...", flush=True)
    perf = evaluate_dna_alpha(
        stocks, stock_rules, market_turns, sectors, index_df, sector_indices
    )
    dna_summary = summarize_dna_alpha(perf)

    print(
        f"  forward-performance rows={len(perf)} | summary rows={len(dna_summary)}",
        flush=True
    )

    print("\n[8/12] Building DAILY CAUSAL market breadth / sector features...", flush=True)
    daily = build_daily_panel(stocks, cal, sectors)
    daily_planet = add_planetary_daily_features(daily)

    print("\n[9/12] Mapping historical turn anatomy at T-20/T-10/T-5/T-3/T-1/T...", flush=True)
    anatomy = anatomy_profiles(daily, market_turns)

    print("\n[10/12] Creating future-turn labels and running chronological predictive models...", flush=True)
    labels = build_future_turn_labels(daily, market_turns)
    model_results, test_predictions = run_predictive_models(daily_planet, labels)

    print("\n[11/12] Making pre-defined final decision...", flush=True)
    decision = make_final_decision(dna_summary, model_results)
    decision.update({
        "index_source": index_source,
        "index_is_independent": int("SYNTHETIC" not in index_source.upper()),
        "stocks_loaded": len(stocks),
        "stocks_with_frozen_dna": dna_eligible,
        "frozen_dna_rules": len(stock_rules),
        "market_turn_windows": len(market_turns),
        "tier_a_turn_windows": int(market_turns["tier_a"].sum()) if len(market_turns) else 0,
        "note": (
            "Synthetic index is used as a return benchmark/diagnostic only; it is not treated as "
            "an independent confirmation source."
        )
    })

    print("\n[12/12] Writing full audit outputs...", flush=True)

    ranges.to_csv(out/"asset_ranges.csv", index=False)
    pivots.to_csv(out/"all_stock_major_pivots.csv", index=False)
    sector_windows.to_csv(out/"sector_turn_windows.csv", index=False)
    market_turns.to_csv(out/"major_market_turn_windows.csv", index=False)
    stock_rules.to_csv(out/"stock_dna_rules.csv", index=False)
    perf.to_csv(out/"dna_active_vs_inactive_forward_returns.csv", index=False)
    dna_summary.to_csv(out/"DNA_ALPHA_SUMMARY.csv", index=False)
    daily.to_csv(out/"daily_causal_features.csv", index=False)
    anatomy.to_csv(out/"turn_anatomy_profiles.csv", index=False)
    model_results.to_csv(out/"predictive_model_results.csv", index=False)
    test_predictions.to_csv(out/"predictive_test_predictions.csv", index=False)

    (out/"FINAL_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    lines = []
    lines.append("EGX FINAL DECISION EXPERIMENT V2")
    lines.append("="*70)
    lines.append(f"Final verdict: {decision['final_verdict']}")
    lines.append(f"Meaning: {decision['final_meaning']}")
    lines.append("")
    lines.append(f"Stocks loaded: {decision['stocks_loaded']}")
    lines.append(f"Stocks with frozen DNA: {decision['stocks_with_frozen_dna']}")
    lines.append(f"Frozen DNA rules: {decision['frozen_dna_rules']}")
    lines.append(f"Tier-A market turns: {decision['tier_a_turn_windows']}")
    lines.append("")
    lines.append(f"Stock DNA alpha pass: {decision['stock_dna_alpha_pass']}")
    lines.append(decision["stock_dna_alpha_detail"])
    lines.append("")
    lines.append(f"Causal market-turn model pass: {decision['causal_turn_model_pass']}")
    lines.append(f"Passing price/breadth test configurations: {decision['causal_turn_model_pass_count']}")
    lines.append("")
    lines.append(f"Planetary incremental value pass: {decision['planetary_incremental_value_pass']}")
    lines.append("")
    lines.append("The historical pivot detector uses future bars only to define labels/outcomes.")
    lines.append("All predictive model FEATURES use only information available on or before that date.")

    (out/"FINAL_DECISION_SUMMARY.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n" + "="*80)
    print("=== EGX FINAL DECISION EXPERIMENT V2 ===")
    print("="*80)
    print(f"FINAL VERDICT: {decision['final_verdict']}")
    print(decision["final_meaning"])
    print()
    print(f"stocks_loaded: {decision['stocks_loaded']}")
    print(f"stocks_with_frozen_dna: {decision['stocks_with_frozen_dna']}")
    print(f"frozen_dna_rules: {decision['frozen_dna_rules']}")
    print(f"tier_a_turn_windows: {decision['tier_a_turn_windows']}")
    print()
    print(f"STOCK DNA ALPHA PASS: {decision['stock_dna_alpha_pass']}")
    print(decision["stock_dna_alpha_detail"])
    print()
    print(f"CAUSAL MARKET TURN MODEL PASS: {decision['causal_turn_model_pass']}")
    print(f"passing configurations: {decision['causal_turn_model_pass_count']}")
    print()
    print(f"PLANETARY INCREMENTAL VALUE PASS: {decision['planetary_incremental_value_pass']}")

    print("\n=== PREDICTIVE MODEL RESULTS ===")
    if len(model_results):
        cols = [
            "turn_type", "horizon_sessions", "model_type",
            "test_auc", "test_precision", "test_recall",
            "test_precision_lift", "test_alert_rate"
        ]
        print(model_results[cols].to_string(index=False))

    print("\n=== DNA 12M LOW SUMMARY ===")
    if len(dna_summary):
        q = dna_summary[
            (dna_summary["turn_type"] == "LOW") &
            (dna_summary["horizon"] == "12M")
        ]
        if len(q):
            print(q.to_string(index=False))
        else:
            print("No 12M LOW comparison.")
    else:
        print("No DNA performance comparison.")

    print("\nMost important files:")
    print(out/"FINAL_DECISION.json")
    print(out/"FINAL_DECISION_SUMMARY.txt")
    print(out/"DNA_ALPHA_SUMMARY.csv")
    print(out/"predictive_model_results.csv")
    print(out/"turn_anatomy_profiles.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/app/egx_final_decision_experiment_v2")
    p.add_argument("--max-symbols", type=int, default=None)
    main(p.parse_args())
