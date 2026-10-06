#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JENKINS / EGX FINAL EXPERIMENT V1.1 FAST
=================================

This is the "last experiment" research pipeline.  It is intentionally stricter
than the earlier prototypes and is designed to answer one question:

    Do major Egyptian-market / sector turning windows contain planetary
    configurations that repeat more often than would be expected by chance,
    and do those configurations survive out-of-sample testing?

Key safeguards added:
---------------------
1) Split/dividend-adjusted OHLC using adj_c/c from the project DB.
2) Fixed, non-transitive HIGH/LOW windows (no chain-merging).
3) HIGH and LOW studied separately.
4) Synthetic index is NEVER allowed to strengthen statistical selection.
   It is reported only as a diagnostic unless a real independent index source
   is later supplied.
5) Train -> validation -> holdout chronology.
6) Multiple random baselines:
      - year-matched random trading dates
      - circular-shift dates preserving spacing
      - harmonic-placebo rotations
7) Multiple-testing control: Benjamini-Hochberg FDR.
8) Robustness across harmonic tolerances 1°, 2°, 3°.
9) Feature families:
      - GEO pair near Jenkins/Gann harmonics
      - HELIO pair near harmonics
      - retro/direct state
      - station-like slow motion
      - applying vs separating from nearest harmonic
10) Sector DNA is tested only when enough sector windows exist.
11) Stability score: feature must recur across tolerances and survive holdout.
12) Full audit CSVs are written.

The script uses the existing /app project:
    from core import database
    from monitor.gann_analysis import _daily_table, _load_egx_panel

Run:
    python /app/egx_jenkins_final_experiment_v1_1_fast.py \
      --output-dir /app/egx_jenkins_final_experiment_v1_1_fast

Useful quick test:
    python /app/egx_jenkins_final_experiment_v1_1_fast.py \
      --output-dir /app/egx_jenkins_final_test \
      --max-symbols 50
"""

from __future__ import annotations

import argparse, json, math, warnings
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import swisseph as swe
except ImportError as exc:
    raise SystemExit(
        "Missing pyswisseph. Install gcc/g++/make then: pip install pyswisseph"
    ) from exc

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------
# Research constants
# -------------------------

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

TOLERANCES = (1.0, 2.0, 3.0)
SEED = 20261006
RNG = np.random.default_rng(SEED)

# Global caches populated once after the market calendar is built.
# This is the main V1.1 speed-up: Swiss Ephemeris is calculated once per
# trading day, not again for every tolerance / random trial / sector.
GLOBAL_CAL_SNAPSHOT = None
GLOBAL_FEATURE_CACHE = {}
GLOBAL_DATE_TO_ROW = {}

# Station threshold in deg/day. Different planets have very different speeds;
# values are normalized by planet-specific historical median absolute speed.
STATION_RATIO = 0.10

# Window construction
PIVOT_LEFT = 20
PIVOT_RIGHT = 20
MIN_PIVOT_SPACING = 15
MIN_PIVOT_MOVE = 10.0
WINDOW_GAP = 3

# Major market-window filters
MIN_MARKET_BREADTH = 0.15
MIN_MARKET_STOCKS = 12
MIN_STRONG_SECTORS = 3
MIN_SECTOR_BREADTH = 0.25
MIN_SECTOR_SYMBOLS = 3

# Statistical thresholds
MIN_DISCOVERY_HITS = 3
MIN_DISCOVERY_ENRICHMENT = 1.25
DISCOVERY_P_MAX = 0.10
HOLDOUT_MIN_HITS = 2
HOLDOUT_ENRICHMENT = 1.15
HOLDOUT_P_MAX = 0.20
MIN_SECTOR_WINDOWS = 8

RANDOM_TRIALS = 3000
CIRCULAR_TRIALS = 3000
PLACEBO_TRIALS = 1500


# ============================================================
# Generic helpers
# ============================================================

def pct(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or a == 0:
        return np.nan
    return (b/a - 1.0) * 100.0

def adist(a, b):
    d = abs((a-b) % 360.0)
    return min(d, 360.0-d)

def signed_adist(a, b):
    return ((b-a+180.0) % 360.0) - 180.0

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

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for pos in range(n-1, -1, -1):
        idx = order[pos]
        rank = pos + 1
        val = p[idx] * n / rank
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    return q

def percentile_ci(values, lo=2.5, hi=97.5):
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


# ============================================================
# Data loading: ADJUSTED OHLC
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

        syms = [
            r[0] for r in s.execute(
                select(cs).where(cs.ilike("%.CA")).distinct().order_by(cs)
            ).all()
            if r and r[0]
        ]
        if max_symbols:
            syms = syms[:max_symbols]

        print(f"EGX symbols found: {len(syms)}")

        for k, sym in enumerate(syms, 1):
            cols = [cd, cc]
            names = ["date", "close"]

            if cadj is not None:
                cols.append(cadj); names.append("adj_close")
            for c, n in ((co, "open"), (ch, "high"), (cl, "low")):
                if c is not None:
                    cols.append(c); names.append(n)

            rows = s.execute(
                select(*cols).where(cs == sym).order_by(cd)
            ).all()
            if not rows:
                continue

            df = pd.DataFrame(rows, columns=names)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            for c in ["open", "high", "low", "close", "adj_close"]:
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if "open" not in df: df["open"] = df["close"]
            if "high" not in df: df["high"] = df["close"]
            if "low" not in df: df["low"] = df["close"]

            # CRITICAL: adjust the entire OHLC history for corporate actions.
            if "adj_close" in df:
                fac = df["adj_close"] / df["close"]
                fac = fac.replace([np.inf, -np.inf], np.nan)
                good = fac.notna() & (fac > 0)
                if good.any():
                    df.loc[good, "open"] = df.loc[good, "open"] * fac[good]
                    df.loc[good, "high"] = df.loc[good, "high"] * fac[good]
                    df.loc[good, "low"] = df.loc[good, "low"] * fac[good]
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
            json=payload, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        ans = {}
        for item in r.json().get("data", []):
            d = item.get("d") or []
            if len(d) >= 3:
                ans[base_symbol(d[0])] = str(d[2]).strip() if d[2] else "UNKNOWN"
        print(f"Sector labels loaded: {len(ans)}")
        return ans
    except Exception as e:
        print(f"[sector warning] sector map unavailable: {e}")
        return {}


# ============================================================
# Pivots
# ============================================================

def detect_pivots(df):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    rows = []

    for i in range(PIVOT_LEFT, len(df)-PIVOT_RIGHT):
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
            prev = collapsed[-1]
            better = (
                (d["type"] == "HIGH" and d["price"] > prev["price"]) or
                (d["type"] == "LOW" and d["price"] < prev["price"])
            )
            if better:
                collapsed[-1] = d

    clean = []
    for r in collapsed:
        if not clean:
            clean.append(r); continue
        prev = clean[-1]
        spacing = int(r["index"]) - int(prev["index"])
        move = abs(pct(float(prev["price"]), float(r["price"])))
        if spacing < MIN_PIVOT_SPACING:
            continue
        if not np.isfinite(move) or move < MIN_PIVOT_MOVE:
            continue
        clean.append(r)

    return pd.DataFrame(clean)


# ============================================================
# Market calendar and FIXED windows
# ============================================================

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


def fixed_window_candidates(df, cal, gap=WINDOW_GAP, by_type=True):
    if df.empty:
        return []

    groups = []
    types = ["HIGH", "LOW"] if by_type and "type" in df.columns else [None]

    for typ in types:
        x = df if typ is None else df[df["type"] == typ]
        if x.empty:
            continue

        at = {
            int(ci): set(g["symbol"].astype(str))
            for ci, g in x.groupby("cal_index")
        }
        if not at:
            continue

        min_ci, max_ci = min(at), max(at)
        scored = []

        for center in range(min_ci, max_ci+1):
            syms = set()
            for ci in range(center-gap, center+gap+1):
                syms.update(at.get(ci, set()))
            if syms:
                scored.append((center, len(syms), syms))

        # local maxima
        candidates = []
        for center, n, syms in scored:
            neigh = [nn for cc, nn, _ in scored if abs(cc-center) <= gap]
            if n >= max(neigh):
                candidates.append((center, n, syms))

        # suppress overlapping centers
        candidates.sort(key=lambda z: (-z[1], z[0]))
        kept = []
        for center, n, syms in candidates:
            if any(abs(center-k[0]) <= 2*gap for k in kept):
                continue
            kept.append((center, n, syms))

        for center, n, syms in sorted(kept):
            lo = max(0, center-gap)
            hi = min(len(cal)-1, center+gap)
            gx = x[(x["cal_index"] >= lo) & (x["cal_index"] <= hi)].copy()
            if gx.empty:
                continue
            groups.append({
                "turn_type": typ,
                "center_ci": center,
                "center_date": pd.Timestamp(cal.iloc[center]["date"]),
                "window_start": pd.Timestamp(cal.iloc[lo]["date"]),
                "window_end": pd.Timestamp(cal.iloc[hi]["date"]),
                "rows": gx,
                "symbols": set(gx["symbol"].astype(str)),
            })
    return groups


def eligible_count(ranges, date, sector=None):
    x = ranges
    if sector is not None:
        x = x[x["sector"] == sector]
    return int(((x["start_date"] <= date) & (x["end_date"] >= date)).sum())


def build_windows(pivots, ranges, idxp, cal, index_source):
    # sector windows first
    sec_rows = []
    for sec, sdf in pivots.groupby("sector"):
        for w in fixed_window_candidates(sdf, cal, by_type=True):
            d = w["center_date"]
            syms = sorted(w["symbols"])
            elig = eligible_count(ranges, d, sec)
            breadth = len(syms)/elig if elig else 0
            if len(syms) < MIN_SECTOR_SYMBOLS or breadth < MIN_SECTOR_BREADTH:
                continue
            sec_rows.append({
                "sector": sec,
                "date": d,
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                "dominant_turn": w["turn_type"],
                "unique_stocks": len(syms),
                "eligible_sector_stocks": elig,
                "sector_breadth_pct": 100*breadth,
                "symbols": " ".join(syms),
            })
    sector_windows = pd.DataFrame(sec_rows)
    sec_ci = add_cal_index(sector_windows, cal) if len(sector_windows) else pd.DataFrame()

    # index diagnostics
    idx_high = set(idxp.loc[idxp["type"] == "HIGH", "cal_index"].astype(int)) if len(idxp) else set()
    idx_low = set(idxp.loc[idxp["type"] == "LOW", "cal_index"].astype(int)) if len(idxp) else set()

    # synthetic proxy must not boost statistical selection
    index_independent = 0 if "SYNTHETIC" in index_source.upper() else 1

    market_rows = []
    for wid, w in enumerate(fixed_window_candidates(pivots, cal, by_type=True), 1):
        ci = int(w["center_ci"])
        d = w["center_date"]
        syms = sorted(w["symbols"])
        elig = eligible_count(ranges, d)
        breadth = len(syms)/elig if elig else 0
        typ = w["turn_type"]

        if len(syms) < MIN_MARKET_STOCKS or breadth < MIN_MARKET_BREADTH:
            continue

        if len(sec_ci):
            near_sec = sec_ci[
                (np.abs(sec_ci["cal_index"]-ci) <= WINDOW_GAP) &
                (sec_ci["dominant_turn"] == typ)
            ]
            strong_secs = sorted(set(near_sec["sector"]))
        else:
            strong_secs = []

        idxset = idx_high if typ == "HIGH" else idx_low
        idx_hit = int(any(abs(x-ci) <= WINDOW_GAP for x in idxset))

        # Selection score excludes synthetic index.
        score = (
            100*breadth
            + 8*min(len(strong_secs), 5)
            + 0.5*len(syms)
            + (20*idx_hit if index_independent else 0)
        )

        tier_a = int(
            (len(strong_secs) >= MIN_STRONG_SECTORS and breadth >= MIN_MARKET_BREADTH) or
            (breadth >= 0.25 and len(strong_secs) >= 2)
        )

        market_rows.append({
            "window_id": wid,
            "date": d,
            "window_start": w["window_start"],
            "window_end": w["window_end"],
            "dominant_turn": typ,
            "unique_stocks": len(syms),
            "eligible_market_stocks": elig,
            "market_breadth_pct": 100*breadth,
            "strong_sector_count": len(strong_secs),
            "strong_sectors": " | ".join(strong_secs),
            "index_pivot_confirmed": idx_hit,
            "index_independent": index_independent,
            "selection_score": score,
            "tier_a": tier_a,
            "symbols": " ".join(syms),
        })

    return sector_windows, pd.DataFrame(market_rows)


# ============================================================
# Planetary snapshots
# ============================================================

def calc_snapshots(dates):
    rows = []
    for d in pd.to_datetime(dates):
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
                for k in range(i+1, len(names)):
                    a, b = names[i], names[k]
                    row[f"{system}_{a}_{b}_angle"] = adist(pos[a], pos[b])
                    row[f"{system}_{a}_{b}_relspeed"] = spd[b] - spd[a]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_speed_scales(calendar_dates):
    global GLOBAL_CAL_SNAPSHOT
    if GLOBAL_CAL_SNAPSHOT is None:
        print("    [ephemeris] building full-calendar snapshot once...", flush=True)
        GLOBAL_CAL_SNAPSHOT = calc_snapshots(calendar_dates)
    else:
        print("    [ephemeris] using cached full-calendar snapshot", flush=True)

    snap = GLOBAL_CAL_SNAPSHOT
    scales = {}
    for system in ("GEO", "HELIO"):
        for p in PLANETS:
            col = f"{system}_{p}_speed"
            med = float(np.nanmedian(np.abs(snap[col].to_numpy(float))))
            scales[col] = med if med > 0 else 1.0
    return scales


# ============================================================
# Feature extraction
# ============================================================

def nearest_harmonic_info(angle):
    j = int(np.argmin(np.abs(HARMONICS-angle)))
    h = float(HARMONICS[j])
    err = float(abs(angle-h))
    return h, err


def feature_frame(snap, tol, speed_scales):
    feats = {}

    for system in ("GEO", "HELIO"):
        # retro/direct + station-like
        for p in PLANETS:
            rcol = f"{system}_{p}_retro"
            scol = f"{system}_{p}_speed"
            retro = snap[rcol].astype(int).to_numpy()
            speed = np.abs(snap[scol].astype(float).to_numpy())
            scale = speed_scales.get(scol, 1.0)

            feats[f"{system}:{p}:RETRO"] = retro == 1
            feats[f"{system}:{p}:DIRECT"] = retro == 0
            feats[f"{system}:{p}:STATIONLIKE"] = speed <= (scale * STATION_RATIO)

        names = list(PLANETS)
        for i in range(len(names)):
            for k in range(i+1, len(names)):
                a, b = names[i], names[k]
                acol = f"{system}_{a}_{b}_angle"
                vcol = f"{system}_{a}_{b}_relspeed"

                ang = snap[acol].astype(float).to_numpy()
                relv = snap[vcol].astype(float).to_numpy()

                for h in HARMONICS:
                    near = np.abs(ang-h) <= tol
                    feats[f"{system}:{a}-{b}:H{h:g}"] = near

                    # Applying / separating around the harmonic:
                    # signed angle error * signed relative speed < 0 => moving toward target.
                    # Because pair angle is folded, this is approximate but useful as a
                    # directional diagnostic, not a standalone selection rule.
                    err = ang-h
                    applying = near & ((err * relv) < 0)
                    separating = near & ((err * relv) > 0)
                    feats[f"{system}:{a}-{b}:H{h:g}:APPLY"] = applying
                    feats[f"{system}:{a}-{b}:H{h:g}:SEP"] = separating

    return pd.DataFrame(feats, index=snap.index)


def prepare_global_feature_cache(calendar_dates, speed_scales):
    """
    Build feature matrices for every tolerance ONCE for every trading day.
    All later random/circular/sector tests become row lookups + NumPy means.
    """
    global GLOBAL_CAL_SNAPSHOT, GLOBAL_FEATURE_CACHE, GLOBAL_DATE_TO_ROW

    if GLOBAL_CAL_SNAPSHOT is None:
        GLOBAL_CAL_SNAPSHOT = calc_snapshots(calendar_dates)

    GLOBAL_CAL_SNAPSHOT = GLOBAL_CAL_SNAPSHOT.sort_values("date").reset_index(drop=True)
    GLOBAL_DATE_TO_ROW = {
        pd.Timestamp(d): i
        for i, d in enumerate(pd.to_datetime(GLOBAL_CAL_SNAPSHOT["date"]))
    }

    for tol in TOLERANCES:
        print(f"    [feature-cache] tolerance ±{tol:g}° ...", flush=True)
        GLOBAL_FEATURE_CACHE[tol] = feature_frame(
            GLOBAL_CAL_SNAPSHOT, tol, speed_scales
        ).astype(np.uint8)

    print(
        f"    [feature-cache] ready: {len(GLOBAL_CAL_SNAPSHOT)} trading dates × "
        f"{len(next(iter(GLOBAL_FEATURE_CACHE.values())).columns)} features",
        flush=True
    )


def rows_for_dates(dates):
    idx = []
    missing = 0
    for d in pd.to_datetime(dates):
        j = GLOBAL_DATE_TO_ROW.get(pd.Timestamp(d))
        if j is None:
            missing += 1
        else:
            idx.append(j)
    if missing:
        print(f"    [cache-warning] {missing} dates not found in calendar cache", flush=True)
    return np.asarray(idx, dtype=int)


# ============================================================
# Baselines
# ============================================================

def year_matched_random_dates(target_dates, calendar_dates, n_trials):
    cal = pd.Series(pd.to_datetime(calendar_dates)).drop_duplicates().sort_values()
    byyear = {int(y): np.array(g.to_list(), dtype="datetime64[ns]") for y, g in cal.groupby(cal.dt.year)}

    tdates = pd.to_datetime(pd.Series(target_dates))
    years = list(tdates.dt.year.astype(int))

    sims = []
    for _ in range(n_trials):
        chosen = []
        for y in years:
            pool = byyear.get(y)
            if pool is None or len(pool) == 0:
                pool = cal.to_numpy(dtype="datetime64[ns]")
            chosen.append(pd.Timestamp(RNG.choice(pool)))
        sims.append(chosen)
    return sims


def circular_shift_dates(target_dates, calendar_dates, n_trials):
    cal = list(pd.to_datetime(pd.Series(calendar_dates).drop_duplicates().sort_values()))
    cmap = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    idx = [cmap[pd.Timestamp(d)] for d in pd.to_datetime(target_dates) if pd.Timestamp(d) in cmap]
    n = len(cal)

    sims = []
    if not idx or n == 0:
        return sims

    for _ in range(n_trials):
        shift = int(RNG.integers(1, max(2, n-1)))
        sims.append([cal[(i+shift) % n] for i in idx])
    return sims


def prevalence_on_dates(feature, dates, snapshot_cache, tol, speed_scales):
    key = tuple(pd.to_datetime(dates).astype(str))
    if key not in snapshot_cache:
        snapshot_cache[key] = calc_snapshots(pd.to_datetime(dates))
    ff = feature_frame(snapshot_cache[key], tol, speed_scales)
    if feature not in ff:
        return 0.0
    return float(ff[feature].mean())


# ============================================================
# Full statistical testing
# ============================================================

def test_features(target_windows, calendar_dates, tol, speed_scales, label):
    """
    Fast V1.1 implementation.
    Swiss Ephemeris is NOT recalculated here.
    Everything is sampled from precomputed full-calendar feature matrices.
    """
    if len(target_windows) == 0:
        return pd.DataFrame()

    feat = GLOBAL_FEATURE_CACHE[tol]
    feature_names = list(feat.columns)
    target_idx = rows_for_dates(target_windows["date"])
    if len(target_idx) == 0:
        return pd.DataFrame()

    target_arr = feat.iloc[target_idx].to_numpy(np.uint8)
    obs = target_arr.mean(axis=0)
    hits = target_arr.sum(axis=0).astype(int)

    cal_dates = pd.to_datetime(pd.Series(calendar_dates)).drop_duplicates().sort_values().reset_index(drop=True)
    cal_rows = rows_for_dates(cal_dates)
    row_by_year = {}
    for year, positions in cal_dates.groupby(cal_dates.dt.year).groups.items():
        row_by_year[int(year)] = cal_rows[np.asarray(list(positions), dtype=int)]

    target_years = pd.to_datetime(target_windows["date"]).dt.year.astype(int).tolist()
    n_feat = len(feature_names)

    print(
        f"    [{label} | ±{tol:g}°] features={n_feat} windows={len(target_idx)} "
        f"year-random={RANDOM_TRIALS} circular={CIRCULAR_TRIALS}",
        flush=True
    )

    # ---- Year-matched random baseline ----
    yr_sum = np.zeros(n_feat, dtype=float)
    yr_ge = np.zeros(n_feat, dtype=np.int32)
    yr_values = np.empty((RANDOM_TRIALS, n_feat), dtype=np.float32)

    for trial in range(RANDOM_TRIALS):
        sample_rows = []
        for y in target_years:
            pool = row_by_year.get(y)
            if pool is None or len(pool) == 0:
                pool = cal_rows
            sample_rows.append(int(RNG.choice(pool)))
        rate = feat.iloc[sample_rows].to_numpy(np.uint8).mean(axis=0)
        yr_values[trial] = rate
        yr_sum += rate
        yr_ge += (rate >= obs)
        if (trial + 1) % 500 == 0 or trial + 1 == RANDOM_TRIALS:
            print(
                f"      year-random {trial+1}/{RANDOM_TRIALS}",
                flush=True
            )

    ymean = yr_sum / RANDOM_TRIALS
    yp = (1.0 + yr_ge) / (1.0 + RANDOM_TRIALS)
    ylo = np.percentile(yr_values, 2.5, axis=0)
    yhi = np.percentile(yr_values, 97.5, axis=0)

    # ---- Circular-shift baseline ----
    # Preserve the spacing of target dates in trading-session coordinates.
    base_positions = target_idx.copy()
    ncal = len(feat)

    cr_sum = np.zeros(n_feat, dtype=float)
    cr_ge = np.zeros(n_feat, dtype=np.int32)
    cr_values = np.empty((CIRCULAR_TRIALS, n_feat), dtype=np.float32)

    for trial in range(CIRCULAR_TRIALS):
        shift = int(RNG.integers(1, max(2, ncal - 1)))
        sample_rows = (base_positions + shift) % ncal
        rate = feat.iloc[sample_rows].to_numpy(np.uint8).mean(axis=0)
        cr_values[trial] = rate
        cr_sum += rate
        cr_ge += (rate >= obs)
        if (trial + 1) % 500 == 0 or trial + 1 == CIRCULAR_TRIALS:
            print(
                f"      circular {trial+1}/{CIRCULAR_TRIALS}",
                flush=True
            )

    cmean = cr_sum / CIRCULAR_TRIALS
    cp = (1.0 + cr_ge) / (1.0 + CIRCULAR_TRIALS)
    clo = np.percentile(cr_values, 2.5, axis=0)
    chi = np.percentile(cr_values, 97.5, axis=0)

    baseline = np.maximum(ymean, cmean)
    with np.errstate(divide="ignore", invalid="ignore"):
        enrich = np.where(baseline > 0, obs / baseline, np.nan)

    conservative_p = np.maximum(yp, cp)

    out = pd.DataFrame({
        "sample": label,
        "tolerance_deg": tol,
        "feature": feature_names,
        "n_windows": len(target_idx),
        "hits": hits,
        "observed_pct": 100 * obs,
        "year_random_mean_pct": 100 * ymean,
        "circular_mean_pct": 100 * cmean,
        "conservative_baseline_pct": 100 * baseline,
        "enrichment": enrich,
        "year_random_p": yp,
        "circular_p": cp,
        "conservative_p": conservative_p,
        "year_random_ci_low_pct": 100 * ylo,
        "year_random_ci_high_pct": 100 * yhi,
        "circular_ci_low_pct": 100 * clo,
        "circular_ci_high_pct": 100 * chi,
    })

    out["fdr_q"] = bh_fdr(out["conservative_p"].fillna(1.0).to_numpy())
    return out.sort_values(
        ["fdr_q", "conservative_p", "enrichment", "hits"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)


def chronological_split(w):
    """
    60% discovery, 20% validation, 20% holdout by DATE ORDER.
    Done separately for HIGH and LOW downstream.
    """
    w = w.sort_values("date").reset_index(drop=True)
    n = len(w)
    if n < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    a = max(4, int(round(n*0.60)))
    b = max(a+2, int(round(n*0.80)))
    b = min(b, n-2)

    return (
        w.iloc[:a].copy(),
        w.iloc[a:b].copy(),
        w.iloc[b:].copy(),
    )


def candidate_from_discovery(stats):
    return stats[
        (stats["hits"] >= MIN_DISCOVERY_HITS) &
        (stats["enrichment"] >= MIN_DISCOVERY_ENRICHMENT) &
        (stats["conservative_p"] <= DISCOVERY_P_MAX)
    ].copy()


def validate_candidate_features(candidates, sample_windows, calendar_dates, tol, speed_scales, label):
    if len(candidates) == 0 or len(sample_windows) == 0:
        return pd.DataFrame()

    all_stats = test_features(sample_windows, calendar_dates, tol, speed_scales, label)
    keep = all_stats[all_stats["feature"].isin(candidates["feature"])].copy()
    return keep


# ============================================================
# Placebo harmonic test
# ============================================================

def harmonic_placebo_test(target_windows, calendar_dates, speed_scales, tol=2.0):
    if len(target_windows) == 0:
        return {}

    idx = rows_for_dates(target_windows["date"])
    snap = GLOBAL_CAL_SNAPSHOT.iloc[idx].reset_index(drop=True)
    pair_cols = [c for c in snap.columns if c.endswith("_angle")]
    arr = snap[pair_cols].to_numpy(float)

    def coverage(grid):
        # window counts as covered if ANY pair is close to ANY grid angle
        d = np.abs(arr[:, :, None] - np.asarray(grid)[None, None, :])
        return float(np.mean(np.min(d, axis=(1, 2)) <= tol))

    actual = coverage(HARMONICS)
    sims = np.empty(PLACEBO_TRIALS, dtype=float)

    print(
        f"    [placebo] windows={len(idx)} trials={PLACEBO_TRIALS}",
        flush=True
    )
    for i in range(PLACEBO_TRIALS):
        shift = float(RNG.uniform(0, 15.0))
        grid = (HARMONICS + shift) % 180.0
        grid = np.where(grid == 0, 180.0, grid)
        sims[i] = coverage(grid)
        if (i + 1) % 250 == 0 or i + 1 == PLACEBO_TRIALS:
            print(f"      placebo {i+1}/{PLACEBO_TRIALS}", flush=True)

    p = (1 + np.sum(sims >= actual)) / (1 + len(sims))
    return {
        "actual_coverage_pct": 100 * actual,
        "placebo_mean_pct": 100 * float(np.mean(sims)),
        "placebo_p": float(p),
    }


# ============================================================
# Sector DNA
# ============================================================

def sector_dna(sector_windows, calendar_dates, speed_scales):
    rows = []
    eligible_jobs = []
    for sec, sw in sector_windows.groupby("sector"):
        for typ in ("LOW", "HIGH"):
            x = sw[sw["dominant_turn"] == typ].copy()
            if len(x) >= MIN_SECTOR_WINDOWS:
                eligible_jobs.append((sec, typ, x))

    print(f"    [sector-DNA] eligible jobs: {len(eligible_jobs)}", flush=True)

    for j, (sec, typ, x) in enumerate(eligible_jobs, 1):
        print(
            f"    [sector-DNA {j}/{len(eligible_jobs)}] {sec} {typ} windows={len(x)}",
            flush=True
        )
        stats = test_features(
            x, calendar_dates, 2.0, speed_scales, f"SECTOR:{sec}:{typ}"
        )
        if stats.empty:
            continue

        top = stats[
            (stats["hits"] >= 3) &
            (stats["enrichment"] >= 1.20) &
            (stats["conservative_p"] <= 0.10)
        ].head(25).copy()

        if len(top):
            top["sector"] = sec
            top["turn_type"] = typ
            rows.append(top)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# Main
# ============================================================

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/10] Loading adjusted EGX stocks...")
    stocks = load_stocks(args.max_symbols)
    index_df, index_source = load_index()
    sectors = fetch_sector_map()

    print("[2/10] Detecting major pivots...")
    pframes = []
    ranges = []

    for k, (sym, df) in enumerate(stocks.items(), 1):
        p = detect_pivots(df)
        sec = sectors.get(base_symbol(sym), "UNKNOWN")

        if len(p):
            p["symbol"] = sym
            p["sector"] = sec
            pframes.append(p)

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

    print("[3/10] Detecting index/proxy pivots...")
    idxp = detect_pivots(index_df)
    idxp["symbol"] = "EGX_MARKET_INDEX"
    idxp["sector"] = "INDEX"

    print("[4/10] Building fixed HIGH/LOW windows...")
    cal = market_calendar(stocks, index_df)
    pivots = add_cal_index(pivots, cal)
    idxp = add_cal_index(idxp, cal)

    sector_windows, market_windows = build_windows(
        pivots, ranges, idxp, cal, index_source
    )

    tier_a = market_windows[market_windows["tier_a"] == 1].copy()

    print("[5/10] Computing ONE full-calendar ephemeris + speed normalization...")
    speed_scales = compute_speed_scales(cal["date"])
    prepare_global_feature_cache(cal["date"], speed_scales)

    print("[6/10] Pulling Tier-A planetary snapshots from cache...")
    tier_rows = rows_for_dates(tier_a["date"])
    tier_snap = GLOBAL_CAL_SNAPSHOT.iloc[tier_rows].reset_index(drop=True) if len(tier_rows) else pd.DataFrame()

    print("[7/10] Discovery -> validation -> holdout market DNA...")
    all_stats = []
    validation_rows = []
    final_rows = []

    for typ in ("LOW", "HIGH"):
        tw = tier_a[tier_a["dominant_turn"] == typ].sort_values("date").copy()
        disc, val, hold = chronological_split(tw)

        if len(disc) == 0:
            print(f"  {typ}: not enough windows for 60/20/20 split")
            continue

        print(f"  {typ}: total={len(tw)} discovery={len(disc)} validation={len(val)} holdout={len(hold)}")

        for tol in TOLERANCES:
            print(f"\n    >>> {typ} tolerance ±{tol:g}° : DISCOVERY", flush=True)
            ds = test_features(disc, cal["date"], tol, speed_scales, f"{typ}:DISCOVERY")
            ds["turn_type"] = typ
            all_stats.append(ds)

            cand = candidate_from_discovery(ds)
            print(f"    >>> {typ} tolerance ±{tol:g}° : VALIDATION candidates={len(cand)}", flush=True)
            vs = validate_candidate_features(cand, val, cal["date"], tol, speed_scales, f"{typ}:VALIDATION")
            if len(vs):
                vs["turn_type"] = typ
                vs["stage"] = "VALIDATION"
                validation_rows.append(vs)

                # Must survive validation to reach holdout
                survivor = vs[
                    (vs["hits"] >= 2) &
                    (vs["enrichment"] >= 1.10) &
                    (vs["conservative_p"] <= 0.25)
                ].copy()

                print(f"    >>> {typ} tolerance ±{tol:g}° : HOLDOUT survivors={len(survivor)}", flush=True)
                hs = validate_candidate_features(
                    survivor, hold, cal["date"], tol, speed_scales, f"{typ}:HOLDOUT"
                )

                if len(hs):
                    hs["turn_type"] = typ
                    hs["stage"] = "HOLDOUT"
                    hs["validated"] = (
                        (hs["hits"] >= HOLDOUT_MIN_HITS) &
                        (hs["enrichment"] >= HOLDOUT_ENRICHMENT) &
                        (hs["conservative_p"] <= HOLDOUT_P_MAX)
                    ).astype(int)
                    final_rows.append(hs)

    discovery_stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    validation_stats = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    holdout_stats = pd.concat(final_rows, ignore_index=True) if final_rows else pd.DataFrame()

    print("[8/10] Robustness / stability across tolerances...")
    stability_rows = []
    if len(holdout_stats):
        v = holdout_stats[holdout_stats["validated"] == 1].copy()
        for (typ, feat), g in v.groupby(["turn_type", "feature"]):
            tolerances = sorted(set(g["tolerance_deg"].astype(float)))
            stability_rows.append({
                "turn_type": typ,
                "feature": feat,
                "validated_tolerances": ",".join(str(x) for x in tolerances),
                "n_validated_tolerances": len(tolerances),
                "mean_holdout_enrichment": float(g["enrichment"].mean()),
                "worst_holdout_p": float(g["conservative_p"].max()),
                "stable_final": int(len(tolerances) >= 2),
            })
    stability = pd.DataFrame(stability_rows)

    print("[9/10] Harmonic placebo + sector DNA...")
    placebo = {}
    for typ in ("LOW", "HIGH"):
        x = tier_a[tier_a["dominant_turn"] == typ]
        placebo[typ] = harmonic_placebo_test(x, cal["date"], speed_scales, 2.0)

    s_dna = sector_dna(sector_windows, cal["date"], speed_scales)

    print("[10/10] Writing audit outputs...")
    ranges.to_csv(out/"asset_ranges_adjusted.csv", index=False)
    pivots.to_csv(out/"all_stock_major_pivots_adjusted.csv", index=False)
    idxp.to_csv(out/"index_major_pivots.csv", index=False)
    sector_windows.to_csv(out/"sector_turn_windows.csv", index=False)
    market_windows.to_csv(out/"market_turn_windows.csv", index=False)
    tier_a.to_csv(out/"TIER_A_MARKET_WINDOWS.csv", index=False)
    tier_snap.to_csv(out/"tier_a_planetary_snapshots.csv", index=False)
    discovery_stats.to_csv(out/"discovery_stats_all_tolerances.csv", index=False)
    validation_stats.to_csv(out/"validation_stats.csv", index=False)
    holdout_stats.to_csv(out/"holdout_stats.csv", index=False)
    stability.to_csv(out/"FINAL_STABLE_MARKET_DNA.csv", index=False)
    s_dna.to_csv(out/"sector_dna_diagnostics.csv", index=False)

    placebo_df = pd.DataFrame([
        {"turn_type": k, **v} for k, v in placebo.items()
    ])
    placebo_df.to_csv(out/"harmonic_placebo_test.csv", index=False)

    # Final "pass" features: validated in >=2 tolerances.
    final_pass = stability[stability["stable_final"] == 1].copy() if len(stability) else pd.DataFrame()
    final_pass.to_csv(out/"FINAL_PASS_FEATURES.csv", index=False)

    report = {
        "index_source": index_source,
        "index_independent": int("SYNTHETIC" not in index_source.upper()),
        "stocks_loaded": len(stocks),
        "stock_major_pivots": len(pivots),
        "sector_windows": len(sector_windows),
        "market_windows": len(market_windows),
        "tier_a_windows": len(tier_a),
        "tier_a_low": int((tier_a["dominant_turn"] == "LOW").sum()) if len(tier_a) else 0,
        "tier_a_high": int((tier_a["dominant_turn"] == "HIGH").sum()) if len(tier_a) else 0,
        "holdout_validated_rows": int((holdout_stats["validated"] == 1).sum()) if len(holdout_stats) else 0,
        "stable_final_features": len(final_pass),
        "low_harmonic_placebo": placebo.get("LOW", {}),
        "high_harmonic_placebo": placebo.get("HIGH", {}),
        "note": (
            "Synthetic index confirmation is diagnostic only and does not strengthen "
            "Tier-A selection or statistical score."
        )
    }
    (out/"run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    print("\n=== JENKINS / EGX FINAL EXPERIMENT V1.1 FAST ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\n=== FINAL STABLE MARKET DNA ===")
    if len(final_pass) == 0:
        print("NONE")
    else:
        print(
            final_pass.sort_values(
                ["turn_type", "n_validated_tolerances", "mean_holdout_enrichment"],
                ascending=[True, False, False]
            ).to_string(index=False)
        )

    print("\n=== HARMONIC PLACEBO TEST ===")
    print(placebo_df.to_string(index=False))

    print("\nMost important output:")
    print(out/"FINAL_PASS_FEATURES.csv")
    print(out/"TIER_A_MARKET_WINDOWS.csv")
    print(out/"harmonic_placebo_test.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/app/egx_jenkins_final_experiment_v1_1_fast")
    p.add_argument("--max-symbols", type=int, default=None)
    main(p.parse_args())
