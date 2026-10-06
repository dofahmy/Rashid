#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jenkins Stock DNA Engine V1.1
===========================

Goal
----
Learn a stock-specific planetary fingerprint ("DNA") from the EARLY part of the chart,
freeze it, then test it on the LATER unseen part of the chart.

Default symbol: COMI.CA (Commercial International Bank - Egypt)

Method
------
1) Download/load daily OHLCV.
2) Split history:
      first 35% = DISCOVERY
      last 65%  = HOLDOUT
3) Detect major pivots with fixed rules.
4) Pick an early principal LOW and HIGH as origin anchors.
5) Compute GEO + HELIO planetary positions, speeds, retrograde/direct state.
6) Generate stock-specific candidate rules from the anchor configurations:
      - planet cumulative-motion harmonics from origin
      - pair-angle offsets from origin
      - pair-angle station/reversal behavior
7) Score candidate rules ONLY inside DISCOVERY.
8) Freeze the best rules into stock_planetary_fingerprint.json.
9) Apply those exact rules unchanged to HOLDOUT.
10) Compare holdout hits with random/circular-shift baselines.

This is an experimental research engine, not financial advice.

Dependencies
------------
pip install pandas numpy yfinance pyswisseph

Example
-------
python jenkins_stock_dna_v1.py \
  --symbol COMI.CA \
  --output-dir /app/jenkins_comi_dna_v1

Optional local CSV
------------------
python jenkins_stock_dna_v1.py \
  --input /app/COMI.csv \
  --output-dir /app/jenkins_comi_dna_v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import swisseph as swe
except ImportError as exc:
    raise SystemExit("Missing pyswisseph. Install: pip install pyswisseph") from exc


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    discovery_fraction: float = 0.35

    # Pivot definition
    pivot_left: int = 20
    pivot_right: int = 20
    min_pivot_move_pct: float = 10.0
    min_pivot_spacing: int = 15

    # Signal evaluation
    hit_window_sessions: int = 3
    cluster_window_sessions: int = 2

    # Candidate filtering / fingerprint
    min_discovery_events_per_rule: int = 3
    min_discovery_hits_per_rule: int = 2
    top_rules: int = 12

    # Jenkins-style harmonics tested around each stock's origin
    harmonics: Tuple[float, ...] = (
        7.5, 15.0, 22.5, 30.0, 45.0, 60.0,
        72.0, 90.0, 120.0, 135.0, 144.0, 180.0,
        270.0, 360.0
    )

    planets: Tuple[str, ...] = (
        "MERCURY", "VENUS", "MARS", "JUPITER",
        "SATURN", "URANUS", "NEPTUNE", "PLUTO"
    )

    random_trials: int = 5000
    random_seed: int = 1729


PLANET_IDS = {
    "MERCURY": swe.MERCURY,
    "VENUS": swe.VENUS,
    "MARS": swe.MARS,
    "JUPITER": swe.JUPITER,
    "SATURN": swe.SATURN,
    "URANUS": swe.URANUS,
    "NEPTUNE": swe.NEPTUNE,
    "PLUTO": swe.PLUTO,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def signed_delta(prev: float, curr: float) -> float:
    return ((curr - prev + 180.0) % 360.0) - 180.0


def angular_distance(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def circular_signed_offset(current: float, origin: float) -> float:
    return ((current - origin + 180.0) % 360.0) - 180.0


def circular_abs_offset(current: float, origin: float) -> float:
    return abs(circular_signed_offset(current, origin))


def crossed(prev: float, curr: float, target: float) -> bool:
    if not (np.isfinite(prev) and np.isfinite(curr)):
        return False
    if prev == curr:
        return False
    lo, hi = sorted((prev, curr))
    return lo < target <= hi


def pct(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a == 0:
        return np.nan
    return (b / a - 1.0) * 100.0


def jd(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    return swe.julday(ts.year, ts.month, ts.day, 12.0)


# ============================================================
# DATA
# ============================================================

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    cols = {str(c).strip().lower(): c for c in df.columns}

    date_col = cols.get("date") or cols.get("datetime")
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        cols = {str(c).strip().lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("datetime")

    if date_col is None:
        # yfinance sometimes names reset index "Date"
        for c in df.columns:
            if str(c).lower().startswith("date"):
                date_col = c
                break

    if date_col is None:
        raise ValueError("Could not find date column.")

    rename = {date_col: "date"}
    for k in ("open","high","low","close","volume","adj close"):
        if k in cols:
            rename[cols[k]] = "adj_close" if k == "adj close" else k

    df = df.rename(columns=rename).copy()

    if "close" not in df.columns:
        raise ValueError("No close column found.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    for c in ("open","high","low","close","adj_close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "open" not in df: df["open"] = df["close"]
    if "high" not in df: df["high"] = df["close"]
    if "low" not in df: df["low"] = df["close"]

    # IMPORTANT: long-history stock studies must use split/dividend-adjusted prices.
    # Yahoo's raw OHLC can jump after capital actions and create fake pivots/DNA.
    # Rebuild adjusted OHLC from the Adj Close / Close factor when available.
    if "adj_close" in df.columns:
        fac = df["adj_close"] / df["close"]
        fac = fac.replace([np.inf, -np.inf], np.nan)
        good = fac.notna() & (fac > 0)
        if good.any():
            df.loc[good, "open"] = df.loc[good, "open"] * fac[good]
            df.loc[good, "high"] = df.loc[good, "high"] * fac[good]
            df.loc[good, "low"] = df.loc[good, "low"] * fac[good]
            df.loc[good, "close"] = df.loc[good, "adj_close"]

    df = (
        df.dropna(subset=["date","open","high","low","close"])
          .sort_values("date")
          .drop_duplicates("date", keep="last")
          .reset_index(drop=True)
    )
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


def download_yahoo(symbol: str, cache_dir: Path) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Missing yfinance. Install: pip install yfinance") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol.replace('.','_')}_max.csv"

    if cache.exists():
        try:
            x = pd.read_csv(cache)
            x = normalize_ohlc(x)
            if len(x) >= 250:
                print(f"Loaded cache: {cache} ({len(x)} rows)")
                return x
        except Exception as e:
            print("Cache ignored:", e)

    print(f"Downloading {symbol} full daily history from Yahoo...")
    raw = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No Yahoo data returned for {symbol}")

    raw = raw.reset_index()
    out = normalize_ohlc(raw)
    out.to_csv(cache, index=False)
    print(f"Saved cache: {cache}")
    return out


def load_data(input_path: Optional[str], symbol: str, output_dir: Path) -> pd.DataFrame:
    if input_path:
        return normalize_ohlc(pd.read_csv(input_path))
    return download_yahoo(symbol, output_dir / "cache")


# ============================================================
# PIVOTS
# ============================================================

def raw_local_pivots(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    rows = []

    for i in range(cfg.pivot_left, len(df)-cfg.pivot_right):
        lh = highs[i-cfg.pivot_left:i]
        rh = highs[i+1:i+1+cfg.pivot_right]
        ll = lows[i-cfg.pivot_left:i]
        rl = lows[i+1:i+1+cfg.pivot_right]

        if highs[i] >= np.max(lh) and highs[i] > np.max(rh):
            rows.append({
                "index": i, "date": df.loc[i,"date"], "type": "HIGH",
                "price": highs[i]
            })

        if lows[i] <= np.min(ll) and lows[i] < np.min(rl):
            rows.append({
                "index": i, "date": df.loc[i,"date"], "type": "LOW",
                "price": lows[i]
            })

    if not rows:
        return pd.DataFrame(columns=["index","date","type","price"])

    p = pd.DataFrame(rows).sort_values(["index","type"]).reset_index(drop=True)

    # Collapse consecutive same-type pivots: keep more extreme.
    collapsed = []
    for _, r in p.iterrows():
        if not collapsed or collapsed[-1]["type"] != r["type"]:
            collapsed.append(r.to_dict())
        else:
            prev = collapsed[-1]
            better = (
                (r["type"] == "HIGH" and r["price"] > prev["price"]) or
                (r["type"] == "LOW" and r["price"] < prev["price"])
            )
            if better:
                collapsed[-1] = r.to_dict()

    # Enforce minimum swing size and spacing.
    clean = []
    for r in collapsed:
        if not clean:
            clean.append(r)
            continue

        prev = clean[-1]
        spacing = int(r["index"]) - int(prev["index"])
        move = abs(pct(float(prev["price"]), float(r["price"])))

        if spacing < cfg.min_pivot_spacing:
            # Keep the more extreme one only if same type; otherwise skip too-close reversal.
            continue

        if move < cfg.min_pivot_move_pct:
            continue

        clean.append(r)

    return pd.DataFrame(clean)


def choose_origin_anchors(
    pivots: pd.DataFrame,
    discovery_end_index: int
) -> Tuple[dict, dict]:
    """
    Choose early LOW + HIGH anchors from the first half of discovery,
    not from the whole history.
    """
    if pivots.empty:
        raise RuntimeError("No pivots found.")

    early_limit = max(1, int(discovery_end_index * 0.55))
    early = pivots[pivots["index"] <= early_limit].copy()

    lows = early[early["type"]=="LOW"]
    highs = early[early["type"]=="HIGH"]

    if lows.empty or highs.empty:
        # Relax to all discovery if needed.
        disc = pivots[pivots["index"] <= discovery_end_index]
        lows = disc[disc["type"]=="LOW"]
        highs = disc[disc["type"]=="HIGH"]

    if lows.empty or highs.empty:
        raise RuntimeError("Could not find both an early major LOW and HIGH.")

    # Choose the strongest early opposite-pivot swing, using ONLY the early
    # discovery segment. This is closer to "principal high + principal low"
    # than simply taking the first LOW and first HIGH encountered.
    early2 = early.sort_values("index").reset_index(drop=True)
    best_pair = None
    best_move = -1.0

    for i in range(len(early2)):
        for j in range(i + 1, len(early2)):
            a = early2.iloc[i]
            b = early2.iloc[j]
            if a["type"] == b["type"]:
                continue
            move = abs(pct(float(a["price"]), float(b["price"])))
            if np.isfinite(move) and move > best_move:
                best_move = move
                best_pair = (a.to_dict(), b.to_dict())

    if best_pair is None:
        low = lows.iloc[0].to_dict()
        high = highs.iloc[0].to_dict()
    else:
        a, b = best_pair
        low = a if a["type"] == "LOW" else b
        high = a if a["type"] == "HIGH" else b

    return low, high


# ============================================================
# EPHEMERIS
# ============================================================

def calc_position(julian_day: float, planet_id: int, helio: bool) -> Tuple[float,float]:
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED
    if helio:
        flags |= swe.FLG_HELCTR

    xx, _ = swe.calc_ut(julian_day, planet_id, flags)
    return float(xx[0]) % 360.0, float(xx[3])


def build_ephemeris(dates: Sequence[pd.Timestamp], cfg: Config) -> pd.DataFrame:
    rows = []

    for d in dates:
        j = jd(pd.Timestamp(d))
        r = {"date": pd.Timestamp(d)}

        for system, helio in (("GEO",False),("HELIO",True)):
            for p in cfg.planets:
                lon, speed = calc_position(j, PLANET_IDS[p], helio)
                r[f"{system}_{p}_lon"] = lon
                r[f"{system}_{p}_speed"] = speed
                r[f"{system}_{p}_retro"] = int(speed < 0)

        rows.append(r)

    e = pd.DataFrame(rows)

    # cumulative signed / absolute longitudinal motion per planet/system
    for system in ("GEO","HELIO"):
        for p in cfg.planets:
            arr = e[f"{system}_{p}_lon"].to_numpy(float)
            delta = np.zeros(len(arr), dtype=float)
            for i in range(1,len(arr)):
                delta[i] = signed_delta(arr[i-1], arr[i])
            e[f"{system}_{p}_delta"] = delta
            e[f"{system}_{p}_cum_signed"] = np.cumsum(delta)
            e[f"{system}_{p}_cum_abs"] = np.cumsum(np.abs(delta))

    return e


# ============================================================
# ORIGIN SNAPSHOT
# ============================================================

def make_anchor_snapshot(anchor: dict, eph: pd.DataFrame, cfg: Config) -> dict:
    i = int(anchor["index"])
    er = eph.iloc[i]

    snap = {
        "date": str(pd.Timestamp(anchor["date"]).date()),
        "type": anchor["type"],
        "price": float(anchor["price"]),
        "index": i,
        "planets": {},
        "pairs": {},
    }

    for system in ("GEO","HELIO"):
        snap["planets"][system] = {}
        for p in cfg.planets:
            snap["planets"][system][p] = {
                "lon": float(er[f"{system}_{p}_lon"]),
                "speed": float(er[f"{system}_{p}_speed"]),
                "retrograde": bool(er[f"{system}_{p}_retro"]),
                "cum_signed": float(er[f"{system}_{p}_cum_signed"]),
                "cum_abs": float(er[f"{system}_{p}_cum_abs"]),
            }

        snap["pairs"][system] = {}
        for a in range(len(cfg.planets)):
            for b in range(a+1,len(cfg.planets)):
                p1,p2 = cfg.planets[a],cfg.planets[b]
                sep = angular_distance(
                    float(er[f"{system}_{p1}_lon"]),
                    float(er[f"{system}_{p2}_lon"])
                )
                snap["pairs"][system][f"{p1}-{p2}"] = float(sep)

    return snap


# ============================================================
# RULE GENERATION
# ============================================================

def build_candidate_rules(
    anchors: List[dict],
    eph: pd.DataFrame,
    cfg: Config
) -> List[dict]:
    """
    Each origin anchor creates stock-specific rules.

    A) PLANET_MOTION:
       cumulative signed degrees from that exact anchor crossing ±harmonic levels.

    B) PAIR_OFFSET:
       future pair separation crossing origin_pair_angle ± harmonic.
       This is the "117 + 30 / 45 / 90..." Jenkins stock-specific idea.
    """
    rules = []

    for anchor in anchors:
        ai = int(anchor["index"])
        anchor_id = f'{anchor["type"]}_{pd.Timestamp(anchor["date"]).date()}'
        er = eph.iloc[ai]

        for system in ("GEO","HELIO"):
            # individual planet movement
            for p in cfg.planets:
                for h in cfg.harmonics:
                    for direction in (+1,-1):
                        rules.append({
                            "rule_id": f"{anchor_id}|{system}|PLANET_MOTION|{p}|{direction*h:g}",
                            "anchor_id": anchor_id,
                            "anchor_index": ai,
                            "anchor_type": anchor["type"],
                            "anchor_date": str(pd.Timestamp(anchor["date"]).date()),
                            "system": system,
                            "family": "PLANET_MOTION",
                            "planet_1": p,
                            "planet_2": None,
                            "origin_value": 0.0,
                            "offset_deg": float(direction*h),
                            "target_value": float(direction*h),
                        })

            # pair-angle offsets from stock-specific origin angle
            for a in range(len(cfg.planets)):
                for b in range(a+1,len(cfg.planets)):
                    p1,p2 = cfg.planets[a],cfg.planets[b]
                    origin = angular_distance(
                        float(er[f"{system}_{p1}_lon"]),
                        float(er[f"{system}_{p2}_lon"])
                    )

                    for h in cfg.harmonics:
                        # pair separation itself is 0..180, so generate valid folded targets.
                        raw_targets = [
                            origin + h,
                            origin - h,
                        ]
                        seen = set()
                        for raw in raw_targets:
                            t = raw % 360.0
                            if t > 180.0:
                                t = 360.0 - t
                            t = round(float(t), 6)
                            if t in seen:
                                continue
                            seen.add(t)

                            rules.append({
                                "rule_id": f"{anchor_id}|{system}|PAIR_OFFSET|{p1}-{p2}|orig={origin:.4f}|h={h:g}|target={t:.4f}",
                                "anchor_id": anchor_id,
                                "anchor_index": ai,
                                "anchor_type": anchor["type"],
                                "anchor_date": str(pd.Timestamp(anchor["date"]).date()),
                                "system": system,
                                "family": "PAIR_OFFSET",
                                "planet_1": p1,
                                "planet_2": p2,
                                "origin_value": float(origin),
                                "offset_deg": float(h),
                                "target_value": float(t),
                            })

    return rules


# ============================================================
# RULE EVENTS
# ============================================================

def events_for_rule(rule: dict, eph: pd.DataFrame, start_idx: int, end_idx: int) -> List[int]:
    start_idx = max(start_idx, int(rule["anchor_index"])+1)
    end_idx = min(end_idx, len(eph)-1)
    if start_idx > end_idx:
        return []

    system = rule["system"]
    fam = rule["family"]
    events = []

    if fam == "PLANET_MOTION":
        p = rule["planet_1"]
        base = float(eph.iloc[int(rule["anchor_index"])][f"{system}_{p}_cum_signed"])
        target = float(rule["target_value"])

        prev = float(eph.iloc[start_idx-1][f"{system}_{p}_cum_signed"]) - base
        for i in range(start_idx,end_idx+1):
            curr = float(eph.iloc[i][f"{system}_{p}_cum_signed"]) - base
            if crossed(prev,curr,target):
                events.append(i)
            prev = curr

    elif fam == "PAIR_OFFSET":
        p1,p2 = rule["planet_1"],rule["planet_2"]
        target = float(rule["target_value"])

        def sep(i):
            return angular_distance(
                float(eph.iloc[i][f"{system}_{p1}_lon"]),
                float(eph.iloc[i][f"{system}_{p2}_lon"])
            )

        prev = sep(start_idx-1)
        for i in range(start_idx,end_idx+1):
            curr = sep(i)
            if crossed(prev,curr,target):
                events.append(i)
            prev = curr

    return events


# ============================================================
# SCORING
# ============================================================

def period_pivot_indices(pivots: pd.DataFrame, start_idx: int, end_idx: int) -> np.ndarray:
    x = pivots[(pivots["index"]>=start_idx)&(pivots["index"]<=end_idx)]
    return x["index"].astype(int).to_numpy()


def near_any_pivot(i: int, piv_idx: np.ndarray, window: int) -> bool:
    return len(piv_idx)>0 and int(np.min(np.abs(piv_idx-i))) <= window


def baseline_date_probability(start_idx: int, end_idx: int, piv_idx: np.ndarray, window: int) -> float:
    if end_idx < start_idx:
        return 0.0
    n = end_idx-start_idx+1
    hit = sum(near_any_pivot(i,piv_idx,window) for i in range(start_idx,end_idx+1))
    return hit/n if n else 0.0


def score_rule(
    rule: dict,
    eph: pd.DataFrame,
    pivots: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    cfg: Config
) -> dict:
    events = events_for_rule(rule,eph,start_idx,end_idx)
    piv_idx = period_pivot_indices(pivots,start_idx,end_idx)
    base = baseline_date_probability(start_idx,end_idx,piv_idx,cfg.hit_window_sessions)

    hits = [i for i in events if near_any_pivot(i,piv_idx,cfg.hit_window_sessions)]
    hit_rate = len(hits)/len(events) if events else 0.0

    covered = set()
    for i in hits:
        if len(piv_idx):
            j = int(np.argmin(np.abs(piv_idx-i)))
            covered.add(int(piv_idx[j]))

    enrichment = hit_rate/base if base>0 else 0.0

    # Conservative DNA score: reward precision, enrichment and unique-pivot coverage.
    score = (
        100.0*hit_rate
        + 12.0*math.log1p(max(enrichment,0.0))
        + 4.0*len(covered)
        - 0.15*max(0,len(events)-20)
    )

    out = dict(rule)
    out.update({
        "discovery_events": len(events),
        "discovery_hits": len(hits),
        "discovery_hit_rate_pct": 100.0*hit_rate,
        "discovery_baseline_pct": 100.0*base,
        "discovery_enrichment": enrichment,
        "discovery_unique_pivots_covered": len(covered),
        "dna_score": score,
    })
    return out


def select_fingerprint(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if scored.empty:
        return scored

    q = scored[
        (scored["discovery_events"] >= cfg.min_discovery_events_per_rule) &
        (scored["discovery_hits"] >= cfg.min_discovery_hits_per_rule) &
        (scored["discovery_enrichment"] > 1.0)
    ].copy()

    if q.empty:
        # Do not silently optimize thresholds; return best candidates transparently.
        q = scored[scored["discovery_events"] >= cfg.min_discovery_events_per_rule].copy()

    q = q.sort_values(
        ["dna_score","discovery_unique_pivots_covered","discovery_hit_rate_pct"],
        ascending=[False,False,False]
    )

    # Diversity cap: don't let one identical pair/planet dominate all top rules.
    chosen = []
    family_key_counts = {}

    for _,r in q.iterrows():
        key = (r["system"],r["family"],r["planet_1"],r["planet_2"])
        count = family_key_counts.get(key,0)
        if count >= 2:
            continue
        chosen.append(r.to_dict())
        family_key_counts[key] = count+1
        if len(chosen) >= cfg.top_rules:
            break

    return pd.DataFrame(chosen)


# ============================================================
# HOLDOUT SIGNALS + CLUSTERING
# ============================================================

def collect_rule_events(
    fingerprint: pd.DataFrame,
    eph: pd.DataFrame,
    start_idx: int,
    end_idx: int
) -> pd.DataFrame:
    rows = []
    for _,r in fingerprint.iterrows():
        rule = r.to_dict()
        for i in events_for_rule(rule,eph,start_idx,end_idx):
            rows.append({
                "index": int(i),
                "date": eph.iloc[i]["date"],
                "rule_id": r["rule_id"],
                "system": r["system"],
                "family": r["family"],
                "planet_1": r["planet_1"],
                "planet_2": r["planet_2"],
                "offset_deg": r["offset_deg"],
                "target_value": r["target_value"],
                "dna_score": r["dna_score"],
            })
    return pd.DataFrame(rows)


def cluster_events(events: pd.DataFrame, market: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    ev = events.sort_values("index").reset_index(drop=True)
    groups = []
    cur = []

    for _,r in ev.iterrows():
        if not cur:
            cur=[r]
        elif int(r["index"]) - int(cur[-1]["index"]) <= cfg.cluster_window_sessions:
            cur.append(r)
        else:
            groups.append(cur)
            cur=[r]
    if cur:
        groups.append(cur)

    rows=[]
    for cid,g in enumerate(groups,1):
        inds=np.array([int(x["index"]) for x in g])
        scores=np.array([float(x["dna_score"]) for x in g])
        # choose event from strongest learned rule as center
        center=int(inds[int(np.argmax(scores))])

        rules=set(str(x["rule_id"]) for x in g)
        planets=set()
        families=set()
        systems=set()
        for x in g:
            planets.add(str(x["planet_1"]))
            if pd.notna(x["planet_2"]):
                planets.add(str(x["planet_2"]))
            families.add(str(x["family"]))
            systems.add(str(x["system"]))

        rows.append({
            "cluster_id":cid,
            "index":center,
            "date":market.iloc[center]["date"],
            "event_count":len(g),
            "unique_rules":len(rules),
            "unique_planets":len(planets),
            "families":"|".join(sorted(families)),
            "systems":"|".join(sorted(systems)),
            "cluster_score":float(scores.sum()),
            "max_rule_score":float(scores.max()),
            "start_index":int(inds.min()),
            "end_index":int(inds.max()),
        })
    return pd.DataFrame(rows)


def evaluate_clusters(
    clusters: pd.DataFrame,
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    cfg: Config
) -> pd.DataFrame:
    if clusters.empty:
        return pd.DataFrame()

    piv = pivots[(pivots["index"]>=start_idx)&(pivots["index"]<=end_idx)].copy()
    piv_idx = piv["index"].astype(int).to_numpy()

    rows=[]
    for _,c in clusters.iterrows():
        i=int(c["index"])
        rec=c.to_dict()

        if len(piv_idx):
            ds=np.abs(piv_idx-i)
            j=int(np.argmin(ds))
            near=piv.iloc[j]
            rec["nearest_pivot_date"]=near["date"]
            rec["nearest_pivot_type"]=near["type"]
            rec["distance_sessions"]=int(ds[j])
            rec["timing_hit"]=int(ds[j]<=cfg.hit_window_sessions)
        else:
            rec["nearest_pivot_date"]=pd.NaT
            rec["nearest_pivot_type"]=None
            rec["distance_sessions"]=np.nan
            rec["timing_hit"]=0

        # returns after signal
        close=float(market.iloc[i]["close"])
        for n in (1,3,5,10,20):
            if i+n<len(market):
                rec[f"ret_{n}s_pct"]=pct(close,float(market.iloc[i+n]["close"]))
            else:
                rec[f"ret_{n}s_pct"]=np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


# ============================================================
# BASELINES
# ============================================================

def random_baseline(
    n_signals: int,
    start_idx: int,
    end_idx: int,
    pivots: pd.DataFrame,
    cfg: Config
) -> pd.DataFrame:
    if n_signals<=0:
        return pd.DataFrame()

    rng=np.random.default_rng(cfg.random_seed)
    eligible=np.arange(start_idx,end_idx+1)
    piv_idx=period_pivot_indices(pivots,start_idx,end_idx)
    replace=n_signals>len(eligible)

    rows=[]
    for t in range(cfg.random_trials):
        chosen=rng.choice(eligible,size=n_signals,replace=replace)
        hits=sum(near_any_pivot(int(i),piv_idx,cfg.hit_window_sessions) for i in chosen)
        rows.append({"trial":t+1,"hit_rate":hits/n_signals,"hits":hits})
    return pd.DataFrame(rows)


def circular_shift_baseline(
    clusters: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    pivots: pd.DataFrame,
    cfg: Config
) -> pd.DataFrame:
    """
    Preserve exact spacing between DNA signal clusters, shift entire signal pattern
    around holdout. Harder baseline than independent random dates.
    """
    if clusters.empty:
        return pd.DataFrame()

    base=np.sort(clusters["index"].astype(int).to_numpy())
    span=end_idx-start_idx+1
    rel=base-start_idx
    piv_idx=period_pivot_indices(pivots,start_idx,end_idx)

    rng=np.random.default_rng(cfg.random_seed+99)
    rows=[]

    for t in range(cfg.random_trials):
        shift=int(rng.integers(0,span))
        shifted=start_idx+((rel+shift)%span)
        hits=sum(near_any_pivot(int(i),piv_idx,cfg.hit_window_sessions) for i in shifted)
        rows.append({"trial":t+1,"hit_rate":hits/len(shifted),"hits":hits})
    return pd.DataFrame(rows)


# ============================================================
# REPORT
# ============================================================

def baseline_stats(obs_rate: float, baseline: pd.DataFrame) -> dict:
    if baseline.empty:
        return {}
    arr=baseline["hit_rate"].to_numpy(float)
    return {
        "mean_pct":100.0*float(arr.mean()),
        "p95_pct":100.0*float(np.quantile(arr,0.95)),
        "empirical_p":float((np.sum(arr>=obs_rate)+1)/(len(arr)+1)),
    }


def run(args, cfg: Config):
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)

    print("[1/9] Loading COMI / stock data...")
    market=load_data(args.input,args.symbol,outdir)
    if len(market)<500:
        raise RuntimeError(f"Only {len(market)} bars found; need longer history.")

    split=int(len(market)*cfg.discovery_fraction)
    split=max(250,min(split,len(market)-150))
    disc_start=0
    disc_end=split-1
    hold_start=split
    hold_end=len(market)-1

    print(f"Rows: {len(market)} | {market.date.min().date()} -> {market.date.max().date()}")
    print(f"Discovery: {market.iloc[disc_start].date.date()} -> {market.iloc[disc_end].date.date()} ({split} bars)")
    print(f"Holdout:   {market.iloc[hold_start].date.date()} -> {market.iloc[hold_end].date.date()} ({len(market)-split} bars)")

    print("[2/9] Detecting major pivots...")
    pivots=raw_local_pivots(market,cfg)
    print(f"Major pivots: {len(pivots)}")

    print("[3/9] Selecting early principal LOW + HIGH anchors...")
    low_anchor,high_anchor=choose_origin_anchors(pivots,disc_end)
    print("LOW anchor :",pd.Timestamp(low_anchor["date"]).date(),low_anchor["price"])
    print("HIGH anchor:",pd.Timestamp(high_anchor["date"]).date(),high_anchor["price"])

    print("[4/9] Computing GEO + HELIO ephemeris...")
    eph=build_ephemeris(market["date"].tolist(),cfg)

    low_snap=make_anchor_snapshot(low_anchor,eph,cfg)
    high_snap=make_anchor_snapshot(high_anchor,eph,cfg)

    print("[5/9] Building stock-specific candidate DNA rules...")
    rules=build_candidate_rules([low_anchor,high_anchor],eph,cfg)
    print(f"Candidate rules: {len(rules)}")

    # Score only AFTER both anchors exist, still inside discovery.
    learning_start=max(int(low_anchor["index"]),int(high_anchor["index"]))+1
    learning_end=disc_end

    if learning_end-learning_start<100:
        raise RuntimeError(
            "Not enough discovery history after both anchors. "
            "Try --discovery-fraction 0.45."
        )

    print("[6/9] Learning DNA inside discovery only...")
    scored=[]
    for k,rule in enumerate(rules,1):
        scored.append(score_rule(rule,eph,pivots,learning_start,learning_end,cfg))
        if k%500==0:
            print(f"  scored {k}/{len(rules)}")
    scored_df=pd.DataFrame(scored)
    fingerprint=select_fingerprint(scored_df,cfg)
    print(f"Frozen DNA rules: {len(fingerprint)}")

    print("[7/9] Applying frozen DNA to unseen holdout...")
    hold_events=collect_rule_events(fingerprint,eph,hold_start,hold_end)
    clusters=cluster_events(hold_events,market,cfg)
    detail=evaluate_clusters(clusters,market,pivots,hold_start,hold_end,cfg)

    signals=len(detail)
    hits=int(detail["timing_hit"].sum()) if signals else 0
    obs=hits/signals if signals else 0.0

    print("[8/9] Random + circular-shift baselines...")
    rnd=random_baseline(signals,hold_start,hold_end,pivots,cfg)
    circ=circular_shift_baseline(clusters,hold_start,hold_end,pivots,cfg)
    rnd_stats=baseline_stats(obs,rnd)
    circ_stats=baseline_stats(obs,circ)

    print("[9/9] Writing audit outputs...")
    market.to_csv(outdir/"market.csv",index=False)
    pivots.to_csv(outdir/"major_pivots.csv",index=False)
    eph.to_csv(outdir/"ephemeris_geo_helio.csv",index=False)
    scored_df.to_csv(outdir/"all_candidate_rules_discovery_scored.csv",index=False)
    fingerprint.to_csv(outdir/"frozen_stock_dna_rules.csv",index=False)
    hold_events.to_csv(outdir/"holdout_raw_dna_events.csv",index=False)
    clusters.to_csv(outdir/"holdout_clustered_signals.csv",index=False)
    detail.to_csv(outdir/"holdout_evaluation.csv",index=False)
    rnd.to_csv(outdir/"holdout_random_baseline.csv",index=False)
    circ.to_csv(outdir/"holdout_circular_shift_baseline.csv",index=False)

    fingerprint_json={
        "symbol":args.symbol,
        "method":"Jenkins Stock DNA V1.1",
        "discovery_fraction":cfg.discovery_fraction,
        "discovery_period":{
            "start":str(market.iloc[disc_start]["date"].date()),
            "end":str(market.iloc[disc_end]["date"].date()),
        },
        "holdout_period":{
            "start":str(market.iloc[hold_start]["date"].date()),
            "end":str(market.iloc[hold_end]["date"].date()),
        },
        "anchors":{
            "principal_low":low_snap,
            "principal_high":high_snap,
        },
        "frozen_rules":fingerprint.to_dict(orient="records"),
    }

    with open(outdir/"stock_planetary_fingerprint.json","w",encoding="utf-8") as f:
        json.dump(fingerprint_json,f,ensure_ascii=False,indent=2,default=str)

    report={
        "symbol":args.symbol,
        "rows":len(market),
        "start_date":str(market.date.min().date()),
        "end_date":str(market.date.max().date()),
        "discovery_rows":split,
        "holdout_rows":len(market)-split,
        "major_pivots":len(pivots),
        "low_anchor_date":str(pd.Timestamp(low_anchor["date"]).date()),
        "high_anchor_date":str(pd.Timestamp(high_anchor["date"]).date()),
        "candidate_rules":len(scored_df),
        "frozen_dna_rules":len(fingerprint),
        "holdout_raw_events":len(hold_events),
        "holdout_clustered_signals":signals,
        "holdout_hits":hits,
        "holdout_hit_rate_pct":100.0*obs,
        "random_baseline":rnd_stats,
        "circular_shift_baseline":circ_stats,
    }

    with open(outdir/"run_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)

    print("\n=== JENKINS STOCK DNA V1 REPORT ===")
    for k,v in report.items():
        if isinstance(v,dict):
            print(k+":",json.dumps(v,ensure_ascii=False))
        else:
            print(f"{k}: {v}")

    print("\n=== FROZEN DNA RULES ===")
    if fingerprint.empty:
        print("No rules survived.")
    else:
        cols=[
            "system","family","planet_1","planet_2","anchor_type",
            "origin_value","offset_deg","target_value",
            "discovery_events","discovery_hits","discovery_hit_rate_pct",
            "discovery_enrichment","discovery_unique_pivots_covered","dna_score"
        ]
        print(fingerprint[cols].to_string(index=False))

    print(f"\nFiles written to: {outdir.resolve()}")


def parse_args():
    p=argparse.ArgumentParser(description="Jenkins Stock DNA Engine V1.1")
    p.add_argument("--symbol",default="COMI.CA")
    p.add_argument("--input",default=None,help="Optional local OHLC CSV")
    p.add_argument("--output-dir",default="/app/jenkins_comi_dna_v1")
    p.add_argument("--discovery-fraction",type=float,default=0.35)
    p.add_argument("--top-rules",type=int,default=12)
    p.add_argument("--pivot-left",type=int,default=20)
    p.add_argument("--pivot-right",type=int,default=20)
    p.add_argument("--min-pivot-move",type=float,default=10.0)
    p.add_argument("--hit-window",type=int,default=3)
    p.add_argument("--random-trials",type=int,default=5000)
    return p.parse_args()


def main():
    args=parse_args()
    cfg=Config(
        discovery_fraction=args.discovery_fraction,
        top_rules=args.top_rules,
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        min_pivot_move_pct=args.min_pivot_move,
        hit_window_sessions=args.hit_window,
        random_trials=args.random_trials,
    )
    run(args,cfg)


if __name__=="__main__":
    main()
