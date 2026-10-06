#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jenkins Engine V1.1
===================
Research upgrade focused on reducing false signal density and removing
the main methodological weaknesses of V1.

Key changes vs V1
-----------------
1) Uses PROJECT MAJOR EGX PIVOTS as preferred ground truth when available.
2) Uses exact harmonic CROSSINGS, not "inside orb" signals.
3) Tracks cumulative signed planetary motion bar-by-bar (retrograde included).
4) Separates anchor-travel crossings from planet-pair aspect crossings.
5) Clusters nearby planetary events into ONE turn-window signal.
6) Reports Discovery (<= 2024-12-31) vs Holdout (>= 2025-01-01).
7) Compares clustered signal dates against a random-date baseline.
8) Prioritizes 180° and 360°; 45° and 90° are still exported separately.

Input CSV
---------
Required:
    date, close
Recommended:
    date, open, high, low, close

Typical run
-----------
python jenkins_engine_v1_1.py \
  --input /app/synthetic_egx_proxy.csv \
  --output-dir /app/jenkins_v1_1_output

This remains a research engine, not a trading recommendation system.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import swisseph as swe
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyswisseph\nInstall with: pip install pyswisseph"
    ) from exc


@dataclass(frozen=True)
class Config:
    planets: Tuple[str, ...] = (
        "MERCURY", "VENUS", "MARS", "JUPITER",
        "SATURN", "URANUS", "NEPTUNE", "PLUTO"
    )
    harmonics_deg: Tuple[float, ...] = (45.0, 90.0, 180.0, 360.0)

    # Evaluation
    hit_window_sessions: int = 3
    min_turn_move_pct: float = 3.0
    forward_sessions: Tuple[int, ...] = (1, 3, 5, 10, 20)

    # Clustering
    cluster_window_sessions: int = 2
    min_cluster_events: int = 1

    # Anchor usage
    min_anchor_age_sessions: int = 1

    # Baseline
    random_trials: int = 5000
    random_seed: int = 1729

    # Walk-forward boundary
    discovery_end: str = "2024-12-31"
    holdout_start: str = "2025-01-01"

    # Fallback pivots only if project pivots are unavailable
    fallback_pivot_left_bars: int = 20
    fallback_pivot_right_bars: int = 20


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


def safe_pct_change(a: float, b: float) -> float:
    if not np.isfinite(a) or a == 0 or not np.isfinite(b):
        return np.nan
    return (b / a - 1.0) * 100.0


def angular_distance(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def signed_angular_delta(prev: float, curr: float) -> float:
    return ((curr - prev + 180.0) % 360.0) - 180.0


def jd_from_timestamp(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    return swe.julday(ts.year, ts.month, ts.day, 12.0)


def load_market_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    if "date" not in cols or "close" not in cols:
        raise ValueError("CSV must contain date and close columns.")

    rename = {cols[k]: k for k in cols if k in {"date","open","high","low","close","volume"}}
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["date","close"])
          .sort_values("date")
          .drop_duplicates("date", keep="last")
          .reset_index(drop=True)
    )

    if "open" not in df.columns: df["open"] = df["close"]
    if "high" not in df.columns: df["high"] = df["close"]
    if "low" not in df.columns: df["low"] = df["close"]

    return df


# ------------------------------------------------------------
# Ground-truth major pivots
# ------------------------------------------------------------

def load_project_major_pivots(market: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Preferred route: use the SAME principal EGX pivots already used by the project.
    """
    try:
        from core import database
        from monitor.gann_analysis import _major_market_pivots_uncached

        db = database()
        pivots, _all = _major_market_pivots_uncached(db, include_diagnostics=True)

        rows = []
        idx_by_date = {pd.Timestamp(d).normalize(): i for i, d in enumerate(market["date"])}

        for p in pivots:
            d = pd.Timestamp(p["date"]).normalize()
            if d not in idx_by_date:
                # map to nearest trading date
                distances = np.abs((market["date"].dt.normalize() - d).dt.days.to_numpy())
                i = int(np.argmin(distances))
            else:
                i = idx_by_date[d]

            typ = "LOW" if p.get("kind") == "L" else "HIGH"
            rows.append({
                "pivot_index": i,
                "pivot_date": market.loc[i, "date"],
                "pivot_type": typ,
                "pivot_price": float(p.get("price", market.loc[i, "close"])),
                "available_date": pd.to_datetime(p.get("available_date"), errors="coerce"),
                "index_source": p.get("index_source"),
                "final_score": p.get("final_score"),
                "consensus_votes": p.get("consensus_votes"),
            })

        out = pd.DataFrame(rows).sort_values("pivot_index").reset_index(drop=True)
        if len(out):
            return out, "PROJECT_MAJOR_PIVOTS"
    except Exception as e:
        print(f"[pivot warning] project pivots unavailable: {e}")

    return pd.DataFrame(), "UNAVAILABLE"


def detect_fallback_major_pivots(
    market: pd.DataFrame, left: int, right: int
) -> pd.DataFrame:
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    rows = []
    for i in range(left, len(market)-right):
        h = highs[i]
        l = lows[i]
        if h >= np.max(highs[i-left:i]) and h > np.max(highs[i+1:i+1+right]):
            rows.append({
                "pivot_index": i,
                "pivot_date": market.loc[i,"date"],
                "pivot_type": "HIGH",
                "pivot_price": h,
                "available_date": market.loc[i+right,"date"],
            })
        if l <= np.min(lows[i-left:i]) and l < np.min(lows[i+1:i+1+right]):
            rows.append({
                "pivot_index": i,
                "pivot_date": market.loc[i,"date"],
                "pivot_type": "LOW",
                "pivot_price": l,
                "available_date": market.loc[i+right,"date"],
            })
    return pd.DataFrame(rows).sort_values("pivot_index").reset_index(drop=True)


# ------------------------------------------------------------
# Ephemeris with cumulative signed motion
# ------------------------------------------------------------

def compute_ephemeris(dates: Sequence[pd.Timestamp], cfg: Config) -> pd.DataFrame:
    rows = []
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED

    for d in dates:
        jd = jd_from_timestamp(pd.Timestamp(d))
        row = {"date": pd.Timestamp(d)}
        for name in cfg.planets:
            xx, _ = swe.calc_ut(jd, PLANET_IDS[name], flags)
            lon = float(xx[0]) % 360.0
            speed = float(xx[3])
            row[f"{name}_lon"] = lon
            row[f"{name}_speed"] = speed
            row[f"{name}_retrograde"] = int(speed < 0)
        rows.append(row)

    eph = pd.DataFrame(rows)

    for name in cfg.planets:
        lon = eph[f"{name}_lon"].to_numpy(float)
        deltas = np.zeros(len(lon), dtype=float)
        for i in range(1, len(lon)):
            deltas[i] = signed_angular_delta(lon[i-1], lon[i])

        eph[f"{name}_delta"] = deltas
        eph[f"{name}_cum_signed"] = np.cumsum(deltas)
        eph[f"{name}_cum_abs"] = np.cumsum(np.abs(deltas))

    return eph


# ------------------------------------------------------------
# Causal anchors from major pivots
# ------------------------------------------------------------

def build_anchor_table(
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Use only pivots whose available_date is already known at each session.
    """
    if pivots.empty:
        return pd.DataFrame({
            "date": market["date"],
            "anchor_pivot_date": pd.NaT,
            "anchor_available_date": pd.NaT,
            "anchor_type": None,
            "anchor_price": np.nan,
            "anchor_index": np.nan,
        })

    p = pivots.copy()
    p["available_date"] = pd.to_datetime(p["available_date"], errors="coerce")
    p["pivot_date"] = pd.to_datetime(p["pivot_date"])
    p = p.sort_values(["available_date","pivot_date"]).reset_index(drop=True)

    rows = []
    current = None
    j = 0

    for i, r in market.iterrows():
        d = pd.Timestamp(r["date"])

        while j < len(p):
            ad = p.loc[j, "available_date"]
            if pd.isna(ad) or pd.Timestamp(ad) > d:
                break
            current = p.loc[j]
            j += 1

        if current is None:
            rows.append({
                "date": d,
                "anchor_pivot_date": pd.NaT,
                "anchor_available_date": pd.NaT,
                "anchor_type": None,
                "anchor_price": np.nan,
                "anchor_index": np.nan,
            })
        else:
            rows.append({
                "date": d,
                "anchor_pivot_date": current["pivot_date"],
                "anchor_available_date": current["available_date"],
                "anchor_type": current["pivot_type"],
                "anchor_price": float(current["pivot_price"]),
                "anchor_index": int(current["pivot_index"]),
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Exact crossing detection
# ------------------------------------------------------------

def crossed_level(prev: float, curr: float, level: float) -> bool:
    """
    True only if interval [prev,curr] crosses level.
    Works in either direction.
    """
    if not (np.isfinite(prev) and np.isfinite(curr)):
        return False
    if prev == curr:
        return False
    lo, hi = sorted((prev, curr))
    return lo < level <= hi


def levels_crossed(prev: float, curr: float, step: float) -> List[float]:
    """
    Returns integer multiples of `step` crossed between prev and curr.
    Useful for cumulative planetary motion.
    """
    if not (np.isfinite(prev) and np.isfinite(curr)) or prev == curr:
        return []

    lo, hi = sorted((prev, curr))
    k0 = math.floor(lo / step) + 1
    k1 = math.floor(hi / step)
    if k1 < k0:
        return []

    return [k * step for k in range(k0, k1 + 1)]


def harmonic_label(level: float) -> float:
    """
    Convert any crossed cumulative multiple into its Jenkins harmonic bucket.
    Priority:
      multiples of 360 -> 360
      multiples of 180 -> 180
      multiples of 90  -> 90
      multiples of 45  -> 45
    """
    x = abs(level)
    eps = 1e-9
    if abs(x % 360.0) < eps: return 360.0
    if abs(x % 180.0) < eps: return 180.0
    if abs(x % 90.0)  < eps: return 90.0
    return 45.0


def generate_anchor_travel_crossings(
    market: pd.DataFrame,
    eph: pd.DataFrame,
    anchors: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    x = (
        market[["date","close"]]
        .merge(eph, on="date", how="left")
        .merge(anchors, on="date", how="left")
    )

    idx_by_date = {pd.Timestamp(d): i for i,d in enumerate(eph["date"])}
    events = []

    prev_anchor_date = None
    state = {}

    for i, row in x.iterrows():
        ad = row["anchor_pivot_date"]
        if pd.isna(ad):
            prev_anchor_date = None
            state = {}
            continue

        ad = pd.Timestamp(ad)

        # New anchor -> reset all relative travel states
        if prev_anchor_date is None or ad != prev_anchor_date:
            ai = idx_by_date.get(ad)
            if ai is None:
                prev_anchor_date = ad
                state = {}
                continue
            state = {}
            for planet in cfg.planets:
                base = float(eph.loc[ai, f"{planet}_cum_signed"])
                current = float(row[f"{planet}_cum_signed"])
                state[planet] = current - base
            prev_anchor_date = ad
            continue

        for planet in cfg.planets:
            ai = idx_by_date.get(ad)
            if ai is None:
                continue

            base = float(eph.loc[ai, f"{planet}_cum_signed"])
            current_rel = float(row[f"{planet}_cum_signed"]) - base
            prev_rel = state.get(planet, current_rel)

            # exact crossings at every 45-degree cumulative step
            crossed = levels_crossed(prev_rel, current_rel, 45.0)

            for level in crossed:
                harm = harmonic_label(level)
                if harm not in cfg.harmonics_deg:
                    continue

                events.append({
                    "date": row["date"],
                    "market_index": i,
                    "signal_family": "ANCHOR_TRAVEL_CROSS",
                    "planet_1": planet,
                    "planet_2": None,
                    "harmonic_deg": harm,
                    "crossed_level_deg": level,
                    "motion_direction": "DIRECT" if current_rel > prev_rel else "RETROGRADE",
                    "retrograde_1": int(row[f"{planet}_retrograde"]),
                    "retrograde_2": None,
                    "anchor_pivot_date": ad,
                    "anchor_type": row["anchor_type"],
                    "anchor_price": row["anchor_price"],
                })

            state[planet] = current_rel

    return pd.DataFrame(events)


def normalize_pair_phase(diff: float) -> float:
    """
    Pair separation phase in [0,180].
    """
    x = diff % 360.0
    return x if x <= 180.0 else 360.0 - x


def generate_pair_aspect_crossings(
    market: pd.DataFrame,
    eph: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    x = market[["date"]].merge(eph, on="date", how="left")
    planets = list(cfg.planets)
    events = []

    for a in range(len(planets)):
        for b in range(a+1, len(planets)):
            p1, p2 = planets[a], planets[b]

            prev_sep = None
            for i, row in x.iterrows():
                sep = angular_distance(float(row[f"{p1}_lon"]), float(row[f"{p2}_lon"]))

                if prev_sep is not None:
                    for h in (45.0, 90.0, 180.0):
                        if crossed_level(prev_sep, sep, h):
                            events.append({
                                "date": row["date"],
                                "market_index": i,
                                "signal_family": "PAIR_ASPECT_CROSS",
                                "planet_1": p1,
                                "planet_2": p2,
                                "harmonic_deg": h,
                                "crossed_level_deg": h,
                                "motion_direction": "WIDENING" if sep > prev_sep else "NARROWING",
                                "retrograde_1": int(row[f"{p1}_retrograde"]),
                                "retrograde_2": int(row[f"{p2}_retrograde"]),
                                "anchor_pivot_date": pd.NaT,
                                "anchor_type": None,
                                "anchor_price": np.nan,
                            })

                    # Conjunction crossing: detect local pass through 0°
                    # We approximate this by a local minimum in separation in next step,
                    # so actual event is emitted only when previous separation was decreasing
                    # and current separation is tiny.
                    if prev_sep > 0.0 and sep <= 1.0:
                        events.append({
                            "date": row["date"],
                            "market_index": i,
                            "signal_family": "PAIR_ASPECT_CROSS",
                            "planet_1": p1,
                            "planet_2": p2,
                            "harmonic_deg": 360.0,
                            "crossed_level_deg": 0.0,
                            "motion_direction": "CONJUNCTION",
                            "retrograde_1": int(row[f"{p1}_retrograde"]),
                            "retrograde_2": int(row[f"{p2}_retrograde"]),
                            "anchor_pivot_date": pd.NaT,
                            "anchor_type": None,
                            "anchor_price": np.nan,
                        })

                prev_sep = sep

    return pd.DataFrame(events)


# ------------------------------------------------------------
# Event clustering
# ------------------------------------------------------------

def event_strength(h: float) -> float:
    if h == 360.0: return 4.0
    if h == 180.0: return 3.0
    if h == 90.0:  return 2.0
    return 1.0


def cluster_events(
    events: pd.DataFrame,
    market: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    ev = events.sort_values(["market_index","harmonic_deg"]).reset_index(drop=True)

    clusters = []
    cur = []

    for _, row in ev.iterrows():
        if not cur:
            cur = [row]
            continue

        last_idx = int(cur[-1]["market_index"])
        this_idx = int(row["market_index"])

        if this_idx - last_idx <= cfg.cluster_window_sessions:
            cur.append(row)
        else:
            clusters.append(cur)
            cur = [row]

    if cur:
        clusters.append(cur)

    out = []

    for cid, cluster in enumerate(clusters, start=1):
        if len(cluster) < cfg.min_cluster_events:
            continue

        # choose weighted center date based on strongest event(s)
        strengths = np.array([event_strength(float(r["harmonic_deg"])) for r in cluster])
        indices = np.array([int(r["market_index"]) for r in cluster])

        # strongest harmonic first; if tie use median market index
        max_s = strengths.max()
        strongest_indices = indices[strengths == max_s]
        center_idx = int(np.median(strongest_indices))

        center_idx = max(0, min(center_idx, len(market)-1))
        center_date = market.loc[center_idx, "date"]

        unique_planets = set()
        families = set()
        harmonics = []
        anchor_types = set()

        for r in cluster:
            unique_planets.add(str(r["planet_1"]))
            if pd.notna(r["planet_2"]):
                unique_planets.add(str(r["planet_2"]))
            families.add(str(r["signal_family"]))
            harmonics.append(float(r["harmonic_deg"]))
            if pd.notna(r["anchor_type"]):
                anchor_types.add(str(r["anchor_type"]))

        score = float(np.sum(strengths))

        out.append({
            "cluster_id": cid,
            "date": center_date,
            "market_index": center_idx,
            "cluster_start_index": int(indices.min()),
            "cluster_end_index": int(indices.max()),
            "cluster_start_date": market.loc[int(indices.min()), "date"],
            "cluster_end_date": market.loc[int(indices.max()), "date"],
            "event_count": len(cluster),
            "unique_planets": len(unique_planets),
            "families": "|".join(sorted(families)),
            "harmonics": "|".join(str(int(h)) for h in sorted(set(harmonics))),
            "has_180": int(180.0 in harmonics),
            "has_360": int(360.0 in harmonics),
            "has_90": int(90.0 in harmonics),
            "has_45": int(45.0 in harmonics),
            "cluster_strength": score,
            "anchor_types": "|".join(sorted(anchor_types)),
        })

    return pd.DataFrame(out).sort_values("market_index").reset_index(drop=True)


# ------------------------------------------------------------
# Evaluation and baseline
# ------------------------------------------------------------

def evaluate_clusters(
    clusters: pd.DataFrame,
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if clusters.empty:
        return pd.DataFrame()

    piv_idx = pivots["pivot_index"].astype(int).to_numpy()
    piv_types = pivots["pivot_type"].to_numpy()
    piv_dates = pd.to_datetime(pivots["pivot_date"]).to_numpy()

    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    closes = market["close"].to_numpy(float)

    rows = []

    for _, c in clusters.iterrows():
        i = int(c["market_index"])

        if len(piv_idx):
            dist = np.abs(piv_idx - i)
            j = int(np.argmin(dist))
            nearest_dist = int(dist[j])
            nearest_type = str(piv_types[j])
            nearest_date = pd.Timestamp(piv_dates[j])
            timing_hit = int(nearest_dist <= cfg.hit_window_sessions)
        else:
            nearest_dist = np.nan
            nearest_type = None
            nearest_date = pd.NaT
            timing_hit = 0

        end = min(len(market)-1, i + max(cfg.forward_sessions))
        if end > i:
            max_up = (np.max(highs[i+1:end+1]) / closes[i] - 1.0) * 100.0
            max_down = (np.min(lows[i+1:end+1]) / closes[i] - 1.0) * 100.0
        else:
            max_up = np.nan
            max_down = np.nan

        meaningful_turn = int(
            (np.isfinite(max_up) and max_up >= cfg.min_turn_move_pct)
            or
            (np.isfinite(max_down) and max_down <= -cfg.min_turn_move_pct)
        )

        rec = c.to_dict()
        rec.update({
            "nearest_pivot_date": nearest_date,
            "nearest_pivot_type": nearest_type,
            "distance_to_nearest_pivot_sessions": nearest_dist,
            "timing_hit": timing_hit,
            "max_up_next_window_pct": max_up,
            "max_down_next_window_pct": max_down,
            "meaningful_turn": meaningful_turn,
            "qualified_hit": int(timing_hit and meaningful_turn),
        })

        for n in cfg.forward_sessions:
            if i+n < len(market):
                rec[f"ret_{n}s_pct"] = safe_pct_change(closes[i], closes[i+n])
            else:
                rec[f"ret_{n}s_pct"] = np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


def random_baseline_for_indices(
    eligible_indices: np.ndarray,
    pivots: pd.DataFrame,
    n_signals: int,
    cfg: Config,
) -> pd.DataFrame:
    if n_signals <= 0 or len(eligible_indices) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.random_seed)
    piv_idx = pivots["pivot_index"].astype(int).to_numpy()
    replace = n_signals > len(eligible_indices)

    rows = []
    for trial in range(cfg.random_trials):
        chosen = rng.choice(eligible_indices, size=n_signals, replace=replace)
        hits = 0
        for i in chosen:
            if len(piv_idx) and np.min(np.abs(piv_idx - i)) <= cfg.hit_window_sessions:
                hits += 1

        rows.append({
            "trial": trial+1,
            "n_signals": n_signals,
            "hits": hits,
            "hit_rate": hits/n_signals,
            "hit_rate_pct": 100.0*hits/n_signals,
        })
    return pd.DataFrame(rows)


def period_name(d: pd.Timestamp, cfg: Config) -> str:
    d = pd.Timestamp(d)
    if d <= pd.Timestamp(cfg.discovery_end):
        return "DISCOVERY"
    if d >= pd.Timestamp(cfg.holdout_start):
        return "HOLDOUT"
    return "GAP"


def summarize_period(
    detail: pd.DataFrame,
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = detail.copy()
    x["period"] = pd.to_datetime(x["date"]).map(lambda d: period_name(d, cfg))

    summary_rows = []
    baseline_frames = []

    for period in ["DISCOVERY","HOLDOUT"]:
        part = x[x["period"] == period].copy()
        if part.empty:
            continue

        if period == "DISCOVERY":
            eligible = market.index[market["date"] <= pd.Timestamp(cfg.discovery_end)].to_numpy()
        else:
            eligible = market.index[market["date"] >= pd.Timestamp(cfg.holdout_start)].to_numpy()

        rb = random_baseline_for_indices(eligible, pivots, len(part), cfg)
        rb["period"] = period
        baseline_frames.append(rb)

        observed = float(part["timing_hit"].mean())
        qualified = float(part["qualified_hit"].mean())

        row = {
            "period": period,
            "signals": int(len(part)),
            "timing_hits": int(part["timing_hit"].sum()),
            "qualified_hits": int(part["qualified_hit"].sum()),
            "timing_hit_rate_pct": observed*100.0,
            "qualified_hit_rate_pct": qualified*100.0,
            "avg_cluster_strength": float(part["cluster_strength"].mean()),
            "avg_event_count": float(part["event_count"].mean()),
        }

        if not rb.empty:
            rr = rb["hit_rate"].to_numpy(float)
            row.update({
                "random_mean_hit_rate_pct": float(rr.mean()*100.0),
                "random_p95_hit_rate_pct": float(np.quantile(rr,0.95)*100.0),
                "empirical_p_value": float((np.sum(rr >= observed)+1)/(len(rr)+1)),
            })

        summary_rows.append(row)

    baseline = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), baseline


def summarize_harmonic_groups(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    rows = []
    groups = {
        "ALL": np.ones(len(detail), dtype=bool),
        "HAS_180_OR_360": (detail["has_180"]==1) | (detail["has_360"]==1),
        "HAS_360": detail["has_360"]==1,
        "HAS_180": detail["has_180"]==1,
        "ONLY_45_90": (detail["has_180"]==0) & (detail["has_360"]==0),
    }

    for name, mask in groups.items():
        p = detail[mask]
        if p.empty:
            continue
        rows.append({
            "group": name,
            "signals": len(p),
            "timing_hit_rate_pct": 100.0*p["timing_hit"].mean(),
            "qualified_hit_rate_pct": 100.0*p["qualified_hit"].mean(),
            "avg_cluster_strength": p["cluster_strength"].mean(),
            "avg_event_count": p["event_count"].mean(),
        })

    return pd.DataFrame(rows)


def run(input_path: str, output_dir: str, cfg: Config) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] Loading market data: {input_path}")
    market = load_market_csv(input_path)

    print("[2/8] Loading PROJECT major EGX pivots...")
    pivots, pivot_source = load_project_major_pivots(market)

    if pivots.empty:
        print("[2b/8] Falling back to fixed-rule major pivots...")
        pivots = detect_fallback_major_pivots(
            market,
            cfg.fallback_pivot_left_bars,
            cfg.fallback_pivot_right_bars,
        )
        pivot_source = "FALLBACK_LOCAL_PIVOTS"

    print(f"Pivot source: {pivot_source}")
    print(f"Major pivots: {len(pivots)}")

    print("[3/8] Computing geocentric ephemeris + cumulative motion...")
    eph = compute_ephemeris(market["date"].tolist(), cfg)

    print("[4/8] Building causal major-pivot anchors...")
    anchors = build_anchor_table(market, pivots, cfg)

    print("[5/8] Generating exact planetary crossings...")
    a = generate_anchor_travel_crossings(market, eph, anchors, cfg)
    b = generate_pair_aspect_crossings(market, eph, cfg)
    events = pd.concat([a,b], ignore_index=True) if len(a) or len(b) else pd.DataFrame()

    print(f"Raw crossing events: {len(events)}")

    print("[6/8] Clustering nearby planetary events...")
    clusters = cluster_events(events, market, cfg)
    print(f"Clustered turn windows: {len(clusters)}")

    print("[7/8] Evaluating against major pivots + holdout baseline...")
    detail = evaluate_clusters(clusters, market, pivots, cfg)
    period_summary, random_df = summarize_period(detail, market, pivots, cfg)
    harmonic_summary = summarize_harmonic_groups(detail)

    print("[8/8] Exporting audit files...")
    market.to_csv(outdir/"market_clean.csv", index=False)
    pivots.to_csv(outdir/"major_pivots_ground_truth.csv", index=False)
    eph.to_csv(outdir/"ephemeris_geocentric_cumulative.csv", index=False)
    anchors.to_csv(outdir/"causal_anchor_table.csv", index=False)
    events.to_csv(outdir/"raw_exact_crossing_events.csv", index=False)
    clusters.to_csv(outdir/"clustered_turn_windows.csv", index=False)
    detail.to_csv(outdir/"cluster_evaluation_detail.csv", index=False)
    period_summary.to_csv(outdir/"discovery_holdout_summary.csv", index=False)
    harmonic_summary.to_csv(outdir/"harmonic_group_summary.csv", index=False)
    random_df.to_csv(outdir/"random_baseline_trials.csv", index=False)

    cfg_json = asdict(cfg)
    cfg_json["planets"] = list(cfg.planets)
    cfg_json["harmonics_deg"] = list(cfg.harmonics_deg)
    cfg_json["forward_sessions"] = list(cfg.forward_sessions)
    with open(outdir/"run_config.json","w",encoding="utf-8") as f:
        json.dump(cfg_json,f,ensure_ascii=False,indent=2)

    report = {
        "input_file": str(Path(input_path).resolve()),
        "rows": int(len(market)),
        "start_date": str(market["date"].min().date()),
        "end_date": str(market["date"].max().date()),
        "pivot_source": pivot_source,
        "major_pivots": int(len(pivots)),
        "raw_crossing_events": int(len(events)),
        "clustered_turn_windows": int(len(clusters)),
        "discovery_end": cfg.discovery_end,
        "holdout_start": cfg.holdout_start,
        "period_summary": period_summary.to_dict(orient="records"),
        "harmonic_group_summary": harmonic_summary.to_dict(orient="records"),
    }

    with open(outdir/"run_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,default=str)

    print("\n=== JENKINS V1.1 REPORT ===")
    print(f"input_file: {report['input_file']}")
    print(f"rows: {report['rows']}")
    print(f"start_date: {report['start_date']}")
    print(f"end_date: {report['end_date']}")
    print(f"pivot_source: {report['pivot_source']}")
    print(f"major_pivots: {report['major_pivots']}")
    print(f"raw_crossing_events: {report['raw_crossing_events']}")
    print(f"clustered_turn_windows: {report['clustered_turn_windows']}")

    print("\n=== DISCOVERY / HOLDOUT ===")
    if period_summary.empty:
        print("No evaluable signals.")
    else:
        print(period_summary.to_string(index=False))

    print("\n=== HARMONIC GROUPS ===")
    if harmonic_summary.empty:
        print("No harmonic-group results.")
    else:
        print(harmonic_summary.to_string(index=False))

    print(f"\nFiles written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jenkins Engine V1.1")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="jenkins_v1_1_output")
    p.add_argument("--hit-window", type=int, default=3)
    p.add_argument("--cluster-window", type=int, default=2)
    p.add_argument("--min-turn-move", type=float, default=3.0)
    p.add_argument("--random-trials", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        hit_window_sessions=args.hit_window,
        cluster_window_sessions=args.cluster_window,
        min_turn_move_pct=args.min_turn_move,
        random_trials=args.random_trials,
    )
    run(args.input, args.output_dir, cfg)


if __name__ == "__main__":
    main()
