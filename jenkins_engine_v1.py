#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jenkins Engine V1
=================
Independent planetary-timing research engine for Egyptian-market testing.

Purpose
-------
1) Load a market series (Synthetic EGX Proxy, COMI, or any OHLC CSV).
2) Detect evaluation pivots using fixed, pre-declared rules.
3) Compute geocentric planetary longitudes and direct/retrograde status.
4) Generate planetary harmonic signals WITHOUT looking ahead.
5) Backtest signal proximity to future/nearby market pivots.
6) Compare hit-rate against a random-date baseline.
7) Export all intermediate tables for auditability.

V1 intentionally EXCLUDES:
- Square of Nine price translation
- manual planet selection
- hand-picked anchors
- heliocentric tuning
- adaptive harmonic tuning

Dependencies
------------
pip install pandas numpy pyswisseph

Input CSV
---------
Required:
    date, close
Recommended:
    date, open, high, low, close, volume

Date examples:
    2026-01-05
    2026-01-05 00:00:00

Example
-------
python jenkins_engine_v1.py --input synthetic_egx_proxy.csv --output-dir jenkins_v1_output

Author note
-----------
This script is designed as a research engine, not a trading recommendation system.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import swisseph as swe
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyswisseph\n"
        "Install it with:\n"
        "    pip install pyswisseph\n"
    ) from exc


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Config:
    # Planet set deliberately fixed in V1.
    planets: Tuple[str, ...] = (
        "MERCURY",
        "VENUS",
        "MARS",
        "JUPITER",
        "SATURN",
        "URANUS",
        "NEPTUNE",
        "PLUTO",
    )

    # Fixed Jenkins/Gann-style harmonics for first-pass research.
    harmonics_deg: Tuple[float, ...] = (45.0, 90.0, 180.0, 360.0)

    # Orb around exact planetary aspect / traveled-degree crossing.
    aspect_orb_deg: float = 1.0

    # Major evaluation pivots.
    pivot_left_bars: int = 10
    pivot_right_bars: int = 10

    # Minimum move required after a signal to call the turn meaningful.
    min_turn_move_pct: float = 3.0

    # Signal-to-pivot proximity tolerance.
    hit_window_sessions: int = 3

    # Forward performance windows.
    forward_sessions: Tuple[int, ...] = (1, 3, 5, 10, 20)

    # Random baseline simulations.
    random_trials: int = 2000
    random_seed: int = 1729

    # To prevent duplicate daily events from nearly identical rules.
    dedupe_same_day: bool = True

    # Minimum available history before using an anchor.
    min_anchor_age_sessions: int = 5


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
# Utilities
# ============================================================

def angular_distance(a: float, b: float) -> float:
    """Smallest angular separation in degrees: [0, 180]."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def signed_angular_delta(prev: float, curr: float) -> float:
    """Shortest signed change from prev to curr in [-180, 180)."""
    return ((curr - prev + 180.0) % 360.0) - 180.0


def circular_abs_delta_from_anchor(anchor: float, current: float) -> float:
    """
    Forward modular distance [0,360), useful as a simple phase measure.
    V1 keeps this deterministic and does not manually flip direction.
    """
    return (current - anchor) % 360.0


def jd_from_timestamp(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    return swe.julday(ts.year, ts.month, ts.day, 12.0)


def safe_pct_change(a: float, b: float) -> float:
    if not np.isfinite(a) or a == 0 or not np.isfinite(b):
        return np.nan
    return (b / a - 1.0) * 100.0


# ============================================================
# Data loading
# ============================================================

def load_market_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    if "date" not in cols:
        raise ValueError("Input CSV must contain a 'date' column.")
    if "close" not in cols:
        raise ValueError("Input CSV must contain a 'close' column.")

    rename = {cols[k]: k for k in cols if k in {"date", "open", "high", "low", "close", "volume"}}
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "open" not in df.columns:
        df["open"] = df["close"]

    df = df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Need at least 100 rows for a meaningful V1 test.")

    return df


# ============================================================
# Pivot detection (evaluation ground truth)
# ============================================================

def detect_pivots(
    df: pd.DataFrame,
    left: int,
    right: int,
) -> pd.DataFrame:
    """
    Detect confirmed local highs/lows.

    IMPORTANT:
    A pivot at row i is only knowable after `right` future sessions.
    For evaluation, we record:
      pivot_date      = the actual local turn date
      confirm_date    = when that pivot became observable
    This avoids pretending that a future-confirmed pivot was known on the turn date.
    """
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    out = []

    for i in range(left, len(df) - right):
        h = highs[i]
        l = lows[i]
        left_highs = highs[i-left:i]
        right_highs = highs[i+1:i+1+right]
        left_lows = lows[i-left:i]
        right_lows = lows[i+1:i+1+right]

        is_high = h >= np.max(left_highs) and h > np.max(right_highs)
        is_low = l <= np.min(left_lows) and l < np.min(right_lows)

        if is_high:
            out.append({
                "pivot_index": i,
                "pivot_date": df.loc[i, "date"],
                "confirm_index": i + right,
                "confirm_date": df.loc[i + right, "date"],
                "pivot_type": "HIGH",
                "pivot_price": h,
            })

        if is_low:
            out.append({
                "pivot_index": i,
                "pivot_date": df.loc[i, "date"],
                "confirm_index": i + right,
                "confirm_date": df.loc[i + right, "date"],
                "pivot_type": "LOW",
                "pivot_price": l,
            })

    piv = pd.DataFrame(out)
    if piv.empty:
        return piv

    piv = piv.sort_values(["pivot_index", "pivot_type"]).reset_index(drop=True)
    return piv


# ============================================================
# Ephemeris
# ============================================================

def compute_ephemeris(
    dates: Sequence[pd.Timestamp],
    cfg: Config,
) -> pd.DataFrame:
    """
    Geocentric ecliptic longitude + instantaneous longitudinal speed.
    speed < 0 => retrograde.
    """
    rows = []
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED

    for d in dates:
        jd = jd_from_timestamp(pd.Timestamp(d))
        row = {"date": pd.Timestamp(d)}

        for name in cfg.planets:
            pid = PLANET_IDS[name]
            xx, retflag = swe.calc_ut(jd, pid, flags)
            lon = float(xx[0]) % 360.0
            speed = float(xx[3])

            row[f"{name}_lon"] = lon
            row[f"{name}_speed"] = speed
            row[f"{name}_retrograde"] = int(speed < 0.0)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Anchor engine
# ============================================================

def build_confirmed_anchor_table(
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    At each date, select the most recently CONFIRMED pivot as the anchor.

    This makes anchor selection causal:
    no future pivot is available before its confirmation date.
    """
    out = []
    if pivots.empty:
        for i, row in market.iterrows():
            out.append({
                "date": row["date"],
                "anchor_pivot_date": pd.NaT,
                "anchor_confirm_date": pd.NaT,
                "anchor_type": None,
                "anchor_price": np.nan,
                "anchor_index": np.nan,
            })
        return pd.DataFrame(out)

    piv_by_confirm = pivots.sort_values("confirm_index").reset_index(drop=True)
    j = 0
    current = None

    for i, row in market.iterrows():
        while j < len(piv_by_confirm) and int(piv_by_confirm.loc[j, "confirm_index"]) <= i:
            candidate = piv_by_confirm.loc[j]
            # Optional age guard prevents immediately reusing an anchor on the same confirmation bar.
            if i - int(candidate["pivot_index"]) >= cfg.min_anchor_age_sessions:
                current = candidate
            j += 1

        if current is None:
            out.append({
                "date": row["date"],
                "anchor_pivot_date": pd.NaT,
                "anchor_confirm_date": pd.NaT,
                "anchor_type": None,
                "anchor_price": np.nan,
                "anchor_index": np.nan,
            })
        else:
            out.append({
                "date": row["date"],
                "anchor_pivot_date": current["pivot_date"],
                "anchor_confirm_date": current["confirm_date"],
                "anchor_type": current["pivot_type"],
                "anchor_price": float(current["pivot_price"]),
                "anchor_index": int(current["pivot_index"]),
            })

    return pd.DataFrame(out)


# ============================================================
# Signal engine
# ============================================================

def generate_planetary_signals(
    market: pd.DataFrame,
    eph: pd.DataFrame,
    anchors: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Generates two V1 signal families:

    A) ANCHOR_TRAVEL:
       Planet longitude relative to the longitude it had at the current confirmed anchor.
       Trigger when modular travel is near 45/90/180/360.

    B) PAIR_ASPECT:
       Angular separation between each planet pair near 45/90/180.
       360 is excluded from pair separation because 0°/360° is conjunction.
       We record conjunction as harmonic=360 for naming consistency.

    No price data other than anchor metadata is used to generate the event.
    """
    m = market[["date", "close"]].copy()
    x = m.merge(eph, on="date", how="left").merge(anchors, on="date", how="left")

    # Map market row index by date and ephemeris row by date.
    market_idx = {pd.Timestamp(d): i for i, d in enumerate(market["date"])}
    eph_idx = {pd.Timestamp(d): i for i, d in enumerate(eph["date"])}

    events: List[dict] = []

    # A) planet traveled degrees from current anchor
    for i, row in x.iterrows():
        adate = row["anchor_pivot_date"]
        if pd.isna(adate):
            continue
        adate = pd.Timestamp(adate)
        if adate not in eph_idx:
            continue

        ai = eph_idx[adate]
        ei = eph_idx[pd.Timestamp(row["date"])]

        for planet in cfg.planets:
            anchor_lon = float(eph.loc[ai, f"{planet}_lon"])
            curr_lon = float(eph.loc[ei, f"{planet}_lon"])
            traveled = circular_abs_delta_from_anchor(anchor_lon, curr_lon)

            for h in cfg.harmonics_deg:
                target = 0.0 if h == 360.0 else h
                # Circular closeness for 360 equivalent to returning near anchor longitude.
                if h == 360.0:
                    err = min(traveled, 360.0 - traveled)
                else:
                    err = abs(traveled - target)

                if err <= cfg.aspect_orb_deg:
                    events.append({
                        "date": row["date"],
                        "signal_family": "ANCHOR_TRAVEL",
                        "planet_1": planet,
                        "planet_2": None,
                        "harmonic_deg": h,
                        "orb_error_deg": err,
                        "planetary_value_deg": traveled,
                        "retrograde_1": int(row[f"{planet}_retrograde"]),
                        "retrograde_2": None,
                        "anchor_pivot_date": adate,
                        "anchor_type": row["anchor_type"],
                        "anchor_price": row["anchor_price"],
                    })

    # B) pair aspects
    pair_harmonics = (45.0, 90.0, 180.0, 360.0)
    planets = list(cfg.planets)

    for _, row in x.iterrows():
        for a in range(len(planets)):
            for b in range(a + 1, len(planets)):
                p1 = planets[a]
                p2 = planets[b]
                lon1 = float(row[f"{p1}_lon"])
                lon2 = float(row[f"{p2}_lon"])
                sep = angular_distance(lon1, lon2)

                for h in pair_harmonics:
                    target = 0.0 if h == 360.0 else h
                    err = abs(sep - target)
                    if err <= cfg.aspect_orb_deg:
                        events.append({
                            "date": row["date"],
                            "signal_family": "PAIR_ASPECT",
                            "planet_1": p1,
                            "planet_2": p2,
                            "harmonic_deg": h,
                            "orb_error_deg": err,
                            "planetary_value_deg": sep,
                            "retrograde_1": int(row[f"{p1}_retrograde"]),
                            "retrograde_2": int(row[f"{p2}_retrograde"]),
                            "anchor_pivot_date": row["anchor_pivot_date"],
                            "anchor_type": row["anchor_type"],
                            "anchor_price": row["anchor_price"],
                        })

    sig = pd.DataFrame(events)
    if sig.empty:
        return sig

    sig = sig.sort_values(["date", "signal_family", "orb_error_deg"]).reset_index(drop=True)

    if cfg.dedupe_same_day:
        # Preserve one best event per family/planet(s)/harmonic/day.
        sig = sig.drop_duplicates(
            subset=["date", "signal_family", "planet_1", "planet_2", "harmonic_deg"],
            keep="first",
        ).reset_index(drop=True)

    return sig


# ============================================================
# Evaluation
# ============================================================

def add_forward_returns(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    out = signals.copy()
    idx_by_date = {pd.Timestamp(d): i for i, d in enumerate(market["date"])}
    closes = market["close"].to_numpy(dtype=float)

    for n in cfg.forward_sessions:
        vals = []
        for d in out["date"]:
            i = idx_by_date.get(pd.Timestamp(d))
            if i is None or i + n >= len(market):
                vals.append(np.nan)
            else:
                vals.append(safe_pct_change(closes[i], closes[i+n]))
        out[f"ret_{n}s_pct"] = vals

    return out


def evaluate_signal_hits(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A signal is a timing hit if a ground-truth pivot is within ±N market sessions.

    Separately, "meaningful_turn" checks if price subsequently moves at least
    min_turn_move_pct away from signal close within 20 sessions in either direction.

    This avoids scoring tiny wiggles as successes.
    """
    if signals.empty:
        return signals.copy(), pd.DataFrame()

    idx_by_date = {pd.Timestamp(d): i for i, d in enumerate(market["date"])}
    piv_idx = pivots["pivot_index"].astype(int).to_numpy() if not pivots.empty else np.array([], dtype=int)
    piv_types = pivots["pivot_type"].to_numpy() if not pivots.empty else np.array([], dtype=object)
    piv_dates = pd.to_datetime(pivots["pivot_date"]).to_numpy() if not pivots.empty else np.array([], dtype="datetime64[ns]")

    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    closes = market["close"].to_numpy(dtype=float)

    rows = []

    for _, s in signals.iterrows():
        d = pd.Timestamp(s["date"])
        i = idx_by_date[d]

        if len(piv_idx):
            distances = np.abs(piv_idx - i)
            j = int(np.argmin(distances))
            nearest_dist = int(distances[j])
            nearest_type = str(piv_types[j])
            nearest_date = pd.Timestamp(piv_dates[j])
            timing_hit = int(nearest_dist <= cfg.hit_window_sessions)
        else:
            nearest_dist = np.nan
            nearest_type = None
            nearest_date = pd.NaT
            timing_hit = 0

        end = min(len(market) - 1, i + max(cfg.forward_sessions))
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

        rec = s.to_dict()
        rec.update({
            "signal_index": i,
            "nearest_pivot_date": nearest_date,
            "nearest_pivot_type": nearest_type,
            "distance_to_nearest_pivot_sessions": nearest_dist,
            "timing_hit": timing_hit,
            "max_up_next_window_pct": max_up,
            "max_down_next_window_pct": max_down,
            "meaningful_turn": meaningful_turn,
            "qualified_hit": int(timing_hit and meaningful_turn),
        })
        rows.append(rec)

    detail = pd.DataFrame(rows)

    grp_cols = ["signal_family", "planet_1", "planet_2", "harmonic_deg"]
    summary = (
        detail.groupby(grp_cols, dropna=False)
        .agg(
            signals=("date", "count"),
            timing_hits=("timing_hit", "sum"),
            qualified_hits=("qualified_hit", "sum"),
            timing_hit_rate=("timing_hit", "mean"),
            qualified_hit_rate=("qualified_hit", "mean"),
            mean_orb_error_deg=("orb_error_deg", "mean"),
        )
        .reset_index()
    )
    summary["timing_hit_rate_pct"] = summary["timing_hit_rate"] * 100.0
    summary["qualified_hit_rate_pct"] = summary["qualified_hit_rate"] * 100.0
    summary = summary.sort_values(
        ["qualified_hit_rate", "signals"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return detail, summary


def random_baseline(
    market: pd.DataFrame,
    pivots: pd.DataFrame,
    n_signals: int,
    cfg: Config,
) -> pd.DataFrame:
    """
    Same number of random signal dates, sampled from actual trading sessions.
    Reports distribution of hit rates.
    """
    if n_signals <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.random_seed)
    n = len(market)
    piv_idx = pivots["pivot_index"].astype(int).to_numpy() if not pivots.empty else np.array([], dtype=int)

    # Avoid first/last few bars where evaluation is structurally different.
    eligible = np.arange(
        cfg.hit_window_sessions,
        max(cfg.hit_window_sessions + 1, n - cfg.hit_window_sessions)
    )

    if len(eligible) == 0:
        return pd.DataFrame()

    replace = n_signals > len(eligible)
    rows = []

    for trial in range(cfg.random_trials):
        chosen = rng.choice(eligible, size=n_signals, replace=replace)

        if len(piv_idx):
            hits = 0
            for i in chosen:
                if np.min(np.abs(piv_idx - i)) <= cfg.hit_window_sessions:
                    hits += 1
            rate = hits / n_signals
        else:
            hits = 0
            rate = 0.0

        rows.append({
            "trial": trial + 1,
            "n_signals": n_signals,
            "hits": hits,
            "hit_rate": rate,
            "hit_rate_pct": rate * 100.0,
        })

    return pd.DataFrame(rows)


def overall_stats(detail: pd.DataFrame, random_df: pd.DataFrame) -> dict:
    if detail.empty:
        return {"signals": 0}

    observed = float(detail["timing_hit"].mean())
    qualified = float(detail["qualified_hit"].mean())

    stats = {
        "signals": int(len(detail)),
        "timing_hits": int(detail["timing_hit"].sum()),
        "qualified_hits": int(detail["qualified_hit"].sum()),
        "timing_hit_rate_pct": observed * 100.0,
        "qualified_hit_rate_pct": qualified * 100.0,
    }

    if not random_df.empty:
        rr = random_df["hit_rate"].to_numpy(dtype=float)
        stats.update({
            "random_mean_hit_rate_pct": float(np.mean(rr) * 100.0),
            "random_median_hit_rate_pct": float(np.median(rr) * 100.0),
            "random_p95_hit_rate_pct": float(np.quantile(rr, 0.95) * 100.0),
            "empirical_p_value": float((np.sum(rr >= observed) + 1) / (len(rr) + 1)),
        })

    return stats


# ============================================================
# Walk-forward
# ============================================================

def walk_forward_report(
    detail: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pure time-split reporting. No parameter optimization is performed here.
    Splits history into thirds to expose stability/decay.
    """
    if detail.empty:
        return pd.DataFrame()

    dates = pd.to_datetime(market["date"])
    q1 = dates.iloc[len(dates) // 3]
    q2 = dates.iloc[(2 * len(dates)) // 3]

    def period(d):
        if d < q1:
            return "EARLY"
        if d < q2:
            return "MIDDLE"
        return "LATE"

    x = detail.copy()
    x["wf_period"] = pd.to_datetime(x["date"]).map(period)

    return (
        x.groupby(["wf_period", "signal_family"], dropna=False)
        .agg(
            signals=("date", "count"),
            timing_hits=("timing_hit", "sum"),
            qualified_hits=("qualified_hit", "sum"),
            timing_hit_rate_pct=("timing_hit", lambda s: 100.0 * s.mean()),
            qualified_hit_rate_pct=("qualified_hit", lambda s: 100.0 * s.mean()),
        )
        .reset_index()
    )


# ============================================================
# Main
# ============================================================

def run(input_path: str, output_dir: str, cfg: Config) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading market data: {input_path}")
    market = load_market_csv(input_path)

    print("[2/7] Detecting fixed-rule evaluation pivots...")
    pivots = detect_pivots(
        market,
        left=cfg.pivot_left_bars,
        right=cfg.pivot_right_bars,
    )

    print("[3/7] Computing geocentric planetary ephemeris...")
    eph = compute_ephemeris(market["date"].tolist(), cfg)

    print("[4/7] Building causal anchor table...")
    anchors = build_confirmed_anchor_table(market, pivots, cfg)

    print("[5/7] Generating planetary signals...")
    signals = generate_planetary_signals(market, eph, anchors, cfg)
    signals = add_forward_returns(signals, market, cfg)

    print("[6/7] Evaluating turns and random baseline...")
    detail, summary = evaluate_signal_hits(signals, market, pivots, cfg)
    random_df = random_baseline(market, pivots, len(detail), cfg)
    wf = walk_forward_report(detail, market)
    stats = overall_stats(detail, random_df)

    print("[7/7] Exporting audit files...")
    market.to_csv(outdir / "market_clean.csv", index=False)
    pivots.to_csv(outdir / "evaluation_pivots.csv", index=False)
    eph.to_csv(outdir / "ephemeris_geocentric.csv", index=False)
    anchors.to_csv(outdir / "anchor_table.csv", index=False)
    signals.to_csv(outdir / "planetary_signals.csv", index=False)
    detail.to_csv(outdir / "signal_evaluation_detail.csv", index=False)
    summary.to_csv(outdir / "signal_summary_by_rule.csv", index=False)
    random_df.to_csv(outdir / "random_baseline_trials.csv", index=False)
    wf.to_csv(outdir / "walk_forward_summary.csv", index=False)

    config_dict = asdict(cfg)
    config_dict["planets"] = list(cfg.planets)
    config_dict["harmonics_deg"] = list(cfg.harmonics_deg)
    config_dict["forward_sessions"] = list(cfg.forward_sessions)

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    report = {
        "input_file": str(Path(input_path).resolve()),
        "rows": int(len(market)),
        "start_date": str(market["date"].min().date()),
        "end_date": str(market["date"].max().date()),
        "evaluation_pivots": int(len(pivots)),
        **stats,
    }

    with open(outdir / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== JENKINS V1 REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    if "empirical_p_value" in report:
        if report["empirical_p_value"] < 0.05:
            print("\nResult: observed timing hit rate beat random baseline at empirical p < 0.05.")
        else:
            print("\nResult: no statistically convincing edge vs random baseline in this V1 run.")

    print(f"\nFiles written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jenkins Engine V1 planetary timing backtester")
    p.add_argument("--input", required=True, help="OHLC CSV, e.g. synthetic_egx_proxy.csv")
    p.add_argument("--output-dir", default="jenkins_v1_output")
    p.add_argument("--orb", type=float, default=1.0, help="Aspect orb in degrees (default: 1.0)")
    p.add_argument("--pivot-left", type=int, default=10)
    p.add_argument("--pivot-right", type=int, default=10)
    p.add_argument("--hit-window", type=int, default=3, help="± sessions around a pivot")
    p.add_argument("--min-turn-move", type=float, default=3.0, help="Minimum %% move after signal")
    p.add_argument("--random-trials", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        aspect_orb_deg=args.orb,
        pivot_left_bars=args.pivot_left,
        pivot_right_bars=args.pivot_right,
        hit_window_sessions=args.hit_window,
        min_turn_move_pct=args.min_turn_move,
        random_trials=args.random_trials,
    )

    run(args.input, args.output_dir, cfg)


if __name__ == "__main__":
    main()
