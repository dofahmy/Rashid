# monitor/egx_live.py
# Production EGX market-turn scorer built from the validated V2 research architecture.

from __future__ import annotations
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from core import Setting

STATE_KEY = "egx_live_market_turn_state_v1"
MODEL_VERSION = "egx-turn-v2-production-1"

BASE_SCRIPT = Path(os.getenv(
    "EGX_RESEARCH_SCRIPT",
    "/app/egx_final_decision_experiment_v2.py"
))

# Validated production architecture from the final experiment:
# LOW  -> price/breadth only
# HIGH -> price/breadth + planetary
MODEL_SPECS = (
    ("LOW", 3, "PRICE_BREADTH_ONLY"),
    ("LOW", 5, "PRICE_BREADTH_ONLY"),
    ("LOW", 10, "PRICE_BREADTH_ONLY"),
    ("HIGH", 3, "PRICE_BREADTH_PLUS_PLANETARY"),
    ("HIGH", 5, "PRICE_BREADTH_PLUS_PLANETARY"),
    ("HIGH", 10, "PRICE_BREADTH_PLUS_PLANETARY"),
)


def _load_base():
    if not BASE_SCRIPT.exists():
        raise RuntimeError(f"Missing EGX research script: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("egx_prod_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_current_model(base, data, label, model_type, horizon):
    """
    Production calibration:
    1) Remove the last `horizon` sessions from labeled training because their
       future label is not fully observable yet.
    2) Chronological 80/20 train/calibration split.
    3) Choose alert rate / threshold on calibration only.
    4) Refit on all fully-labeled history.
    5) Recalibrate final threshold to the same calibration alert-rate.
    6) Score the latest row.
    """
    price_cols = [
        c for c in data.columns
        if c != "date"
        and not c.startswith("GEO_")
        and not c.startswith("HELIO_")
        and not c.startswith("future_")
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    price_cols = [c for c in price_cols if c != "n_stocks"]

    planet_cols = [
        c for c in data.columns
        if (c.startswith("GEO_") or c.startswith("HELIO_"))
        and pd.api.types.is_numeric_dtype(data[c])
    ]

    cols = price_cols if model_type == "PRICE_BREADTH_ONLY" else price_cols + planet_cols

    z = data[["date"] + cols + [label]].copy()
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.iloc[80:].reset_index(drop=True)

    # latest row is for live scoring; labeled history excludes last horizon sessions
    latest = z.iloc[[-1]].copy()
    hist = z.iloc[:-horizon].copy() if len(z) > horizon else z.iloc[0:0].copy()
    if len(hist) < 400:
        raise RuntimeError(f"Not enough fully-labeled history for {label}: {len(hist)}")

    cut = max(250, int(len(hist) * 0.80))
    cut = min(cut, len(hist)-100)
    tr = hist.iloc[:cut].copy()
    va = hist.iloc[cut:].copy()

    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).to_numpy(float)
    Xva = va[cols].fillna(med).to_numpy(float)

    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    mu[~np.isfinite(mu)] = 0.0

    Xtr = (Xtr-mu)/sd
    Xva = (Xva-mu)/sd
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xva = np.column_stack([np.ones(len(Xva)), Xva])

    ytr = tr[label].to_numpy(int)
    yva = va[label].to_numpy(int)

    w = base.fit_logit(Xtr, ytr)
    pva = base.sigmoid(Xva @ w)
    raw_threshold = base.choose_threshold(yva, pva)
    cal_metrics = base.confusion_metrics(yva, pva, raw_threshold)
    alert_rate = float(cal_metrics["alert_rate"])

    # Refit on all fully-observable history.
    med2 = hist[cols].median()
    Xall = hist[cols].fillna(med2).to_numpy(float)
    mu2 = np.nanmean(Xall, axis=0)
    sd2 = np.nanstd(Xall, axis=0)
    sd2[~np.isfinite(sd2) | (sd2 < 1e-9)] = 1.0
    mu2[~np.isfinite(mu2)] = 0.0
    Xall = (Xall-mu2)/sd2
    Xall_i = np.column_stack([np.ones(len(Xall)), Xall])
    yall = hist[label].to_numpy(int)

    w2 = base.fit_logit(Xall_i, yall)
    pall = base.sigmoid(Xall_i @ w2)

    # Use recent fitted probability distribution to reproduce validated alert rate.
    recent = pall[-min(504, len(pall)):]
    if alert_rate <= 0:
        threshold = 1.0
    elif alert_rate >= 1:
        threshold = 0.0
    else:
        threshold = float(np.quantile(recent, 1.0-alert_rate))

    Xlive = latest[cols].fillna(med2).to_numpy(float)
    Xlive = (Xlive-mu2)/sd2
    Xlive = np.column_stack([np.ones(len(Xlive)), Xlive])
    probability = float(base.sigmoid(Xlive @ w2)[0])

    return {
        "turn_type": label.split("_")[1],
        "horizon_sessions": int(horizon),
        "model_type": model_type,
        "probability": probability,
        "threshold": threshold,
        "alert": bool(probability >= threshold),
        "probability_vs_threshold": probability / threshold if threshold > 0 else None,
        "calibration_alert_rate": alert_rate,
        "calibration_precision": float(cal_metrics["precision"]),
        "calibration_recall": float(cal_metrics["recall"]),
        "calibration_lift": float(cal_metrics["precision_lift"]) if np.isfinite(cal_metrics["precision_lift"]) else None,
        "calibration_auc": float(base.auc_score(yva, pva)) if np.isfinite(base.auc_score(yva, pva)) else None,
        "training_rows": int(len(hist)),
        "feature_count": int(len(cols)),
    }


def _current_stock_rankings(base, stocks, sectors, daily_date, market_bias):
    """
    Causal ranking only.  No future-confirmed stock pivot is used live.

    During LOW/buy watch:
      reward stocks near 20/60d low, making higher lows, recovering above MA20,
      and positive 5d momentum.

    During HIGH/sell watch:
      reward stocks near 20/60d high, making lower highs, losing MA20,
      and negative 5d momentum.
    """
    rows = []

    for sym, df in stocks.items():
        x = df.copy().sort_values("date").reset_index(drop=True)
        if len(x) < 70:
            continue

        # latest session on/before live date
        x = x[pd.to_datetime(x["date"]) <= pd.Timestamp(daily_date)].copy()
        if len(x) < 70:
            continue

        c = x["close"].astype(float)
        h = x["high"].astype(float)
        l = x["low"].astype(float)

        last = float(c.iloc[-1])
        ma20 = float(c.iloc[-20:].mean())
        ret5 = base.pct(float(c.iloc[-6]), last) if len(c) >= 6 else 0.0

        low20 = float(l.iloc[-20:].min())
        high20 = float(h.iloc[-20:].max())
        low60 = float(l.iloc[-60:].min())
        high60 = float(h.iloc[-60:].max())

        near20low = max(0.0, 1.0 - max(0.0, last/low20-1.0)/0.10)
        near20high = max(0.0, 1.0 - max(0.0, high20/last-1.0)/0.10)
        near60low = max(0.0, 1.0 - max(0.0, last/low60-1.0)/0.20)
        near60high = max(0.0, 1.0 - max(0.0, high60/last-1.0)/0.20)

        higher_low = float(l.iloc[-5:].min() > l.iloc[-10:-5].min())
        lower_high = float(h.iloc[-5:].max() < h.iloc[-10:-5].max())
        above_ma20 = float(last > ma20)

        buy_score = 100.0 * (
            0.22*near20low +
            0.18*near60low +
            0.25*higher_low +
            0.20*above_ma20 +
            0.15*max(0.0, min(ret5/8.0, 1.0))
        )

        sell_score = 100.0 * (
            0.22*near20high +
            0.18*near60high +
            0.25*lower_high +
            0.20*(1.0-above_ma20) +
            0.15*max(0.0, min(-ret5/8.0, 1.0))
        )

        rows.append({
            "symbol": sym,
            "sector": sectors.get(base.base_symbol(sym), "UNKNOWN"),
            "last_close": last,
            "ret5_pct": ret5,
            "above_ma20": bool(above_ma20),
            "higher_low_5": bool(higher_low),
            "lower_high_5": bool(lower_high),
            "buy_rank_score": round(buy_score, 2),
            "sell_rank_score": round(sell_score, 2),
        })

    rows.sort(
        key=lambda r: r["buy_rank_score"] if market_bias == "BUY_WATCH" else r["sell_rank_score"],
        reverse=True
    )
    return rows[:40]


def compute_live_state():
    base = _load_base()

    stocks = base.load_stocks(None)
    index_df, index_source = base.load_index()
    sectors = base.fetch_sector_map()

    cal = base.market_calendar(stocks, index_df)

    # Historical market-turn labels.
    pframes = []
    ranges = []
    for sym, df in stocks.items():
        p = base.detect_pivots(df)
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

    pivots = pd.concat(pframes, ignore_index=True) if pframes else pd.DataFrame()
    ranges = pd.DataFrame(ranges)
    pivots = base.add_cal_index(pivots, cal)
    _, market_turns = base.build_market_turns(pivots, ranges, cal)

    base.prepare_ephemeris(cal["date"])
    daily = base.build_daily_panel(stocks, cal, sectors)
    dailyp = base.add_planetary_daily_features(daily)
    labels = base.build_future_turn_labels(daily, market_turns)
    data = dailyp.merge(labels, on="date", how="inner").sort_values("date").reset_index(drop=True)

    scores = []
    for turn_type, h, model_type in MODEL_SPECS:
        label = f"future_{turn_type}_{h}"
        scores.append(_fit_current_model(base, data, label, model_type, h))

    low = [x for x in scores if x["turn_type"] == "LOW"]
    high = [x for x in scores if x["turn_type"] == "HIGH"]

    low_alerts = sum(int(x["alert"]) for x in low)
    high_alerts = sum(int(x["alert"]) for x in high)

    if low_alerts >= 2 and low_alerts > high_alerts:
        bias = "BUY_WATCH"
        action_ar = "مراقبة شراء / احتمال قاع سوق قريب"
    elif high_alerts >= 2 and high_alerts > low_alerts:
        bias = "SELL_WATCH"
        action_ar = "مراقبة بيع / احتمال قمة سوق قريبة"
    elif low_alerts == 3 and high_alerts == 3:
        bias = "CONFLICT"
        action_ar = "تعارض إشارات — لا قرار اتجاهي"
    else:
        bias = "NEUTRAL"
        action_ar = "محايد — لا توجد إشارة سوق كافية"

    latest_date = pd.Timestamp(data.iloc[-1]["date"])
    rankings = _current_stock_rankings(base, stocks, sectors, latest_date, bias)

    latest_features = daily.iloc[-1].to_dict()

    return {
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_date": latest_date.date().isoformat(),
        "index_source": index_source,
        "index_independent": ("SYNTHETIC" not in index_source.upper()),
        "market_bias": bias,
        "action_ar": action_ar,
        "low_alert_count": low_alerts,
        "high_alert_count": high_alerts,
        "scores": scores,
        "breadth_snapshot": {
            k: (None if pd.isna(v) else float(v))
            for k, v in latest_features.items()
            if k != "date" and isinstance(v, (int, float, np.integer, np.floating))
        },
        "top_stocks": rankings,
        "note": (
            "LOW production model uses price/breadth/sector features only. "
            "HIGH production model adds planetary features because that was the "
            "only side where planets improved out-of-sample performance."
        ),
    }


def save_state(DB, state):
    payload = json.dumps(state, ensure_ascii=False)
    with DB.begin() as s:
        row = s.get(Setting, STATE_KEY)
        if row is None:
            row = Setting(key=STATE_KEY, value=payload)
            s.add(row)
        else:
            row.value = payload


def load_state(session):
    row = session.get(Setting, STATE_KEY)
    if row is None or not row.value:
        return None
    try:
        return json.loads(row.value)
    except Exception:
        return None


def refresh(DB):
    state = compute_live_state()
    save_state(DB, state)
    return state
