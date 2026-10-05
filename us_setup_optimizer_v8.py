#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Market Setup Optimizer
-------------------------
Searches combinations of:
- R² minimum
- Slope entry range
- CCI period/range (optional)
- TP
- Time Exit
- Slope Exit
- Early Failure Exit
- Giveback Exit

Goal:
- roughly 250-350 trades per FULL year (target 300)
- average NET trade >= 4% after round-trip costs
- winner/loser payoff ratio >= 4:1
- prefer stability across years, not one lucky year

Data source:
- Railway PostgreSQL table: market_candles_1d

Usage example:
  cd /app
  python us_setup_optimizer.py --cost 0.30 --configs 600

Cost is round-trip trading cost in percent points.
Example --cost 0.30 means subtract 0.30 percentage points from every closed trade.

The script writes:
  /data/us_optimizer_results.csv
  /data/us_optimizer_top20.csv
  /data/us_optimizer_report.txt
"""

import os, sys, math, json, time, random, argparse, subprocess
from pathlib import Path

def ensure(pkg, import_name=None):
    import_name = import_name or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(import_name)
    except Exception:
        print(f"[setup] installing {pkg} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

ensure("numpy")
ensure("pandas")
ensure("sqlalchemy")
try:
    import psycopg2  # noqa
except Exception:
    ensure("psycopg2-binary", "psycopg2")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text

OUT_DIR = Path(os.getenv("OPTIMIZER_OUT_DIR", "/data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "us_optimizer_results.csv"
TOP_CSV = OUT_DIR / "us_optimizer_top20.csv"
REPORT_TXT = OUT_DIR / "us_optimizer_report.txt"
CACHE_PARQUET = OUT_DIR / "us_optimizer_features.parquet"

TABLE = os.getenv("US_DAILY_TABLE", "market_candles_1d")
VERSION = "8.0-cross-sectional-ranking-20261005"
SEED = 20261005
LOOKBACK = 126
FULL_YEARS = []
HOLDOUT_YEAR = None

def db_url():
    u = os.getenv("DATABASE_URL", "").strip()
    if not u:
        raise RuntimeError("DATABASE_URL is missing.")
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://"):]
    return u

def pick(cols, *names):
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def detect_year_windows(df):
    """
    Detect complete calendar years from actual stored dates.
    A complete US year must have data near both the start and end of the year.
    The latest incomplete year is used as holdout.
    """
    global FULL_YEARS, HOLDOUT_YEAR
    by_year = df.groupby(df["d"].dt.year)["d"].agg(["min","max","nunique"])
    full = []
    for y, row in by_year.iterrows():
        # first US trading days are in early January, last in late December
        start_ok = row["min"] <= pd.Timestamp(int(y), 1, 10)
        end_ok = row["max"] >= pd.Timestamp(int(y), 12, 20)
        enough_days = int(row["nunique"]) >= 200
        if start_ok and end_ok and enough_days:
            full.append(int(y))
    if not full:
        raise RuntimeError(f"No complete calendar years detected. Year coverage:\n{by_year}")
    FULL_YEARS = full
    max_year = int(df["d"].dt.year.max())
    HOLDOUT_YEAR = max_year if max_year not in FULL_YEARS else None
    print(f"[version] {VERSION}", flush=True)
    print(f"[years] complete years used for 250-350 constraint: {FULL_YEARS}", flush=True)
    print(f"[years] holdout/partial year: {HOLDOUT_YEAR}", flush=True)
    print("[years] coverage:", flush=True)
    for y,row in by_year.iterrows():
        print(f"  {int(y)}: {row['min'].date()} -> {row['max'].date()} | trading_dates={int(row['nunique'])}", flush=True)

def load_daily():
    eng = create_engine(db_url(), pool_pre_ping=True)
    insp = inspect(eng)
    if TABLE not in insp.get_table_names():
        raise RuntimeError(f"Table {TABLE!r} not found.")
    cols = [c["name"] for c in insp.get_columns(TABLE)]
    c_symbol = pick(cols, "symbol", "ticker", "feed_symbol")
    c_date = pick(cols, "date", "day", "trade_date", "ts", "timestamp", "time")
    c_open = pick(cols, "open", "o", "adj_open")
    c_high = pick(cols, "high", "h", "adj_high")
    c_low = pick(cols, "low", "l", "adj_low")
    c_close = pick(cols, "close", "c", "adj_close")
    c_vol = pick(cols, "volume", "v", "vol")
    missing = [k for k,v in {
        "symbol":c_symbol,"date":c_date,"open":c_open,"high":c_high,
        "low":c_low,"close":c_close,"volume":c_vol}.items() if not v]
    if missing:
        raise RuntimeError(f"Cannot map required columns {missing}. Columns are: {cols}")

    # If a market column exists, filter US. Otherwise the table is assumed US-only.
    c_market = pick(cols, "market", "market_key")
    where = ""
    if c_market:
        where = f" WHERE UPPER(CAST({c_market} AS TEXT)) IN ('US','USA') "

    q = f"""
        SELECT
            {c_symbol} AS symbol,
            {c_date} AS d,
            {c_open} AS o,
            {c_high} AS h,
            {c_low} AS l,
            {c_close} AS c,
            {c_vol} AS v
        FROM {TABLE}
        {where}
        ORDER BY {c_symbol}, {c_date}
    """
    print("[data] loading daily US candles from PostgreSQL ...", flush=True)
    chunks = []
    with eng.connect() as con:
        for ch in pd.read_sql_query(text(q), con, chunksize=250_000):
            chunks.append(ch)
            print(f"[data] rows loaded: {sum(len(x) for x in chunks):,}", flush=True)
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # Robust date parsing. In market_candles_1d the date column may be stored
    # as Unix seconds; plain pd.to_datetime(numeric) treats numbers as ns and
    # collapses them into 1970-01-01.
    raw_d = df["d"]
    if pd.api.types.is_numeric_dtype(raw_d):
        vals = pd.to_numeric(raw_d, errors="coerce")
        sample = vals.dropna()
        med = float(sample.median()) if len(sample) else float("nan")
        if math.isfinite(med):
            # YYYYMMDD integer, e.g. 20261002
            if 19000101 <= med <= 21001231:
                df["d"] = pd.to_datetime(vals.round().astype("Int64").astype(str),
                                         format="%Y%m%d", errors="coerce", utc=True)
            # Unix seconds
            elif 1e8 <= abs(med) < 1e11:
                df["d"] = pd.to_datetime(vals, unit="s", errors="coerce", utc=True)
            # Unix milliseconds
            elif 1e11 <= abs(med) < 1e14:
                df["d"] = pd.to_datetime(vals, unit="ms", errors="coerce", utc=True)
            # Unix microseconds
            elif 1e14 <= abs(med) < 1e17:
                df["d"] = pd.to_datetime(vals, unit="us", errors="coerce", utc=True)
            else:
                df["d"] = pd.to_datetime(vals, errors="coerce", utc=True)
        else:
            df["d"] = pd.NaT
    else:
        df["d"] = pd.to_datetime(raw_d, errors="coerce", utc=True)

    df["d"] = df["d"].dt.tz_convert(None).dt.normalize()

    for c in ["o","h","l","c","v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["symbol","d","o","h","l","c"])
    df = df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)]
    df = df.drop_duplicates(["symbol","d"], keep="last").sort_values(["symbol","d"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No usable daily rows after date parsing.")
    dmin, dmax = df["d"].min(), df["d"].max()
    if dmax.year < 2018 or dmin.year > 2030:
        raise RuntimeError(f"Date parsing still looks wrong: {dmin} -> {dmax}")
    print(f"[data] final rows={len(df):,} symbols={df.symbol.nunique():,} range={dmin.date()} -> {dmax.date()}", flush=True)
    detect_year_windows(df)
    return df

def linreg_features(close):
    """Previous 126 sessions only; current bar excluded."""
    n = len(close)
    slope = np.full(n, np.nan, dtype=np.float32)
    r2 = np.full(n, np.nan, dtype=np.float32)
    x = np.arange(LOOKBACK, dtype=np.float64)
    sx = x.sum()
    sx2 = (x*x).sum()
    denx = LOOKBACK*sx2 - sx*sx
    for i in range(LOOKBACK, n):
        y = close[i-LOOKBACK:i].astype(np.float64, copy=False)
        if np.any(~np.isfinite(y)) or y[0] <= 0:
            continue
        sy = y.sum()
        sy2 = (y*y).sum()
        sxy = (x*y).sum()
        b = (LOOKBACK*sxy - sx*sy) / denx
        a = (sy - b*sx) / LOOKBACK
        pred = a + b*x
        ss_tot = ((y-y.mean())**2).sum()
        ss_res = ((y-pred)**2).sum()
        rr = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
        slope[i] = np.float32(100.0 * b * (LOOKBACK-1) / y[0])
        r2[i] = np.float32(rr)
    return slope, r2

def cci_series(h,l,c,period):
    tp = (h+l+c)/3.0
    s = pd.Series(tp)
    ma = s.rolling(period, min_periods=period).mean()
    # exact rolling mean absolute deviation around rolling mean
    vals = np.full(len(tp), np.nan, dtype=np.float32)
    a = tp.astype(np.float64)
    m = ma.to_numpy(dtype=np.float64)
    for i in range(period-1, len(tp)):
        w = a[i-period+1:i+1]
        md = np.mean(np.abs(w - m[i]))
        vals[i] = 0.0 if md == 0 else np.float32((a[i]-m[i])/(0.015*md))
    return vals

def build_features(df):
    print("[features] computing slope/R²/CCI/confirmation ...", flush=True)
    n = len(df)
    slope = np.full(n, np.nan, dtype=np.float32)
    r2 = np.full(n, np.nan, dtype=np.float32)
    cci14 = np.full(n, np.nan, dtype=np.float32)
    cci20 = np.full(n, np.nan, dtype=np.float32)
    cci30 = np.full(n, np.nan, dtype=np.float32)
    cci40 = np.full(n, np.nan, dtype=np.float32)
    cci50 = np.full(n, np.nan, dtype=np.float32)
    cci60 = np.full(n, np.nan, dtype=np.float32)
    confirm = np.zeros(n, dtype=bool)
    rvol20 = np.full(n, np.nan, dtype=np.float32)
    dollar_vol20 = np.full(n, np.nan, dtype=np.float64)
    atr_pct14 = np.full(n, np.nan, dtype=np.float32)
    breakout5 = np.zeros(n, dtype=bool)
    breakout10 = np.zeros(n, dtype=bool)
    breakout20 = np.zeros(n, dtype=bool)
    sma50_ok = np.zeros(n, dtype=bool)
    sma100_ok = np.zeros(n, dtype=bool)
    sma200_ok = np.zeros(n, dtype=bool)
    ret5 = np.full(n, np.nan, dtype=np.float32)
    ret20 = np.full(n, np.nan, dtype=np.float32)
    ret60 = np.full(n, np.nan, dtype=np.float32)
    range_exp = np.full(n, np.nan, dtype=np.float32)
    close_loc = np.full(n, np.nan, dtype=np.float32)
    high60_strength = np.full(n, np.nan, dtype=np.float32)
    sym_id = np.empty(n, dtype=np.int32)
    end_idx = np.empty(n, dtype=np.int32)

    groups = []
    sid = 0
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index.to_numpy()
        a,b = idx[0], idx[-1]+1
        groups.append((sym,a,b))
        o = df.loc[idx,"o"].to_numpy(np.float64)
        h = df.loc[idx,"h"].to_numpy(np.float64)
        l = df.loc[idx,"l"].to_numpy(np.float64)
        c = df.loc[idx,"c"].to_numpy(np.float64)

        sl, rr = linreg_features(c)
        slope[a:b] = sl; r2[a:b] = rr
        cci14[a:b] = cci_series(h,l,c,14)
        cci20[a:b] = cci_series(h,l,c,20)
        cci30[a:b] = cci_series(h,l,c,30)
        cci40[a:b] = cci_series(h,l,c,40)
        cci50[a:b] = cci_series(h,l,c,50)
        cci60[a:b] = cci_series(h,l,c,60)

        cf = np.zeros(len(c), dtype=bool)
        if len(c)>1:
            cf[1:] = (c[1:] > h[:-1]) & (c[1:] > o[1:]) & ((c[1:]-o[1:]) > (h[1:]-c[1:]))
        confirm[a:b] = cf

        # Cross-sectional quality / tradability features.
        vv = df.loc[idx,"v"].to_numpy(np.float64)
        ps_v = pd.Series(vv)
        ps_c = pd.Series(c)
        vol20_prev = ps_v.shift(1).rolling(20, min_periods=20).mean().to_numpy(np.float64)
        dv20_prev = (pd.Series(vv*c).shift(1).rolling(20, min_periods=20).mean()).to_numpy(np.float64)
        rv = np.divide(vv, vol20_prev, out=np.full(len(c), np.nan), where=vol20_prev>0)
        rvol20[a:b] = rv.astype(np.float32)
        dollar_vol20[a:b] = dv20_prev

        prev_c = np.r_[np.nan, c[:-1]]
        tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
        atr = pd.Series(tr).shift(1).rolling(14, min_periods=14).mean().to_numpy(np.float64)
        atrp = np.divide(100.0*atr, c, out=np.full(len(c), np.nan), where=c>0)
        atr_pct14[a:b] = atrp.astype(np.float32)

        for win, dest in ((5,breakout5),(10,breakout10),(20,breakout20)):
            prior_high = pd.Series(h).shift(1).rolling(win, min_periods=win).max().to_numpy(np.float64)
            dest[a:b] = c > prior_high

        sma50 = pd.Series(c).shift(1).rolling(50, min_periods=50).mean().to_numpy(np.float64)
        sma100 = pd.Series(c).shift(1).rolling(100, min_periods=100).mean().to_numpy(np.float64)
        sma200 = pd.Series(c).shift(1).rolling(200, min_periods=200).mean().to_numpy(np.float64)
        sma50_ok[a:b] = c > sma50
        sma100_ok[a:b] = c > sma100
        sma200_ok[a:b] = c > sma200

        # Momentum / breakout-quality features used for cross-sectional ranking.
        def pct_ret(period):
            prev = pd.Series(c).shift(period).to_numpy(np.float64)
            return np.divide(c, prev, out=np.full(len(c), np.nan), where=prev>0) - 1.0

        ret5[a:b] = (100.0 * pct_ret(5)).astype(np.float32)
        ret20[a:b] = (100.0 * pct_ret(20)).astype(np.float32)
        ret60[a:b] = (100.0 * pct_ret(60)).astype(np.float32)

        # Today's true-range expansion versus PRIOR 14-day ATR.
        rx = np.divide(tr, atr, out=np.full(len(c), np.nan), where=atr>0)
        range_exp[a:b] = rx.astype(np.float32)

        day_range = h-l
        cl = np.divide(c-l, day_range, out=np.full(len(c), 0.5), where=day_range>0)
        close_loc[a:b] = cl.astype(np.float32)

        prior_h60 = pd.Series(h).shift(1).rolling(60, min_periods=60).max().to_numpy(np.float64)
        hs = np.divide(c, prior_h60, out=np.full(len(c), np.nan), where=prior_h60>0)
        high60_strength[a:b] = hs.astype(np.float32)

        sym_id[a:b] = sid
        end_idx[a:b] = b
        sid += 1
        if sid % 500 == 0:
            print(f"[features] {sid:,} symbols", flush=True)

    out = df[["symbol","d","o","h","l","c","v"]].copy()
    out["slope"] = slope
    out["r2"] = r2
    out["confirm"] = confirm
    out["rvol20"] = rvol20
    out["dollar_vol20"] = dollar_vol20
    out["atr_pct14"] = atr_pct14
    out["breakout5"] = breakout5
    out["breakout10"] = breakout10
    out["breakout20"] = breakout20
    out["sma50_ok"] = sma50_ok
    out["sma100_ok"] = sma100_ok
    out["sma200_ok"] = sma200_ok
    out["ret5"] = ret5
    out["ret20"] = ret20
    out["ret60"] = ret60
    out["range_exp"] = range_exp
    out["close_loc"] = close_loc
    out["high60_strength"] = high60_strength

    print("[features] computing daily cross-sectional ranks ...", flush=True)
    rank_sources = {
        "rr2":"r2", "rslope":"slope", "rmom5":"ret5", "rmom20":"ret20",
        "rmom60":"ret60", "rrvol":"rvol20", "rrange":"range_exp",
        "rclose":"close_loc", "rhigh":"high60_strength"
    }
    for rk, col in rank_sources.items():
        out[rk] = out.groupby("d", sort=False)[col].rank(pct=True, method="average").astype("float32")

    # Four different ranking philosophies. The optimizer chooses the model,
    # rather than trying millions of arbitrary weight combinations.
    out["rank_trend"] = (
        .20*out["rr2"] + .15*out["rslope"] + .20*out["rmom20"] +
        .15*out["rmom60"] + .10*out["rrvol"] + .10*out["rhigh"] + .10*out["rclose"]
    ).astype("float32")
    out["rank_breakout"] = (
        .10*out["rr2"] + .10*out["rslope"] + .15*out["rmom5"] +
        .15*out["rmom20"] + .20*out["rrvol"] + .15*out["rrange"] +
        .10*out["rhigh"] + .05*out["rclose"]
    ).astype("float32")
    out["rank_smooth"] = (
        .30*out["rr2"] + .20*out["rslope"] + .20*out["rmom20"] +
        .15*out["rmom60"] + .10*out["rhigh"] + .05*out["rclose"]
    ).astype("float32")
    out["rank_accel"] = (
        .10*out["rr2"] + .10*out["rslope"] + .25*out["rmom5"] +
        .20*out["rmom20"] + .20*out["rrvol"] + .10*out["rrange"] + .05*out["rclose"]
    ).astype("float32")

    out["cci14"] = cci14
    out["cci20"] = cci20
    out["cci30"] = cci30
    out["cci40"] = cci40
    out["cci50"] = cci50
    out["cci60"] = cci60
    out["sym_id"] = sym_id
    out["end_idx"] = end_idx
    return out, groups

def random_config(rng):
    slope_min = rng.choice([10,20,30,40,50])
    slope_max_choices = [x for x in [55,65,75,90,110,140] if x >= slope_min+5]
    slope_max = rng.choice(slope_max_choices)
    cci_enabled = rng.random() < 0.45
    cci_period = rng.choice([20,40,60])
    cci_min = rng.choice([-100,-60,-30,-10])
    cci_max_choices = [x for x in [0,20,40,80,120] if x > cci_min]
    return {
        "r2_min": rng.choice([0.10,0.20,0.30,0.40,0.50,0.60,0.70]),
        "slope_min": float(slope_min),
        "slope_max": float(slope_max),
        "cci_enabled": int(cci_enabled),
        "cci_period": int(cci_period),
        "cci_min": float(cci_min),
        "cci_max": float(rng.choice(cci_max_choices)),
        "confirm_on": int(rng.random() < 0.35),
        "breakout_days": int(rng.choice([0,0,0,5,10,20])),
        "min_price": float(rng.choice([2,3,5,7,10,15,20])),
        "min_dollar_vol": float(rng.choice([3e6,5e6,10e6,15e6,25e6,50e6])),
        "rvol_min": float(rng.choice([0,0.75,1.0,1.25,1.5,2.0])),
        "atr_min": float(rng.choice([0,1,2,3])),
        "atr_max": float(rng.choice([8,10,12,15,20])),
        "regime_sma": int(rng.choice([0,50,100,200])),
        "cooldown": int(rng.choice([0,5,10,20,30])),
        # NEW: choose only the strongest cross-sectional opportunities each day.
        "rank_model": int(rng.choice([1,2,3,4])),
        "rank_score_min": float(rng.choice([0.60,0.65,0.70,0.75,0.80,0.85])),
        "max_entries_per_day": int(rng.choice([1,1,1,2,2,3])),
        "tp": float(rng.choice([12,15,18,21,24,27,30,35,40])),
        "time_exit": int(rng.choice([7,10,14,18,21,30,45])),
        "slope_exit": float(rng.choice([75,85,95,110,125,140,160])),
        "early_on": int(rng.random() < 0.55),
        "early_sessions": int(rng.choice([3,4,5,7])),
        "early_max_gain": float(rng.choice([2,3,4,5,7])),
        "early_return": float(rng.choice([-3,-4,-5,-7,-10])),
        "giveback_on": int(rng.random() < 0.55),
        "giveback_peak": float(rng.choice([5,7,10,12,15,20])),
        "giveback_return": float(rng.choice([1,0,-2,-3,-5])),
    }

def local_mutations(base, rng, count):
    vals=[]
    numeric_fields=[
        "r2_min","slope_min","slope_max","cci_min","cci_max","tp","time_exit",
        "slope_exit","early_sessions","early_max_gain","early_return","giveback_peak",
        "giveback_return","min_price","min_dollar_vol","rvol_min","atr_min","atr_max",
        "cooldown","rank_score_min"
    ]
    for _ in range(count):
        c=dict(base)
        for f in rng.sample(numeric_fields, k=rng.randint(3,7)):
            if f=="r2_min": c[f]=float(np.clip(c[f]+rng.choice([-0.10,-0.05,0.05,0.10]),0.0,0.95))
            elif f in ("slope_min","slope_max"): c[f]=float(max(0,c[f]+rng.choice([-10,-5,5,10])))
            elif f in ("cci_min","cci_max"): c[f]=float(c[f]+rng.choice([-20,-10,10,20]))
            elif f=="tp": c[f]=float(max(5,c[f]+rng.choice([-6,-3,3,6])))
            elif f=="time_exit": c[f]=int(max(3,c[f]+rng.choice([-7,-3,3,7,10])))
            elif f=="slope_exit": c[f]=float(max(50,c[f]+rng.choice([-20,-10,10,20])))
            elif f=="early_sessions": c[f]=int(max(2,c[f]+rng.choice([-2,-1,1,2])))
            elif f=="early_max_gain": c[f]=float(max(.5,c[f]+rng.choice([-2,-1,1,2])))
            elif f=="early_return": c[f]=float(c[f]+rng.choice([-3,-2,2,3]))
            elif f=="giveback_peak": c[f]=float(max(2,c[f]+rng.choice([-3,-2,2,3])))
            elif f=="giveback_return": c[f]=float(c[f]+rng.choice([-3,-2,2,3]))
            elif f=="min_price": c[f]=float(max(.5,c[f]+rng.choice([-5,-2,2,5])))
            elif f=="min_dollar_vol": c[f]=float(max(0,c[f]+rng.choice([-10e6,-5e6,5e6,10e6])))
            elif f=="rvol_min": c[f]=float(max(0,c[f]+rng.choice([-.5,-.25,.25,.5])))
            elif f=="atr_min": c[f]=float(max(0,c[f]+rng.choice([-2,-1,1,2])))
            elif f=="atr_max": c[f]=float(max(3,c[f]+rng.choice([-5,-2,2,5])))
            elif f=="cooldown": c[f]=int(max(0,c[f]+rng.choice([-10,-5,5,10])))
            elif f=="rank_score_min": c[f]=float(np.clip(c[f]+rng.choice([-.10,-.05,.05,.10]),.40,.95))
        if rng.random()<0.30: c["rank_model"]=rng.choice([1,2,3,4])
        if rng.random()<0.30: c["max_entries_per_day"]=rng.choice([1,2,3])
        if rng.random()<0.20: c["confirm_on"]=1-int(c["confirm_on"])
        if rng.random()<0.20: c["breakout_days"]=rng.choice([0,5,10,20])
        if rng.random()<0.15: c["regime_sma"]=rng.choice([0,50,100,200])
        if rng.random()<0.12: c["cci_enabled"]=1-int(c["cci_enabled"])
        if rng.random()<0.12: c["early_on"]=1-int(c["early_on"])
        if rng.random()<0.12: c["giveback_on"]=1-int(c["giveback_on"])
        if c["slope_max"]<=c["slope_min"]: c["slope_max"]=c["slope_min"]+5
        if c["cci_max"]<=c["cci_min"]: c["cci_max"]=c["cci_min"]+10
        if c["atr_max"]<=c["atr_min"]: c["atr_max"]=c["atr_min"]+2
        vals.append(c)
    return vals

def evaluate(cfg, A, cost_pct, collect=False):
    # Defensive normalization for configs loaded/mutated through pandas.
    cfg=dict(cfg)
    for k in ("cci_enabled","cci_period","time_exit","early_on","early_sessions","giveback_on",
              "confirm_on","breakout_days","regime_sma","cooldown","rank_model","max_entries_per_day"):
        if k in cfg:
            cfg[k]=int(round(float(cfg[k])))
    slope=A["slope"]; r2=A["r2"]; confirm=A["confirm"]
    d=A["date"]; years=A["year"]; h=A["h"]; c=A["c"]
    sym_id=A["sym_id"]; end_idx=A["end_idx"]

    mask = (
        np.isfinite(slope) & np.isfinite(r2) &
        (r2 >= cfg["r2_min"]) &
        (slope >= cfg["slope_min"]) &
        (slope <= cfg["slope_max"]) &
        (c >= cfg["min_price"])
    )
    if cfg["confirm_on"]:
        mask &= confirm
    if cfg["cci_enabled"]:
        cv=A[f"cci{cfg['cci_period']}"]
        mask &= np.isfinite(cv) & (cv >= cfg["cci_min"]) & (cv <= cfg["cci_max"])
    if cfg["min_dollar_vol"] > 0:
        dv=A["dollar_vol20"]; mask &= np.isfinite(dv) & (dv >= cfg["min_dollar_vol"])
    if cfg["rvol_min"] > 0:
        rv=A["rvol20"]; mask &= np.isfinite(rv) & (rv >= cfg["rvol_min"])
    at=A["atr_pct14"]; mask &= np.isfinite(at) & (at >= cfg["atr_min"]) & (at <= cfg["atr_max"])
    if cfg["breakout_days"] in (5,10,20):
        mask &= A[f"breakout{cfg['breakout_days']}"]
    if cfg["regime_sma"] in (50,100,200):
        mask &= A[f"sma{cfg['regime_sma']}_ok"]

    # Cross-sectional selection: from thousands of stocks, take only the
    # strongest N opportunities per trading day.
    model_map={1:"rank_trend",2:"rank_breakout",3:"rank_smooth",4:"rank_accel"}
    rs=A[model_map[cfg["rank_model"]]]
    mask &= np.isfinite(rs) & (rs >= cfg["rank_score_min"])
    idxs=np.flatnonzero(mask)

    if len(idxs):
        days=A["day_id"][idxs]
        scores=rs[idxs]
        order=np.lexsort((-scores, days))  # day asc, score desc
        si=idxs[order]
        sd=days[order]
        starts=np.r_[True, sd[1:] != sd[:-1]]
        start_pos=np.maximum.accumulate(np.where(starts, np.arange(len(sd)), 0))
        within=np.arange(len(sd)) - start_pos
        idxs=si[within < cfg["max_entries_per_day"]]

    last_exit = np.full(int(sym_id.max())+1, -1, dtype=np.int64)
    last_entry = np.full(int(sym_id.max())+1, -10**9, dtype=np.int64)
    rets=[]; entry_years=[]; trades=[]

    for i in idxs:
        y=int(years[i])
        if y < min(FULL_YEARS) or (HOLDOUT_YEAR is not None and y > HOLDOUT_YEAR):
            continue
        sid=int(sym_id[i])
        if i <= last_exit[sid]:
            continue
        if cfg["cooldown"] > 0 and (i-last_entry[sid]) < cfg["cooldown"]:
            continue
        entry=float(c[i])
        if not math.isfinite(entry) or entry<=0:
            continue
        stop=min(int(end_idx[i])-1, i+int(cfg["time_exit"]))
        if stop <= i:
            continue

        peak=0.0
        chosen=None
        for j in range(i+1, stop+1):
            peak=max(peak,100.0*(float(h[j])/entry-1.0))
            cur=100.0*(float(c[j])/entry-1.0)

            # same-bar priority: TP > early fail > giveback > slope > time
            if float(h[j]) >= entry*(1.0+cfg["tp"]/100.0):
                chosen=("TP",j,cfg["tp"])
                break
            if cfg["early_on"] and (j-i)<=cfg["early_sessions"] and peak < cfg["early_max_gain"] and cur <= cfg["early_return"]:
                chosen=("EARLY",j,cur)
                break
            if cfg["giveback_on"] and peak >= cfg["giveback_peak"] and cur <= cfg["giveback_return"]:
                chosen=("GIVEBACK",j,cur)
                break
            if math.isfinite(float(slope[j])) and float(slope[j]) >= cfg["slope_exit"]:
                chosen=("SLOPE",j,cur)
                break
            if j==stop:
                chosen=("TIME",j,cur)
                break

        if chosen is None:
            continue
        kind,j,gross=chosen
        net=float(gross)-cost_pct
        rets.append(net)
        entry_years.append(y)
        last_entry[sid]=i
        last_exit[sid]=j
        if collect:
            trades.append((i,j,y,kind,float(gross),net))

    if not rets:
        return None

    r=np.asarray(rets,dtype=np.float64)
    yy=np.asarray(entry_years,dtype=np.int16)
    wins=r[r>0]; losses=r[r<=0]
    avg_win=float(wins.mean()) if len(wins) else 0.0
    avg_loss=float(abs(losses.mean())) if len(losses) else 999.0
    payoff=(avg_win/avg_loss) if avg_loss>0 else 999.0
    win_rate=100.0*len(wins)/len(r)

    yearly={}
    for y in FULL_YEARS+(([HOLDOUT_YEAR]) if HOLDOUT_YEAR is not None else []):
        z=r[yy==y]
        yearly[y]={
            "n":int(len(z)),
            "avg":float(z.mean()) if len(z) else np.nan,
            "median":float(np.median(z)) if len(z) else np.nan,
            "win":float(100*(z>0).mean()) if len(z) else np.nan,
        }

    full_counts=[yearly[y]["n"] for y in FULL_YEARS]
    full_avgs=[yearly[y]["avg"] for y in FULL_YEARS if yearly[y]["n"]>0]
    avg_trades=float(np.mean(full_counts))
    min_trades=int(min(full_counts))
    max_trades=int(max(full_counts))
    avg_net=float(np.mean(full_avgs)) if full_avgs else -999.0
    min_year_avg=float(np.min(full_avgs)) if full_avgs else -999.0

    # Trade-equity drawdown proxy: one equal notional trade after another.
    curve=np.cumsum(r)
    peak=np.maximum.accumulate(np.r_[0.0,curve])
    dd=curve-peak[1:]
    worst_trade_dd=float(dd.min()) if len(dd) else 0.0

    frequency_ok = all(250 <= yearly[y]["n"] <= 350 for y in FULL_YEARS)
    near_frequency = all(150 <= yearly[y]["n"] <= 500 for y in FULL_YEARS)
    min_sample_ok = all(yearly[y]["n"] >= 100 for y in FULL_YEARS)

    qualified = (
        frequency_ok and
        avg_net >= 4.0 and
        payoff >= 4.0
    )

    # Frequency-first ranking:
    # Tiny-sample setups must NEVER outrank 250-350-trade setups just because
    # one lucky trade made +40%. This was the v6 ranking bug.
    freq_distance = sum(abs(yearly[y]["n"]-300) for y in FULL_YEARS)
    if frequency_ok:
        tier = 3
    elif near_frequency:
        tier = 2
    elif min_sample_ok:
        tier = 1
    else:
        tier = 0

    # Within the same frequency tier, reward return/payoff/stability.
    # Tier dominates all other terms.
    score = (
        tier * 100000.0
        - freq_distance * 25.0
        + avg_net * 120.0
        + min_year_avg * 60.0
        + min(payoff, 10.0) * 50.0
        + win_rate * 2.0
        + worst_trade_dd * 0.10
        + (500000.0 if qualified else 0.0)
    )

    row=dict(cfg)
    row.update({
        "qualified":int(qualified),
        "score":score,
        "avg_trades_full_year":avg_trades,
        "min_trades_full_year":min_trades,
        "max_trades_full_year":max_trades,
        "frequency_ok":int(frequency_ok),
        "near_frequency":int(near_frequency),
        "frequency_tier":int(tier),
        "avg_net_full_year_pct":avg_net,
        "min_year_avg_net_pct":min_year_avg,
        "payoff_ratio":payoff,
        "win_rate_pct":win_rate,
        "avg_win_pct":avg_win,
        "avg_loss_abs_pct":avg_loss,
        "all_trades":len(r),
        "trade_equity_dd_points":worst_trade_dd,
    })
    for y in FULL_YEARS+(([HOLDOUT_YEAR]) if HOLDOUT_YEAR is not None else []):
        row[f"n_{y}"]=yearly[y]["n"]
        row[f"avg_{y}"]=yearly[y]["avg"]
        row[f"win_{y}"]=yearly[y]["win"]

    if collect:
        return row,trades
    return row

def prepare_arrays(feat):
    return {
        "slope":feat["slope"].to_numpy(np.float32),
        "r2":feat["r2"].to_numpy(np.float32),
        "confirm":feat["confirm"].to_numpy(bool),
        "rvol20":feat["rvol20"].to_numpy(np.float32),
        "dollar_vol20":feat["dollar_vol20"].to_numpy(np.float64),
        "atr_pct14":feat["atr_pct14"].to_numpy(np.float32),
        "breakout5":feat["breakout5"].to_numpy(bool),
        "breakout10":feat["breakout10"].to_numpy(bool),
        "breakout20":feat["breakout20"].to_numpy(bool),
        "sma50_ok":feat["sma50_ok"].to_numpy(bool),
        "sma100_ok":feat["sma100_ok"].to_numpy(bool),
        "sma200_ok":feat["sma200_ok"].to_numpy(bool),
        "rank_trend":feat["rank_trend"].to_numpy(np.float32),
        "rank_breakout":feat["rank_breakout"].to_numpy(np.float32),
        "rank_smooth":feat["rank_smooth"].to_numpy(np.float32),
        "rank_accel":feat["rank_accel"].to_numpy(np.float32),
        "day_id":pd.factorize(feat["d"], sort=True)[0].astype(np.int32),
        "cci14":feat["cci14"].to_numpy(np.float32),
        "cci20":feat["cci20"].to_numpy(np.float32),
        "cci30":feat["cci30"].to_numpy(np.float32),
        "cci40":feat["cci40"].to_numpy(np.float32),
        "cci50":feat["cci50"].to_numpy(np.float32),
        "cci60":feat["cci60"].to_numpy(np.float32),
        "date":feat["d"].to_numpy(),
        "year":feat["d"].dt.year.to_numpy(np.int16),
        "h":feat["h"].to_numpy(np.float64),
        "c":feat["c"].to_numpy(np.float64),
        "sym_id":feat["sym_id"].to_numpy(np.int32),
        "end_idx":feat["end_idx"].to_numpy(np.int32),
    }

def run_search(A, cost, configs, stage2):
    rng=random.Random(SEED)
    cfgs=[]
    seen=set()
    while len(cfgs)<configs:
        c=random_config(rng)
        key=json.dumps(c,sort_keys=True)
        if key not in seen:
            seen.add(key);cfgs.append(c)

    results=[]
    t0=time.time()
    for k,cfg in enumerate(cfgs,1):
        row=evaluate(cfg,A,cost)
        if row:results.append(row)
        if k%25==0 or k==len(cfgs):
            elapsed=(time.time()-t0)/60
            print(f"[stage1] {k}/{len(cfgs)} configs | results={len(results)} | {elapsed:.1f} min", flush=True)
            if results:
                pd.DataFrame(results).sort_values(["frequency_tier","qualified","score"],ascending=[False,False,False]).to_csv(RESULTS_CSV,index=False)

    if not results:
        raise RuntimeError(
            "No configurations produced trades. Check the parsed date range and feature filters."
        )
    df=pd.DataFrame(results).sort_values(["frequency_tier","qualified","score"],
                                         ascending=[False,False,False]).reset_index(drop=True)

    # Local search must start around configs with enough trades.
    # Prefer exact 250-350/year; if none, use near-frequency 150-500/year.
    exact = df[df["frequency_ok"]==1]
    near = df[df["near_frequency"]==1]
    if len(exact):
        seeds = exact.head(min(24,len(exact)))
        print(f"[stage2] seeding from {len(seeds)} exact-frequency configs", flush=True)
    elif len(near):
        seeds = near.head(min(24,len(near)))
        print(f"[stage2] no exact-frequency seeds; using {len(seeds)} near-frequency configs", flush=True)
    else:
        max_tier=int(df["frequency_tier"].max())
        seeds=df[df["frequency_tier"]==max_tier].head(min(24,len(df)))
        print(f"[stage2] WARNING: no near-frequency configs; using tier={max_tier} seeds", flush=True)
    local=[]
    for _,r in seeds.iterrows():
        base={k:r[k] for k in [
            "r2_min","slope_min","slope_max","cci_enabled","cci_period","cci_min","cci_max",
            "confirm_on","breakout_days","min_price","min_dollar_vol","rvol_min","atr_min","atr_max",
            "regime_sma","cooldown","rank_model","rank_score_min","max_entries_per_day",
            "tp","time_exit","slope_exit","early_on","early_sessions",
            "early_max_gain","early_return","giveback_on","giveback_peak","giveback_return"
        ]}
        # pandas upcasts mixed numeric rows to floats. Restore integer-like fields
        # so keys such as cci20 remain cci20 instead of cci20.0.
        for k in ("cci_enabled","cci_period","time_exit","early_on","early_sessions","giveback_on","confirm_on","breakout_days","regime_sma","cooldown","rank_model","max_entries_per_day"):
            base[k]=int(round(float(base[k])))
        for k in ("r2_min","slope_min","slope_max","cci_min","cci_max","tp","slope_exit",
                  "early_max_gain","early_return","giveback_peak","giveback_return","rank_score_min"):
            base[k]=float(base[k])
        local.extend(local_mutations(base,rng,max(1,stage2//max(1,len(seeds)))))
    local=local[:stage2]

    for k,cfg in enumerate(local,1):
        row=evaluate(cfg,A,cost)
        if row:results.append(row)
        if k%25==0 or k==len(local):
            print(f"[stage2] {k}/{len(local)} local configs", flush=True)

    all_df=pd.DataFrame(results).drop_duplicates(subset=[
        "r2_min","slope_min","slope_max","cci_enabled","cci_period","cci_min","cci_max",
        "confirm_on","breakout_days","min_price","min_dollar_vol","rvol_min","atr_min","atr_max",
        "regime_sma","cooldown","tp","time_exit","slope_exit","early_on","early_sessions",
        "early_max_gain","early_return","giveback_on","giveback_peak","giveback_return"
    ])
    all_df=all_df.sort_values(["frequency_tier","qualified","score"],
                              ascending=[False,False,False]).reset_index(drop=True)
    all_df.to_csv(RESULTS_CSV,index=False)
    all_df.head(20).to_csv(TOP_CSV,index=False)
    return all_df

def report(df, cost):
    top=df.head(20)
    lines=[]
    lines.append("US SETUP OPTIMIZER REPORT")
    lines.append("="*80)
    lines.append(f"Round-trip cost deducted from every trade: {cost:.3f}%")
    lines.append(f"Complete years used for frequency constraint: {FULL_YEARS}")
    lines.append(f"Partial/holdout year: {HOLDOUT_YEAR}")
    lines.append("Target constraints: 250-350 trades in EACH complete year; avg net >=4%; payoff >=4:1")
    lines.append("Rank models: 1=trend, 2=breakout, 3=smooth trend, 4=acceleration")
    lines.append("")
    qualified=df[df["qualified"]==1]
    lines.append(f"Qualified configurations: {len(qualified)} / {len(df)}")
    lines.append("")
    cols=[
        "qualified","frequency_tier","frequency_ok","near_frequency","score",
        "avg_trades_full_year","avg_net_full_year_pct","payoff_ratio",
        "win_rate_pct","min_year_avg_net_pct","r2_min","slope_min","slope_max",
        "cci_enabled","cci_period","cci_min","cci_max","confirm_on","breakout_days",
        "min_price","min_dollar_vol","rvol_min","atr_min","atr_max","regime_sma","cooldown",
        "rank_model","rank_score_min","max_entries_per_day",
        "tp","time_exit","slope_exit",
        "early_on","early_sessions","early_max_gain","early_return",
        "giveback_on","giveback_peak","giveback_return"
    ]
    for y in FULL_YEARS + (([HOLDOUT_YEAR]) if HOLDOUT_YEAR is not None else []):
        cols += [f"n_{y}", f"avg_{y}", f"win_{y}"]
    lines.append(top[cols].to_string(index=False))

    freq = df[(df["frequency_ok"]==1)].head(30)
    lines.append("")
    lines.append("="*80)
    lines.append("BEST CONFIGS WITH 250-350 TRADES IN EACH COMPLETE YEAR")
    lines.append("="*80)
    if len(freq):
        lines.append(freq[cols].to_string(index=False))
    else:
        lines.append("No exact-frequency configs found.")

    REPORT_TXT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)

def main():
    print(f"[version] {VERSION}", flush=True)
    ap=argparse.ArgumentParser()
    ap.add_argument("--cost",type=float,default=float(os.getenv("ROUND_TRIP_COST_PCT","0")),
                    help="Round-trip cost percentage points per trade, e.g. 0.30")
    ap.add_argument("--configs",type=int,default=8000,help="Stage-1 random configurations")
    ap.add_argument("--stage2",type=int,default=4000,help="Local mutations around frequency-matched stage-1 configs")
    args=ap.parse_args()

    df=load_daily()
    feat,_groups=build_features(df)
    A=prepare_arrays(feat)
    del df,feat
    print(f"[search] stage1={args.configs} stage2={args.stage2} cost={args.cost:.3f}%", flush=True)
    res=run_search(A,args.cost,args.configs,args.stage2)
    report(res,args.cost)
    print(f"\nSaved:\n{RESULTS_CSV}\n{TOP_CSV}\n{REPORT_TXT}", flush=True)

if __name__=="__main__":
    main()
