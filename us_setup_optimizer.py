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
SEED = 20261005
LOOKBACK = 126
FULL_YEARS = [2021, 2022, 2023, 2024, 2025]
HOLDOUT_YEAR = 2026

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
        sym_id[a:b] = sid
        end_idx[a:b] = b
        sid += 1
        if sid % 500 == 0:
            print(f"[features] {sid:,} symbols", flush=True)

    out = df[["symbol","d","o","h","l","c","v"]].copy()
    out["slope"] = slope
    out["r2"] = r2
    out["confirm"] = confirm
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
    slope_min = rng.choice([20,25,30,35,40,45,50,55])
    slope_max_choices = [x for x in [55,60,65,70,80,90] if x >= slope_min+5]
    slope_max = rng.choice(slope_max_choices or [slope_min+10])
    cci_enabled = rng.random() < 0.85
    cci_period = rng.choice([14,20,30,40,50,60])
    cci_min = rng.choice([-100,-80,-70,-60,-55,-50,-40,-30,-20])
    cci_max_choices = [x for x in [-40,-30,-20,-10,0,20,40,60] if x > cci_min]
    cci_max = rng.choice(cci_max_choices or [cci_min+20])
    return {
        "r2_min": rng.choice([0.20,0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90]),
        "slope_min": float(slope_min),
        "slope_max": float(slope_max),
        "cci_enabled": int(cci_enabled),
        "cci_period": int(cci_period),
        "cci_min": float(cci_min),
        "cci_max": float(cci_max),
        "tp": float(rng.choice([12,15,18,21,24,27,30])),
        "time_exit": int(rng.choice([5,7,10,12,14,18,21,30])),
        "slope_exit": float(rng.choice([65,70,75,80,85,90,95,100,110,120])),
        "early_on": int(rng.random() < 0.75),
        "early_sessions": int(rng.choice([3,4,5,7])),
        "early_max_gain": float(rng.choice([2,3,4,5,7])),
        "early_return": float(rng.choice([-3,-4,-5,-7,-10])),
        "giveback_on": int(rng.random() < 0.70),
        "giveback_peak": float(rng.choice([5,7,10,12,15])),
        "giveback_return": float(rng.choice([0,-2,-3,-5])),
    }

def local_mutations(base, rng, count):
    vals = []
    for _ in range(count):
        c = dict(base)
        # Mutate 2-5 fields around the winner.
        fields = rng.sample([
            "r2_min","slope_min","slope_max","cci_min","cci_max","tp","time_exit",
            "slope_exit","early_sessions","early_max_gain","early_return",
            "giveback_peak","giveback_return"
        ], k=rng.randint(2,5))
        for f in fields:
            if f=="r2_min":
                c[f]=float(np.clip(c[f]+rng.choice([-0.10,-0.05,0.05,0.10]),0.05,0.95))
            elif f=="slope_min":
                c[f]=float(max(10,c[f]+rng.choice([-10,-5,5,10])))
            elif f=="slope_max":
                c[f]=float(min(120,c[f]+rng.choice([-10,-5,5,10])))
            elif f in ("cci_min","cci_max"):
                c[f]=float(c[f]+rng.choice([-20,-10,10,20]))
            elif f=="tp":
                c[f]=float(max(5,c[f]+rng.choice([-6,-3,3,6])))
            elif f=="time_exit":
                c[f]=int(max(3,c[f]+rng.choice([-5,-3,3,5])))
            elif f=="slope_exit":
                c[f]=float(max(50,c[f]+rng.choice([-15,-10,-5,5,10,15])))
            elif f=="early_sessions":
                c[f]=int(max(2,c[f]+rng.choice([-2,-1,1,2])))
            elif f=="early_max_gain":
                c[f]=float(max(0.5,c[f]+rng.choice([-2,-1,1,2])))
            elif f=="early_return":
                c[f]=float(c[f]+rng.choice([-3,-2,2,3]))
            elif f=="giveback_peak":
                c[f]=float(max(2,c[f]+rng.choice([-3,-2,2,3])))
            elif f=="giveback_return":
                c[f]=float(c[f]+rng.choice([-3,-2,2,3]))
        if c["slope_max"] <= c["slope_min"]:
            c["slope_max"] = c["slope_min"] + 5
        if c["cci_max"] <= c["cci_min"]:
            c["cci_max"] = c["cci_min"] + 10
        vals.append(c)
    return vals

def evaluate(cfg, A, cost_pct, collect=False):
    slope=A["slope"]; r2=A["r2"]; confirm=A["confirm"]
    d=A["date"]; years=A["year"]; h=A["h"]; c=A["c"]
    sym_id=A["sym_id"]; end_idx=A["end_idx"]

    mask = (
        np.isfinite(slope) & np.isfinite(r2) &
        (r2 >= cfg["r2_min"]) &
        (slope >= cfg["slope_min"]) &
        (slope <= cfg["slope_max"]) &
        confirm
    )
    if cfg["cci_enabled"]:
        cv=A[f"cci{cfg['cci_period']}"]
        mask &= np.isfinite(cv) & (cv >= cfg["cci_min"]) & (cv <= cfg["cci_max"])

    # first day entering the condition episode, same symbol only
    prev = np.zeros_like(mask)
    prev[1:] = mask[:-1] & (sym_id[1:] == sym_id[:-1])
    activation = mask & ~prev
    idxs = np.flatnonzero(activation)

    last_exit = np.full(int(sym_id.max())+1, -1, dtype=np.int64)
    rets=[]; entry_years=[]; trades=[]

    for i in idxs:
        y=int(years[i])
        if y < min(FULL_YEARS) or y > HOLDOUT_YEAR:
            continue
        sid=int(sym_id[i])
        if i <= last_exit[sid]:
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
    for y in FULL_YEARS+[HOLDOUT_YEAR]:
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

    qualified = (
        250 <= avg_trades <= 350 and
        avg_net >= 4.0 and
        payoff >= 4.0
    )
    # Stability-weighted score. Constraints dominate ranking.
    score = (
        (1000 if qualified else 0)
        + avg_net*8
        + min_year_avg*3
        + min(payoff,10)*4
        + win_rate*0.15
        - abs(avg_trades-300)*0.25
        + worst_trade_dd*0.03
    )

    row=dict(cfg)
    row.update({
        "qualified":int(qualified),
        "score":score,
        "avg_trades_full_year":avg_trades,
        "min_trades_full_year":min_trades,
        "max_trades_full_year":max_trades,
        "avg_net_full_year_pct":avg_net,
        "min_year_avg_net_pct":min_year_avg,
        "payoff_ratio":payoff,
        "win_rate_pct":win_rate,
        "avg_win_pct":avg_win,
        "avg_loss_abs_pct":avg_loss,
        "all_trades":len(r),
        "trade_equity_dd_points":worst_trade_dd,
    })
    for y in FULL_YEARS+[HOLDOUT_YEAR]:
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
                pd.DataFrame(results).sort_values(["qualified","score"],ascending=[False,False]).to_csv(RESULTS_CSV,index=False)

    if not results:
        raise RuntimeError(
            "No configurations produced trades. Check the parsed date range and feature filters."
        )
    df=pd.DataFrame(results).sort_values(["qualified","score"],ascending=[False,False]).reset_index(drop=True)
    seeds=df.head(min(12,len(df)))
    local=[]
    for _,r in seeds.iterrows():
        base={k:r[k] for k in [
            "r2_min","slope_min","slope_max","cci_enabled","cci_period","cci_min","cci_max",
            "tp","time_exit","slope_exit","early_on","early_sessions","early_max_gain",
            "early_return","giveback_on","giveback_peak","giveback_return"
        ]}
        local.extend(local_mutations(base,rng,max(1,stage2//max(1,len(seeds)))))
    local=local[:stage2]

    for k,cfg in enumerate(local,1):
        row=evaluate(cfg,A,cost)
        if row:results.append(row)
        if k%25==0 or k==len(local):
            print(f"[stage2] {k}/{len(local)} local configs", flush=True)

    all_df=pd.DataFrame(results).drop_duplicates(subset=[
        "r2_min","slope_min","slope_max","cci_enabled","cci_period","cci_min","cci_max",
        "tp","time_exit","slope_exit","early_on","early_sessions","early_max_gain",
        "early_return","giveback_on","giveback_peak","giveback_return"
    ])
    all_df=all_df.sort_values(["qualified","score"],ascending=[False,False]).reset_index(drop=True)
    all_df.to_csv(RESULTS_CSV,index=False)
    all_df.head(20).to_csv(TOP_CSV,index=False)
    return all_df

def report(df, cost):
    top=df.head(20)
    lines=[]
    lines.append("US SETUP OPTIMIZER REPORT")
    lines.append("="*80)
    lines.append(f"Round-trip cost deducted from every trade: {cost:.3f}%")
    lines.append("Target constraints: 250-350 trades/full year; avg net >=4%; payoff >=4:1")
    lines.append("")
    qualified=df[df["qualified"]==1]
    lines.append(f"Qualified configurations: {len(qualified)} / {len(df)}")
    lines.append("")
    cols=[
        "qualified","score","avg_trades_full_year","avg_net_full_year_pct","payoff_ratio",
        "win_rate_pct","min_year_avg_net_pct","r2_min","slope_min","slope_max",
        "cci_enabled","cci_period","cci_min","cci_max","tp","time_exit","slope_exit",
        "early_on","early_sessions","early_max_gain","early_return",
        "giveback_on","giveback_peak","giveback_return",
        "n_2025","avg_2025","n_2026","avg_2026"
    ]
    lines.append(top[cols].to_string(index=False))
    REPORT_TXT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cost",type=float,default=float(os.getenv("ROUND_TRIP_COST_PCT","0")),
                    help="Round-trip cost percentage points per trade, e.g. 0.30")
    ap.add_argument("--configs",type=int,default=600,help="Stage-1 random configurations")
    ap.add_argument("--stage2",type=int,default=400,help="Local mutations around top stage-1 configs")
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
