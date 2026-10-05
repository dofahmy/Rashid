#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Outcome-Learning Optimizer V9
================================
Purpose
-------
Use the full US universe as a cross-sectional opportunity set, learn from 2024
which daily candidates are most likely to hit a profit barrier before a loss
barrier, validate on 2025, and report 2026 as holdout.

Key changes vs V8
-----------------
1) Outcome learning instead of hand-ranking only.
2) Labels are path-based: TP first vs SL first within N sessions.
3) Tests asymmetric TP/SL pairs such as +12/-3, +15/-4, +18/-4, +20/-5, +24/-6.
4) Trains ONLY on 2024.
5) Selects model/threshold/top-N using 2025 validation.
6) 2026 is holdout and is never used for training or model selection.
7) Targets ~250-350 trades/year by selecting top-ranked opportunities per day.
8) Applies round-trip cost to every closed trade.

Outputs
-------
/data/us_v9_all_results.csv
/data/us_v9_validation_top20.csv
/data/us_v9_holdout_report.txt
/data/us_v9_selected_trades_2025.csv
/data/us_v9_selected_trades_2026.csv

Example
-------
python us_setup_optimizer_v9.py --cost 0.30
"""

import os, sys, math, json, time, argparse, subprocess
from pathlib import Path

VERSION = "9.0-outcome-learning-barriers-20261005"
TRAIN_YEAR = 2024
VALID_YEAR = 2025
HOLDOUT_YEAR = 2026

OUT_DIR = Path(os.getenv("OPTIMIZER_OUT_DIR", "/data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CSV = OUT_DIR / "us_v9_all_results.csv"
TOP_CSV = OUT_DIR / "us_v9_validation_top20.csv"
REPORT = OUT_DIR / "us_v9_holdout_report.txt"
TRADES_2025 = OUT_DIR / "us_v9_selected_trades_2025.csv"
TRADES_2026 = OUT_DIR / "us_v9_selected_trades_2026.csv"

TABLE = os.getenv("US_DAILY_TABLE", "market_candles_1d")
RANDOM_STATE = 20261005

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
ensure("scikit-learn", "sklearn")
try:
    import psycopg2  # noqa
except Exception:
    ensure("psycopg2-binary", "psycopg2")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def db_url():
    u = os.getenv("DATABASE_URL", "").strip()
    if not u:
        raise RuntimeError("DATABASE_URL is missing")
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
        raise RuntimeError(f"Table {TABLE!r} not found")
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
        "low":c_low,"close":c_close,"volume":c_vol
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Cannot map columns {missing}; table columns={cols}")

    c_market = pick(cols, "market", "market_key")
    where = f" WHERE UPPER(CAST({c_market} AS TEXT)) IN ('US','USA') " if c_market else ""
    q = f"""
    SELECT {c_symbol} AS symbol, {c_date} AS d, {c_open} AS o,
           {c_high} AS h, {c_low} AS l, {c_close} AS c, {c_vol} AS v
    FROM {TABLE}
    {where}
    ORDER BY {c_symbol}, {c_date}
    """
    print("[data] loading US daily candles ...", flush=True)
    chunks=[]
    total=0
    with eng.connect() as con:
        for ch in pd.read_sql_query(text(q), con, chunksize=250_000):
            chunks.append(ch); total += len(ch)
            print(f"[data] rows loaded: {total:,}", flush=True)
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    raw = df["d"]
    if pd.api.types.is_numeric_dtype(raw):
        vals = pd.to_numeric(raw, errors="coerce")
        med = float(vals.dropna().median()) if vals.notna().any() else float("nan")
        if 19000101 <= med <= 21001231:
            dt = pd.to_datetime(vals.round().astype("Int64").astype(str), format="%Y%m%d", errors="coerce", utc=True)
        elif 1e8 <= abs(med) < 1e11:
            dt = pd.to_datetime(vals, unit="s", errors="coerce", utc=True)
        elif 1e11 <= abs(med) < 1e14:
            dt = pd.to_datetime(vals, unit="ms", errors="coerce", utc=True)
        elif 1e14 <= abs(med) < 1e17:
            dt = pd.to_datetime(vals, unit="us", errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(vals, errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(raw, errors="coerce", utc=True)
    df["d"] = dt.dt.tz_convert(None).dt.normalize()

    for c in ["o","h","l","c","v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["symbol","d","o","h","l","c","v"])
    df = df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)&(df["v"]>=0)]
    df = df.drop_duplicates(["symbol","d"], keep="last").sort_values(["symbol","d"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No usable rows")
    print(f"[data] final rows={len(df):,} symbols={df.symbol.nunique():,} range={df.d.min().date()} -> {df.d.max().date()}", flush=True)
    return df

def cci_np(h,l,c,period):
    tp=(h+l+c)/3.0
    s=pd.Series(tp)
    ma=s.rolling(period,min_periods=period).mean().to_numpy(np.float64)
    out=np.full(len(tp),np.nan,np.float32)
    for i in range(period-1,len(tp)):
        w=tp[i-period+1:i+1]
        md=np.mean(np.abs(w-ma[i]))
        out[i]=0.0 if md==0 else np.float32((tp[i]-ma[i])/(0.015*md))
    return out

def linreg_prev(close, lookback=126):
    n=len(close)
    slope=np.full(n,np.nan,np.float32)
    r2=np.full(n,np.nan,np.float32)
    x=np.arange(lookback,dtype=np.float64)
    sx=x.sum(); sx2=(x*x).sum(); den=lookback*sx2-sx*sx
    for i in range(lookback,n):
        y=close[i-lookback:i].astype(np.float64,copy=False)
        if np.any(~np.isfinite(y)) or y[0]<=0: continue
        sy=y.sum(); sy2=(y*y).sum(); sxy=(x*y).sum()
        b=(lookback*sxy-sx*sy)/den
        a=(sy-b*sx)/lookback
        pred=a+b*x
        sst=((y-y.mean())**2).sum()
        ssr=((y-pred)**2).sum()
        r2[i]=np.float32(1-ssr/sst if sst>0 else 0)
        slope[i]=np.float32(100*b*(lookback-1)/y[0])
    return slope,r2

def build_features(df):
    print("[features] computing per-stock features ...", flush=True)
    n=len(df)
    names = [
        "slope","r2","cci20","cci40","cci60","rvol20","dollar_vol20","atr_pct14",
        "ret5","ret20","ret60","range_exp","close_loc","high20_strength",
        "high60_strength","gap_pct","vol_accel5","dist_sma20","dist_sma50","dist_sma200"
    ]
    arr={k:np.full(n,np.nan,np.float32) for k in names}
    arr["dollar_vol20"]=np.full(n,np.nan,np.float64)

    for si,(sym,g) in enumerate(df.groupby("symbol",sort=False),1):
        idx=g.index.to_numpy()
        a,b=idx[0],idx[-1]+1
        o=df.loc[idx,"o"].to_numpy(np.float64)
        h=df.loc[idx,"h"].to_numpy(np.float64)
        l=df.loc[idx,"l"].to_numpy(np.float64)
        c=df.loc[idx,"c"].to_numpy(np.float64)
        v=df.loc[idx,"v"].to_numpy(np.float64)

        sl,rr=linreg_prev(c,126)
        arr["slope"][a:b]=sl; arr["r2"][a:b]=rr
        arr["cci20"][a:b]=cci_np(h,l,c,20)
        arr["cci40"][a:b]=cci_np(h,l,c,40)
        arr["cci60"][a:b]=cci_np(h,l,c,60)

        sv=pd.Series(v); sc=pd.Series(c); sh=pd.Series(h)
        vol20=sv.shift(1).rolling(20,min_periods=20).mean().to_numpy(np.float64)
        vol5=sv.shift(1).rolling(5,min_periods=5).mean().to_numpy(np.float64)
        dv20=pd.Series(v*c).shift(1).rolling(20,min_periods=20).mean().to_numpy(np.float64)
        arr["rvol20"][a:b]=np.divide(v,vol20,out=np.full(len(c),np.nan),where=vol20>0).astype(np.float32)
        arr["vol_accel5"][a:b]=np.divide(vol5,vol20,out=np.full(len(c),np.nan),where=vol20>0).astype(np.float32)
        arr["dollar_vol20"][a:b]=dv20

        prev_c=np.r_[np.nan,c[:-1]]
        tr=np.maximum(h-l,np.maximum(np.abs(h-prev_c),np.abs(l-prev_c)))
        atr14=pd.Series(tr).shift(1).rolling(14,min_periods=14).mean().to_numpy(np.float64)
        arr["atr_pct14"][a:b]=np.divide(100*atr14,c,out=np.full(len(c),np.nan),where=c>0).astype(np.float32)
        arr["range_exp"][a:b]=np.divide(tr,atr14,out=np.full(len(c),np.nan),where=atr14>0).astype(np.float32)
        arr["close_loc"][a:b]=np.divide(c-l,h-l,out=np.full(len(c),.5),where=(h-l)>0).astype(np.float32)
        arr["gap_pct"][a:b]=(100*np.divide(o,prev_c,out=np.full(len(c),np.nan),where=prev_c>0)-100).astype(np.float32)

        for p in [5,20,60]:
            prev=sc.shift(p).to_numpy(np.float64)
            arr[f"ret{p}"][a:b]=(100*(np.divide(c,prev,out=np.full(len(c),np.nan),where=prev>0)-1)).astype(np.float32)

        ph20=sh.shift(1).rolling(20,min_periods=20).max().to_numpy(np.float64)
        ph60=sh.shift(1).rolling(60,min_periods=60).max().to_numpy(np.float64)
        arr["high20_strength"][a:b]=np.divide(c,ph20,out=np.full(len(c),np.nan),where=ph20>0).astype(np.float32)
        arr["high60_strength"][a:b]=np.divide(c,ph60,out=np.full(len(c),np.nan),where=ph60>0).astype(np.float32)

        for p in [20,50,200]:
            sma=sc.shift(1).rolling(p,min_periods=p).mean().to_numpy(np.float64)
            arr[f"dist_sma{p}"][a:b]=(100*(np.divide(c,sma,out=np.full(len(c),np.nan),where=sma>0)-1)).astype(np.float32)

        if si%500==0:
            print(f"[features] {si:,} symbols", flush=True)

    out=df.copy()
    for k,v in arr.items():
        out[k]=v
    out["year"]=out["d"].dt.year.astype(np.int16)

    # Daily percentile ranks across the whole universe.
    print("[features] computing daily cross-sectional ranks ...", flush=True)
    for col in ["r2","slope","ret5","ret20","ret60","rvol20","range_exp","close_loc",
                "high20_strength","high60_strength","vol_accel5"]:
        out["rank_"+col]=out.groupby("d",sort=False)[col].rank(pct=True,method="average").astype("float32")

    return out

FEATURE_COLS = [
    "r2","slope","cci20","cci40","cci60","rvol20","dollar_vol20","atr_pct14",
    "ret5","ret20","ret60","range_exp","close_loc","high20_strength","high60_strength",
    "gap_pct","vol_accel5","dist_sma20","dist_sma50","dist_sma200",
    "rank_r2","rank_slope","rank_ret5","rank_ret20","rank_ret60","rank_rvol20",
    "rank_range_exp","rank_close_loc","rank_high20_strength","rank_high60_strength",
    "rank_vol_accel5"
]

BARRIERS = [
    (12.0,3.0),(15.0,4.0),(18.0,4.0),(20.0,5.0),(24.0,6.0),
    (15.0,3.0),(18.0,3.0),(20.0,4.0),(24.0,4.0)
]
HOLD_WINDOWS = [10,15,20,30,45]

def broad_candidates(feat, min_price=2.0, min_dollar_vol=3_000_000.0):
    m = (
        feat["year"].isin([TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR]) &
        (feat["c"] >= min_price) &
        (feat["dollar_vol20"] >= min_dollar_vol) &
        feat[FEATURE_COLS].notna().all(axis=1)
    )
    return feat.loc[m].copy()

def path_outcome_for_symbol(g, tp, sl, hold, cost):
    """
    One label/realized-return for every row in g.
    Conservative same-day ambiguity: if both TP and SL touched on the same day,
    count SL first.
    """
    n=len(g)
    c=g["c"].to_numpy(np.float64)
    h=g["h"].to_numpy(np.float64)
    l=g["l"].to_numpy(np.float64)

    y=np.full(n,np.nan,np.float32)
    ret=np.full(n,np.nan,np.float32)
    exit_step=np.full(n,-1,np.int16)
    exit_kind=np.empty(n,dtype=object)

    for i in range(n-1):
        entry=c[i]
        end=min(n-1,i+hold)
        kind="TIME"; rr=None; step=None
        for j in range(i+1,end+1):
            hit_tp=h[j] >= entry*(1+tp/100.0)
            hit_sl=l[j] <= entry*(1-sl/100.0)
            if hit_tp and hit_sl:
                kind="SL"; rr=-sl-cost; step=j-i; break
            if hit_sl:
                kind="SL"; rr=-sl-cost; step=j-i; break
            if hit_tp:
                kind="TP"; rr=tp-cost; step=j-i; break
        if rr is None:
            if end<=i: continue
            kind="TIME"; rr=100*(c[end]/entry-1)-cost; step=end-i
        y[i]=1.0 if kind=="TP" else 0.0
        ret[i]=np.float32(rr)
        exit_step[i]=step
        exit_kind[i]=kind
    return y,ret,exit_step,exit_kind

def add_outcomes(cand, tp, sl, hold, cost):
    ys=np.full(len(cand),np.nan,np.float32)
    rs=np.full(len(cand),np.nan,np.float32)
    steps=np.full(len(cand),-1,np.int16)
    kinds=np.empty(len(cand),dtype=object)

    # Need path from full feature table, not only candidate rows, so this function
    # expects cand to carry original index and caller supplies grouped full data.
    return ys,rs,steps,kinds

def compute_barrier_outcomes(full_feat, candidate_idx, tp, sl, hold, cost):
    """
    Compute outcomes using the complete daily history per symbol, then return
    values aligned to candidate_idx.
    """
    out_y=pd.Series(index=full_feat.index,dtype="float32")
    out_r=pd.Series(index=full_feat.index,dtype="float32")
    out_step=pd.Series(index=full_feat.index,dtype="float32")
    out_kind=pd.Series(index=full_feat.index,dtype="object")

    candidate_set=set(int(x) for x in candidate_idx)
    total_syms=0
    for sym,g in full_feat.groupby("symbol",sort=False):
        idx=g.index.to_numpy()
        relevant=[k for k,x in enumerate(idx) if int(x) in candidate_set]
        if not relevant:
            continue
        y,r,st,ki=path_outcome_for_symbol(g,tp,sl,hold,cost)
        pos=np.asarray(relevant,dtype=int)
        take_idx=idx[pos]
        out_y.loc[take_idx]=y[pos]
        out_r.loc[take_idx]=r[pos]
        out_step.loc[take_idx]=st[pos]
        out_kind.loc[take_idx]=ki[pos]
        total_syms += 1
    return (
        out_y.loc[candidate_idx].to_numpy(np.float32),
        out_r.loc[candidate_idx].to_numpy(np.float32),
        out_step.loc[candidate_idx].to_numpy(np.float32),
        out_kind.loc[candidate_idx].to_numpy(object)
    )

def train_model(train_df, y):
    X=train_df[FEATURE_COLS].to_numpy(np.float32)
    model=HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
    model.fit(X,y.astype(int))
    return model

def choose_daily(df, prob, max_per_day, prob_min):
    z=df[["symbol","d","c"]].copy()
    z["prob"]=prob
    z=z[z["prob"]>=prob_min]
    if z.empty: return z
    z=z.sort_values(["d","prob"],ascending=[True,False])
    z["day_rank"]=z.groupby("d").cumcount()+1
    return z[z["day_rank"]<=max_per_day].copy()

def backtest_selected(selected, base_df, realized_return, kind, exit_step):
    if selected.empty:
        return None, pd.DataFrame()
    pos = pd.Series(np.arange(len(base_df)), index=base_df.index)
    rows=[]
    last_exit_day_by_symbol={}
    for ix,row in selected.sort_values(["d","prob"],ascending=[True,False]).iterrows():
        sym=row["symbol"]; d=row["d"]
        # prevent overlapping same-stock trades
        if sym in last_exit_day_by_symbol and d <= last_exit_day_by_symbol[sym]:
            continue
        p=int(pos.loc[ix])
        rr=float(realized_return[p])
        if not math.isfinite(rr): continue
        st=int(exit_step[p]) if math.isfinite(float(exit_step[p])) else 0
        k=kind[p]
        # approximate exit date from trading rows of same symbol in base_df
        sym_rows=base_df[base_df["symbol"]==sym]
        local=np.flatnonzero(sym_rows.index.to_numpy()==ix)
        if len(local):
            lp=int(local[0]); ep=min(len(sym_rows)-1,lp+st)
            exit_date=pd.Timestamp(sym_rows.iloc[ep]["d"])
        else:
            exit_date=pd.Timestamp(d)
        last_exit_day_by_symbol[sym]=exit_date
        rows.append({
            "index":int(ix),"symbol":sym,"entry_date":pd.Timestamp(d),
            "entry_price":float(row["c"]),"prob":float(row["prob"]),
            "exit_kind":str(k),"exit_date":exit_date,"net_return_pct":rr
        })
    tr=pd.DataFrame(rows)
    if tr.empty:
        return None,tr
    r=tr["net_return_pct"].to_numpy(np.float64)
    wins=r[r>0]; losses=r[r<=0]
    avg_win=float(wins.mean()) if len(wins) else 0.0
    avg_loss=float(abs(losses.mean())) if len(losses) else 999.0
    payoff=avg_win/avg_loss if avg_loss>0 else 999.0
    metrics={
        "n":int(len(tr)),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "avg_win":avg_win,
        "avg_loss_abs":avg_loss,
        "payoff":float(payoff),
        "tp_rate":float(100*(tr["exit_kind"]=="TP").mean()),
        "sl_rate":float(100*(tr["exit_kind"]=="SL").mean()),
        "time_rate":float(100*(tr["exit_kind"]=="TIME").mean()),
    }
    return metrics,tr

def candidate_thresholds():
    return [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]

def validation_score(m):
    if not m: return -1e9
    n=m["n"]
    freq_bonus = 100000 if 250<=n<=350 else 0
    freq_pen = abs(n-300)*100
    return (
        freq_bonus - freq_pen
        + m["avg_net"]*1000
        + min(m["payoff"],10)*300
        + m["win_rate"]*10
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cost",type=float,default=float(os.getenv("ROUND_TRIP_COST_PCT","0.30")))
    ap.add_argument("--min-price",type=float,default=2.0)
    ap.add_argument("--min-dollar-vol",type=float,default=3_000_000.0)
    args=ap.parse_args()

    print(f"[version] {VERSION}", flush=True)
    print(f"[years] train={TRAIN_YEAR} validate={VALID_YEAR} holdout={HOLDOUT_YEAR}", flush=True)
    print(f"[cost] round-trip={args.cost:.3f}%", flush=True)

    full=build_features(load_daily())
    cand=broad_candidates(full,args.min_price,args.min_dollar_vol)
    print(f"[candidates] total={len(cand):,} train={sum(cand.year==TRAIN_YEAR):,} valid={sum(cand.year==VALID_YEAR):,} holdout={sum(cand.year==HOLDOUT_YEAR):,}", flush=True)

    rows=[]
    selected_best=None
    best_valid_trades=None
    best_holdout_trades=None

    for bi,(tp,sl) in enumerate(BARRIERS,1):
        for hold in HOLD_WINDOWS:
            print(f"[barrier {bi}/{len(BARRIERS)}] TP={tp}% SL={sl}% hold={hold}", flush=True)
            y,realized,step,kind = compute_barrier_outcomes(full,cand.index.to_numpy(),tp,sl,hold,args.cost)
            usable=np.isfinite(y)&np.isfinite(realized)
            cur=cand.loc[usable].copy()
            yy=y[usable]; rr=realized[usable]; ss=step[usable]; kk=kind[usable]

            train_mask=(cur["year"].to_numpy()==TRAIN_YEAR)
            val_mask=(cur["year"].to_numpy()==VALID_YEAR)
            hold_mask=(cur["year"].to_numpy()==HOLDOUT_YEAR)

            if train_mask.sum()<500 or val_mask.sum()<200:
                print("  skipped: insufficient train/validation candidates", flush=True)
                continue

            model=train_model(cur.loc[train_mask],yy[train_mask])
            p_train=model.predict_proba(cur.loc[train_mask,FEATURE_COLS].to_numpy(np.float32))[:,1]
            p_val=model.predict_proba(cur.loc[val_mask,FEATURE_COLS].to_numpy(np.float32))[:,1]
            p_hold=model.predict_proba(cur.loc[hold_mask,FEATURE_COLS].to_numpy(np.float32))[:,1] if hold_mask.any() else np.array([])

            try:
                auc_train=roc_auc_score(yy[train_mask],p_train)
                auc_val=roc_auc_score(yy[val_mask],p_val)
            except Exception:
                auc_train=np.nan; auc_val=np.nan

            val_df=cur.loc[val_mask]
            hold_df=cur.loc[hold_mask]

            best_local=None
            for topn in [1,2,3]:
                for thr in candidate_thresholds():
                    sel=choose_daily(val_df,p_val,topn,thr)
                    vm,vtr=backtest_selected(sel,val_df,rr[val_mask],kk[val_mask],ss[val_mask])
                    if not vm: continue
                    rec={
                        "tp":tp,"sl":sl,"hold":hold,"prob_min":thr,"max_entries_per_day":topn,
                        "auc_train":auc_train,"auc_valid":auc_val,
                        "valid_n":vm["n"],"valid_avg_net":vm["avg_net"],"valid_median_net":vm["median_net"],
                        "valid_win_rate":vm["win_rate"],"valid_payoff":vm["payoff"],
                        "valid_tp_rate":vm["tp_rate"],"valid_sl_rate":vm["sl_rate"],"valid_time_rate":vm["time_rate"],
                    }
                    rec["valid_score"]=validation_score(vm)
                    rows.append(rec)
                    if best_local is None or rec["valid_score"]>best_local[0]:
                        best_local=(rec["valid_score"],rec,vtr,p_hold,hold_df,rr[hold_mask],kk[hold_mask],ss[hold_mask])

            if best_local is None: continue
            _,rec,vtr,p_hold,hold_df,rrh,kkh,ssh=best_local

            # Evaluate 2026 with the SAME threshold/top-N chosen on 2025.
            if len(hold_df):
                hsel=choose_daily(hold_df,p_hold,int(rec["max_entries_per_day"]),float(rec["prob_min"]))
                hm,htr=backtest_selected(hsel,hold_df,rrh,kkh,ssh)
            else:
                hm,htr=None,pd.DataFrame()

            rec2=dict(rec)
            if hm:
                rec2.update({
                    "holdout_n":hm["n"],"holdout_avg_net":hm["avg_net"],
                    "holdout_win_rate":hm["win_rate"],"holdout_payoff":hm["payoff"],
                    "holdout_tp_rate":hm["tp_rate"],"holdout_sl_rate":hm["sl_rate"],
                    "holdout_time_rate":hm["time_rate"],
                })
            else:
                rec2.update({"holdout_n":0,"holdout_avg_net":np.nan,"holdout_win_rate":np.nan,"holdout_payoff":np.nan})

            # Global best is selected ONLY by validation score, never 2026.
            if selected_best is None or rec2["valid_score"]>selected_best["valid_score"]:
                selected_best=rec2
                best_valid_trades=vtr
                best_holdout_trades=htr

    if not rows:
        raise RuntimeError("No model/search results produced")

    all_df=pd.DataFrame(rows).sort_values("valid_score",ascending=False).reset_index(drop=True)
    all_df.to_csv(ALL_CSV,index=False)

    # Attach holdout metrics to validation-top models by re-evaluating only the
    # actual globally selected model in detail; TOP_CSV remains validation ranking.
    top=all_df.head(20).copy()
    top.to_csv(TOP_CSV,index=False)

    if best_valid_trades is not None:
        best_valid_trades.to_csv(TRADES_2025,index=False)
    if best_holdout_trades is not None:
        best_holdout_trades.to_csv(TRADES_2026,index=False)

    lines=[]
    lines.append("US OUTCOME-LEARNING OPTIMIZER V9")
    lines.append("="*100)
    lines.append(f"Version: {VERSION}")
    lines.append(f"Train year: {TRAIN_YEAR} | Validation year: {VALID_YEAR} | Holdout year: {HOLDOUT_YEAR}")
    lines.append(f"Round-trip cost: {args.cost:.3f}%")
    lines.append("Selection rule: choose model/TP/SL/hold/probability/top-N using 2025 only; 2026 is untouched holdout.")
    lines.append("")
    lines.append("BEST VALIDATION-SELECTED SETUP")
    lines.append("-"*100)
    for k,v in selected_best.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("TOP 20 BY 2025 VALIDATION SCORE")
    lines.append("-"*100)
    cols=[
        "tp","sl","hold","prob_min","max_entries_per_day","auc_train","auc_valid",
        "valid_n","valid_avg_net","valid_win_rate","valid_payoff",
        "valid_tp_rate","valid_sl_rate","valid_time_rate","valid_score"
    ]
    lines.append(top[cols].to_string(index=False))
    lines.append("")
    lines.append("TARGET CHECK")
    lines.append("-"*100)
    v_ok = (
        250 <= selected_best.get("valid_n",0) <= 350 and
        selected_best.get("valid_avg_net",-999) >= 4 and
        selected_best.get("valid_payoff",-999) >= 4
    )
    h_ok = (
        250 <= selected_best.get("holdout_n",0) <= 350 and
        selected_best.get("holdout_avg_net",-999) >= 4 and
        selected_best.get("holdout_payoff",-999) >= 4
    )
    lines.append(f"2025 validation meets target: {v_ok}")
    lines.append(f"2026 holdout meets target: {h_ok}")
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{ALL_CSV}\n{TOP_CSV}\n{REPORT}\n{TRADES_2025}\n{TRADES_2026}", flush=True)

if __name__=="__main__":
    main()
