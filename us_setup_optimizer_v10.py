#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Outcome-Learning Optimizer V10
=================================
Fixes V9's main issues:
1) NO absolute probability threshold. TP-before-SL is a rare-event label, so
   calibrated probabilities can be far below 0.50 even for the best stocks.
   V10 ranks all candidates cross-sectionally and takes the strongest daily pick(s).
2) Frequency-first selection. Top-1-per-day naturally targets ~250 trades/full year.
3) Correct payoff calculation when there are no losing trades.
4) 2026 is a partial holdout, so its trade count is NOT required to be 250-350.
5) Model selection uses 2025 only. 2026 is untouched until the final chosen setup.
6) Adds probability-rank / margin variants rather than hard probability cutoffs.

Outputs:
  /data/us_v10_all_results.csv
  /data/us_v10_validation_top20.csv
  /data/us_v10_holdout_report.txt
  /data/us_v10_selected_trades_2025.csv
  /data/us_v10_selected_trades_2026.csv
"""

import os, sys, math, time, argparse, subprocess
from pathlib import Path

VERSION = "10.0-daily-rank-frequency-first-20261005"
TRAIN_YEAR = 2024
VALID_YEAR = 2025
HOLDOUT_YEAR = 2026
TABLE = os.getenv("US_DAILY_TABLE", "market_candles_1d")
RANDOM_STATE = 20261005

OUT_DIR = Path(os.getenv("OPTIMIZER_OUT_DIR", "/data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_CSV = OUT_DIR / "us_v10_all_results.csv"
TOP_CSV = OUT_DIR / "us_v10_validation_top20.csv"
REPORT = OUT_DIR / "us_v10_holdout_report.txt"
TRADES_2025 = OUT_DIR / "us_v10_selected_trades_2025.csv"
TRADES_2026 = OUT_DIR / "us_v10_selected_trades_2026.csv"

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
    u=os.getenv("DATABASE_URL","").strip()
    if not u:
        raise RuntimeError("DATABASE_URL is missing")
    if u.startswith("postgres://"):
        u="postgresql://"+u[len("postgres://"):]
    return u

def pick(cols,*names):
    low={c.lower():c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def load_daily():
    eng=create_engine(db_url(),pool_pre_ping=True)
    insp=inspect(eng)
    if TABLE not in insp.get_table_names():
        raise RuntimeError(f"Table {TABLE!r} not found")
    cols=[c["name"] for c in insp.get_columns(TABLE)]
    cs=pick(cols,"symbol","ticker","feed_symbol")
    cd=pick(cols,"date","day","trade_date","ts","timestamp","time")
    co=pick(cols,"open","o","adj_open")
    ch=pick(cols,"high","h","adj_high")
    cl=pick(cols,"low","l","adj_low")
    cc=pick(cols,"close","c","adj_close")
    cv=pick(cols,"volume","v","vol")
    missing=[k for k,v in {"symbol":cs,"date":cd,"open":co,"high":ch,"low":cl,"close":cc,"volume":cv}.items() if not v]
    if missing:
        raise RuntimeError(f"Cannot map {missing}; columns={cols}")
    cm=pick(cols,"market","market_key")
    where=f" WHERE UPPER(CAST({cm} AS TEXT)) IN ('US','USA') " if cm else ""
    q=f"""SELECT {cs} AS symbol,{cd} AS d,{co} AS o,{ch} AS h,{cl} AS l,{cc} AS c,{cv} AS v
          FROM {TABLE} {where} ORDER BY {cs},{cd}"""
    print("[data] loading US daily candles ...",flush=True)
    chunks=[]; total=0
    with eng.connect() as con:
        for x in pd.read_sql_query(text(q),con,chunksize=250_000):
            chunks.append(x); total+=len(x)
            print(f"[data] rows loaded: {total:,}",flush=True)
    df=pd.concat(chunks,ignore_index=True)

    df["symbol"]=df["symbol"].astype(str).str.upper().str.strip()
    raw=df["d"]
    if pd.api.types.is_numeric_dtype(raw):
        vals=pd.to_numeric(raw,errors="coerce")
        med=float(vals.dropna().median()) if vals.notna().any() else float("nan")
        if 19000101<=med<=21001231:
            dt=pd.to_datetime(vals.round().astype("Int64").astype(str),format="%Y%m%d",errors="coerce",utc=True)
        elif 1e8<=abs(med)<1e11:
            dt=pd.to_datetime(vals,unit="s",errors="coerce",utc=True)
        elif 1e11<=abs(med)<1e14:
            dt=pd.to_datetime(vals,unit="ms",errors="coerce",utc=True)
        elif 1e14<=abs(med)<1e17:
            dt=pd.to_datetime(vals,unit="us",errors="coerce",utc=True)
        else:
            dt=pd.to_datetime(vals,errors="coerce",utc=True)
    else:
        dt=pd.to_datetime(raw,errors="coerce",utc=True)
    df["d"]=dt.dt.tz_convert(None).dt.normalize()
    for c in ["o","h","l","c","v"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["symbol","d","o","h","l","c","v"])
    df=df[(df["o"]>0)&(df["h"]>0)&(df["l"]>0)&(df["c"]>0)&(df["v"]>=0)]
    df=df.drop_duplicates(["symbol","d"],keep="last").sort_values(["symbol","d"]).reset_index(drop=True)
    print(f"[data] final rows={len(df):,} symbols={df.symbol.nunique():,} range={df.d.min().date()} -> {df.d.max().date()}",flush=True)
    return df

def cci_np(h,l,c,p):
    tp=(h+l+c)/3.0
    ma=pd.Series(tp).rolling(p,min_periods=p).mean().to_numpy(np.float64)
    out=np.full(len(tp),np.nan,np.float32)
    for i in range(p-1,len(tp)):
        w=tp[i-p+1:i+1]
        md=np.mean(np.abs(w-ma[i]))
        out[i]=0.0 if md==0 else np.float32((tp[i]-ma[i])/(0.015*md))
    return out

def linreg_prev(c,lookback=126):
    n=len(c); sl=np.full(n,np.nan,np.float32); r2=np.full(n,np.nan,np.float32)
    x=np.arange(lookback,dtype=np.float64); sx=x.sum(); sx2=(x*x).sum(); den=lookback*sx2-sx*sx
    for i in range(lookback,n):
        y=c[i-lookback:i].astype(np.float64,copy=False)
        if np.any(~np.isfinite(y)) or y[0]<=0: continue
        sy=y.sum(); sxy=(x*y).sum()
        b=(lookback*sxy-sx*sy)/den
        a=(sy-b*sx)/lookback
        pred=a+b*x
        sst=((y-y.mean())**2).sum()
        ssr=((y-pred)**2).sum()
        r2[i]=np.float32(1-ssr/sst if sst>0 else 0)
        sl[i]=np.float32(100*b*(lookback-1)/y[0])
    return sl,r2

FEATURE_COLS=[
    "r2","slope","cci20","cci40","cci60","rvol20","dollar_vol20","atr_pct14",
    "ret5","ret20","ret60","range_exp","close_loc","high20_strength","high60_strength",
    "gap_pct","vol_accel5","dist_sma20","dist_sma50","dist_sma200",
    "rank_r2","rank_slope","rank_ret5","rank_ret20","rank_ret60","rank_rvol20",
    "rank_range_exp","rank_close_loc","rank_high20_strength","rank_high60_strength",
    "rank_vol_accel5"
]

def build_features(df):
    print("[features] computing per-stock features ...",flush=True)
    n=len(df)
    arr={k:np.full(n,np.nan,np.float32) for k in [
        "slope","r2","cci20","cci40","cci60","rvol20","atr_pct14",
        "ret5","ret20","ret60","range_exp","close_loc","high20_strength",
        "high60_strength","gap_pct","vol_accel5","dist_sma20","dist_sma50","dist_sma200"
    ]}
    arr["dollar_vol20"]=np.full(n,np.nan,np.float64)

    for si,(sym,g) in enumerate(df.groupby("symbol",sort=False),1):
        idx=g.index.to_numpy(); a,b=idx[0],idx[-1]+1
        o=df.loc[idx,"o"].to_numpy(np.float64)
        h=df.loc[idx,"h"].to_numpy(np.float64)
        l=df.loc[idx,"l"].to_numpy(np.float64)
        c=df.loc[idx,"c"].to_numpy(np.float64)
        v=df.loc[idx,"v"].to_numpy(np.float64)

        sl,rr=linreg_prev(c)
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
            print(f"[features] {si:,} symbols",flush=True)

    out=df.copy()
    for k,v in arr.items():
        out[k]=v
    out["year"]=out["d"].dt.year.astype(np.int16)

    print("[features] computing cross-sectional ranks ...",flush=True)
    for col in ["r2","slope","ret5","ret20","ret60","rvol20","range_exp","close_loc",
                "high20_strength","high60_strength","vol_accel5"]:
        out["rank_"+col]=out.groupby("d",sort=False)[col].rank(pct=True,method="average").astype(np.float32)
    return out

BARRIERS=[(12.,3.),(15.,4.),(18.,4.),(20.,5.),(24.,6.),(15.,3.),(18.,3.),(20.,4.),(24.,4.)]
HOLD_WINDOWS=[10,15,20,30,45]

def broad_candidates(feat,min_price,min_dv):
    m=(
        feat["year"].isin([TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR]) &
        (feat["c"]>=min_price) &
        (feat["dollar_vol20"]>=min_dv) &
        feat[FEATURE_COLS].notna().all(axis=1)
    )
    return feat.loc[m].copy()

def path_outcome_for_symbol(g,tp,sl,hold,cost):
    n=len(g)
    c=g["c"].to_numpy(np.float64)
    h=g["h"].to_numpy(np.float64)
    l=g["l"].to_numpy(np.float64)
    y=np.full(n,np.nan,np.float32)
    ret=np.full(n,np.nan,np.float32)
    step=np.full(n,-1,np.int16)
    kind=np.empty(n,dtype=object)

    for i in range(n-1):
        entry=c[i]; end=min(n-1,i+hold)
        rr=None; kk="TIME"; st=0
        for j in range(i+1,end+1):
            hit_tp=h[j]>=entry*(1+tp/100.0)
            hit_sl=l[j]<=entry*(1-sl/100.0)
            # conservative intraday ambiguity: SL wins ties
            if hit_sl:
                rr=-sl-cost; kk="SL"; st=j-i; break
            if hit_tp:
                rr=tp-cost; kk="TP"; st=j-i; break
        if rr is None:
            if end<=i: continue
            rr=100*(c[end]/entry-1)-cost; kk="TIME"; st=end-i
        y[i]=1.0 if kk=="TP" else 0.0
        ret[i]=np.float32(rr); step[i]=st; kind[i]=kk
    return y,ret,step,kind

def compute_outcomes(full,candidate_idx,tp,sl,hold,cost):
    out_y=pd.Series(index=full.index,dtype="float32")
    out_r=pd.Series(index=full.index,dtype="float32")
    out_s=pd.Series(index=full.index,dtype="float32")
    out_k=pd.Series(index=full.index,dtype="object")
    wanted=set(map(int,candidate_idx))
    for sym,g in full.groupby("symbol",sort=False):
        idx=g.index.to_numpy()
        loc=[k for k,x in enumerate(idx) if int(x) in wanted]
        if not loc: continue
        y,r,s,k=path_outcome_for_symbol(g,tp,sl,hold,cost)
        loc=np.asarray(loc,dtype=int); ii=idx[loc]
        out_y.loc[ii]=y[loc]; out_r.loc[ii]=r[loc]; out_s.loc[ii]=s[loc]; out_k.loc[ii]=k[loc]
    return (
        out_y.loc[candidate_idx].to_numpy(np.float32),
        out_r.loc[candidate_idx].to_numpy(np.float32),
        out_s.loc[candidate_idx].to_numpy(np.float32),
        out_k.loc[candidate_idx].to_numpy(object)
    )

def train_model(df,y):
    model=HistGradientBoostingClassifier(
        learning_rate=.05,max_iter=220,max_leaf_nodes=31,
        min_samples_leaf=60,l2_regularization=1.5,
        random_state=RANDOM_STATE
    )
    model.fit(df[FEATURE_COLS].to_numpy(np.float32),y.astype(int))
    return model

def choose_daily_ranked(df,prob,mode):
    """
    No absolute probability cutoff.
    Modes:
      top1         -> strongest candidate each day
      top1_margin  -> top1 only if its score beats #2 by >= 0.01
      top1_strong  -> top1 only if daily percentile >= 0.98
      top2_strong  -> up to 2 if both are in top 1% of that day's model scores
    """
    z=df[["symbol","d","c"]].copy()
    z["prob"]=prob
    z=z.sort_values(["d","prob"],ascending=[True,False])
    z["day_rank"]=z.groupby("d").cumcount()+1
    z["daily_count"]=z.groupby("d")["prob"].transform("size")
    z["prob_pct"]=z.groupby("d")["prob"].rank(pct=True,method="first")
    second=z[z["day_rank"]==2].set_index("d")["prob"]
    z["second_prob"]=z["d"].map(second)
    z["margin"]=z["prob"]-z["second_prob"]

    if mode=="top1":
        return z[z["day_rank"]==1].copy()
    if mode=="top1_margin":
        return z[(z["day_rank"]==1)&((z["margin"]>=.01)|z["second_prob"].isna())].copy()
    if mode=="top1_strong":
        return z[(z["day_rank"]==1)&(z["prob_pct"]>=.98)].copy()
    if mode=="top2_strong":
        return z[(z["day_rank"]<=2)&(z["prob_pct"]>=.99)].copy()
    raise ValueError(mode)

def backtest_selected(selected,base_df,realized,kind,step):
    if selected.empty:
        return None,pd.DataFrame()
    pos=pd.Series(np.arange(len(base_df)),index=base_df.index)
    sym_groups={s:g for s,g in base_df.groupby("symbol",sort=False)}
    rows=[]; last_exit={}
    for ix,row in selected.sort_values(["d","prob"],ascending=[True,False]).iterrows():
        sym=row["symbol"]; d=pd.Timestamp(row["d"])
        if sym in last_exit and d<=last_exit[sym]:
            continue
        p=int(pos.loc[ix]); rr=float(realized[p])
        if not math.isfinite(rr): continue
        st=int(step[p]) if math.isfinite(float(step[p])) else 0
        kk=str(kind[p])
        sg=sym_groups[sym]
        loc=np.flatnonzero(sg.index.to_numpy()==ix)
        if len(loc):
            ep=min(len(sg)-1,int(loc[0])+st)
            ed=pd.Timestamp(sg.iloc[ep]["d"])
        else:
            ed=d
        last_exit[sym]=ed
        rows.append({
            "index":int(ix),"symbol":sym,"entry_date":d,"entry_price":float(row["c"]),
            "model_score":float(row["prob"]),"daily_rank":int(row["day_rank"]),
            "exit_kind":kk,"exit_date":ed,"net_return_pct":rr
        })
    tr=pd.DataFrame(rows)
    if tr.empty: return None,tr
    r=tr["net_return_pct"].to_numpy(np.float64)
    wins=r[r>0]; losses=r[r<=0]
    avg_win=float(wins.mean()) if len(wins) else 0.0
    if len(losses):
        avg_loss=float(abs(losses.mean()))
        payoff=avg_win/avg_loss if avg_loss>0 else float("inf")
    else:
        avg_loss=0.0
        payoff=float("inf")
    m={
        "n":int(len(tr)),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "avg_win":avg_win,
        "avg_loss_abs":avg_loss,
        "payoff":float(payoff),
        "tp_rate":float(100*(tr["exit_kind"]=="TP").mean()),
        "sl_rate":float(100*(tr["exit_kind"]=="SL").mean()),
        "time_rate":float(100*(tr["exit_kind"]=="TIME").mean())
    }
    return m,tr

def validation_score(m):
    if not m: return -1e12
    n=m["n"]
    # Frequency dominates.
    if 250<=n<=350: tier=3
    elif 200<=n<=400: tier=2
    elif 150<=n<=450: tier=1
    else: tier=0
    payoff_capped=min(m["payoff"] if math.isfinite(m["payoff"]) else 10.0,10.0)
    return (
        tier*1_000_000
        - abs(n-300)*500
        + m["avg_net"]*10_000
        + payoff_capped*1_000
        + m["win_rate"]*100
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cost",type=float,default=.30)
    ap.add_argument("--min-price",type=float,default=2.0)
    ap.add_argument("--min-dollar-vol",type=float,default=3_000_000.0)
    args=ap.parse_args()

    print(f"[version] {VERSION}",flush=True)
    print(f"[years] train={TRAIN_YEAR} validate={VALID_YEAR} holdout={HOLDOUT_YEAR}",flush=True)
    print(f"[cost] round-trip={args.cost:.3f}%",flush=True)

    full=build_features(load_daily())
    cand=broad_candidates(full,args.min_price,args.min_dollar_vol)
    print(f"[candidates] total={len(cand):,} train={sum(cand.year==TRAIN_YEAR):,} valid={sum(cand.year==VALID_YEAR):,} holdout={sum(cand.year==HOLDOUT_YEAR):,}",flush=True)

    modes=["top1","top1_margin","top1_strong","top2_strong"]
    rows=[]
    best=None
    best_model_payload=None

    for bi,(tp,sl) in enumerate(BARRIERS,1):
        for hold in HOLD_WINDOWS:
            print(f"[barrier {bi}/{len(BARRIERS)}] TP={tp}% SL={sl}% hold={hold}",flush=True)
            y,r,s,k=compute_outcomes(full,cand.index.to_numpy(),tp,sl,hold,args.cost)
            usable=np.isfinite(y)&np.isfinite(r)
            cur=cand.loc[usable].copy()
            yy=y[usable]; rr=r[usable]; ss=s[usable]; kk=k[usable]
            yrs=cur["year"].to_numpy()
            mt=yrs==TRAIN_YEAR; mv=yrs==VALID_YEAR; mh=yrs==HOLDOUT_YEAR
            if mt.sum()<500 or mv.sum()<200: continue

            model=train_model(cur.loc[mt],yy[mt])
            ptrain=model.predict_proba(cur.loc[mt,FEATURE_COLS].to_numpy(np.float32))[:,1]
            pval=model.predict_proba(cur.loc[mv,FEATURE_COLS].to_numpy(np.float32))[:,1]
            try:
                auc_train=roc_auc_score(yy[mt],ptrain)
                auc_valid=roc_auc_score(yy[mv],pval)
            except Exception:
                auc_train=np.nan; auc_valid=np.nan

            val_df=cur.loc[mv]
            for mode in modes:
                sel=choose_daily_ranked(val_df,pval,mode)
                vm,vtr=backtest_selected(sel,val_df,rr[mv],kk[mv],ss[mv])
                if not vm: continue
                rec={
                    "tp":tp,"sl":sl,"hold":hold,"selection_mode":mode,
                    "auc_train":auc_train,"auc_valid":auc_valid,
                    "valid_n":vm["n"],"valid_avg_net":vm["avg_net"],
                    "valid_median_net":vm["median_net"],"valid_win_rate":vm["win_rate"],
                    "valid_payoff":vm["payoff"],"valid_avg_win":vm["avg_win"],
                    "valid_avg_loss_abs":vm["avg_loss_abs"],
                    "valid_tp_rate":vm["tp_rate"],"valid_sl_rate":vm["sl_rate"],
                    "valid_time_rate":vm["time_rate"]
                }
                rec["valid_score"]=validation_score(vm)
                rows.append(rec)
                if best is None or rec["valid_score"]>best["valid_score"]:
                    best=rec
                    best_model_payload=(model,cur,rr,ss,kk,mh,pval,vtr)

    if best is None:
        raise RuntimeError("No validation results")

    all_df=pd.DataFrame(rows).sort_values("valid_score",ascending=False).reset_index(drop=True)
    all_df.to_csv(ALL_CSV,index=False)
    all_df.head(20).to_csv(TOP_CSV,index=False)

    # Evaluate 2026 only now, with the validation-selected setup.
    model,cur,rr,ss,kk,mh,pval,vtr=best_model_payload
    hold_df=cur.loc[mh]
    phold=model.predict_proba(hold_df[FEATURE_COLS].to_numpy(np.float32))[:,1] if len(hold_df) else np.array([])
    hsel=choose_daily_ranked(hold_df,phold,best["selection_mode"]) if len(hold_df) else pd.DataFrame()
    hm,htr=backtest_selected(hsel,hold_df,rr[mh],kk[mh],ss[mh]) if len(hold_df) else (None,pd.DataFrame())

    vtr.to_csv(TRADES_2025,index=False)
    htr.to_csv(TRADES_2026,index=False)

    lines=[]
    lines.append("US OUTCOME-LEARNING OPTIMIZER V10")
    lines.append("="*100)
    lines.append(f"Version: {VERSION}")
    lines.append(f"Train={TRAIN_YEAR} | Validate={VALID_YEAR} | Holdout={HOLDOUT_YEAR} (partial year)")
    lines.append(f"Round-trip cost: {args.cost:.3f}%")
    lines.append("")
    lines.append("BEST VALIDATION-SELECTED SETUP")
    lines.append("-"*100)
    for k,v in best.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("2026 HOLDOUT (NOT USED FOR MODEL SELECTION)")
    lines.append("-"*100)
    if hm:
        for k,v in hm.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("No holdout trades.")
    lines.append("")
    lines.append("TOP 20 BY 2025 VALIDATION")
    lines.append("-"*100)
    cols=["tp","sl","hold","selection_mode","auc_train","auc_valid","valid_n",
          "valid_avg_net","valid_win_rate","valid_payoff","valid_tp_rate",
          "valid_sl_rate","valid_time_rate","valid_score"]
    lines.append(all_df.head(20)[cols].to_string(index=False))
    lines.append("")
    lines.append("TARGET CHECK")
    lines.append("-"*100)
    v_ok=(250<=best["valid_n"]<=350 and best["valid_avg_net"]>=4 and best["valid_payoff"]>=4)
    lines.append(f"2025 validation meets full target: {v_ok}")
    if hm:
        # 2026 is partial, so do NOT require 250-350 trades.
        h_quality=(hm["avg_net"]>=4 and hm["payoff"]>=4)
        lines.append(f"2026 partial holdout meets return/payoff target: {h_quality}")
        lines.append(f"2026 partial holdout trades so far: {hm['n']}")
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{ALL_CSV}\n{TOP_CSV}\n{REPORT}\n{TRADES_2025}\n{TRADES_2026}",flush=True)

if __name__=="__main__":
    main()
