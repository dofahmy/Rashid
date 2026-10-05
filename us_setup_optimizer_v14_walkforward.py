#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, json, argparse, subprocess
from pathlib import Path

VERSION="14.0-monthly-walkforward-resume-20261005"
OUT=Path(os.getenv("OPTIMIZER_OUT_DIR","/data")); OUT.mkdir(parents=True,exist_ok=True)
CKCSV=OUT/"us_v14_checkpoint_results.csv"
CKJSON=OUT/"us_v14_checkpoint_state.json"
MONTHLY=OUT/"us_v14_monthly_results.csv"
SUMMARY=OUT/"us_v14_summary.csv"
REPORT=OUT/"us_v14_report.txt"
TRADES=OUT/"us_v14_trades.csv"

def ensure(pkg, name=None):
    name=name or pkg.split("==")[0].replace("-","_")
    try: __import__(name)
    except Exception:
        print(f"[setup] installing {pkg} ...",flush=True)
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet",pkg])

for pkg,name in [("numpy",None),("pandas",None),("scikit-learn","sklearn")]:
    ensure(pkg,name)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

try:
    import us_setup_optimizer_v11 as v11
except Exception as e:
    raise RuntimeError("Put us_setup_optimizer_v11.py beside V14 in /app") from e

SETUPS=[
 {"name":"A_20_3_10","tp":20.0,"sl":3.0,"hold":10},
 {"name":"B_18_3_12","tp":18.0,"sl":3.0,"hold":12},
]
PROFILES=[
 {"name":"balanced","pw":0.50,"rw":0.50},
 {"name":"tp_heavy","pw":0.65,"rw":0.35},
 {"name":"return_heavy","pw":0.35,"rw":0.65},
]
TRAIN_START=pd.Timestamp("2024-01-01")
TEST_START=pd.Timestamp("2025-01-01")
MAX_OPEN=10
FALLBACK_DEPTH=50

def months_until(last_date):
    cur=TEST_START
    end=pd.Timestamp(last_date).normalize()+pd.offsets.MonthBegin(1)
    out=[]
    while cur<end:
        nxt=cur+pd.offsets.MonthBegin(1)
        out.append((cur,nxt))
        cur=nxt
    return out

def key(setup,profile,month):
    return f"{setup}|{profile}|{month:%Y-%m}"

def load_ck():
    done=set(); rows=[]; trades=[]
    if CKJSON.exists():
        try:
            s=json.loads(CKJSON.read_text(encoding="utf-8"))
            if s.get("version")==VERSION: done=set(s.get("done",[]))
        except Exception as e: print(f"[resume] state warning: {e}",flush=True)
    if CKCSV.exists():
        try:
            d=pd.read_csv(CKCSV)
            if "version" in d: d=d[d.version==VERSION]
            rows=d.to_dict("records")
        except Exception as e: print(f"[resume] result warning: {e}",flush=True)
    if TRADES.exists():
        try:
            t=pd.read_csv(TRADES)
            if "version" in t: t=t[t.version==VERSION]
            trades=t.to_dict("records")
        except Exception as e: print(f"[resume] trade warning: {e}",flush=True)
    return done,rows,trades

def save_ck(done,rows,trades):
    if rows:
        p=CKCSV.with_suffix(".tmp.csv"); pd.DataFrame(rows).to_csv(p,index=False); p.replace(CKCSV)
    if trades:
        p=TRADES.with_suffix(".tmp.csv"); pd.DataFrame(trades).to_csv(p,index=False); p.replace(TRADES)
    p=CKJSON.with_suffix(".tmp.json")
    p.write_text(json.dumps({
        "version":VERSION,"done":sorted(done),
        "saved_at":pd.Timestamp.now("UTC").isoformat(),
        "jobs":len(done),"rows":len(rows),"trades":len(trades)
    },ensure_ascii=False,indent=2),encoding="utf-8")
    p.replace(CKJSON)

def train_models(df,y,ret):
    X=df[v11.FEATURE_COLS].to_numpy(np.float32)
    clf=HistGradientBoostingClassifier(
        learning_rate=.05,max_iter=220,max_leaf_nodes=31,
        min_samples_leaf=60,l2_regularization=1.5,random_state=20261005)
    reg=HistGradientBoostingRegressor(
        learning_rate=.05,max_iter=220,max_leaf_nodes=31,
        min_samples_leaf=60,l2_regularization=1.5,random_state=20261006)
    clf.fit(X,y.astype(int)); reg.fit(X,ret.astype(np.float32))
    return clf,reg

def score(df,clf,reg,pw,rw):
    X=df[v11.FEATURE_COLS].to_numpy(np.float32)
    p=clf.predict_proba(X)[:,1]; er=reg.predict(X)
    z=df[["symbol","d","c"]].copy()
    z["p_tp"]=p; z["pred_return"]=er
    z["tp_rank"]=z.groupby("d")["p_tp"].rank(pct=True,method="average")
    z["ret_rank"]=z.groupby("d")["pred_return"].rank(pct=True,method="average")
    z["score"]=pw*z.tp_rank+rw*z.ret_rank
    z=z.sort_values(["d","score"],ascending=[True,False])
    z["day_rank"]=z.groupby("d").cumcount()+1
    return z[z.day_rank<=FALLBACK_DEPTH]

def execute_month(scored,base,ret,kind,step,prior_trades):
    pos=pd.Series(np.arange(len(base)),index=base.index)
    symgroups={s:g for s,g in base.groupby("symbol",sort=False)}
    active=[]; last_exit={}
    for r in prior_trades:
        ed=pd.Timestamp(r["exit_date"]); sym=str(r["symbol"])
        last_exit[sym]=max(last_exit.get(sym,pd.Timestamp.min),ed)
        if ed>pd.Timestamp(scored["d"].min() if len(scored) else base["d"].min()):
            active.append((sym,ed))
    rows=[]
    for d,g in scored.groupby("d",sort=True):
        d=pd.Timestamp(d); active=[x for x in active if x[1]>d]
        if len(active)>=MAX_OPEN: continue
        for ix,row in g.sort_values("score",ascending=False).iterrows():
            sym=row.symbol
            if any(s==sym for s,_ in active): continue
            if sym in last_exit and d<=last_exit[sym]: continue
            p=int(pos.loc[ix]); rr=float(ret[p])
            if not math.isfinite(rr): continue
            st=int(step[p]) if math.isfinite(float(step[p])) else 0
            sg=symgroups[sym]; loc=np.flatnonzero(sg.index.to_numpy()==ix)
            ed=pd.Timestamp(sg.iloc[min(len(sg)-1,int(loc[0])+st)]["d"]) if len(loc) else d
            last_exit[sym]=ed; active.append((sym,ed))
            rows.append({
                "index":int(ix),"symbol":sym,"entry_date":d,"entry_price":float(row.c),
                "score":float(row.score),"p_tp":float(row.p_tp),
                "predicted_return":float(row.pred_return),
                "exit_kind":str(kind[p]),"exit_date":ed,"net_return_pct":rr})
            break
    return pd.DataFrame(rows)

def mets(t):
    if t is None or t.empty:
        return dict(n=0,avg_net=np.nan,median_net=np.nan,win_rate=np.nan,payoff=np.nan,
                    tp_rate=np.nan,sl_rate=np.nan,time_rate=np.nan)
    r=t.net_return_pct.to_numpy(float); w=r[r>0]; l=r[r<=0]
    aw=float(w.mean()) if len(w) else 0.0; al=float(abs(l.mean())) if len(l) else 0.0
    po=float("inf") if not len(l) else (aw/al if al>0 else float("inf"))
    return dict(
        n=len(t),avg_net=float(r.mean()),median_net=float(np.median(r)),
        win_rate=float(100*(r>0).mean()),payoff=po,
        tp_rate=float(100*(t.exit_kind=="TP").mean()),
        sl_rate=float(100*(t.exit_kind=="SL").mean()),
        time_rate=float(100*(t.exit_kind=="TIME").mean()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cost",type=float,default=.30)
    ap.add_argument("--min-price",type=float,default=2.0)
    ap.add_argument("--min-dollar-vol",type=float,default=3_000_000.0)
    ap.add_argument("--reset-checkpoint",action="store_true")
    a=ap.parse_args()
    if a.reset_checkpoint:
        for p in [CKCSV,CKJSON,MONTHLY,SUMMARY,REPORT,TRADES]:
            try:p.unlink()
            except FileNotFoundError:pass
        print("[resume] checkpoint cleared",flush=True)

    print(f"[version] {VERSION}",flush=True)
    print("[mode] monthly expanding walk-forward; no lookahead",flush=True)

    full=v11.build_features(v11.load_daily())
    cand=v11.broad_candidates(full,a.min_price,a.min_dollar_vol)
    months=months_until(cand.d.max())

    done,rows,trade_rows=load_ck()
    total=len(SETUPS)*len(PROFILES)*len(months)
    print(f"[walkforward] months={len(months)} total_jobs={total}",flush=True)
    print(f"[resume] completed={len(done)}/{total}",flush=True)

    caches={}
    for s in SETUPS:
        print(f"[outcomes] {s['name']}",flush=True)
        y,r,st,k=v11.compute_outcomes(full,cand.index.to_numpy(),s["tp"],s["sl"],s["hold"],a.cost)
        ok=np.isfinite(y)&np.isfinite(r)
        caches[s["name"]]=(cand.loc[ok].copy(),y[ok],r[ok],st[ok],k[ok])

    job=0
    for s in SETUPS:
        cur,y,r,st,k=caches[s["name"]]
        for pr in PROFILES:
            for ms,me in months:
                job+=1; jk=key(s["name"],pr["name"],ms)
                if jk in done:
                    print(f"[resume] skip {job}/{total} {jk}",flush=True); continue
                trmask=((cur.d>=TRAIN_START)&(cur.d<ms)).to_numpy()
                temask=((cur.d>=ms)&(cur.d<me)).to_numpy()
                print(f"[job {job}/{total}] {jk} train={trmask.sum():,} test={temask.sum():,}",flush=True)

                if trmask.sum()<500 or temask.sum()<10:
                    rows.append({"version":VERSION,"job_key":jk,"setup_name":s["name"],
                                 "profile_name":pr["name"],"month":ms.strftime("%Y-%m"),
                                 "tp":s["tp"],"sl":s["sl"],"hold":s["hold"],**mets(pd.DataFrame())})
                    done.add(jk); save_ck(done,rows,trade_rows); continue

                clf,reg=train_models(cur.loc[trmask],y[trmask],r[trmask])
                testdf=cur.loc[temask]
                scored=score(testdf,clf,reg,pr["pw"],pr["rw"])

                prior=[x for x in trade_rows
                       if x["setup_name"]==s["name"] and x["profile_name"]==pr["name"]
                       and pd.Timestamp(x["entry_date"])<ms]
                tr=execute_month(scored,testdf,r[temask],k[temask],st[temask],prior)

                try:
                    auc=roc_auc_score(y[temask],clf.predict_proba(
                        testdf[v11.FEATURE_COLS].to_numpy(np.float32))[:,1])
                except Exception: auc=np.nan

                m=mets(tr)
                rows.append({"version":VERSION,"job_key":jk,"setup_name":s["name"],
                             "profile_name":pr["name"],"month":ms.strftime("%Y-%m"),
                             "tp":s["tp"],"sl":s["sl"],"hold":s["hold"],
                             "prob_weight":pr["pw"],"return_weight":pr["rw"],
                             "auc_test":auc,**m})
                if tr is not None and not tr.empty:
                    for _,x in tr.iterrows():
                        trade_rows.append({
                            "version":VERSION,"job_key":jk,"setup_name":s["name"],
                            "profile_name":pr["name"],"tp":s["tp"],"sl":s["sl"],"hold":s["hold"],
                            "entry_date":pd.Timestamp(x.entry_date).isoformat(),
                            "exit_date":pd.Timestamp(x.exit_date).isoformat(),
                            "symbol":x.symbol,"entry_price":x.entry_price,"score":x.score,
                            "p_tp":x.p_tp,"predicted_return":x.predicted_return,
                            "exit_kind":x.exit_kind,"net_return_pct":x.net_return_pct})
                done.add(jk); save_ck(done,rows,trade_rows)
                print(f"[checkpoint] {len(done)}/{total} | month trades={m['n']} avg={m['avg_net']:.3f}%",flush=True)

    rdf=pd.DataFrame(rows)
    rdf=rdf[rdf.version==VERSION].drop_duplicates("job_key",keep="last")
    rdf.to_csv(MONTHLY,index=False)

    tdf=pd.DataFrame(trade_rows)
    if len(tdf):
        tdf=tdf[tdf.version==VERSION].drop_duplicates(["job_key","symbol","entry_date"],keep="last")
        tdf.to_csv(TRADES,index=False)

    sums=[]
    if len(tdf):
        tdf["entry_date"]=pd.to_datetime(tdf.entry_date)
        for s in SETUPS:
            for pr in PROFILES:
                t=tdf[(tdf.setup_name==s["name"])&(tdf.profile_name==pr["name"])]
                for yr in [2025,2026]:
                    ty=t[t.entry_date.dt.year==yr]
                    mm=mets(ty if len(ty) else pd.DataFrame())
                    sums.append({"setup_name":s["name"],"profile_name":pr["name"],"year":yr,**mm})

    sdf=pd.DataFrame(sums); sdf.to_csv(SUMMARY,index=False)
    lines=["US WALK-FORWARD OPTIMIZER V14","="*100,f"Version: {VERSION}",
           "Monthly expanding retraining; no lookahead.","","YEAR SUMMARY","-"*100]
    if len(sdf):
        lines.append(sdf[["setup_name","profile_name","year","n","avg_net","win_rate","payoff","tp_rate","sl_rate","time_rate"]].to_string(index=False))
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{MONTHLY}\n{SUMMARY}\n{REPORT}\n{TRADES}",flush=True)

if __name__=="__main__":
    main()
