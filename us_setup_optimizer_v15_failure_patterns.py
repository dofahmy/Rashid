#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math, json, argparse, subprocess
from pathlib import Path

VERSION="15.0-failure-pattern-learning-resume-20261005"
OUT=Path(os.getenv("OPTIMIZER_OUT_DIR","/data")); OUT.mkdir(parents=True, exist_ok=True)
CKCSV=OUT/"us_v15_checkpoint_results.csv"
CKJSON=OUT/"us_v15_checkpoint_state.json"
RESULTS=OUT/"us_v15_results.csv"
TOP20=OUT/"us_v15_top20.csv"
REPORT=OUT/"us_v15_report.txt"
TRADES25=OUT/"us_v15_trades_2025.csv"
TRADES26=OUT/"us_v15_trades_2026.csv"
FAILURE=OUT/"us_v15_failure_timing.csv"

def ensure(pkg,name=None):
    name=name or pkg.split("==")[0].replace("-","_")
    try: __import__(name)
    except Exception:
        print(f"[setup] installing {pkg} ...", flush=True)
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet",pkg])

for pkg,name in [("numpy",None),("pandas",None),("scikit-learn","sklearn")]:
    ensure(pkg,name)

import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

try:
    import us_setup_optimizer_v11 as v11
except Exception as e:
    raise RuntimeError("Put us_setup_optimizer_v11.py beside V15 in /app") from e

TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR=2024,2025,2026
SETUPS=[
    {"name":"A_20_3_10","tp":20.0,"sl":3.0,"hold":10},
    {"name":"B_18_3_12","tp":18.0,"sl":3.0,"hold":12},
]
PROFILES=[
    {"name":"balanced","w_tp":.45,"w_ret":.30,"w_neutral":.05,"w_sl":.10,"w_fast":.10},
    {"name":"tp_focus","w_tp":.60,"w_ret":.25,"w_neutral":0.0,"w_sl":.075,"w_fast":.075},
    {"name":"return_focus","w_tp":.35,"w_ret":.45,"w_neutral":.05,"w_sl":.075,"w_fast":.075},
    {"name":"failure_aware","w_tp":.45,"w_ret":.25,"w_neutral":.05,"w_sl":.10,"w_fast":.15},
    {"name":"fastsl_defensive","w_tp":.40,"w_ret":.25,"w_neutral":.05,"w_sl":.10,"w_fast":.20},
]
FAST_VETOES=[None,.45,.50,.55,.60,.65]
SL_VETOES=[None,.70,.75,.80,.85]
MAX_OPEN=10
FALLBACK_DEPTH=50

def load_ck():
    done=set(); rows=[]
    if CKJSON.exists():
        try:
            s=json.loads(CKJSON.read_text(encoding="utf-8"))
            if s.get("version")==VERSION: done=set(s.get("done",[]))
        except Exception as e: print(f"[resume] state warning: {e}",flush=True)
    if CKCSV.exists():
        try:
            d=pd.read_csv(CKCSV)
            if "version" in d.columns: d=d[d.version==VERSION]
            rows=d.to_dict("records")
        except Exception as e: print(f"[resume] result warning: {e}",flush=True)
    return done,rows

def save_ck(done,rows):
    if rows:
        p=CKCSV.with_suffix(".tmp.csv"); pd.DataFrame(rows).to_csv(p,index=False); p.replace(CKCSV)
    p=CKJSON.with_suffix(".tmp.json")
    p.write_text(json.dumps({"version":VERSION,"done":sorted(done),
        "saved_at":pd.Timestamp.now("UTC").isoformat(),"completed_jobs":len(done),
        "rows":len(rows)},ensure_ascii=False,indent=2),encoding="utf-8")
    p.replace(CKJSON)

def labels(kind,step):
    k=np.asarray(kind,dtype=object); st=np.asarray(step,dtype=float)
    cls=np.ones(len(k),dtype=np.int8)
    cls[np.array([str(x)=="TP" for x in k])]=0
    cls[np.array([str(x)=="SL" for x in k])]=2
    fast=np.array([(str(x)=="SL" and np.isfinite(s) and s<=3) for x,s in zip(k,st)],dtype=np.int8)
    w=np.ones(len(k),dtype=np.float32)
    for i,(x,s) in enumerate(zip(k,st)):
        if str(x)!="SL" or not np.isfinite(s): continue
        if s<=3: w[i]=2.5
        elif s<=7: w[i]=1.5
    return cls,fast,w

def train_models(df,cls,fast,w,ret):
    X=df[v11.FEATURE_COLS].to_numpy(np.float32)
    multi=HistGradientBoostingClassifier(learning_rate=.05,max_iter=240,max_leaf_nodes=31,
        min_samples_leaf=60,l2_regularization=1.5,random_state=20261005)
    fastclf=HistGradientBoostingClassifier(learning_rate=.05,max_iter=220,max_leaf_nodes=31,
        min_samples_leaf=60,l2_regularization=1.5,random_state=20261006)
    reg=HistGradientBoostingRegressor(learning_rate=.05,max_iter=220,max_leaf_nodes=31,
        min_samples_leaf=60,l2_regularization=1.5,random_state=20261007)
    multi.fit(X,cls.astype(int),sample_weight=w)
    fastclf.fit(X,fast.astype(int))
    reg.fit(X,ret.astype(np.float32))
    return multi,fastclf,reg

def score(df,multi,fastclf,reg,p,fast_veto,sl_veto):
    X=df[v11.FEATURE_COLS].to_numpy(np.float32)
    pr=multi.predict_proba(X); cmap={int(c):i for i,c in enumerate(multi.classes_)}
    ptp=pr[:,cmap[0]] if 0 in cmap else np.zeros(len(df))
    ptime=pr[:,cmap[1]] if 1 in cmap else np.zeros(len(df))
    psl=pr[:,cmap[2]] if 2 in cmap else np.zeros(len(df))
    pfast=fastclf.predict_proba(X)[:,1]; er=reg.predict(X)
    z=df[["symbol","d","c"]].copy()
    z["p_tp"]=ptp; z["p_time"]=ptime; z["p_sl"]=psl; z["p_fast_sl"]=pfast; z["pred_return"]=er
    if fast_veto is not None: z=z[z.p_fast_sl<=float(fast_veto)].copy()
    if sl_veto is not None: z=z[z.p_sl<=float(sl_veto)].copy()
    if z.empty: return z
    z["ret_rank"]=z.groupby("d")["pred_return"].rank(pct=True,method="average")
    z["score"]=(p["w_tp"]*z.p_tp+p["w_ret"]*z.ret_rank+p["w_neutral"]*z.p_time
                -p["w_sl"]*z.p_sl-p["w_fast"]*z.p_fast_sl)
    z=z.sort_values(["d","score"],ascending=[True,False]); z["day_rank"]=z.groupby("d").cumcount()+1
    return z[z.day_rank<=FALLBACK_DEPTH].copy()

def execute(scored,base,ret,kind,step):
    if scored.empty: return pd.DataFrame()
    pos=pd.Series(np.arange(len(base)),index=base.index)
    symgroups={s:g for s,g in base.groupby("symbol",sort=False)}
    active=[]; last_exit={}; rows=[]
    for d,g in scored.groupby("d",sort=True):
        d=pd.Timestamp(d); active=[x for x in active if x[1]>d]
        if len(active)>=MAX_OPEN: continue
        for ix,row in g.sort_values("score",ascending=False).iterrows():
            sym=row.symbol
            if any(s==sym for s,_ in active): continue
            if sym in last_exit and d<=last_exit[sym]: continue
            q=int(pos.loc[ix]); rr=float(ret[q])
            if not np.isfinite(rr): continue
            st=int(step[q]) if np.isfinite(float(step[q])) else 0
            kk=str(kind[q]); sg=symgroups[sym]; loc=np.flatnonzero(sg.index.to_numpy()==ix)
            ed=pd.Timestamp(sg.iloc[min(len(sg)-1,int(loc[0])+st)]["d"]) if len(loc) else d
            active.append((sym,ed)); last_exit[sym]=ed
            rows.append({"index":int(ix),"symbol":sym,"entry_date":d,"entry_price":float(row.c),
                         "score":float(row.score),"p_tp":float(row.p_tp),"p_time":float(row.p_time),
                         "p_sl":float(row.p_sl),"p_fast_sl":float(row.p_fast_sl),
                         "predicted_return":float(row.pred_return),"exit_kind":kk,"exit_step":st,
                         "exit_date":ed,"net_return_pct":rr})
            break
    return pd.DataFrame(rows)

def mets(t):
    if t is None or t.empty:
        return dict(n=0,avg_net=np.nan,median_net=np.nan,win_rate=np.nan,payoff=np.nan,
                    tp_rate=np.nan,sl_rate=np.nan,time_rate=np.nan,fast_sl_rate=np.nan,
                    sl_4_7_rate=np.nan,sl_late_rate=np.nan)
    r=t.net_return_pct.to_numpy(float); w=r[r>0]; l=r[r<=0]
    aw=float(w.mean()) if len(w) else 0.; al=float(abs(l.mean())) if len(l) else 0.
    payoff=float("inf") if not len(l) else (aw/al if al>0 else float("inf"))
    isl=t.exit_kind=="SL"; fast=isl&(t.exit_step<=3); mid=isl&(t.exit_step>=4)&(t.exit_step<=7); late=isl&(t.exit_step>=8)
    return dict(n=len(t),avg_net=float(r.mean()),median_net=float(np.median(r)),
        win_rate=float(100*(r>0).mean()),payoff=payoff,tp_rate=float(100*(t.exit_kind=="TP").mean()),
        sl_rate=float(100*isl.mean()),time_rate=float(100*(t.exit_kind=="TIME").mean()),
        fast_sl_rate=float(100*fast.mean()),sl_4_7_rate=float(100*mid.mean()),sl_late_rate=float(100*late.mean()))

def vscore(m):
    if not m or m["n"]==0: return -1e12
    n=m["n"]; payoff=m["payoff"] if np.isfinite(m["payoff"]) else 10.
    if 250<=n<=350 and m["avg_net"]>=4 and payoff>=4: tier=6
    elif 250<=n<=350 and payoff>=4 and m["tp_rate"]>=18: tier=5
    elif 250<=n<=350 and payoff>=4: tier=4
    elif 220<=n<=380 and payoff>=4: tier=3
    elif 250<=n<=350: tier=2
    elif 180<=n<=420: tier=1
    else: tier=0
    return tier*1_000_000+m["avg_net"]*50_000+min(payoff,8)*5_000+m["tp_rate"]*800-m["fast_sl_rate"]*600-abs(n-300)*400

def jk(s,p,fv,sv): return f"{s}|{p}|fast={fv}|sl={sv}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cost",type=float,default=.30)
    ap.add_argument("--min-price",type=float,default=2.0); ap.add_argument("--min-dollar-vol",type=float,default=3_000_000.)
    ap.add_argument("--reset-checkpoint",action="store_true"); a=ap.parse_args()
    if a.reset_checkpoint:
        for p in [CKCSV,CKJSON,RESULTS,TOP20,REPORT,TRADES25,TRADES26,FAILURE]:
            try:p.unlink()
            except FileNotFoundError:pass
        print("[resume] checkpoint cleared",flush=True)
    print(f"[version] {VERSION}",flush=True)
    print("[plan] TP/TIME/SL multiclass + fast-SL model + expected return",flush=True)
    full=v11.build_features(v11.load_daily()); cand=v11.broad_candidates(full,a.min_price,a.min_dollar_vol)
    done,rows=load_ck(); total=len(SETUPS)*len(PROFILES)*len(FAST_VETOES)*len(SL_VETOES)
    print(f"[resume] completed={len(done)}/{total}",flush=True)
    cache={}
    for s in SETUPS:
        print(f"[outcomes] {s['name']}",flush=True)
        y,r,st,k=v11.compute_outcomes(full,cand.index.to_numpy(),s["tp"],s["sl"],s["hold"],a.cost)
        ok=np.isfinite(y)&np.isfinite(r); cur=cand.loc[ok].copy(); rr=r[ok]; ss=st[ok]; kk=k[ok]
        cls,fast,w=labels(kk,ss); yrs=cur.year.to_numpy(); mt=yrs==2024; mv=yrs==2025; mh=yrs==2026
        print(f"[train] {s['name']} train={mt.sum():,} valid={mv.sum():,} holdout={mh.sum():,}",flush=True)
        multi,fastclf,reg=train_models(cur.loc[mt],cls[mt],fast[mt],w[mt],rr[mt])
        try:
            auc=roc_auc_score(fast[mv],fastclf.predict_proba(cur.loc[mv,v11.FEATURE_COLS].to_numpy(np.float32))[:,1])
        except Exception: auc=np.nan
        cache[s["name"]]=dict(cur=cur,rr=rr,ss=ss,kk=kk,mv=mv,mh=mh,multi=multi,fastclf=fastclf,reg=reg,auc=auc)
    job=0
    for s in SETUPS:
        c=cache[s["name"]]; valdf=c["cur"].loc[c["mv"]]
        for p in PROFILES:
            for fv in FAST_VETOES:
                for sv in SL_VETOES:
                    job+=1; key=jk(s["name"],p["name"],fv,sv)
                    if key in done:
                        print(f"[resume] skip {job}/{total} {key}",flush=True); continue
                    sc=score(valdf,c["multi"],c["fastclf"],c["reg"],p,fv,sv)
                    tr=execute(sc,valdf,c["rr"][c["mv"]],c["kk"][c["mv"]],c["ss"][c["mv"]]); m=mets(tr)
                    rec={"version":VERSION,"job_key":key,"setup_name":s["name"],"tp":s["tp"],"sl":s["sl"],"hold":s["hold"],
                         "profile_name":p["name"],"fast_veto":fv,"sl_veto":sv,"fast_auc_valid":c["auc"],**m}
                    rec["qualified"]=int(250<=m["n"]<=350 and m["avg_net"]>=4 and m["payoff"]>=4)
                    rec["valid_score"]=vscore(m); rows.append(rec); done.add(key); save_ck(done,rows)
                    print(f"[checkpoint] {len(done)}/{total} | {key} | n={m['n']} avg={m['avg_net']:.3f}% TP={m['tp_rate']:.1f}% SL={m['sl_rate']:.1f}% fastSL={m['fast_sl_rate']:.1f}%",flush=True)
    rdf=pd.DataFrame(rows); rdf=rdf[rdf.version==VERSION].drop_duplicates("job_key",keep="last")
    rdf=rdf.sort_values(["qualified","valid_score"],ascending=[False,False]).reset_index(drop=True)
    rdf.to_csv(RESULTS,index=False); rdf.head(20).to_csv(TOP20,index=False)
    best=rdf.iloc[0].to_dict(); c=cache[str(best["setup_name"])]; p=next(x for x in PROFILES if x["name"]==best["profile_name"])
    fv=None if pd.isna(best["fast_veto"]) else float(best["fast_veto"]); sv=None if pd.isna(best["sl_veto"]) else float(best["sl_veto"])
    valdf=c["cur"].loc[c["mv"]]; sc25=score(valdf,c["multi"],c["fastclf"],c["reg"],p,fv,sv)
    tr25=execute(sc25,valdf,c["rr"][c["mv"]],c["kk"][c["mv"]],c["ss"][c["mv"]]); tr25.to_csv(TRADES25,index=False)
    holddf=c["cur"].loc[c["mh"]]; sc26=score(holddf,c["multi"],c["fastclf"],c["reg"],p,fv,sv)
    tr26=execute(sc26,holddf,c["rr"][c["mh"]],c["kk"][c["mh"]],c["ss"][c["mh"]]); tr26.to_csv(TRADES26,index=False)
    m25,m26=mets(tr25),mets(tr26)
    ft=pd.DataFrame([{"year":2025,"fast_sl_1_3":m25["fast_sl_rate"],"sl_4_7":m25["sl_4_7_rate"],"sl_8_plus":m25["sl_late_rate"],"all_sl":m25["sl_rate"],"tp":m25["tp_rate"],"time":m25["time_rate"],"n":m25["n"]},
                     {"year":2026,"fast_sl_1_3":m26["fast_sl_rate"],"sl_4_7":m26["sl_4_7_rate"],"sl_8_plus":m26["sl_late_rate"],"all_sl":m26["sl_rate"],"tp":m26["tp_rate"],"time":m26["time_rate"],"n":m26["n"]}])
    ft.to_csv(FAILURE,index=False)
    lines=["US FAILURE-PATTERN OPTIMIZER V15","="*100,f"Version: {VERSION}",
           f"Train=2024 | Validate=2025 | Holdout=2026 partial",f"Round-trip cost: {a.cost:.3f}%","",
           "BEST VALIDATION-SELECTED SETUP","-"*100]
    for k,v in best.items(): lines.append(f"{k}: {v}")
    lines+=["","2025 SELECTED RESULT","-"*100]+[f"{k}: {v}" for k,v in m25.items()]
    lines+=["","2026 HOLDOUT","-"*100]+[f"{k}: {v}" for k,v in m26.items()]
    lines+=["","FAILURE TIMING","-"*100,ft.to_string(index=False),"","TOP 20 BY 2025 VALIDATION","-"*100]
    cols=["qualified","setup_name","profile_name","fast_veto","sl_veto","n","avg_net","win_rate","payoff","tp_rate","sl_rate","fast_sl_rate","sl_4_7_rate","sl_late_rate","valid_score"]
    lines.append(rdf.head(20)[cols].to_string(index=False))
    lines+=["","TARGET CHECK","-"*100,
            f"2025 validation meets full target: {250<=m25['n']<=350 and m25['avg_net']>=4 and m25['payoff']>=4}",
            f"2026 partial holdout meets return/payoff target: {m26['avg_net']>=4 and m26['payoff']>=4}",
            f"2026 partial holdout trades so far: {m26['n']}"]
    REPORT.write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines),flush=True)
    print(f"\nSaved:\n{RESULTS}\n{TOP20}\n{REPORT}\n{TRADES25}\n{TRADES26}\n{FAILURE}",flush=True)

if __name__=="__main__":
    main()
