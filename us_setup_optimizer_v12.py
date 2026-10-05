#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, argparse, subprocess
from pathlib import Path

VERSION='12.0-spy-regime-sl-risk-20261005'
TRAIN_YEAR=2024; VALID_YEAR=2025; HOLDOUT_YEAR=2026
BASE_TP=20.0; BASE_SL=3.0; BASE_HOLD=10
OUT=Path(os.getenv('OPTIMIZER_OUT_DIR','/data')); OUT.mkdir(parents=True,exist_ok=True)
RESULTS_CSV=OUT/'us_v12_results.csv'; TOP20_CSV=OUT/'us_v12_top20.csv'; REPORT_TXT=OUT/'us_v12_report.txt'; TRADES_2025=OUT/'us_v12_trades_2025.csv'; TRADES_2026=OUT/'us_v12_trades_2026.csv'

def ensure(pkg, imp=None):
    imp=imp or pkg.split('==')[0].replace('-','_')
    try: __import__(imp)
    except Exception:
        print(f'[setup] installing {pkg} ...',flush=True)
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',pkg])
for pkg,imp in [('numpy',None),('pandas',None),('sqlalchemy',None),('scikit-learn','sklearn')]: ensure(pkg,imp)
try: import psycopg2
except Exception: ensure('psycopg2-binary','psycopg2')

import numpy as np, pandas as pd
from sqlalchemy import create_engine, inspect, text
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

try: import us_setup_optimizer_v11 as prev
except Exception:
    try: import us_setup_optimizer_v10_resume as prev
    except Exception as e: raise RuntimeError('Put us_setup_optimizer_v11.py or us_setup_optimizer_v10_resume.py beside this file in /app') from e

def db_url():
    u=os.getenv('DATABASE_URL','').strip()
    if not u: raise RuntimeError('DATABASE_URL is missing')
    if u.startswith('postgres://'): u='postgresql://'+u[len('postgres://'):]
    return u

def load_spy_from_db():
    table=getattr(prev,'TABLE',os.getenv('US_DAILY_TABLE','market_candles_1d'))
    eng=create_engine(db_url(),pool_pre_ping=True); insp=inspect(eng); cols=[c['name'] for c in insp.get_columns(table)]
    low={c.lower():c for c in cols}
    def pick(*names):
        for n in names:
            if n.lower() in low: return low[n.lower()]
    cs=pick('symbol','ticker','feed_symbol'); cd=pick('date','day','trade_date','ts','timestamp','time'); co=pick('open','o','adj_open'); ch=pick('high','h','adj_high'); cl=pick('low','l','adj_low'); cc=pick('close','c','adj_close'); cv=pick('volume','v','vol')
    if not all([cs,cd,co,ch,cl,cc,cv]): raise RuntimeError(f'Could not map SPY columns from {cols}')
    q=text(f"SELECT {cd} AS d,{co} AS o,{ch} AS h,{cl} AS l,{cc} AS c,{cv} AS v FROM {table} WHERE UPPER(CAST({cs} AS TEXT))='SPY' ORDER BY {cd}")
    with eng.connect() as con: spy=pd.read_sql_query(q,con)
    if spy.empty: raise RuntimeError('SPY not found in market_candles_1d')
    raw=spy['d']
    if pd.api.types.is_numeric_dtype(raw):
        vals=pd.to_numeric(raw,errors='coerce'); med=float(vals.dropna().median()) if vals.notna().any() else float('nan')
        if 19000101<=med<=21001231: dt=pd.to_datetime(vals.round().astype('Int64').astype(str),format='%Y%m%d',errors='coerce',utc=True)
        elif 1e8<=abs(med)<1e11: dt=pd.to_datetime(vals,unit='s',errors='coerce',utc=True)
        elif 1e11<=abs(med)<1e14: dt=pd.to_datetime(vals,unit='ms',errors='coerce',utc=True)
        elif 1e14<=abs(med)<1e17: dt=pd.to_datetime(vals,unit='us',errors='coerce',utc=True)
        else: dt=pd.to_datetime(vals,errors='coerce',utc=True)
    else: dt=pd.to_datetime(raw,errors='coerce',utc=True)
    spy['d']=dt.dt.tz_convert(None).dt.normalize()
    for c in ['o','h','l','c','v']: spy[c]=pd.to_numeric(spy[c],errors='coerce')
    return spy.dropna().drop_duplicates('d',keep='last').sort_values('d').reset_index(drop=True)

def add_spy_features(feat):
    spy=load_spy_from_db().copy(); sc=spy['c']
    for p in [5,20,60]: spy[f'spy_ret{p}']=100*(sc/sc.shift(p)-1)
    for p in [20,50,100,200]:
        spy[f'spy_sma{p}']=sc.shift(1).rolling(p,min_periods=p).mean(); spy[f'spy_above_sma{p}']=(sc>spy[f'spy_sma{p}']).astype(float)
    spy['spy_dist_sma50']=100*(sc/spy['spy_sma50']-1); spy['spy_dist_sma200']=100*(sc/spy['spy_sma200']-1)
    keep=['d','spy_ret5','spy_ret20','spy_ret60','spy_above_sma20','spy_above_sma50','spy_above_sma100','spy_above_sma200','spy_dist_sma50','spy_dist_sma200']
    out=feat.merge(spy[keep],on='d',how='left')
    for p in [5,20,60]: out[f'rs_vs_spy_{p}']=out[f'ret{p}']-out[f'spy_ret{p}']
    out['spy_regime_score']=(out['spy_above_sma20'].fillna(0)+out['spy_above_sma50'].fillna(0)+out['spy_above_sma100'].fillna(0)+out['spy_above_sma200'].fillna(0)).astype(np.float32)
    return out

BASE_FEATURES=list(prev.FEATURE_COLS)
EXTRA_FEATURES=['spy_ret5','spy_ret20','spy_ret60','spy_above_sma20','spy_above_sma50','spy_above_sma100','spy_above_sma200','spy_dist_sma50','spy_dist_sma200','rs_vs_spy_5','rs_vs_spy_20','rs_vs_spy_60','spy_regime_score']
FEATURE_COLS=BASE_FEATURES+EXTRA_FEATURES

def compute_targets(full,cand,cost):
    y,ret,step,kind=prev.compute_outcomes(full,cand.index.to_numpy(),BASE_TP,BASE_SL,BASE_HOLD,cost)
    ysl=np.array([1.0 if str(k)=='SL' else 0.0 for k in kind],dtype=np.float32)
    return y,ysl,ret,step,kind

def train_models(df,ytp,ysl,yret):
    X=df[FEATURE_COLS].to_numpy(np.float32)
    clf_tp=HistGradientBoostingClassifier(learning_rate=.05,max_iter=220,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=1.5,random_state=20261005)
    clf_sl=HistGradientBoostingClassifier(learning_rate=.05,max_iter=220,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=1.5,random_state=20261006)
    reg=HistGradientBoostingRegressor(learning_rate=.05,max_iter=220,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=1.5,random_state=20261007)
    clf_tp.fit(X,ytp.astype(int)); clf_sl.fit(X,ysl.astype(int)); reg.fit(X,yret.astype(np.float32)); return clf_tp,clf_sl,reg

def day_rank(df,vals):
    z=pd.DataFrame({'d':df['d'].to_numpy(),'v':vals},index=df.index)
    return z.groupby('d')['v'].rank(pct=True,method='average').to_numpy(np.float32)

def score_candidates(df,clf_tp,clf_sl,reg,weights):
    X=df[FEATURE_COLS].to_numpy(np.float32); p_tp=clf_tp.predict_proba(X)[:,1]; p_sl=clf_sl.predict_proba(X)[:,1]; pred=reg.predict(X)
    ret_rank=day_rank(df,pred); rs_rank=day_rank(df,df['rs_vs_spy_20'].to_numpy(np.float32)); regime=np.clip(df['spy_regime_score'].to_numpy(np.float32)/4.0,0,1)
    a,b,c,d,e=weights; score=a*p_tp-b*p_sl+c*ret_rank+d*rs_rank+e*regime
    return score

def enhanced_backtest(ranked,base_df,realized,kind,step,max_entries_per_day=1,max_open=10):
    if ranked.empty: return None,pd.DataFrame()
    pos=pd.Series(np.arange(len(base_df)),index=base_df.index); sym_groups={s:g for s,g in base_df.groupby('symbol',sort=False)}
    rows=[]; open_positions=[]; last_exit={}
    for d,g in ranked.groupby('d',sort=True):
        d=pd.Timestamp(d); open_positions=[(s,ed) for s,ed in open_positions if ed>d]; entries=0
        for ix,row in g.iterrows():
            if entries>=max_entries_per_day or len(open_positions)>=max_open: break
            sym=row['symbol']
            if any(s==sym for s,_ in open_positions): continue
            if sym in last_exit and d<=last_exit[sym]: continue
            p=int(pos.loc[ix]); rr=float(realized[p])
            if not math.isfinite(rr): continue
            st=int(step[p]) if math.isfinite(float(step[p])) else 0; kk=str(kind[p]); sg=sym_groups[sym]; loc=np.flatnonzero(sg.index.to_numpy()==ix)
            ed=pd.Timestamp(sg.iloc[min(len(sg)-1,int(loc[0])+st)]['d']) if len(loc) else d
            last_exit[sym]=ed; open_positions.append((sym,ed)); entries+=1
            rows.append({'index':int(ix),'symbol':sym,'entry_date':d,'entry_price':float(row['c']),'score':float(row['score']),'exit_kind':kk,'exit_date':ed,'net_return_pct':rr})
    tr=pd.DataFrame(rows)
    if tr.empty: return None,tr
    r=tr['net_return_pct'].to_numpy(float); wins=r[r>0]; losses=r[r<=0]; avg_win=float(wins.mean()) if len(wins) else 0.0; avg_loss=float(abs(losses.mean())) if len(losses) else 0.0; payoff=float('inf') if len(losses)==0 else (avg_win/avg_loss if avg_loss>0 else float('inf'))
    m={'n':len(tr),'avg_net':float(r.mean()),'median_net':float(np.median(r)),'win_rate':100*float((r>0).mean()),'avg_win':avg_win,'avg_loss_abs':avg_loss,'payoff':payoff,'tp_rate':100*float((tr.exit_kind=='TP').mean()),'sl_rate':100*float((tr.exit_kind=='SL').mean()),'time_rate':100*float((tr.exit_kind=='TIME').mean())}
    return m,tr

def validation_score(m):
    if not m: return -1e12
    n=m['n']; tier=3 if 250<=n<=350 else 2 if 220<=n<=380 else 1 if 180<=n<=420 else 0; p=m['payoff'] if math.isfinite(m['payoff']) else 10
    return tier*1_000_000-abs(n-300)*800+m['avg_net']*20_000+min(p,8)*4_000-m['sl_rate']*500+m['tp_rate']*500

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cost',type=float,default=.30); ap.add_argument('--min-price',type=float,default=2.0); ap.add_argument('--min-dollar-vol',type=float,default=3_000_000.0); args=ap.parse_args()
    print(f'[version] {VERSION}',flush=True); print(f'[base] TP={BASE_TP} SL={BASE_SL} HOLD={BASE_HOLD}',flush=True)
    full=prev.build_features(prev.load_daily()); full=add_spy_features(full)
    if hasattr(prev,'broad_candidates'): cand=prev.broad_candidates(full,args.min_price,args.min_dollar_vol)
    else:
        cand=full.loc[full['year'].isin([2024,2025,2026])&(full['c']>=args.min_price)&(full['dollar_vol20']>=args.min_dollar_vol)].copy()
    cand=cand[cand[FEATURE_COLS].notna().all(axis=1)].copy()
    print(f"[candidates] total={len(cand):,} 2024={sum(cand.year==2024):,} 2025={sum(cand.year==2025):,} 2026={sum(cand.year==2026):,}",flush=True)
    ytp,ysl,ret,step,kind=compute_targets(full,cand,args.cost); usable=np.isfinite(ytp)&np.isfinite(ysl)&np.isfinite(ret); cur=cand.loc[usable].copy(); ytp=ytp[usable]; ysl=ysl[usable]; rr=ret[usable]; ss=step[usable]; kk=kind[usable]
    yrs=cur['year'].to_numpy(); mt=yrs==2024; mv=yrs==2025; mh=yrs==2026
    clf_tp,clf_sl,reg=train_models(cur.loc[mt],ytp[mt],ysl[mt],rr[mt])
    try: auc_tp=roc_auc_score(ytp[mv],clf_tp.predict_proba(cur.loc[mv,FEATURE_COLS].to_numpy(np.float32))[:,1])
    except: auc_tp=np.nan
    try: auc_sl=roc_auc_score(ysl[mv],clf_sl.predict_proba(cur.loc[mv,FEATURE_COLS].to_numpy(np.float32))[:,1])
    except: auc_sl=np.nan
    profiles={'balanced':(.40,.30,.20,.05,.05),'sl_defensive':(.35,.40,.15,.05,.05),'tp_heavy':(.55,.25,.10,.05,.05),'return_heavy':(.30,.25,.35,.05,.05),'rs_regime':(.30,.30,.15,.15,.10),'sl_plus_rs':(.30,.40,.10,.15,.05)}
    val_df=cur.loc[mv]; results=[]; best=None; besttr=None
    for name,w in profiles.items():
        score=score_candidates(val_df,clf_tp,clf_sl,reg,w); ranked=val_df[['symbol','d','c']].copy(); ranked['score']=score; ranked=ranked.sort_values(['d','score'],ascending=[True,False]); ranked['day_rank']=ranked.groupby('d').cumcount()+1
        for mpe in [1,2]:
            for depth in [10,25,50]:
                sel=ranked[ranked.day_rank<=depth].copy(); m,tr=enhanced_backtest(sel,val_df,rr[mv],kk[mv],ss[mv],mpe,10)
                if not m: continue
                q=int(250<=m['n']<=350 and m['avg_net']>=4 and m['payoff']>=4)
                rec={'qualified':q,'weight_profile':name,'w_tp':w[0],'w_sl':w[1],'w_ret':w[2],'w_rs':w[3],'w_regime':w[4],'max_entries_per_day':mpe,'fallback_depth':depth,'auc_tp_valid':auc_tp,'auc_sl_valid':auc_sl,'valid_n':m['n'],'valid_avg_net':m['avg_net'],'valid_win_rate':m['win_rate'],'valid_payoff':m['payoff'],'valid_tp_rate':m['tp_rate'],'valid_sl_rate':m['sl_rate'],'valid_time_rate':m['time_rate'],'valid_score':validation_score(m)}; results.append(rec)
                if best is None or (rec['qualified'],rec['valid_score'])>(best['qualified'],best['valid_score']): best=rec; besttr=tr.copy()
    res=pd.DataFrame(results).sort_values(['qualified','valid_score'],ascending=[False,False]).reset_index(drop=True); res.to_csv(RESULTS_CSV,index=False); res.head(20).to_csv(TOP20_CSV,index=False); besttr.to_csv(TRADES_2025,index=False)
    w=(best['w_tp'],best['w_sl'],best['w_ret'],best['w_rs'],best['w_regime']); hold_df=cur.loc[mh]; score=score_candidates(hold_df,clf_tp,clf_sl,reg,w); ranked=hold_df[['symbol','d','c']].copy(); ranked['score']=score; ranked=ranked.sort_values(['d','score'],ascending=[True,False]); ranked['day_rank']=ranked.groupby('d').cumcount()+1; sel=ranked[ranked.day_rank<=int(best['fallback_depth'])].copy(); hm,htr=enhanced_backtest(sel,hold_df,rr[mh],kk[mh],ss[mh],int(best['max_entries_per_day']),10); htr.to_csv(TRADES_2026,index=False)
    lines=['US OUTCOME-LEARNING OPTIMIZER V12','='*100,f'Version: {VERSION}',f'Base setup: TP +{BASE_TP}% | SL -{BASE_SL}% | Hold {BASE_HOLD}',f'Train={TRAIN_YEAR} | Validate={VALID_YEAR} | Holdout={HOLDOUT_YEAR} partial',f'Round-trip cost: {args.cost:.3f}%','','BEST VALIDATION-SELECTED SETUP','-'*100]
    lines += [f'{k}: {v}' for k,v in best.items()]; lines += ['','2026 HOLDOUT','-'*100]
    if hm: lines += [f'{k}: {v}' for k,v in hm.items()]
    else: lines += ['No holdout trades.']
    lines += ['','TOP 20 BY 2025 VALIDATION','-'*100]
    cols=['qualified','weight_profile','w_tp','w_sl','w_ret','w_rs','w_regime','max_entries_per_day','fallback_depth','auc_tp_valid','auc_sl_valid','valid_n','valid_avg_net','valid_win_rate','valid_payoff','valid_tp_rate','valid_sl_rate','valid_time_rate','valid_score']; lines += [res.head(20)[cols].to_string(index=False),'','TARGET CHECK','-'*100]
    vok=250<=best['valid_n']<=350 and best['valid_avg_net']>=4 and best['valid_payoff']>=4; lines += [f'2025 validation meets full target: {vok}']
    if hm: lines += [f"2026 partial holdout meets return/payoff target: {hm['avg_net']>=4 and hm['payoff']>=4}",f"2026 partial holdout trades so far: {hm['n']}"]
    REPORT_TXT.write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True); print(f'\nSaved:\n{RESULTS_CSV}\n{TOP20_CSV}\n{REPORT_TXT}\n{TRADES_2025}\n{TRADES_2026}',flush=True)

if __name__=='__main__': main()
