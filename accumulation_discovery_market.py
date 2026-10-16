#!/usr/bin/env python3
"""One-run research: historical examples -> broad *pre-breakout* fingerprints -> frozen rule -> full US cached backtest.
NOT a production recommendation engine. Read-only. Train examples TSLA/MSFT/META, market held out.
"""
from __future__ import annotations
import argparse, csv, json, math, itertools, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import text

SAMPLES={'META':('2012-05-01','2026-10-16'),'TSLA':('2012-01-01','2026-10-16'),'MSFT':('1993-01-01','2026-10-16')}
BASE_COLUMNS=['date','symbol','signal_close','base_width_pct','breakout_pct','resistance_touches','support_touches','volume_ratio_20','future_complete_3y','max_rise_3y_pct','first_500pct_date','sessions_to_500pct','max_drawdown_3y_pct']
FEATURES=['base_width_pct','breakout_pct','resistance_touches','support_touches','volume_ratio_20',
'close_position_base_pct','close_position_prev20_pct','body_pct_candle','upper_wick_pct','lower_wick_pct',
'prev5_return_pct','prev10_return_pct','prev20_return_pct','prev60_return_pct','prev120_return_pct','prev252_return_pct',
'prev20_range_pct','prev10_range_pct','volatility20_pct','volatility60_pct','volatility_ratio20to60',
'range20_to_range60','range10_to_range60','prior_trend60_vs120_pct','base_close_change_pct',
'near_resistance_days20','near_support_days20','up_day_share20','down_day_share20',
'volume20_over60','volume_today_over20','volume_up_vs_down20','gap_pct','breakout_body_pct',
'breakout_candle_range_pct','distance_ma20_pct','distance_ma60_pct','distance_ma120_pct',
'ma20_vs60_pct','ma60_vs120_pct','high60_position252_pct','lower_half_days60_pct',
'last20_compression_pct','recovery_from_60low_pct','prior_drawdown_from120high_pct',
'base_first30_last30_change_pct','prior_20day_negative_streak','atr14_pct']

def read_yahoo(symbol,start,end):
    import yfinance as yf
    x=yf.download(symbol,start=start,end=end,auto_adjust=False,progress=False,threads=False)
    if x.empty: raise ValueError('Empty Yahoo result')
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    x=x.rename(columns={i: str(i).title() for i in x.columns})
    x.index=pd.to_datetime(x.index).tz_localize(None).normalize()
    return clean(x)

def clean(df):
    df=df.copy();df=df.rename(columns={c:c.title() for c in df.columns})
    for k in ('Open','High','Low','Close'):
        if k not in df: df[k]=np.nan
    if 'Volume' not in df:df['Volume']=np.nan
    df=df[['Open','High','Low','Close','Volume']]
    for c in df:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.loc[~df.index.duplicated(keep='last')].sort_index().dropna(subset=['High','Low','Close'])
    return df.loc[(df.Close>0)&(df.High>0)&(df.Low>0)]

def pct(a,b):return (a/b-1)*100 if b is not None and np.isfinite(b) and b>0 and np.isfinite(a) else np.nan

def feats(df,i,base, touches, breakout):
    # Hist closes at signal i; *all* values available by close on signal day.
    close=df.Close.to_numpy(dtype=float); hi=df.High.to_numpy(dtype=float);lo=df.Low.to_numpy(dtype=float)
    vol=df.Volume.to_numpy(dtype=float); op=df.Open.to_numpy(dtype=float)
    c=close[i];h=hi[i];l=lo[i]
    ph=hi[i-base:i];pl=lo[i-base:i];pc=close[i-base:i]
    high=max(ph);low=min(pl)
    hist=lambda n:close[max(0,i-n):i]
    r=lambda n:np.nanmax(hi[max(0,i-n):i])/np.nanmin(lo[max(0,i-n):i])*100-100 if i>=n else np.nan
    ret=lambda n:pct(close[i-1],close[i-1-n]) if i>n else np.nan
    rv=np.diff(np.log(close[max(0,i-60):i]))*100
    rv20=rv[-19:] if len(rv)>=19 else []
    sigma20=float(np.std(rv20)*np.sqrt(252)) if len(rv20)>=10 else np.nan
    sigma60=float(np.std(rv)*np.sqrt(252)) if len(rv)>=40 else np.nan
    hi20=np.max(hi[i-20:i]);lo20=np.min(lo[i-20:i])
    hi10=np.max(hi[i-10:i]);lo10=np.min(lo[i-10:i])
    range60=max(ph)-min(pl)
    daily_range=h-l
    body=c-op[i] if np.isfinite(op[i]) else np.nan
    candle_fraction=lambda v:(100*v/daily_range) if daily_range>0 and np.isfinite(v) else np.nan
    cp=np.diff(pc)
    vol20=vol[i-20:i];vol60=vol[max(0,i-60):i]
    med20=np.nanmedian(vol20) if np.any(np.isfinite(vol20)) else np.nan
    med60=np.nanmedian(vol60) if np.any(np.isfinite(vol60)) else np.nan
    plus=(np.diff(pc[-21:])>0) if len(pc)>20 else np.array([])
    neg=(np.diff(pc[-21:])<0) if len(pc)>20 else np.array([])
    vmax=np.nanmean(vol[i-20:i][plus]) if len(plus)==20 and np.any(plus) else np.nan
    vmin=np.nanmean(vol[i-20:i][neg]) if len(neg)==20 and np.any(neg) else np.nan
    true_ranges=[]
    for j in range(max(1,i-13),i+1): true_ranges.append(max(hi[j]-lo[j],abs(hi[j]-close[j-1]),abs(lo[j]-close[j-1])))
    ddcount=0
    for j in range(i-1,max(i-21,0),-1):
        if close[j]<close[j-1]:ddcount+=1
        else:break
    f={
        'base_width_pct':pct(high,low),'breakout_pct':pct(c,high),
        'resistance_touches':sum(ph>=high*(1-touches/100)),'support_touches':sum(pl<=low*(1+touches/100)),
        'volume_ratio_20':vol[i]/med20 if np.isfinite(med20) and med20>0 else np.nan,
        'close_position_base_pct':(c-low)/(high-low)*100 if high>low else np.nan,
        'close_position_prev20_pct':(close[i-1]-lo20)/(hi20-lo20)*100 if hi20>lo20 else np.nan,
        'body_pct_candle':candle_fraction(body),'upper_wick_pct':candle_fraction(h-max(c,op[i])),
        'lower_wick_pct':candle_fraction(min(c,op[i])-l),
        'prev5_return_pct':ret(5),'prev10_return_pct':ret(10),'prev20_return_pct':ret(20),
        'prev60_return_pct':ret(60),'prev120_return_pct':ret(120),'prev252_return_pct':ret(252),
        'prev20_range_pct':r(20),'prev10_range_pct':r(10),
        'volatility20_pct':sigma20,'volatility60_pct':sigma60,
        'volatility_ratio20to60':sigma20/sigma60 if sigma60>0 else np.nan,
        'range20_to_range60':(hi20-lo20)/range60 if range60>0 else np.nan,
        'range10_to_range60':(hi10-lo10)/range60 if range60>0 else np.nan,
        'prior_trend60_vs120_pct':pct(close[i-60],close[i-120]) if i>=120 else np.nan,
        'base_close_change_pct':pct(close[i-1],close[i-base]),
        'near_resistance_days20':sum(hi[i-20:i]>=high*(1-touches/100)),
        'near_support_days20':sum(lo[i-20:i]<=low*(1+touches/100)),
        'up_day_share20':float(np.mean(plus))*100 if len(plus) else np.nan,
        'down_day_share20':float(np.mean(neg))*100 if len(neg) else np.nan,
        'volume20_over60':med20/med60 if np.isfinite(med20) and np.isfinite(med60) and med60>0 else np.nan,
        'volume_today_over20':vol[i]/med20 if np.isfinite(med20) and med20>0 else np.nan,
        'volume_up_vs_down20':vmax/vmin if np.isfinite(vmax) and np.isfinite(vmin) and vmin>0 else np.nan,
        'gap_pct':pct(op[i],close[i-1]),'breakout_body_pct':pct(c,op[i]),
        'breakout_candle_range_pct':pct(h,l),
        'distance_ma20_pct':pct(c,np.mean(hist(20))),
        'distance_ma60_pct':pct(c,np.mean(hist(60))),
        'distance_ma120_pct':pct(c,np.mean(hist(120))) if i>=120 else np.nan,
        'ma20_vs60_pct':pct(np.mean(hist(20)),np.mean(hist(60))),
        'ma60_vs120_pct':pct(np.mean(hist(60)),np.mean(hist(120))) if i>=120 else np.nan,
        'high60_position252_pct':(high-np.min(lo[i-252:i]))/(np.max(hi[i-252:i])-np.min(lo[i-252:i]))*100 if i>=252 and np.max(hi[i-252:i])>np.min(lo[i-252:i]) else np.nan,
        'lower_half_days60_pct':float(np.mean(pc <= (low+high)/2)*100),
        'last20_compression_pct':r(20)/r(60) if r(60)>0 else np.nan,
        'recovery_from_60low_pct':pct(c,low),
        'prior_drawdown_from120high_pct':pct(close[i-1],np.max(hi[i-120:i])) if i>=120 else np.nan,
        'base_first30_last30_change_pct':pct(np.mean(pc[-30:]),np.mean(pc[:30])) if len(pc)>=60 else np.nan,
        'prior_20day_negative_streak':ddcount,
        'atr14_pct':float(np.mean(true_ranges)/c*100) if true_ranges else np.nan,
    }
    return {k: round(float(f.get(k)),6) if np.isfinite(f.get(k,np.nan)) else '' for k in FEATURES}

def detect(df,symbol,args):
    if len(df)<args.base+2:return []
    prevhi=df.High.shift(1).rolling(args.base,min_periods=args.base).max()
    prevlo=df.Low.shift(1).rolling(args.base,min_periods=args.base).min()
    bw=(prevhi/prevlo-1)*100
    bp=(df.Close/prevhi-1)*100
    mask=(bp>=args.breakout)&(bw<=args.max_width)&(df.Close.shift(1)<=prevhi*(1+args.breakout/100))
    indices=np.flatnonzero(mask.fillna(False).to_numpy());last=-10**9
    hi=df.High.to_numpy();lo=df.Low.to_numpy();c=df.Close.to_numpy();dates=df.index
    result=[]
    for i in indices:
        if i-last<=args.cooldown:continue
        ph=hi[i-args.base:i];pl=lo[i-args.base:i]
        bh=max(ph);bl=min(pl)
        if sum(ph>=bh*(1-args.touch_pct/100))<args.min_touches:continue
        if sum(pl<=bl*(1+args.touch_pct/100))<args.min_touches:continue
        future=hi[i+1:i+1+args.horizon]
        fmin=lo[i+1:i+1+args.horizon]
        first=np.flatnonzero(future>=c[i]*6)
        complete=len(future)>=args.horizon
        target_hit=bool(len(first))
        feat=feats(df,i,args.base,args.touch_pct,args.breakout)
        x={**feat,'date':str(dates[i].date()),'symbol':symbol,'signal_close':round(float(c[i]),6),
           'future_complete_3y':int(complete),'future_hit_500':int(target_hit),
           'first_500pct_date':str(dates[i+1+int(first[0])].date()) if target_hit else '',
           'sessions_to_500pct':int(first[0])+1 if target_hit else '',
           'max_rise_3y_pct':round(pct(max(future),c[i]),4) if len(future) else '',
           'max_drawdown_3y_pct':round(pct(min(fmin),c[i]),4) if len(fmin) else '',
           'available_future_sessions':len(future)}
        result.append(x);last=i
    return result

def save_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row)) if rows else ['date','symbol']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('SAVED',path,'ROWS',len(rows),flush=True)

def numeric(row,k):
    try:return float(row[k])
    except (TypeError,ValueError,KeyError):return float('nan')

def clause_match(row,cl):
    k,op,v=cl;x=numeric(row,k)
    return bool(np.isfinite(x) and (x<=v if op=='<=' else x>=v))

def rule_match(row,clauses):return all(clause_match(row,c) for c in clauses)

def cv_score(rows, clauses, split=None):
    if split:rows=[r for r in rows if r['symbol'] in split]
    # Only complete 3yr cases in model building; success in incomplete is excluded for symmetric labels.
    arr=[r for r in rows if r['future_complete_3y']]
    matches=[r for r in arr if rule_match(r,clauses)]
    hit=sum(r['future_hit_500'] for r in matches)
    total=len(matches)
    return (total,hit,round(hit/total,5) if total else 0)

def discover(rows,args):
    labeled=[r for r in rows if r['future_complete_3y']]
    print('TRAIN:',len(labeled),'COMPLETE CASES,',sum(r['future_hit_500'] for r in labeled),'HITS',flush=True)
    # Univariate thresholds: observed quantiles computed from TRAIN ONLY (all three named stocks).
    all_clauses=[]
    market_unavailable={'volume_ratio_20','volume20_over60','volume_today_over20',
                        'volume_up_vs_down20','body_pct_candle','upper_wick_pct',
                        'lower_wick_pct','gap_pct','breakout_body_pct'}
    for key in FEATURES:
        if key in market_unavailable:continue
        vals=np.array([numeric(r,key) for r in labeled]); vals=vals[np.isfinite(vals)]
        if len(vals)<int(0.6*len(labeled)):continue # volume vars absent in market should not be chosen
        if len(np.unique(vals))<4:continue
        for q in (.15,.25,.35,.45,.55,.65,.75,.85):
            v=round(float(np.quantile(vals,q)),4)
            for op in ('<=','>='):
                cl=(key,op,v)
                total,hits,_=cv_score(labeled,[cl])
                if total>=args.min_train_support and hits>=args.min_train_hits:
                    all_clauses.append((cl,total,hits))
    # Rank candidates using precision with beta-binomial shrinkage + support. Prevent trivial tiny subsets.
    base_rate=sum(r['future_hit_500'] for r in labeled)/max(len(labeled),1)
    def score(n,h):
        # Shrink tiny groups heavily, while requiring uplift over baseline; force adequate recall.
        rate=(h+8*base_rate)/(n+8)
        recall=h/max(sum(r['future_hit_500'] for r in labeled),1)
        return round((rate-base_rate)*math.sqrt(n)*(.55+.45*recall),6)
    all_clauses.sort(key=lambda t:(score(t[1],t[2]),t[2]),reverse=True)
    chosen_single=all_clauses[:args.top_singles]
    evaluated=[]
    for cl,n,h in chosen_single:
        evaluated.append((score(n,h),[cl],n,h))
    # pairs and triples from strongest marginal clauses; correlation redundancy is accepted but penalized.
    for a,b in itertools.combinations(chosen_single[:args.combo_singles],2):
        clauses=[a[0],b[0]]
        if clauses[0][0]==clauses[1][0]:continue
        n,h,_=cv_score(labeled,clauses)
        if n>=args.min_train_support and h>=args.min_train_hits:evaluated.append((score(n,h)-.003,clauses,n,h))
    pairs=sorted([e for e in evaluated if len(e[1])==2],key=lambda e:e[0],reverse=True)[:args.top_pairs]
    for p in pairs:
        for cl,_,_ in chosen_single[:args.combo_singles]:
            if cl[0] in [x[0] for x in p[1]]:continue
            clauses=p[1]+[cl]
            n,h,_=cv_score(labeled,clauses)
            if n>=args.min_train_support and h>=args.min_train_hits:evaluated.append((score(n,h)-.008,clauses,n,h))
    # Group by economic wave proxy: leave-one-stock-out diagnostics, not a formal independent test.
    evaluated.sort(key=lambda x:(x[0],x[3]),reverse=True)
    selected=[];seen=set()
    for sc,cls,n,h in evaluated:
        key=json.dumps(sorted(cls))
        if key in seen:continue
        seen.add(key)
        yr={sym:cv_score(labeled,cls,{sym}) for sym in SAMPLES}
        # Penalize all success coming from only Tesla; no rule can generate META hits because none exist.
        nonzero_groups=sum(s[1]>0 for s in yr.values())
        item={'rank':len(selected)+1,'score':sc,'clauses':cls,'train_n':n,'train_hits':h,
              'train_hit_rate_pct':round(h/n*100,2),'tsla_n_hits':str(yr['TSLA']),
              'msft_n_hits':str(yr['MSFT']),'meta_n_hits':str(yr['META']),
              'hits_in_different_stocks':nonzero_groups}
        selected.append(item)
        if len(selected)>=args.max_rules:break
    stable=[s for s in selected if s['hits_in_different_stocks']>=2]
    winner=(stable[0] if stable else (selected[0] if selected else None))
    print('CANDIDATE RULES:',len(selected),'STABLE MULTI-STOCK:',len(stable),flush=True)
    if winner:print('FROZEN WINNER',json.dumps(winner,ensure_ascii=False),flush=True)
    else:print('NO RULE SATISFIES TRAIN SUPPORT; default to base-breakout-only.',flush=True)
    return winner,selected

def market_scan(args, out, winner):
    # The market never participates in rule selection. Process symbols independently to avoid memory blowups.
    from core import database
    factory=database()
    with factory() as s:
        symbols=[str(x[0]) for x in s.execute(text("SELECT DISTINCT symbol FROM market_stocks WHERE market='US' ORDER BY symbol"))]
    print('US UNIVERSE',len(symbols),'database coverage usually 2021+',flush=True)
    sql=text('SELECT day,h,l,c FROM us_tp_daily_bars WHERE symbol=:symbol ORDER BY day')
    detail=out/'market_breakouts_frozen.csv';detail.parent.mkdir(parents=True,exist_ok=True)
    cols=['symbol','date','signal_close','selected','future_complete_3y','future_hit_500','max_rise_3y_pct','max_drawdown_3y_pct','first_500pct_date','sessions_to_500pct','available_future_sessions']+FEATURES
    counts=defaultdict(lambda:defaultdict(int)); errors=0; signals=0; selected_n=0
    with detail.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        with factory() as sess:
            for ix,symbol in enumerate(symbols,1):
                try:
                    data=sess.execute(sql,{'symbol':symbol}).all()
                    if len(data)<args.base+2:continue
                    df=pd.DataFrame(data,columns=['day','High','Low','Close'])
                    df.index=pd.to_datetime(df.pop('day'),errors='coerce')
                    df['Open']=np.nan;df['Volume']=np.nan
                    df=clean(df)
                    for row in detect(df,symbol,args):
                        selected=int(rule_match(row,winner['clauses'])) if winner else 1
                        row['selected']=selected;w.writerow({k:row.get(k,'') for k in cols})
                        signals+=1;selected_n+=selected
                        year=row['date'][:4];
                        for typ in ['ALL','FILTERED'] if selected else ['ALL']:
                            counts[(year,typ)]['candidates']+=1
                            counts[('ALL_YEARS',typ)]['candidates']+=1
                            if row['future_complete_3y']:
                                counts[(year,typ)]['complete']+=1
                                counts[('ALL_YEARS',typ)]['complete']+=1
                                if row['future_hit_500']:
                                    counts[(year,typ)]['hits']+=1
                                    counts[('ALL_YEARS',typ)]['hits']+=1
                            elif row['future_hit_500']:
                                counts[(year,typ)]['early_hits_censored']+=1
                                counts[('ALL_YEARS',typ)]['early_hits_censored']+=1
                except Exception as exc:
                    errors+=1
                    if errors<=12:print('SCAN ERROR',symbol,str(exc)[:300],flush=True)
                if ix%250==0:
                    f.flush();print('MARKET PROGRESS',ix,'/',len(symbols),'ALL SIGNALS',signals,'FILTERED',selected_n,'ERRORS',errors,flush=True)
    metrics=[]
    for key,ct in sorted(counts.items()):
        year,typ=key;n=ct.get('complete',0);h=ct.get('hits',0)
        metrics.append({'year':year,'selection':typ,'candidates':ct['candidates'],
                        'complete_3y':n,'hits_500_complete':h,'hit_rate_complete_pct':round(h/n*100,3) if n else '',
                        'early_hits_in_incomplete_cases':ct.get('early_hits_censored',0),
                        'incomplete_3y':ct['candidates']-n})
    save_csv(out/'market_yearly_summary.csv',metrics)
    print('MARKET DONE', 'SIGNALS',signals,'SELECTED',selected_n,'ERRORS',errors,flush=True)
    for r in metrics:
        if r['year']=='ALL_YEARS':print('MARKET SUMMARY',r,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/tmp/acc500_discovery_market')
    p.add_argument('--base',type=int,default=60)
    p.add_argument('--max-width',type=float,default=55)
    p.add_argument('--breakout',type=float,default=.5)
    p.add_argument('--min-touches',type=int,default=2)
    p.add_argument('--touch-pct',type=float,default=7)
    p.add_argument('--cooldown',type=int,default=30)
    p.add_argument('--horizon',type=int,default=756)
    p.add_argument('--min-train-support',type=int,default=14)
    p.add_argument('--min-train-hits',type=int,default=3)
    p.add_argument('--top-singles',type=int,default=60)
    p.add_argument('--combo-singles',type=int,default=35)
    p.add_argument('--top-pairs',type=int,default=35)
    p.add_argument('--max-rules',type=int,default=150)
    p.add_argument('--examples-only',action='store_true')
    p.add_argument('--market-only',action='store_true',help='use FROZEN rule.json already generated by examples run')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.market_only:
        winner=json.loads((out/'frozen_rule.json').read_text(encoding='utf8'))
        print('LOADED FROZEN RULE',winner,flush=True)
    else:
        examples=[]
        for symbol,(start,end) in SAMPLES.items():
            try:
                print('DOWNLOAD',symbol,start,end,flush=True)
                df=read_yahoo(symbol,start,end)
                rows=detect(df,symbol,args);examples.extend(rows)
                print('EXAMPLE',symbol,'TOTAL',len(rows),'COMPLETE',sum(r['future_complete_3y'] for r in rows),
                      'HITS_500',sum(r['future_hit_500'] for r in rows),flush=True)
            except Exception as exc:
                print('DOWNLOAD ERROR',symbol,str(exc),flush=True)
                raise
        save_csv(out/'examples_all_features.csv',examples)
        winner,rules=discover(examples,args)
        save_csv(out/'candidate_rules_ranked.csv',[{**r,'clauses':json.dumps(r['clauses'])} for r in rules])
        (out/'frozen_rule.json').write_text(json.dumps(winner,indent=2,ensure_ascii=False),encoding='utf8')
        report=[]
        for symbol in SAMPLES:
            sample=[r for r in examples if r['symbol']==symbol and r['future_complete_3y']]
            for typ in ['BASELINE','FROZEN_RULE']:
                rs=[r for r in sample if typ=='BASELINE' or (winner and rule_match(r,winner['clauses']))]
                hits=sum(r['future_hit_500'] for r in rs)
                report.append({'symbol':symbol,'selection':typ,'complete':len(rs),'hits':hits,
                               'hit_pct':round(hits/len(rs)*100,3) if rs else ''})
        save_csv(out/'examples_frozen_comparison.csv',report)
        for r in report:print('EXAMPLES SUMMARY',r,flush=True)
    if not args.examples_only:market_scan(args,out,winner)
    print('DONE:',out,'READ ONLY, multiple signals from same moves, 3yr censoring, present-day universe survivorship bias; no live trading.',flush=True)
if __name__=='__main__':main()
