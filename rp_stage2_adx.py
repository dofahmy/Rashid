#!/usr/bin/env python3
"""Second-stage winner envelopes on RS-matched stock/date rows. Research, not live advice."""
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import create_engine,text

BEST={'BGMS':252,'YDKG':10,'FROG':5,'MU':252,'IMNM':5,'ZIM':20,'IMSR':60}
FEATURES=['ADX14','PLUS_DI14','MINUS_DI14','RELATIVE_VOLUME20','RELATIVE_VOLATILITY14_60']

def indicators(df):
    h=df.h.to_numpy(float); l=df.l.to_numpy(float); c=df.c.to_numpy(float); v=df.v.to_numpy(float)
    n=len(c);prev=np.r_[np.nan,c[:-1]]
    tr=np.maximum.reduce([h-l,np.abs(h-prev),np.abs(l-prev)])
    up=np.r_[np.nan,np.diff(h)]; down=np.r_[np.nan,-np.diff(l)]
    plus=np.where((up>down)&(up>0),up,0.0);minus=np.where((down>up)&(down>0),down,0.0)
    plus[0]=minus[0]=np.nan
    def wilder(x,period=14):
        result=np.full(len(x),np.nan);series=pd.Series(x)
        # conventional seed: SMA over first period of valid observations
        valid=np.flatnonzero(np.isfinite(x))
        if len(valid)<period:return result
        start=int(valid[0]);seed=start+period-1
        if not np.all(np.isfinite(x[start:seed+1])):return result
        result[seed]=np.mean(x[start:seed+1])
        for i in range(seed+1,len(x)):
            if np.isfinite(x[i]):result[i]=(result[i-1]*(period-1)+x[i])/period
        return result
    atr=wilder(tr);pdm=wilder(plus);mdm=wilder(minus)
    pdi=np.divide(100*pdm,atr,out=np.full(n,np.nan),where=atr>0)
    mdi=np.divide(100*mdm,atr,out=np.full(n,np.nan),where=atr>0)
    den=pdi+mdi;dx=np.divide(100*np.abs(pdi-mdi),den,out=np.full(n,np.nan),where=den>0)
    adx=wilder(dx)
    avgvol=pd.Series(v).shift(1).rolling(20,min_periods=20).mean().to_numpy()
    rvol=np.divide(v,avgvol,out=np.full(n,np.nan),where=avgvol>0)
    # Relative volatility = current Wilder ATR14 % / preceding 60-session mean Wilder ATR14%.
    # This measures volatility RELATIVE TO THE TICKER'S OWN HISTORY, not to S&P500.
    atrpct=np.divide(atr*100,c,out=np.full(n,np.nan),where=c>0)
    baseline=pd.Series(atrpct).shift(1).rolling(60,min_periods=60).mean().to_numpy()
    relv=np.divide(atrpct,baseline,out=np.full(n,np.nan),where=baseline>0)
    return dict(zip(FEATURES,(adx,pdi,mdi,rvol,relv)))

def windows(inputdir,all_windows):
    if all_windows:
        return [(p.parent.name,int(p.stem.split('_RP')[1].split('_')[0]),p) for p in sorted(inputdir.glob('*/*_mirrors.csv'))]
    return [(sym,w,inputdir/sym/f'{sym}_RP{w}_mirrors.csv') for sym,w in BEST.items()]

def safe_num(x):
    try:
        f=float(x)
        return f if np.isfinite(f) else None
    except (ValueError,TypeError):return None

def rounded_range(vals,lowq,highq):
    return float(np.quantile(vals,lowq)),float(np.quantile(vals,highq))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--database-url',required=True)
    p.add_argument('--input',default='/data/seven_rp_mirrors_90')
    p.add_argument('--out',default='/data/rp_stage2_adx')
    p.add_argument('--all-windows',action='store_true',help='All 40 RP windows; default best one per reference')
    p.add_argument('--target',type=float,default=30)
    p.add_argument('--trim',type=float,default=0.05,help='Optional exploratory 5%-95% winner envelope')
    a=p.parse_args()
    if not 0<=a.trim<.5:raise SystemExit('trim must be [0,.5)')
    root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    specs=windows(root,a.all_windows)
    missing=[str(path) for _,_,path in specs if not path.exists()]
    if missing:raise SystemExit('Missing stage-1 files: '+str(missing[:5]))
    matchrows=[]
    for ref,w,path in specs:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                gain=safe_num(row.get('max_high_return_90_pct'))
                if gain is None:continue
                matchrows.append({'reference':ref,'rp_window':w,'symbol':row['symbol'].strip().upper(),
                                  'signal_date':row['signal_date'][:10], 'max_high_return_90_pct':gain,
                                  'hit30':gain>=a.target})
    if not matchrows:raise SystemExit('No first-stage signal rows')
    symbols=sorted(set(x['symbol'] for x in matchrows))
    bysymbol=defaultdict(list)
    for i,r in enumerate(matchrows):bysymbol[r['symbol']].append(i)
    engine=create_engine(a.database_url.replace('postgres://','postgresql://',1))
    missing_db=[]; missing_date=[]
    try:
        with engine.connect() as conn:
            for k,sym in enumerate(symbols,1):
                d=pd.read_sql(text('SELECT session_date AS date,o,h,l,c,v FROM market_candles_1d WHERE symbol=:s ORDER BY session_date'),conn,params={'s':sym})
                if d.empty:missing_db.append(sym);continue
                d['date']=pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
                for col in ('o','h','l','c','v'):d[col]=pd.to_numeric(d[col],errors='coerce')
                d=d.dropna(subset=['date','o','h','l','c','v'])
                d=d[(d.o>0)&(d.c>0)&(d.l>0)&(d.v>=0)&(d.h>=d[['o','c','l']].max(axis=1))&(d.l<=d[['o','c','h']].min(axis=1))]
                d=d.sort_values('date').drop_duplicates('date').reset_index(drop=True)
                f=indicators(d)
                date2ix={x:i for i,x in enumerate(d.date)}
                for rowidx in bysymbol[sym]:
                    row=matchrows[rowidx];ix=date2ix.get(row['signal_date'])
                    if ix is None or ix<1:missing_date.append((sym,row['signal_date']));continue
                    # indicators ONLY as of the completed PREVIOUS stock session
                    for feature,arr in f.items():row[feature]=safe_num(arr[ix-1])
                if k%50==0:print('PROCESSED',k,'/',len(symbols),flush=True)
    finally:engine.dispose()
    raw=pd.DataFrame(matchrows)
    raw.to_csv(out/'stage2_all_RP_cases_indicators.csv',index=False)
    detail=[];summary=[];envelopes=[]
    for (ref,w),g in raw.groupby(['reference','rp_window'],sort=False):
        winners=g[g.hit30];losers=g[~g.hit30]
        base={'reference':ref,'rp_window':int(w),'rs_cases':len(g),'rs_hit30':len(winners),'rs_below30':len(losers)}
        rules={}
        for name in FEATURES:
            pos=pd.to_numeric(winners[name],errors='coerce').dropna() if name in winners else pd.Series(dtype=float)
            if len(pos)==0:
                for approach in ('minmax','trimmed'):
                    envelopes.append({**base,'feature':name,'approach':approach,'lower':None,'upper':None,'winners_with_value':0})
                continue
            for approach,quantiles in [('minmax',(0,1)),('trimmed',(a.trim,1-a.trim))]:
                lower,upper=rounded_range(pos.to_numpy(),*quantiles)
                envelopes.append({**base,'feature':name,'approach':approach,'lower':lower,'upper':upper,'winners_with_value':len(pos)})
                rules[(approach,name)]=(lower,upper)
        for approach in ('minmax','trimmed'):
            for test_feature in FEATURES+['ALL_FIVE_AND']:
                subset=FEATURES if test_feature=='ALL_FIVE_AND' else [test_feature]
                available=all((approach,feature) in rules for feature in subset)
                if not available:
                    detail.append({**base,'approach':approach,'test':test_feature,'status':'no_winner_feature_data'})
                    continue
                mask=pd.Series(True,index=g.index)
                for feature in subset:
                    lo,hi=rules[(approach,feature)]
                    value=pd.to_numeric(g[feature],errors='coerce')
                    mask=mask & value.notna() & value.between(lo,hi,inclusive='both')
                kept=g[mask];win=int(kept.hit30.sum());fail=int((~kept.hit30).sum())
                detail.append({**base,'approach':approach,'test':test_feature,'status':'ok',
                   'kept_cases':len(kept),'kept_hit30':win,'kept_below30':fail,
                   'excluded_below30':len(losers)-fail,'lost_hit30':len(winners)-win,
                   'below30_exclusion_pct':round(100*(len(losers)-fail)/len(losers),2) if len(losers) else None,
                   'winner_retention_pct':round(100*win/len(winners),2) if len(winners) else None,
                   'hit30_after_pct':round(100*win/len(kept),2) if len(kept) else None})
                if test_feature=='ALL_FIVE_AND':
                    summary.append(detail[-1])
                if test_feature=='ALL_FIVE_AND':
                    ixkeys=set(kept.index)
                    # result indicator for each row, per independent strategy
                    raw.loc[g.index,f'{approach}_pass_all']=[int(i in ixkeys) for i in g.index]
    pd.DataFrame(envelopes).to_csv(out/'winner_indicator_ranges.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'filter_by_feature_and_joint.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary_seven_strategy_stage2.csv',index=False)
    raw.to_csv(out/'stage2_all_RP_cases_indicators.csv',index=False)
    report={'status':'complete','strategies':len(specs),'rs_case_rows':len(raw),'unique_symbols':len(symbols),
       'missing_db_symbols':missing_db,'missing_signal_dates':len(missing_date),
       'definitions':{'ADX14/DI14':'Standard Wilder 14-period smoothing of directional movement, ADX14 Wilder-smoothed DX',
         'Relative Volume20':'Previous session volume / mean of 20 sessions BEFORE that previous session',
         'Relative Volatility14_60':'Previous session Wilder ATR14/close *100, divided by mean of prior 60 such ATR% observations',
         'features_asof':'The completed session immediately BEFORE RS signal date',
         'winner':'Max future daily High over next 90 sessions >= entry close*(1+target/100)',
         'envelopes':'Min/max winners retains available positive-sample ranges; trimmed winner quantiles [trim,1-trim] exploratory'},
       'caveats':['Min-max winner envelope is in-sample and may overfit; does not guarantee retaining winners out of sample.',
         'Missing features are rejected in combined filter and count as lost winners.',
         'Signals may repeat same ticker on different dates.',
         'Corporate actions, extreme price errors, liquidity and drawdowns require separate audit.',
         'RS feature computed from previous session while entry is signal-day CLOSE: feature/entry timing must be audited before deployment.',
         'Relative Volatility here is normalized ATR, not comparison with S&P500 volatility.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('DONE',json.dumps({'status':'complete','RS_cases':len(raw),'symbols':len(symbols),'strategies':len(specs),'out':str(out)}),flush=True)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
