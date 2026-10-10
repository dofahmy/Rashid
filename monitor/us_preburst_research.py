"""Retrospective pre-burst characterization ONLY for qualifying 21..250-bar episodes.

This is descriptive research, not a trained advance-warning signal.
Bar history may be too short for long lookbacks or 100..250 *days* forecasting.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam
from core import database

BARS='us_all_seven_hourly_bars'
WINDOWS=(10,20,40,60,100,150,250)

def fnum(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else np.nan
    except (ValueError,TypeError): return np.nan

def mean(a):
    a=np.asarray(a,dtype=float)
    a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan

def slope_r2(y):
    y=np.asarray(y,dtype=float)
    if len(y)<5 or not np.isfinite(y).all() or np.any(y<=0):return np.nan,np.nan
    y=np.log(y);x=np.arange(len(y),dtype=float)
    m,b=np.polyfit(x,y,1);pred=m*x+b
    sst=np.sum((y-y.mean())**2)
    r2=1-np.sum((y-pred)**2)/sst if sst>0 else np.nan
    return float(100*(np.exp(m)-1)),float(r2)

def calculate_window(df,idx,n):
    # Strictly BEFORE episode start; the row at start is not part of features.
    sub=df.iloc[max(0,idx-n):idx].copy()
    out={f'count_{n}':len(sub)}
    if len(sub)<n:
        return out
    c=sub.close.to_numpy(float); h=sub.high.to_numpy(float); l=sub.low.to_numpy(float);v=sub.volume.to_numpy(float)
    if not np.isfinite(c).all() or np.any(c<=0) or not np.isfinite(h).all() or not np.isfinite(l).all():return out
    lo=np.min(l);hi=np.max(h);first=c[0];last=c[-1]
    out.update({f'return_{n}':100*(last/first-1),f'range_{n}':100*(hi/lo-1) if lo>0 else np.nan,
      f'position_{n}':(last-lo)/(hi-lo) if hi>lo else np.nan,
      f'vol_cv_{n}':float(np.std(v)/np.mean(v)) if np.isfinite(v).all() and np.mean(v)>0 else np.nan,
      f'up_candles_{n}':float(np.mean(sub.close.to_numpy()>sub.open.to_numpy())),
      f'body_avg_{n}':mean(np.abs(sub.close-sub.open)/sub.open)*100,
      f'upper_wick_{n}':mean((sub.high-np.maximum(sub.open,sub.close))/(sub.high-sub.low).replace(0,np.nan)),
      f'lower_wick_{n}':mean((np.minimum(sub.open,sub.close)-sub.low)/(sub.high-sub.low).replace(0,np.nan))})
    m,r2=slope_r2(c);out[f'log_trend_pct_bar_{n}']=m;out[f'trend_r2_{n}']=r2
    # relative volume: last 5 bars vs previous bars; require enough valid observations.
    if n>=20 and np.isfinite(v).all() and np.mean(v[:-5])>0:
        out[f'volume_last5_relative_{n}']=float(v[-5:].mean()/v[:-5].mean())
    # Efficiency ratio and OBV trend
    distance=abs(last-first);path=np.abs(np.diff(c)).sum()
    out[f'kaufman_efficiency_{n}']=float(distance/path) if path>0 else np.nan
    return out

def features(df,idx):
    # Only candles strictly earlier than episode start are used.
    out={}
    for w in WINDOWS:out.update(calculate_window(df,idx,w))
    sub=df.iloc[:idx]
    if len(sub)<20:return out
    close=sub.close.to_numpy(float);high=sub.high.to_numpy(float);low=sub.low.to_numpy(float);vol=sub.volume.to_numpy(float)
    s=pd.Series(close);chg=s.diff();up=chg.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();down=(-chg.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=up.iloc[-1]/down.iloc[-1] if down.iloc[-1]>0 else np.inf
    out['rsi14']=float(100-100/(1+rs))
    e12=s.ewm(span=12,adjust=False).mean();e26=s.ewm(span=26,adjust=False).mean()
    macd=e12-e26;signal=macd.ewm(span=9,adjust=False).mean()
    out['macd_hist_pct']=float(100*(macd.iloc[-1]-signal.iloc[-1])/close[-1]) if close[-1]>0 else np.nan
    for span in [20,50,100,200]:
        if len(sub)>=span:out[f'close_vs_sma{span}_pct']=100*(close[-1]/close[-span:].mean()-1)
    if len(sub)>=22:
        tr=pd.DataFrame({'hl':high-low,'hc':abs(high-np.r_[close[0],close[:-1]]),'lc':abs(low-np.r_[close[0],close[:-1]])}).max(axis=1)
        out['atr14_pct']=float(tr.tail(14).mean()/close[-1]*100)
        ma=s.tail(20).mean();std=s.tail(20).std(ddof=0)
        out['bbwidth20_pct']=float(400*std/ma) if ma>0 else np.nan
        out['bbz20']=float((close[-1]-ma)/std) if std>0 else np.nan
    if len(sub)>=50:
        out['volume5_vs_50']=mean(vol[-5:])/mean(vol[-50:-5]) if mean(vol[-50:-5])>0 else np.nan
        # 20 vs prior 20 lows/highs; structural HH/HL comparison
        out['higher_low20']=float(np.min(low[-20:])>np.min(low[-40:-20]))
        out['higher_high20']=float(np.max(high[-20:])>np.max(high[-40:-20]))
        # Percent max drawdown of last 50 closes
        c50=close[-50:];out['max_drawdown50_pct']=float(100*np.min(c50/np.maximum.accumulate(c50)-1))
        out['volume_price_corr50']=float(np.corrcoef(np.diff(np.log(c50)),vol[-49:])[0,1]) if np.std(vol[-49:])>0 else np.nan
    # Momentum and OBV across recent 20 bars
    out['momentum5_pct']=float(100*(close[-1]/close[-6]-1)) if len(close)>=6 else np.nan
    out['momentum20_pct']=float(100*(close[-1]/close[-21]-1)) if len(close)>=21 else np.nan
    if len(close)>=60 and np.isfinite(vol[-60:]).all():
        obv=(np.sign(np.diff(close[-60:]))*vol[-59:]).cumsum()
        out['obv_slope60']=float((obv[-1]-obv[0])/np.sum(vol[-60:])) if np.sum(vol[-60:])>0 else np.nan
    # Squaring style: price near repeated past price, report gaps 1..60, not claiming a signal.
    if len(close)>=61:
        current=close[-1];tol=0.01
        gaps=[g for g in range(1,61) if abs(close[-1-g]/current-1)<=tol]
        out['repeat_close_gaps_1pct_60']=';'.join(map(str,gaps))
        out['repeat_count_1pct_60']=len(gaps)
        out['min_repeat_gap_1pct_60']=min(gaps) if gaps else np.nan
        for factor in range(2,20):
            # Descriptive factor multiples within 60 bars; NOT the platform's proprietary match.
            out[f'repeat_factor_{factor}_multiple_hits']=sum(g%factor==0 for g in gaps)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',default='/tmp/us-explosions-100pct/us_explosions_all_episodes.csv')
    p.add_argument('--output-dir',default='/tmp/us-preburst-research')
    p.add_argument('--sp500-csv',default='',help='Optional SPY or ^GSPC 1H CSV with timestamp + close columns')
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    episodes=pd.read_csv(a.episodes)
    episodes=episodes[(episodes.bars_to_peak.between(21,250)) & (episodes.rise_pct>100)].copy()
    # Protect against optimistic look-ahead: only pre-start features. Note selection is retrospective.
    episodes.start_utc=pd.to_datetime(episodes.start_utc,utc=True).dt.tz_localize(None)
    episodes.peak_utc=pd.to_datetime(episodes.peak_utc,utc=True).dt.tz_localize(None)
    symbols=set(episodes.symbol.astype(str));by_symbol=defaultdict(list)
    for row in episodes.to_dict('records'):by_symbol[row['symbol']].append(row)
    # User asked to analyze those 471 symbols only.
    market=None
    if a.sp500_csv:
        market=pd.read_csv(a.sp500_csv)
        dt=next((k for k in ('bar_time','date','datetime','timestamp','time') if k in market.columns),None)
        price=next((k for k in ('close','Close','adj_close','Adj Close') if k in market.columns),None)
        if not dt or not price:raise ValueError('S&P CSV needs time/date column and close')
        market=pd.DataFrame({'date':pd.to_datetime(market[dt],utc=True).dt.tz_localize(None),'sp_close':pd.to_numeric(market[price],errors='coerce')}).dropna().sort_values('date')
        market['sp_ret20_pct']=100*(market.sp_close/market.sp_close.shift(20)-1)
        market['sp_ret60_pct']=100*(market.sp_close/market.sp_close.shift(60)-1)
    results=[];count=0;failed=0
    def process(sym,buf):
        nonlocal count,failed
        if sym not in symbols:return
        count+=1
        try:
            df=pd.DataFrame(buf,columns=['date','open','high','low','close','volume'])
            df['date']=pd.to_datetime(df.date,utc=True).dt.tz_localize(None)
            for col in ('open','high','low','close','volume'):df[col]=pd.to_numeric(df[col],errors='coerce')
            df=df.sort_values('date').reset_index(drop=True)
            dates=df.date.to_numpy(dtype='datetime64[ns]')
            for e in by_symbol[sym]:
                i=int(np.searchsorted(dates,np.datetime64(e['start_utc']),side='left'))
                if i>=len(df) or df.date.iloc[i]!=e['start_utc']:continue
                feat=features(df,i)
                outrow={**e,**feat,'prehistory_available_bars':i}
                if market is not None:
                    matched=market[market.date<e['start_utc']]
                    if len(matched):
                        sp=matched.iloc[-1]
                        for n in (20,60):
                            outrow[f'sp500_return_{n}h_pct']=sp[f'sp_ret{n}_pct']
                            stock=feat.get(f'return_{n}')
                            outrow[f'relative_strength_vs_sp500_{n}h_pct']=stock-sp[f'sp_ret{n}_pct'] if pd.notna(stock) and pd.notna(sp[f'sp_ret{n}_pct']) else np.nan
                results.append(outrow)
        except Exception as exc:
            failed+=1;print('ERROR',sym,str(exc),flush=True)
        if count%50==0:print(f'Preburst analyzed {count}/{len(symbols)} symbols, rows={len(results)}',flush=True)
    with database()() as s:
        prev=None;buffer=[]
        stmt=text(f'SELECT symbol,bar_time,open,high,low,close,volume FROM {BARS} WHERE symbol IN :symbols ORDER BY symbol,bar_time').bindparams(bindparam('symbols',expanding=True))
        for r in s.execute(stmt, {'symbols': sorted(symbols)}):
            sym=str(r[0])
            if sym not in symbols:continue
            if prev is not None and sym!=prev:
                process(prev,buffer);buffer=[]
            prev=sym;buffer.append(r[1:])
        if prev is not None:process(prev,buffer)
    frame=pd.DataFrame(results)
    if frame.empty:raise RuntimeError('No matching episodes; check CSV matches stored bar timestamps')
    detail=out/'preburst_episodes_features.csv';frame.to_csv(detail,index=False,encoding='utf-8-sig')
    # Describe candidates rather than optimize on winners-only dataset.
    featcols=[c for c in frame if c not in episodes.columns and pd.api.types.is_numeric_dtype(frame[c])]
    summ=[]
    for col in featcols:
        v=pd.to_numeric(frame[col],errors='coerce').dropna()
        if len(v):summ.append({'feature':col,'observations':len(v),'coverage_pct':round(100*len(v)/len(frame),2),
               'median':round(float(v.median()),5),'q25':round(float(v.quantile(.25)),5),
               'q75':round(float(v.quantile(.75)),5),'mean':round(float(v.mean()),5)})
    summary=out/'preburst_feature_summary.csv';pd.DataFrame(summ).to_csv(summary,index=False,encoding='utf-8-sig')
    # Cross-section synchronization of burst starts by UTC market date and 5 trading days (calendar proxy only)
    ev=frame[['symbol','start_utc','peak_utc','rise_pct','bars_to_peak']].copy()
    ev['start_date']=pd.to_datetime(ev.start_utc).dt.date
    day=ev.groupby('start_date').agg(episodes=('symbol','size'),unique_stocks=('symbol','nunique'),mean_rise_pct=('rise_pct','mean')).reset_index()
    day.to_csv(out/'preburst_start_date_clusters.csv',index=False,encoding='utf-8-sig')
    # Group per symbol, do not pretend 1415 episodes are independent.
    per=frame.sort_values('rise_pct',ascending=False).drop_duplicates('symbol')
    per.to_csv(out/'preburst_best_episode_per_stock.csv',index=False,encoding='utf-8-sig')
    report={'symbols_in_source':len(symbols),'episodes_input':len(episodes),'episodes_with_features':len(frame),
        'unique_symbols_analyzed':int(frame.symbol.nunique()),'errors':failed,
        'median_prestart_bars':float(frame.prehistory_available_bars.median()),
        'episodes_with_100_prestart_bars':int((frame.prehistory_available_bars>=100).sum()),
        'sp500_comparison':bool(a.sp500_csv),'files':[str(x) for x in out.glob('*.csv')],
        'interpretation':'Descriptive features just before retrospective low; NOT proof of predictive model. Need broad historical data and out-of-sample controls to predict 100-250 trading days.'}
    (out/'study_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
