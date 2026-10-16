#!/usr/bin/env python3
"""Real, causal accumulation->breakout scanner; optional 3-year forward validation.
Data sources: Yahoo Finance (examples) / Rajih PostgreSQL (US universe).
No Time-Price. Read-only. Future data NEVER participates in signal selection.
"""
from __future__ import annotations
import argparse, csv, math, time
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

SAMPLES={
  'META':('2012-05-01','2026-10-16','2022-08-01','2023-09-01'),
  'TSLA':('2012-01-01','2026-10-16','2019-01-01','2020-09-01'),
  'MSFT':('1993-01-01','2026-10-16','1994-01-01','1998-12-31'),
}
COLS=['date','symbol','signal_close','breakout_level','base_start','base_end',
      'base_low','base_high','base_width_pct','base_days','resistance_touches',
      'support_touches','volume_ratio_20','breakout_pct','future_complete_3y',
      'max_high_3y','max_rise_3y_pct','first_500pct_date','sessions_to_500pct',
      'max_drawdown_3y_pct','date_last_available']

def normalize(df):
    df=df.copy()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    lower={str(c).lower().strip():c for c in df.columns}
    names={}
    for key,target in [('open','Open'),('high','High'),('low','Low'),('close','Close'),('volume','Volume')]:
        if key in lower: names[lower[key]]=target
    df=df.rename(columns=names)
    for required in ('High','Low','Close'):
        if required not in df: raise ValueError('Missing '+required)
    if 'Volume' not in df:df['Volume']=np.nan
    df=df[['High','Low','Close','Volume']]
    for c in df:df[c]=pd.to_numeric(df[c],errors='coerce')
    df.index=pd.to_datetime(df.index).tz_localize(None).normalize()
    df=df[~df.index.duplicated(keep='last')].sort_index()
    return df.dropna(subset=['High','Low','Close']).query('Close>0 and High>0 and Low>0')

def yahoo(symbol,start,end):
    try:import yfinance as yf
    except ImportError as e:raise SystemExit('Install dependency: pip install yfinance') from e
    # Yahoo returns split-adjusted OHLC (auto_adjust=False), not dividend-adjusted Adj Close.
    data=yf.download(symbol,start=start,end=end,auto_adjust=False,progress=False,threads=False)
    if data.empty:raise RuntimeError(f'No Yahoo candles for {symbol}. Download failed or rate-limited.')
    return normalize(data)

def detect(df,symbol,args):
    """Breakout requires same-day close ABOVE PREVIOUS base resistance, no future inputs.
    Only forward outcome columns use future days and are not part of the entry rule.
    """
    if len(df)<args.base+2:return []
    # Base is the N COMPLETED sessions BEFORE signal day; cannot include breakout candle.
    prevhi=df.High.shift(1).rolling(args.base,min_periods=args.base).max()
    prevlo=df.Low.shift(1).rolling(args.base,min_periods=args.base).min()
    width=(prevhi/prevlo-1)*100
    breakout_pct=(df.Close/prevhi-1)*100
    # Needs prior close at or under previous resistance: first closing breakout, not repeated entries.
    prior_close=df.Close.shift(1)
    mask=((breakout_pct>=args.breakout) & (width<=args.max_width) &
          (prior_close<=prevhi*(1+args.breakout/100)) &
          (df.Close>=prevlo) & (df.Close>0))
    if args.volume_ratio>0:
        vbase=df.Volume.shift(1).rolling(20,min_periods=20).median()
        mask &= (df.Volume/vbase >= args.volume_ratio)
    if args.first_date:mask &= (df.index >= pd.Timestamp(args.first_date))
    if args.last_date:mask &= (df.index <= pd.Timestamp(args.last_date))
    positions=np.flatnonzero(mask.fillna(False).to_numpy())
    rows=[]; last_selected=-10**9
    highs=df.High.to_numpy(); lows=df.Low.to_numpy(); closes=df.Close.to_numpy()
    volumes=df.Volume.to_numpy(); dates=df.index
    for i in positions:
        if i-last_selected<=args.cooldown:continue
        start=i-args.base
        h=highs[start:i]; l=lows[start:i]
        resistance=float(np.max(h)); support=float(np.min(l))
        # Distinct touch days (optional); can tune tolerance in CLI
        res_touch=int(np.sum(h >= resistance*(1-args.touch_pct/100)))
        sup_touch=int(np.sum(l <= support*(1+args.touch_pct/100)))
        if res_touch<args.min_touches or sup_touch<args.min_touches: continue
        # Test day is after breakout; 756 later trading sessions approximates 3 years.
        future_high=highs[i+1:i+1+args.horizon]
        future_low=lows[i+1:i+1+args.horizon]
        completed=len(future_high)>=args.horizon
        entry=float(closes[i]); goal=entry*6
        first_hit=np.flatnonzero(future_high>=goal)
        first_pos=(i+1+int(first_hit[0])) if len(first_hit) else None
        vr=np.nan
        if i>=20:
            med=np.nanmedian(volumes[i-20:i]);vr=(volumes[i]/med if med>0 else np.nan)
        rows.append(dict(date=str(dates[i].date()),symbol=symbol,signal_close=round(entry,6),
            breakout_level=round(resistance,6),base_start=str(dates[start].date()),base_end=str(dates[i-1].date()),
            base_low=round(support,6),base_high=round(resistance,6),
            base_width_pct=round(float(width.iloc[i]),3),base_days=args.base,
            resistance_touches=res_touch,support_touches=sup_touch,
            volume_ratio_20=round(float(vr),3) if np.isfinite(vr) else '',
            breakout_pct=round(float(breakout_pct.iloc[i]),3),future_complete_3y=int(completed),
            max_high_3y=round(float(np.max(future_high)),6) if len(future_high) else '',
            max_rise_3y_pct=round((float(np.max(future_high))/entry-1)*100,3) if len(future_high) else '',
            first_500pct_date=str(dates[first_pos].date()) if first_pos is not None else '',
            sessions_to_500pct=int(first_pos-i) if first_pos is not None else '',
            max_drawdown_3y_pct=round((float(np.min(future_low))/entry-1)*100,3) if len(future_low) else '',
            date_last_available=str(dates[-1].date())))
        last_selected=i
    return rows

def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=COLS);writer.writeheader();writer.writerows(rows)
    print('FILE:',path,flush=True)

def make_chart(df,rows,symbol,dest,span_start,span_end):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show=df.loc[span_start:span_end]
    if show.empty:return
    fig,ax=plt.subplots(figsize=(15,6))
    ax.plot(show.index,show.Close,linewidth=1.5,label='Actual split-adjusted close')
    displayed=[r for r in rows if span_start<=r['date']<=span_end]
    for k,r in enumerate(displayed):
        at=pd.Timestamp(r['date']);price=float(r['signal_close'])
        if k<18:
            ax.scatter([at],[price],marker='^',s=60,zorder=3)
            ax.annotate(at.strftime('%Y-%m-%d'),(at,price),xytext=(0,9),textcoords='offset points',fontsize=7,rotation=45)
        if r['first_500pct_date'] and k<18:
            hit=pd.Timestamp(r['first_500pct_date'])
            if hit in show.index:ax.scatter([hit],[show.loc[hit,'High']],marker='*',s=95,zorder=3)
    ax.set_title(symbol+' | real historical close | triangles = causal base breakout | stars = later +500%')
    ax.set_ylabel('Price (USD, split-adjusted)')
    ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(dest,dpi=160);plt.close(fig);print('CHART:',dest,flush=True)

def summary(rows,label,args):
    past=[r for r in rows if r['future_complete_3y']]
    hits=[r for r in rows if r['first_500pct_date']]
    print('SUMMARY:',label,'CANDIDATE BREAKOUTS:',len(rows),'COMPLETE 3 YEARS:',len(past),
          'REACHED +500% WITHIN 3 YEARS:',len(hits),flush=True)
    # A signal with only 1yr of future data cannot be labeled a failed 3yr signal.
    for r in hits[:15]:
        print('500% HIT:',r['symbol'],r['date'],'entry',r['signal_close'],'hit',r['first_500pct_date'],
              'sessions',r['sessions_to_500pct'],'base width',r['base_width_pct'],flush=True)

def examples(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for symbol,(start,end,show_start,show_end) in SAMPLES.items():
        print('DOWNLOAD:',symbol,start,end,flush=True)
        try:df=yahoo(symbol,start,end)
        except Exception as e:
            print('ERROR:',symbol,str(e),flush=True);continue
        rows=detect(df,symbol,args)
        write(out/(symbol+'_all_breakouts.csv'),rows)
        summary(rows,symbol,args)
        # Zoom into the historical study window with real prices, even if zero matching breakouts.
        make_chart(df,rows,symbol,out/(symbol+'_real_chart.png'),show_start,show_end)
        all_rows.extend(rows)
    write(out/'all_examples_breakouts.csv',all_rows)
    summary(all_rows,'THREE STOCKS TOTAL',args)

def dbscan(args):
    from core import database
    from sqlalchemy import text
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    session_factory=database()
    with session_factory() as s:
        symbols=[str(x[0]) for x in s.execute(text("SELECT DISTINCT symbol FROM market_stocks WHERE market='US' ORDER BY symbol"))]
    print('US SYMBOLS:',len(symbols),'READ ONLY. DB BARS START AROUND 2021.',flush=True)
    rows=[]; errors=0
    # This processes one symbol at a time; keeps peak memory low; may take several minutes.
    sql=text('SELECT day,h,l,c FROM us_tp_daily_bars WHERE symbol=:symbol ORDER BY day')
    with session_factory() as s:
        for i,symbol in enumerate(symbols,1):
            try:
                data=s.execute(sql,{'symbol':symbol}).all()
                if len(data)<args.base+2:continue
                df=pd.DataFrame(data,columns=['day','High','Low','Close'])
                df.index=pd.to_datetime(df.pop('day'),errors='coerce')
                df['Volume']=np.nan # cached table does not carry trading volume
                df=normalize(df)
                rows.extend(detect(df,symbol,args))
            except Exception as ex:
                errors+=1
                if errors<=10:print('ERROR:',symbol,str(ex),flush=True)
            if i%250==0:print('PROCESSED:',i,'/',len(symbols),'SIGNALS:',len(rows),'ERRORS:',errors,flush=True)
    write(out/'us_all_breakouts.csv',rows)
    summary(rows,'US CACHED UNIVERSE',args)
    print('SURVIVORSHIP BIAS: current US universe may exclude historically delisted stocks.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['examples','db'],default='examples')
    p.add_argument('--out',default='/tmp/rajih_accumulation500')
    p.add_argument('--base',type=int,default=60,help='base length (past completed trading sessions)')
    p.add_argument('--max-width',type=float,default=55,help='max range high/low width percent')
    p.add_argument('--breakout',type=float,default=0.5,help='close minimum percent above previous base resistance')
    p.add_argument('--min-touches',type=int,default=2)
    p.add_argument('--touch-pct',type=float,default=7)
    p.add_argument('--cooldown',type=int,default=30,help='min sessions between selected breakouts on same stock')
    p.add_argument('--horizon',type=int,default=756,help='approx three years in trading sessions')
    p.add_argument('--volume-ratio',type=float,default=0,help='optional min current/median20 volume; 0 disables; db mode has no volume')
    p.add_argument('--first-date',default=None)
    p.add_argument('--last-date',default=None)
    args=p.parse_args()
    if args.base<20 or args.horizon<1:p.error('base >=20, horizon >=1')
    if args.mode=='db' and args.volume_ratio>0:p.error('volume filtering unavailable in Rajih cached US bars')
    print('CAUSAL ENTRY: Close above previous',args.base,'day resistance by',args.breakout,'percent')
    print('BASE WIDTH <=',args.max_width,'%, min touches per edge:',args.min_touches,
          'touch tolerance',args.touch_pct,'% | cooldown',args.cooldown)
    print('TARGET TEST: at least 500% after entry within next',args.horizon,'trading sessions.')
    print('IMPORTANT: This is an exploratory screen, not fitted to ensure any particular outcome.')
    if args.mode=='examples':examples(args)
    else:dbscan(args)
if __name__=='__main__':main()
