#!/usr/bin/env python3
"""Read-only historical audit of Explosion DNA picks. NOT execution or actionable fill estimates."""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam

PICKS=('train_top20.csv','validation_top20.csv','holdout_top20.csv')

def safe_ratio(a,b):
    return float(a/b-1)*100 if pd.notna(a) and pd.notna(b) and b>0 else np.nan

def first_touch(g, col, target):
    x=g[pd.to_numeric(g[col], errors='coerce')>=target]
    if x.empty:return None
    return str(x.iloc[0].session_date)

def analyze(row, prices):
    sym=str(row.symbol); month=str(row.month)
    s=prices.get(sym)
    res=dict(symbol=sym,month=month,period=row.period,model_score=float(row.score),positive_100=int(row.y),atlas_month_return_pct=float(row.month_return_pct),model_pick_cutoff=month+'-01',signal_definition='end of last available trading session prior to target month; hypothetical next session open')
    if s is None or s.empty:return dict(res,status='NO_PRICE_DATA')
    s=s.sort_values('session_date').reset_index(drop=True)
    before=s[s.session_date<month+'-01']; within=s[(s.session_date>=month+'-01')&(s.session_date<=(month+'-31' if month.endswith(('01','03','05','07','08','10','12')) else month+'-30'))] if False else s[s.session_date.str[:7]==month]
    if before.empty or within.empty:return dict(res,status='MISSING_PRE_OR_MONTH')
    prev=before.iloc[-1]; entry=within.iloc[0]; last=within.iloc[-1]
    entry_idx=within.index[0]; latest_prior=before.iloc[-20:]
    entry_price=float(entry.o); signal_price=float(prev.c)
    res.update(status='AUDIT',signal_date=str(prev.session_date),signal_close=signal_price,signal_volume=float(prev.v) if pd.notna(prev.v) else np.nan,
      pre_20day_median_volume=float(latest_prior.v.median()),pre_20day_median_dollar_volume=float((latest_prior.c*latest_prior.v).median()),
      execution_date=str(entry.session_date),hypothetical_entry_open=entry_price,entry_day_volume=float(entry.v) if pd.notna(entry.v) else np.nan,
      entry_gap_pct=safe_ratio(entry_price,signal_price),last_month_session=str(last.session_date),end_month_close=float(last.c),
      entry_to_month_end_pct=safe_ratio(last.c,entry_price),signal_close_to_month_end_pct=safe_ratio(last.c,signal_price),
      max_month_high=float(within.h.max()),date_max_high=str(within.loc[within.h.idxmax(),'session_date']),
      min_month_low=float(within.l.min()),date_min_low=str(within.loc[within.l.idxmin(),'session_date']),
      max_intramonth_upside_from_entry_pct=safe_ratio(within.h.max(),entry_price),
      max_intramonth_downside_from_entry_pct=safe_ratio(within.l.min(),entry_price),
      month_total_volume=float(within.v.sum()),month_bars=len(within),
      first_50pct_intraday_date=first_touch(within, 'h',entry_price*1.5),
      first_100pct_intraday_date=first_touch(within, 'h',entry_price*2),
      first_200pct_intraday_date=first_touch(within, 'h',entry_price*3),
      raw_to_atlas_return_diff_pp=safe_ratio(last.c,signal_price)-float(row.month_return_pct))
    # Across adjacent day boundaries, large price steps can reflect splits or errors.
    window=s[(s.session_date>=str(before.iloc[max(0,len(before)-20)].session_date)) & (s.session_date<=str(last.session_date))].copy()
    raw_jumps=pd.to_numeric(window.c,errors='coerce').pct_change().abs()
    res['daily_close_jump_gt_80pct_count']=int((raw_jumps>0.8).sum())
    adj=pd.to_numeric(window.adj_c,errors='coerce'); close=pd.to_numeric(window.c,errors='coerce')
    rat=(adj/close.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna()
    res['adj_close_ratio_varies']=bool(rat.max()-rat.min()>0.02) if not rat.empty else None
    res['needs_corporate_action_audit']=bool(res['daily_close_jump_gt_80pct_count'] or res['adj_close_ratio_varies'])
    return res

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--picks-dir',default='/data/explosion_dna_v2')
    p.add_argument('--out',default='/data/explosion_trade_audit')
    p.add_argument('--all-picks',action='store_true',help='Audit all 600 picks rather than 26 positive signals')
    p.add_argument('--database-url',default=os.getenv('DATABASE_URL'))
    a=p.parse_args()
    if not a.database_url:raise SystemExit('DATABASE_URL missing')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    picks=pd.concat([pd.read_csv(Path(a.picks_dir)/f) for f in PICKS],ignore_index=True)
    if not a.all_picks:picks=picks[picks.y==1].copy()
    syms=sorted(picks.symbol.unique().tolist())
    engine=create_engine(a.database_url)
    q=text('SELECT symbol,session_date,o,h,l,c,v,adj_c,source FROM market_candles_1d WHERE symbol IN :syms AND session_date BETWEEN :start AND :end ORDER BY symbol,session_date').bindparams(bindparam('syms',expanding=True))
    rows=[]
    with engine.connect() as con:
      for start_i in range(0,len(syms),120):
        batch=syms[start_i:start_i+120]
        fetched=pd.read_sql_query(q,con,params={'syms':batch,'start':'2024-01-01','end':'2026-09-30'})
        for c in ['o','h','l','c','v','adj_c']:
          fetched[c]=pd.to_numeric(fetched[c],errors='coerce')
        m={k:v for k,v in fetched.groupby('symbol')}
        for item in picks[picks.symbol.isin(batch)].itertuples(index=False):
          rows.append(analyze(item,m))
        print(f'PROGRESS {min(start_i+120,len(syms))}/{len(syms)} symbols',flush=True)
    report=pd.DataFrame(rows).sort_values(['month','symbol'])
    fn='all_600_trade_audit.csv' if a.all_picks else 'winning_26_trade_audit.csv'
    report.to_csv(out/fn,index=False)
    print('SAVED',out/fn,'ROWS',len(report),flush=True)
    print('\nAUDIT SUMMARY')
    cols=['symbol','month','signal_date','signal_close','signal_volume','execution_date','hypothetical_entry_open','entry_day_volume','end_month_close','entry_to_month_end_pct','max_intramonth_upside_from_entry_pct','max_intramonth_downside_from_entry_pct','needs_corporate_action_audit']
    print(report[[c for c in cols if c in report]].to_string(index=False),flush=True)
    print('\nNOTE: Signal date is inferred from month boundary. Model produced retrospective monthly rankings, not timestamped live alerts. Entry next open ignores slippage/spreads/halts; intramonth max high and low are NOT sequential realized outcomes. Volume for signal is PRIOR trading day; entry-day volume becomes known only after that session.',flush=True)

if __name__=='__main__':main()
