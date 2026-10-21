#!/usr/bin/env python3
"""Construct and audit an equal-weight Egyptian market index from stored OHLCV.
Read-only source: public.market_candles_1d. Writes ONLY to public.eg_internal_index_draft.
No feature / benchmark / price table is modified.
"""
import argparse
import os
from datetime import date
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

EXCLUDED_FROM_INDEX = {
 'AIFI.CA','AMES.CA','ARAB.CA','CCRS.CA','COSG.CA',
 'DCCC.CA','FERC.CA','GDWA.CA','HELI.CA','ICFC.CA',
 'ICLE.CA','MCRO.CA','MHOT.CA','MPCI.CA','NIPH.CA',
 'OCDI.CA','ORHD.CA','POUL.CA','UNIP.CA',
 'ADPC.CA','SPHT.CA','RKAZ.CA','ASPI.CA','IBCT.CA','COPR.CA','MFPC.CA'
}

DDL = '''
CREATE TABLE IF NOT EXISTS public.eg_internal_index_draft (
 session_date DATE PRIMARY KEY,
 index_level DOUBLE PRECISION NOT NULL,
 market_return_pct DOUBLE PRECISION,
 contributors INTEGER NOT NULL,
 eligible_symbols INTEGER NOT NULL,
 excluded_invalid INTEGER NOT NULL,
 zero_volume_symbols INTEGER NOT NULL,
 zero_return_symbols INTEGER NOT NULL,
 methodology TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
'''

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--start', default='2024-01-01',help='first source date for warmup and index inception')
 p.add_argument('--end', default=str(date.today()))
 p.add_argument('--base', type=float, default=1000.)
 p.add_argument('--coverage-min', type=float, default=0.80,help='minimum fraction of 210 universe with an available close')
 p.add_argument('--write',action='store_true',help='write only to separate draft table; default audit without writes')
 args=p.parse_args()
 if not (0 < args.coverage_min <= 1):p.error('--coverage-min must be in (0,1]')
 if args.base<=0:p.error('--base must be positive')
 eng=create_engine(os.environ['DATABASE_URL'])
 # The current inventory has ~210 CA instruments. Fix universe to latest date with broad coverage.
 with eng.connect() as cn:
  # Use the market screener's complete Egyptian universe, NOT the most recent
  # raw-candle date (which can contain only a handful of newly refreshed stocks).
  universe=pd.read_sql(text('''
    SELECT DISTINCT symbol FROM public.rs_market_features
    WHERE market='EG' AND session_date = (
      SELECT MAX(session_date) FROM public.rs_market_features
      WHERE market='EG' AND session_date BETWEEN CAST(:start AS DATE) AND CAST(:end AS DATE)
    ) ORDER BY symbol
  '''), cn, params={'start':args.start,'end':args.end})['symbol'].tolist()
  print('EG_SCREENING_UNIVERSE_SIZE',len(universe),flush=True)
  print('EXCLUDED_INDEX_ONLY',len(EXCLUDED_FROM_INDEX),sorted(EXCLUDED_FROM_INDEX),flush=True)
  universe=[x for x in universe if x not in EXCLUDED_FROM_INDEX]
  print('EGXALL_CONSTITUENTS',len(universe),flush=True)
  if len(universe)<150:raise SystemExit('Too few symbols in EG screener universe')
  df=pd.read_sql(text('''
    SELECT symbol, session_date, o, h, l, c, v
    FROM public.market_candles_1d
    WHERE symbol LIKE '%.CA' AND session_date BETWEEN :start AND :end
    ORDER BY session_date,symbol
  '''), cn, params={'start':args.start,'end':args.end})
 if df.empty: raise SystemExit('No source rows')
 df['session_date']=pd.to_datetime(df['session_date'],errors='coerce')
 for x in ['o','h','l','c','v']:df[x]=pd.to_numeric(df[x],errors='coerce')
 bad_date=df.session_date.isna().sum()
 bad_close=((~np.isfinite(df.c)) | (df.c<=0)).sum()
 dupe=df.duplicated(['symbol','session_date']).sum()
 invalid_ohlc=((df.h < df.l) | (df.o > df.h) | (df.o < df.l) | (df.c > df.h) | (df.c < df.l)).sum()
 print('SOURCE_ROWS',len(df),'UNIVERSE',len(universe),'BAD_DATES',bad_date,'BAD_CLOSES',bad_close,'DUPLICATE_SYMBOL_DATES',dupe,'INVALID_OHLC',invalid_ohlc,flush=True)
 if dupe: raise SystemExit('Duplicate symbol/date detected; no index built')
 # Recognize calendar-wide synthetic no-trading dates without altering source or deleting zero-volume securities.
 df=df[df.symbol.isin(universe)].copy()
 cal=df.groupby('session_date').agg(stocks=('symbol','nunique'), positive_volume=('v',lambda x:int((x>0).sum())),total_volume=('v','sum'))
 # Require actual trading across the market. One isolated traded share does not establish a market session:
 market_days=cal.index[(cal['positive_volume']>=max(5,int(np.ceil(len(universe)*0.05)))) & (cal['stocks']>=int(np.ceil(len(universe)*args.coverage_min)))]
 non_market=cal[(~cal.index.isin(market_days)) & (cal.stocks>=int(np.ceil(len(universe)*args.coverage_min)))]
 print('DATES_WITHOUT_BROAD_MARKET_TRADING',len(non_market),'EXAMPLES',[(str(i.date()),int(r.positive_volume)) for i,r in non_market.tail(20).iterrows()],flush=True)
 df=df[df.session_date.isin(market_days) & df.symbol.isin(universe)].copy()
 # Keep all individual zero-volume securities: use their unchanged closing prices. The 1st close is warmup only.
 wide=df.pivot(index='session_date',columns='symbol',values='c').sort_index()
 wide=wide.where((wide>0)&np.isfinite(wide))
 prev=wide.shift(1)
 ret=(wide/prev-1).where(prev.notna() & wide.notna())
 # Guard against unadjusted corporate actions / malformed data by FAILING audit, not silently clipping.
 extreme=(ret.abs()>0.60).sum().sum()
 if extreme:print('WARNING_ABSOLUTE_DAILY_RETURNS_GT_60_PERCENT',int(extreme),'REVIEW_CORPORATE_ACTIONS_BEFORE_PUBLISHING',flush=True)
 # For stable equal-weight total market return, only same-ticker consecutive market-session closes contribute.
 contributors=ret.notna().sum(axis=1)
 eligible=wide.notna().sum(axis=1)
 min_contributors=int(np.ceil(len(universe)*args.coverage_min))
 daily=ret.mean(axis=1,skipna=True)
 valid=(contributors>=min_contributors) & daily.notna()
 valid.iloc[0]=False
 if (~valid.iloc[1:]).any():
  bad=valid.iloc[1:][~valid.iloc[1:]]
  print('INSUFFICIENT_COVERAGE_DATES',[(str(x.date()),int(contributors[x])) for x in bad.index[:30]],flush=True)
  raise SystemExit('Insufficient coverage; abort index to avoid distorted chaining')
 if extreme:raise SystemExit('Extreme daily returns need review; draft not written')
 daily.iloc[0]=0.
 level=args.base*(1+daily).cumprod()
 out=pd.DataFrame({'session_date':wide.index.date,'index_level':level.values,'market_return_pct':daily.values*100,'contributors':contributors.values.astype(int),'eligible_symbols':eligible.values.astype(int),'excluded_invalid':(len(universe)-eligible).values.astype(int),'zero_volume_symbols':df.assign(z=df.v.fillna(0).eq(0)).groupby('session_date').z.sum().reindex(wide.index).values.astype(int),'zero_return_symbols':ret.fillna(999).eq(0).sum(axis=1).values.astype(int)})
 method=f'equal_weight_cross_sectional_mean_returns_excluding_{len(EXCLUDED_FROM_INDEX)}_extreme_move_symbols_v3'
 out['methodology']=method
 print('INDEX_START',out.iloc[0].to_dict(),'INDEX_END',out.iloc[-1].to_dict(),flush=True)
 print('MIN_CONTRIBUTORS',int(out.contributors.iloc[1:].min()),'MAX',int(out.contributors.max()),flush=True)
 print('MONTH_ENDS',flush=True)
 print(out.assign(month=pd.to_datetime(out.session_date).dt.to_period('M')).groupby('month').tail(1).tail(24)[['session_date','index_level','contributors']].to_string(index=False),flush=True)
 if args.write:
  with eng.begin() as cn:
   cn.execute(text(DDL))
   q=text('''INSERT INTO public.eg_internal_index_draft
    (session_date,index_level,market_return_pct,contributors,eligible_symbols,excluded_invalid,zero_volume_symbols,zero_return_symbols,methodology)
    VALUES(:session_date,:index_level,:market_return_pct,:contributors,:eligible_symbols,:excluded_invalid,:zero_volume_symbols,:zero_return_symbols,:methodology)
    ON CONFLICT(session_date) DO UPDATE SET
    index_level=EXCLUDED.index_level,market_return_pct=EXCLUDED.market_return_pct,
    contributors=EXCLUDED.contributors,eligible_symbols=EXCLUDED.eligible_symbols,
    excluded_invalid=EXCLUDED.excluded_invalid,zero_volume_symbols=EXCLUDED.zero_volume_symbols,
    zero_return_symbols=EXCLUDED.zero_return_symbols,methodology=EXCLUDED.methodology,created_at=now()''')
   for i in range(0,len(out),300):
    cn.execute(q,out.iloc[i:i+300].to_dict('records'))
  print('DRAFT_TABLE_UPDATED',len(out),flush=True)
 else:print('AUDIT_ONLY_NO_DATABASE_CHANGES',flush=True)
 eng.dispose()
if __name__=='__main__':main()
