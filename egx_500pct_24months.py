"""EGX all-stocks retrospective +500% within 24 calendar months.
Run: python egx_500pct_24months.py --output-dir /tmp/egx-500-24m
Requires DATABASE_URL, pandas, sqlalchemy and database PostgreSQL driver.
Entry = daily close; target = 6x entry on a *later* day's High; 24 calendar months inclusive.
"""
import os, argparse, json
from pathlib import Path
from collections import deque
import pandas as pd
from sqlalchemy import create_engine, inspect, text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='/tmp/egx-500-24m');ap.add_argument('--table',default='market_candles_1d');args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    url=os.environ.get('DATABASE_URL')
    if not url: raise RuntimeError('DATABASE_URL environment variable is not set')
    if url.startswith('postgres://'):url='postgresql://'+url[len('postgres://'):]
    engine=create_engine(url)
    insp=inspect(engine)
    cols=[v['name'] for v in insp.get_columns(args.table)]
    print('TABLE COLUMNS:',cols,flush=True)
    def choose(options):
        for c in options:
            if c in cols:return c
        raise RuntimeError('Missing column; candidates '+repr(options)+'; available '+repr(cols))
    sym=choose(['symbol','ticker','stock_symbol']); date=choose(['bar_period','date','trade_date','timestamp','time','period','bar_date']);close=choose(['close','close_price','c']);high=choose(['high','high_price','h'])
    q='SELECT "'+sym+'" AS symbol, "'+date+'" AS dt, "'+close+'" AS close, "'+high+'" AS high FROM "'+args.table+'" WHERE "'+close+'" > 0 AND "'+high+'" > 0 ORDER BY "'+sym+'", "'+date+'"'
    with engine.connect() as con: df=pd.read_sql_query(text(q),con)
    df['dt']=pd.to_datetime(df.dt,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    df['close']=pd.to_numeric(df.close,errors='coerce');df['high']=pd.to_numeric(df.high,errors='coerce')
    df=df.dropna(subset=['symbol','dt','close','high']);df=df[(df.close>0)&(df.high>0)]
    df=df.sort_values(['symbol','dt']).drop_duplicates(['symbol','dt'],keep='last')
    print('VALID BARS:',len(df),'SYMBOLS:',df.symbol.nunique(),flush=True)
    # O(n) monotonic deque to find maximum future High; once max >=6*entry, use binary search over monotone running max
    # First successful entry per wave, with wave cooldown ending at first target hit to count separate discoveries.
    all_hits=[]; summary=[]
    for symbol,g in df.groupby('symbol',sort=False):
        g=g.reset_index(drop=True); n=len(g)
        if n<2:continue
        dates=g.dt.tolist(); prices=g.close.to_numpy(); highs=g.high.to_numpy()
        # two-pointer max deque over next 24 calendar months, excluding entry day
        dq=deque();right=0;hits=[]
        for i in range(n-1):
            if right<i+1:right=i+1;dq.clear()
            end=dates[i]+pd.DateOffset(months=24)
            while right<n and dates[right]<=end:
                while dq and highs[dq[-1]]<=highs[right]:dq.pop()
                dq.append(right);right+=1
            while dq and dq[0]<=i:dq.popleft()
            if not dq or highs[dq[0]] < 6*prices[i]:continue
            # first hit: scan only until first hit; rare successes, no dependence on prices after 24 months
            first=next(j for j in range(i+1,right) if highs[j]>=6*prices[i])
            hits.append((i,first,highs[dq[0]]))
        # Distinct symbols and successful entry days; episode grouping by overlapping [entry, first hit] windows
        episodes=[]; current=None
        for i,j,maxhigh in hits:
            if current is None or dates[i]>current['latest_target_date']:
                current={'symbol':str(symbol),'first_entry_date':dates[i].date().isoformat(),'first_entry_close':round(float(prices[i]),6),'first_hit_date':dates[j].date().isoformat(),'first_hit_high':round(float(highs[j]),6),'latest_target_date':dates[j],'successful_entry_days':1,'max_gain_first_entry_pct':round((highs[j]/prices[i]-1)*100,2)}
                episodes.append(current)
            else:
                current['successful_entry_days']+=1
                if dates[j]>current['latest_target_date']:current['latest_target_date']=dates[j]
        for e in episodes:e['latest_target_date']=e['latest_target_date'].date().isoformat()
        all_hits.extend(episodes)
        summary.append({'symbol':str(symbol),'bars':n,'first_price_day':dates[0].date().isoformat(),'last_price_day':dates[-1].date().isoformat(),'successful_entry_days':len(hits),'episodes':len(episodes),'ever_500pct':bool(hits)})
    per=pd.DataFrame(summary);epi=pd.DataFrame(all_hits)
    per.to_csv(out/'all_symbols_500pct.csv',index=False,encoding='utf-8-sig')
    epi.to_csv(out/'500pct_episodes.csv',index=False,encoding='utf-8-sig')
    overview={'definition':'High of a subsequent session reaches >= 6x entry Close within 24 calendar months','first_data_day':df.dt.min().date().isoformat(),'last_data_day':df.dt.max().date().isoformat(),'symbols_scanned':int(per.shape[0]),'distinct_symbols_ever_hit_500pct':int(per.ever_500pct.sum()),'successful_entry_days':int(per.successful_entry_days.sum()),'grouped_episodes':int(per.episodes.sum()),'caution':'Retrospective; recent entries have incomplete 24-month followup; highs not executable prices, and corporate actions/splits/unadjusted price discontinuities can create false hits. Counts are descriptive, not verified investable gains.'}
    (out/'summary.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(overview,indent=2,ensure_ascii=False),flush=True)
    print('TOP SYMBOLS BY EPISODES:');print(per.sort_values(['episodes','successful_entry_days'],ascending=False).head(25).to_string(index=False))
    print('OUTPUT:',out,flush=True)
if __name__=='__main__':main()
