#!/usr/bin/env python3
"""Explore robust entry-indicator ranges from EGX +500% episodes, scan EGX and evaluate 24-month outcomes.
Usage: python egx_500_clean_range_scan.py --output-dir /tmp/egx500-clean-scan
Requires: pandas, numpy, sqlalchemy, psycopg2/psycopg, DATABASE_URL.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

F = ['rsi14','momentum5_pct','momentum20_pct','close_vs_sma20_pct',
     'close_vs_sma50_pct','bbz20','atr14_pct','return_20','position_20_pct',
     'volume5_vs_50','repeat_count_1pct_60']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',default='/tmp/egx-500-entry-indicators/500pct_episodes_with_entry_indicators.csv')
    ap.add_argument('--output-dir',default='/tmp/egx500-clean-scan')
    ap.add_argument('--lower-quantile',type=float,default=.10)
    ap.add_argument('--upper-quantile',type=float,default=.90)
    args=ap.parse_args()
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        ap.error('Invalid quantiles')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if not Path(args.episodes).exists():
        raise SystemExit('Missing episode indicators CSV. Recreate using egx_500pct_24months.py then egx_500_entry_indicators.py in the same Railway instance.')
    eps=pd.read_csv(args.episodes)
    eps['symbol']=eps.symbol.astype(str).str.upper().str.strip()
    eps=eps[eps.symbol.str.endswith('.CA')].copy()
    eps[F]=eps[F].apply(pd.to_numeric,errors='coerce')
    eps['first_entry_date']=pd.to_datetime(eps.first_entry_date,errors='coerce').dt.normalize()
    # Prespecified extreme-value exclusions, not fit to future observations.
    extreme=(eps.rsi14.lt(5)|eps.rsi14.gt(95)|eps.atr14_pct.gt(30)|
             eps.atr14_pct.le(0)|eps.momentum5_pct.lt(-60)|
             eps.momentum20_pct.lt(-60)|eps.return_20.lt(-60)|
             eps.close_vs_sma20_pct.lt(-60)|eps.close_vs_sma50_pct.lt(-60)|
             eps.volume5_vs_50.gt(30))
    incomplete=eps[F].isna().any(axis=1)
    eps['quality_status']=np.select([incomplete,extreme],['missing_features','extreme_values'],default='clean')
    eps.to_csv(out/'episode_quality_review.csv',index=False,encoding='utf-8-sig')
    clean=eps[eps.quality_status=='clean'].copy()
    if len(clean)<12:raise SystemExit(f'Too few clean episodes ({len(clean)})')
    # One representative episode per symbol reduces repeated-stock weighting.
    representatives=clean.sort_values(['symbol','first_entry_date']).drop_duplicates('symbol').copy()
    limits=pd.DataFrame({'low':representatives[F].quantile(args.lower_quantile),
                         'high':representatives[F].quantile(args.upper_quantile),
                         'median':representatives[F].median(),
                         'sample_size':len(representatives)})
    limits.index.name='indicator';limits.to_csv(out/'clean_indicator_ranges.csv',encoding='utf-8-sig')
    print('CLEAN ENTRY RANGES (one episode per stock):',flush=True)
    print(limits.round(5).to_string(),flush=True)
    url=os.environ['DATABASE_URL'];url='postgresql://'+url[len('postgres://'):] if url.startswith('postgres://') else url
    engine=create_engine(url,pool_pre_ping=True)
    fields=', '.join(F)
    with engine.connect() as c:
        indicators=pd.read_sql_query(text(f"SELECT symbol, reference_utc, {fields} FROM us_preburst_full_history_v5 WHERE lower(market)='egypt' AND upper(symbol) LIKE '%.CA' ORDER BY symbol, reference_utc"),c)
        prices=pd.read_sql_query(text("SELECT symbol,session_date,c,h FROM market_candles_1d WHERE upper(symbol) LIKE '%.CA' AND c>0 AND h>0 ORDER BY symbol,session_date"),c)
    indicators['symbol']=indicators.symbol.astype(str).str.upper().str.strip()
    indicators['dt']=pd.to_datetime(indicators.reference_utc,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    indicators=indicators.dropna(subset=['dt']).sort_values(['symbol','dt']).drop_duplicates(['symbol','dt'],keep='last')
    indicators[F]=indicators[F].apply(pd.to_numeric,errors='coerce')
    prices['symbol']=prices.symbol.astype(str).str.upper().str.strip()
    prices['dt']=pd.to_datetime(prices.session_date,errors='coerce').dt.normalize()
    prices['c']=pd.to_numeric(prices.c,errors='coerce'); prices['h']=pd.to_numeric(prices.h,errors='coerce')
    prices=prices.dropna(subset=['dt','c','h']).sort_values(['symbol','dt']).drop_duplicates(['symbol','dt'],keep='last')
    price_groups={s:g.reset_index(drop=True) for s,g in prices.groupby('symbol',sort=False)}
    print(f'Historical indicator rows: {len(indicators)}; EGX price symbols: {len(price_groups)}',flush=True)
    valid=indicators[F].notna().all(axis=1)
    checks=indicators[F].ge(limits['low']).mul(indicators[F].le(limits['high'])).sum(axis=1)
    indicators['matched_features']=checks.where(valid,-1)
    summary={'version':'EGX-500-clean-ranges-v1','target':'+500% (High >= 6x entry Close) within 24 calendar months',
             'source_episode_rows':len(eps),'source_episode_stocks':eps.symbol.nunique(),
             'excluded_missing':int(incomplete.sum()),'excluded_extreme':int((~incomplete & extreme).sum()),
             'clean_episodes':len(clean),'clean_unique_stocks':len(representatives),
             'quantile_low':args.lower_quantile,'quantile_high':args.upper_quantile,
             'indicator_rows':len(indicators),'market_price_stocks':len(price_groups), 'strategies':[]}
    for required in [11,9,7]:
        # A valid daily signal can only be produced when all 11 inputs are available.
        candidate=indicators.loc[indicators.matched_features>=required,['symbol','dt','matched_features']].copy()
        candidate.to_csv(out/f'all_matching_days_{required}of11.csv',index=False,encoding='utf-8-sig')
        cycles=[];unpriced=0
        for symbol,g in candidate.groupby('symbol',sort=False):
            pg=price_groups.get(symbol)
            if pg is None:unpriced+=len(g);continue
            dates=pg.dt.to_numpy(dtype='datetime64[ns]'); highs=pg.h.to_numpy(dtype=float); closes=pg.c.to_numpy(dtype=float)
            d_to_i={d:i for i,d in enumerate(dates)}
            next_allowed=pd.Timestamp.min
            for row in g.itertuples(index=False):
                dt=pd.Timestamp(row.dt)
                if dt < next_allowed:continue
                i=d_to_i.get(np.datetime64(dt,'ns'))
                if i is None:unpriced+=1;continue
                entry=float(closes[i]);end=dt+pd.DateOffset(months=24)
                stop=int(np.searchsorted(dates,np.datetime64(end,'ns'),side='right'))
                future=highs[i+1:stop]
                target=6*entry
                reached=bool(np.any(future>=target))
                hit_idx=(i+1+int(np.argmax(future>=target))) if reached else None
                coverage=bool(pd.Timestamp(dates[-1])>=end)
                label='winner' if reached else ('failure' if coverage else 'pending')
                cycles.append({'symbol':symbol,'entry_date':dt.date().isoformat(),
                               'entry_close':entry,'matched_features':int(row.matched_features),
                               'first_hit_date':pd.Timestamp(dates[hit_idx]).date().isoformat() if reached else '',
                               'first_hit_high':float(highs[hit_idx]) if reached else np.nan,
                               'max_gain_pct':round((float(future.max())/entry-1)*100,3) if len(future) else np.nan,
                               'outcome':label,'complete_24_month_calendar_window':coverage})
                next_allowed=end+pd.Timedelta(days=1)
        frame=pd.DataFrame(cycles,columns=['symbol','entry_date','entry_close','matched_features','first_hit_date','first_hit_high','max_gain_pct','outcome','complete_24_month_calendar_window'])
        frame.to_csv(out/f'cycles_{required}of11.csv',index=False,encoding='utf-8-sig')
        periods={'2019_2022':frame.loc[frame.entry_date.lt('2023-01-01')],
                 '2023_2026':frame.loc[frame.entry_date.ge('2023-01-01')],
                 'whole_period':frame}
        report={'required_matches':required,'matching_daily_rows':len(candidate),'unpriced_matching_days':unpriced,'periods':{}}
        for name,x in periods.items():
            nw=int((x.outcome=='winner').sum());nf=int((x.outcome=='failure').sum());npd=int((x.outcome=='pending').sum())
            report['periods'][name]={'cycles':len(x),'stocks':x.symbol.nunique(),'winners':nw,'failures':nf,'pending':npd,
                                      'decided_win_rate_pct':round(nw/(nw+nf)*100,2) if nw+nf else None}
        summary['strategies'].append(report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    summary['limitations']=[
        'Ranges fit retrospectively to winning episodes and are exploratory only.',
        'Episodes first_entry_date is selected retrospectively; may favor unusual price conditions.',
        'Cooldown is 24 calendar months from first matching entry per symbol; different from grouping historical winners.',
        'Pending includes incomplete 24-month horizons unless target already hit.',
        'EGX corporate actions and unadjusted prices may cause fictitious +500% hits.',
        'Period groupings use ENTRY dates, not dates that reached target.',
        '2019-2022 is a prior-period stress test, not an untouched forward validation.' ]
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    print('SUMMARY:',json.dumps(summary,ensure_ascii=False,indent=2,default=int),flush=True)
    print('OUTPUT:',out,flush=True)

if __name__=='__main__':main()
