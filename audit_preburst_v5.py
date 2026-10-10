"""Read-only Pre-Burst V5 audit: snapshot freshness and default-rule transition checks."""
from sqlalchemy import text
from core import database

TABLE='us_preburst_full_history_v5'
COND='''rsi14 BETWEEN 27 AND 43 AND momentum5_pct BETWEEN -9 AND 0
AND momentum20_pct BETWEEN -22 AND -4 AND close_vs_sma20_pct BETWEEN -13 AND -3
AND return_20 BETWEEN -20 AND -4 AND position_20_pct BETWEEN 2 AND 24'''
SQL=f'''WITH latest AS (
SELECT symbol, reference_utc, rsi14,momentum5_pct,momentum20_pct,close_vs_sma20_pct,return_20,position_20_pct,
 ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY reference_utc DESC) AS n
FROM {TABLE} WHERE market='us'
), chosen AS (
 SELECT *, CASE WHEN {COND} THEN 1 ELSE 0 END AS matched FROM latest WHERE n=1
)
SELECT reference_utc, COUNT(*) AS symbols_available, SUM(matched) AS currently_matched
FROM chosen GROUP BY reference_utc ORDER BY reference_utc DESC LIMIT 18'''
SAMPLES=['AAOI','ABLV','MGRT','IOR','DCGO','ALAB']
with database()() as con:
    global_max=con.execute(text(f"SELECT MAX(reference_utc) FROM {TABLE} WHERE market='us'")).scalar()
    print('GLOBAL_LATEST_CANDLE_UTC:',global_max)
    print('\nLATEST CANDLE DISTRIBUTION (all stored symbols; matched with DEFAULTS):')
    for row in con.execute(text(SQL)):
        print(row[0], 'available=',row[1], 'match=',row[2])
    print('\nEXAMPLE STOCKS: latest 3 market candles')
    for sym in SAMPLES:
        print('\n',sym)
        rows=con.execute(text(f'''SELECT reference_utc, rsi14, momentum5_pct, momentum20_pct,
         close_vs_sma20_pct,return_20,position_20_pct,
         CASE WHEN {COND} THEN 1 ELSE 0 END match FROM {TABLE}
         WHERE market='us' AND symbol=:s ORDER BY reference_utc DESC LIMIT 3'''),{'s':sym}).mappings().all()
        for row in rows:
            print('  ',row['reference_utc'],'matches=',row['match'],
                  'RSI=',round(row['rsi14'],2) if row['rsi14'] is not None else None,
                  'M20=',round(row['momentum20_pct'],2) if row['momentum20_pct'] is not None else None)
        if rows:
            latest=rows[0]['match']==1
            prior=len(rows)>1 and rows[1]['match']==1
            freshness=rows[0]['reference_utc']==global_max
            print('  STATUS:',('NEW_AT_OWN_LAST_BAR' if latest and not prior and len(rows)>1 else
                               'ONGOING' if latest and prior else 'NOT_MATCHING_OR_NO_PREV'),
                  'FRESH_RELATIVE_TO_MARKET=',freshness)
print('\nNOTE: This audits snapshot freshness and transitions under DEFAULT filters only. It does not demonstrate future price gains.')
