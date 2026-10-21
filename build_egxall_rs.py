#!/usr/bin/env python3
"""Build independent Egyptian RS features against the internally computed EGXALL index.

Source tables are read-only. Only public.rs_egxall_features is created/updated.
RS horizons are measured in matched stock/index sessions; no stale benchmark forward-fill.
No stock/volume rows are changed, and excluded index constituents remain eligible for RS.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

WINDOWS = (1, 5, 10, 20, 60, 120, 252)
AVGS = (5, 10, 20, 60, 120)
COLUMNS = ['symbol', 'session_date', 'close', 'benchmark_close',
           'stock_day_pct', 'benchmark_day_pct'] + [f'rp{w}' for w in WINDOWS] + [f'avg_rp{w}' for w in AVGS]
DDL = '''
CREATE TABLE IF NOT EXISTS public.rs_egxall_features (
 symbol TEXT NOT NULL,
 session_date DATE NOT NULL,
 close DOUBLE PRECISION,
 benchmark_close DOUBLE PRECISION,
 stock_day_pct DOUBLE PRECISION,
 benchmark_day_pct DOUBLE PRECISION,
 rp1 DOUBLE PRECISION,
 rp5 DOUBLE PRECISION,
 rp10 DOUBLE PRECISION,
 rp20 DOUBLE PRECISION,
 rp60 DOUBLE PRECISION,
 rp120 DOUBLE PRECISION,
 rp252 DOUBLE PRECISION,
 avg_rp5 DOUBLE PRECISION,
 avg_rp10 DOUBLE PRECISION,
 avg_rp20 DOUBLE PRECISION,
 avg_rp60 DOUBLE PRECISION,
 avg_rp120 DOUBLE PRECISION,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY (symbol, session_date)
)
'''

def num(x):
    return float(x) if np.isfinite(x) else None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2019-01-01')
    p.add_argument('--end', default='2026-10-20')
    p.add_argument('--write', action='store_true', help='Create/upsert a separate RS table; default is read-only audit')
    args = p.parse_args()
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        p.error('start must be <= end')
    url = os.getenv('DATABASE_URL')
    if not url:
        sys.exit('DATABASE_URL is not configured')
    eng = create_engine(url, pool_pre_ping=True)
    with eng.connect() as conn:
        index = pd.read_sql(text('''SELECT session_date, index_level, contributors, methodology
            FROM public.eg_internal_index_draft
            WHERE session_date <= :end ORDER BY session_date'''), conn, params={'end': args.end})
        syms = conn.execute(text('''SELECT DISTINCT symbol FROM public.market_candles_1d
            WHERE upper(symbol) LIKE '%.CA' ORDER BY symbol''')).scalars().all()
    if len(index) < 253:
        sys.exit('INDEX_HISTORY_TOO_SHORT: need at least 253 matching index sessions')
    if not index.index_level.gt(0).all() or index.session_date.duplicated().any():
        sys.exit('INDEX_INVALID: nonpositive level or duplicate dates')
    methods = index.methodology.dropna().unique().tolist()
    print('EGXALL_INDEX_ROWS', len(index), 'FIRST', str(index.session_date.iloc[0]),
          'LAST', str(index.session_date.iloc[-1]), 'METHODS', methods, flush=True)
    print('EG_STOCK_UNIVERSE', len(syms), 'MODE', 'WRITE' if args.write else 'AUDIT', flush=True)
    if len(methods) != 1:
        sys.exit('INDEX_MIXED_METHODOLOGIES: rebuild full index consistently first')
    if not methods[0].endswith('excluding_26_extreme_move_symbols_v3'):
        sys.exit('INDEX_VERSION_UNEXPECTED: expected v3 excluding 26 stocks')
    if args.write:
        with eng.begin() as conn:
            conn.execute(text(DDL))
    benchmark = index[['session_date', 'index_level']].rename(columns={'index_level': 'bc'}).copy()
    benchmark['session_date'] = pd.to_datetime(benchmark.session_date)
    total = matched_rows = written = no_match = 0
    for ticker in syms:
        with eng.connect() as conn:
            stock = pd.read_sql(text('''SELECT session_date, c FROM public.market_candles_1d
                WHERE symbol = :sym AND session_date <= :end
                ORDER BY session_date'''), conn, params={'sym': ticker, 'end': args.end})
        if stock.empty:
            continue
        stock['session_date'] = pd.to_datetime(stock.session_date, errors='coerce')
        stock['c'] = pd.to_numeric(stock.c, errors='coerce')
        stock = stock.dropna(subset=['session_date', 'c']).drop_duplicates('session_date', keep='last')
        stock = stock[stock.c > 0].sort_values('session_date')
        joined = stock.merge(benchmark, on='session_date', how='inner', validate='one_to_one')
        no_match += len(stock) - len(joined)
        if joined.empty:
            continue
        sc = joined.c.to_numpy(dtype=float)
        bc = joined.bc.to_numpy(dtype=float)
        n = len(joined)
        stock_daily = np.full(n, np.nan)
        benchmark_daily = np.full(n, np.nan)
        if n > 1:
            stock_daily[1:] = (sc[1:] / sc[:-1] - 1) * 100
            benchmark_daily[1:] = (bc[1:] / bc[:-1] - 1) * 100
        daily_rs = stock_daily - benchmark_daily
        rps = {}
        for w in WINDOWS:
            vals = np.full(n, np.nan)
            if w == 1:
                vals = daily_rs.copy()
            elif n > w:
                vals[w:] = 100 * (sc[w:] / sc[:-w] - bc[w:] / bc[:-w])
            rps[w] = vals
        avgs = {w: pd.Series(daily_rs).rolling(w, min_periods=w).mean().to_numpy() for w in AVGS}
        dates = joined.session_date.dt.date.to_numpy()
        indexes = np.flatnonzero((dates >= pd.Timestamp(args.start).date()) &
                                  (dates <= pd.Timestamp(args.end).date()))
        rows = []
        for i in indexes:
            values = [ticker, dates[i], num(sc[i]), num(bc[i]), num(stock_daily[i]), num(benchmark_daily[i])]
            values.extend(num(rps[w][i]) for w in WINDOWS)
            values.extend(num(avgs[w][i]) for w in AVGS)
            rows.append(dict(zip(COLUMNS, values)))
        total += 1
        matched_rows += len(rows)
        if args.write and rows:
            sql = text('''INSERT INTO public.rs_egxall_features ('''+','.join(COLUMNS)+''')
                VALUES ('''+','.join(':'+x for x in COLUMNS)+''')
                ON CONFLICT (symbol,session_date) DO UPDATE SET
                '''+','.join(f'{col}=EXCLUDED.{col}' for col in COLUMNS[2:])+''',updated_at=now()''')
            with eng.begin() as conn:
                for j in range(0, len(rows), 400):
                    conn.execute(sql, rows[j:j+400])
                written += len(rows)
        if total % 25 == 0:
            print('PROGRESS_SYMBOLS', total, 'ROWS', matched_rows, flush=True)
    print('DONE_SYMBOLS', total, 'MATCHED_ROWS', matched_rows,
          'UNMATCHED_STOCK_DATES_ALL_HISTORY', no_match, 'ROWS_WRITTEN', written, flush=True)
    print('NOTE: RS horizons are matched sessions; days with no EGXALL benchmark are not included in this separate RS table.')
    print('NOTE: zero-volume stock sessions remain included when the market index has a matching session.')
    eng.dispose()

if __name__ == '__main__':
    main()
