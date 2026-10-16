#!/usr/bin/env python3
"""
Find stocks that look like the 14-stock basket at each month-end, and audit
whether those flagged stocks explode in the FOLLOWING month.

Every feature uses only data available at the close of the last trading day
before the month starts (point-in-time). The month's own prices are used only
to score the outcome.

Fingerprint (thresholds are calibrated each month from the basket itself):
  1. Turnover      : 20-day median dollar volume / market cap   >= basket q-quantile
  2. Basket beta   : 60-day beta to the basket index (ex-self)   >= basket q-quantile
  3. Volatility    : 60-day annualised volatility                >= basket q-quantile
  4. Market cap    : between --cap-min and --cap-max
  5. Listing year  : >= --min-listing-year
Optional trigger : 5-day avg dollar volume / 60-day median dollar volume >= --surge

Outputs (in --out):
  features_all.csv     every eligible stock-month with features, flags, outcome
  flagged.csv          non-basket stock-months that pass the full fingerprint
  monthly.csv          per-month base rate vs flagged hit rate
  live_candidates.csv  stocks that match the fingerprint at the LAST date in the DB
  summary.json         base rate, hit rate, lift for each criterion and combination
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASKET = ['RGTI','QBTS','IONQ','SOUN','BBAI','RKLB','ASTS','LUNR','ACHR',
          'OKLO','SMR','IREN','CIFR','HIMS']
SPLIT_K = [2,3,4,5,8,10,15,20,25,30,40,50,100]


def load_candles(engine, start, end, cols):
    sql = text(f'''SELECT symbol, session_date AS date, {cols['h']} AS high,
                   {cols['c']} AS close, {cols['v']} AS volume
                   FROM market_candles_1d
                   WHERE session_date BETWEEN :start AND :end''')
    parts = []
    with engine.connect() as c:
        for ch in pd.read_sql(sql, c, params={'start': start, 'end': end}, chunksize=500_000):
            parts.append(ch)
        first = pd.read_sql(text('SELECT symbol, MIN(session_date) AS first_date '
                                 'FROM market_candles_1d GROUP BY symbol'), c)
    d = pd.concat(parts, ignore_index=True)
    d = d[~d.symbol.astype(str).str.endswith('.CA')]
    d['date'] = pd.to_datetime(d.date)
    for k in ['high', 'close', 'volume']:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date']).sort_values(['symbol', 'date'])
    first['first_date'] = pd.to_datetime(first.first_date)
    return d, first.set_index('symbol').first_date


def detect_splits(close, vol):
    """Flag split days: price ratio ~ k or 1/k AND share volume moves the opposite way.
    A real spike (price up, volume up) is NOT flagged; a reverse split (price up x k,
    share volume down) is."""
    ratio = close / close.shift(1)
    pre = vol.rolling(10, min_periods=5).median().shift(1)
    post = vol[::-1].rolling(10, min_periods=5).median()[::-1]
    vratio = post / pre
    flag = pd.DataFrame(False, index=close.index, columns=close.columns)
    for k in SPLIT_K:
        rev = (ratio / k - 1).abs().le(0.04) & vratio.le(3.0 / k)       # reverse split
        fwd = (ratio * k - 1).abs().le(0.04) & vratio.ge(k / 3.0)       # forward split
        flag |= (rev | fwd).fillna(False)
    return flag


def load_fundamentals(path):
    f = pd.read_csv(path)
    if not {'symbol', 'asof_date'}.issubset(f.columns) or not (
            'market_cap_usd' in f.columns or 'shares_outstanding' in f.columns):
        raise SystemExit('fundamentals CSV needs: symbol, asof_date, and market_cap_usd or '
                         'shares_outstanding (listing_date optional)')
    f['asof_date'] = pd.to_datetime(f.asof_date, errors='coerce')
    if 'listing_date' in f.columns:
        f['listing_date'] = pd.to_datetime(f.listing_date, errors='coerce')
    for k in ['market_cap_usd', 'shares_outstanding']:
        if k in f.columns:
            f[k] = pd.to_numeric(f[k], errors='coerce')
    return f.dropna(subset=['symbol', 'asof_date']).sort_values('asof_date')


def asof_join(rows, fund):
    """Backward as-of join: only fundamentals published on/before the as-of date."""
    keep = [c for c in ['market_cap_usd', 'shares_outstanding', 'listing_date'] if c in fund.columns]
    r = pd.merge_asof(rows.sort_values('asof'), fund[['symbol', 'asof_date'] + keep],
                      left_on='asof', right_on='asof_date', by='symbol', direction='backward')
    if 'shares_outstanding' in r.columns:
        cap_from_shares = r.shares_outstanding * r.price
        r['mcap'] = cap_from_shares.where(cap_from_shares.notna(), r.get('market_cap_usd'))
    else:
        r['mcap'] = r.market_cap_usd
    return r


def month_end_features(W, asof_dates, basket_in_db, a):
    close, high, vol, ret, dv, split = (W[k] for k in ['close', 'high', 'vol', 'ret', 'dv', 'split'])
    # Basket index excluding self: (sum of basket returns - own return) / (n - own)
    b_ret = ret[basket_in_db]
    b_sum = b_ret.sum(axis=1, min_count=1)
    b_cnt = b_ret.notna().sum(axis=1)
    rows = []
    dates = close.index
    for month, asof in asof_dates:
        i = dates.get_loc(asof)
        if i < 60:
            continue
        w60 = slice(i - 59, i + 1)
        w20 = slice(i - 19, i + 1)
        r60 = ret.iloc[w60]
        own_in = r60.columns.isin(basket_in_db)
        idx_sum = b_sum.iloc[w60].values[:, None] - np.where(own_in, r60.fillna(0).values, 0)
        idx_cnt = b_cnt.iloc[w60].values[:, None] - np.where(own_in, r60.notna().values, 0)
        bidx = pd.DataFrame(idx_sum / np.where(idx_cnt > 0, idx_cnt, np.nan),
                            index=r60.index, columns=r60.columns)
        valid = r60.notna() & bidx.notna()
        x = bidx.where(valid); y = r60.where(valid)
        n = valid.sum()
        cov = ((x - x.mean()) * (y - y.mean())).sum() / (n - 1)
        beta = cov / x.var()
        vol60 = y.std() * np.sqrt(252)
        med_dv20 = dv.iloc[w20].median()
        surge = dv.iloc[i - 4:i + 1].mean() / dv.iloc[w60].median()
        # Split anywhere in lookback (120 sessions) or in the outcome month -> exclude
        m_start = pd.Timestamp(month + '-01')
        m_end = m_start + pd.offsets.MonthEnd(0)
        in_month = (dates > asof) & (dates <= m_end)
        look = slice(max(0, i - 119), i + 1)
        split_any = split.iloc[look].any() | split.loc[in_month].any()
        p0 = close.iloc[i]
        out_max = (high.loc[in_month].max() / p0 - 1) * 100
        out_close = (close.loc[in_month].ffill().iloc[-1] / p0 - 1) * 100 if in_month.any() else np.nan
        f = pd.DataFrame({'month': month, 'asof': asof, 'price': p0, 'med_dv20': med_dv20,
                          'beta_basket': beta, 'vol60': vol60, 'surge': surge,
                          'n60': n, 'split_window': split_any,
                          'month_max_pct': out_max, 'month_close_pct': out_close})
        f.index.name = 'symbol'
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


def lift_table(df, cols, target_col):
    base = df[target_col].mean()
    out = {'eligible': int(len(df)), 'base_rate': round(base, 4)}
    for name, mask in cols.items():
        s = df[mask]
        rate = s[target_col].mean() if len(s) else None
        out[name] = {'n': int(len(s)), 'hit_rate': None if rate is None else round(rate, 4),
                     'lift': None if (rate is None or base == 0) else round(rate / base, 2),
                     'median_month_max_pct': None if s.empty else round(s.month_max_pct.median(), 1),
                     'median_month_close_pct': None if s.empty else round(s.month_close_pct.median(), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', required=True,
                    help='CSV: symbol, asof_date, market_cap_usd OR shares_outstanding, [listing_date]. '
                         'Must be point-in-time (value as known on asof_date).')
    ap.add_argument('--out', default='/data/fingerprint_audit')
    ap.add_argument('--start-month', default='2024-04')
    ap.add_argument('--end-month', default='2026-09')
    ap.add_argument('--target', type=float, default=50, help='explosion = month max >= target %% above prior close')
    ap.add_argument('--min-liquidity', type=float, default=4_500_000)
    ap.add_argument('--min-price', type=float, default=2.0)
    ap.add_argument('--cap-min', type=float, default=1e9)
    ap.add_argument('--cap-max', type=float, default=40e9)
    ap.add_argument('--min-listing-year', type=int, default=2019)
    ap.add_argument('--quantile', type=float, default=0.10,
                    help='threshold = this quantile of basket members each month (lower = looser)')
    ap.add_argument('--surge', type=float, default=2.0)
    ap.add_argument('--splits-csv', help='optional CSV symbol,date of known splits (authoritative)')
    ap.add_argument('--high-col', default='h'); ap.add_argument('--close-col', default='c')
    ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    start = (pd.Timestamp(a.start_month + '-01') - pd.Timedelta(days=220)).strftime('%Y-%m-%d')
    engine = create_engine(os.environ['DATABASE_URL'])
    d, first_date = load_candles(engine, start, '2100-01-01',
                                 {'h': a.high_col, 'c': a.close_col, 'v': a.volume_col})
    print('candles', len(d), 'symbols', d.symbol.nunique(), 'last date', d.date.max().date(), flush=True)

    close = d.pivot(index='date', columns='symbol', values='close')
    high = d.pivot(index='date', columns='symbol', values='high').reindex_like(close)
    vol = d.pivot(index='date', columns='symbol', values='volume').reindex_like(close)
    split = detect_splits(close, vol)
    if a.splits_csv:
        s = pd.read_csv(a.splits_csv); s['date'] = pd.to_datetime(s.date)
        for sym, dt in zip(s.symbol, s.date):
            if sym in split.columns:
                j = split.index.searchsorted(dt)
                if j < len(split.index):
                    split.iloc[j, split.columns.get_loc(sym)] = True
    ret = close.pct_change(fill_method=None).mask(split)        # no fake returns on split days
    dv = close * vol
    W = dict(close=close, high=high, vol=vol, ret=ret, dv=dv, split=split)

    basket_in_db = [b for b in BASKET if b in close.columns]
    missing = sorted(set(BASKET) - set(basket_in_db))
    if len(basket_in_db) < 8:
        raise SystemExit(f'only {len(basket_in_db)} basket symbols in DB (missing {missing})')

    # As-of = last trading day strictly before each month; plus a LIVE row at the last date.
    months = pd.period_range(a.start_month, a.end_month, freq='M').astype(str)
    last = close.index.max()
    live_month = (last + pd.offsets.MonthBegin(1)).strftime('%Y-%m')
    asof_dates = []
    for mth in months:
        prior = close.index[close.index < pd.Timestamp(mth + '-01')]
        if len(prior) and pd.Timestamp(mth + '-01') <= close.index.max() and mth != live_month:
            asof_dates.append((mth, prior[-1]))
    asof_dates.append((live_month, last))
    feats = month_end_features(W, asof_dates, basket_in_db, a)
    feats['live'] = feats['asof'].eq(last) & feats.month.eq(live_month)

    fund = load_fundamentals(a.fundamentals)
    feats = asof_join(feats, fund)
    feats['turnover_pct'] = feats.med_dv20 / feats.mcap * 100
    ld = feats['listing_date'] if 'listing_date' in feats.columns else pd.Series(pd.NaT, index=feats.index)
    listing = ld.fillna(feats.symbol.map(first_date))
    feats['listing_year'] = pd.to_datetime(listing).dt.year
    feats['listing_known'] = ld.notna()
    feats['in_basket'] = feats.symbol.isin(basket_in_db)

    # Monthly calibration from basket members (point-in-time, same as-of date)
    thr = (feats[feats.in_basket].groupby('asof')[['turnover_pct', 'beta_basket', 'vol60']]
           .quantile(a.quantile).add_prefix('thr_'))
    feats = feats.merge(thr, left_on='asof', right_index=True, how='left')

    feats['eligible'] = (feats.price.ge(a.min_price) & feats.med_dv20.ge(a.min_liquidity)
                         & ~feats.split_window & feats.n60.ge(40))
    feats['c_turnover'] = feats.turnover_pct.ge(feats.thr_turnover_pct)
    feats['c_beta'] = feats.beta_basket.ge(feats.thr_beta_basket)
    feats['c_vol'] = feats.vol60.ge(feats.thr_vol60)
    feats['c_cap'] = feats.mcap.between(a.cap_min, a.cap_max)
    feats['c_listing'] = feats.listing_year.ge(a.min_listing_year)
    feats['c_surge'] = feats.surge.ge(a.surge)
    crit = ['c_turnover', 'c_beta', 'c_vol', 'c_cap', 'c_listing']
    feats['n_criteria'] = feats[crit].sum(axis=1)
    feats['full_match'] = feats[crit].all(axis=1)
    feats['exploded'] = feats.month_max_pct.ge(a.target)

    feats.to_csv(out / 'features_all.csv', index=False)
    live = feats[feats.live & feats.eligible & feats.full_match].sort_values('turnover_pct', ascending=False)
    live.to_csv(out / 'live_candidates.csv', index=False)

    hist = feats[~feats.live & feats.eligible & feats.month_max_pct.notna()]
    test = hist[~hist.in_basket]          # basket was picked with hindsight -> exclude from scoring
    test[test.full_match].sort_values(['month', 'month_max_pct'], ascending=[True, False]) \
        .to_csv(out / 'flagged.csv', index=False)

    masks = {c: test[c] for c in crit + ['c_surge']}
    masks['full_match'] = test.full_match
    masks['full_match_and_surge'] = test.full_match & test.c_surge
    masks['4_of_5'] = test.n_criteria.ge(4)
    for name in list(masks):
        masks[name] = masks[name].fillna(False).astype(bool)
    summary = lift_table(test, masks, 'exploded')

    mo = test.groupby('month').apply(lambda g: pd.Series({
        'eligible': len(g), 'base_rate': g.exploded.mean(),
        'flagged': int(g.full_match.sum()),
        'flagged_hit_rate': g.loc[g.full_match, 'exploded'].mean() if g.full_match.any() else np.nan,
        'flagged_median_close_pct': g.loc[g.full_match, 'month_close_pct'].median() if g.full_match.any() else np.nan}),
        include_groups=False)
    mo.to_csv(out / 'monthly.csv')
    half = sorted(test.month.unique()); cut = half[len(half) // 2]
    summary['first_half'] = lift_table(test[test.month < cut],
                                       {'full_match': masks['full_match'][test.month < cut]}, 'exploded')
    summary['second_half'] = lift_table(test[test.month >= cut],
                                        {'full_match': masks['full_match'][test.month >= cut]}, 'exploded')
    summary['months_flagged_beats_base'] = int((mo.flagged_hit_rate > mo.base_rate).sum())
    summary['months_with_flags'] = int(mo.flagged_hit_rate.notna().sum())
    summary['basket_missing_in_db'] = missing
    summary['listing_from_fundamentals_pct'] = round(float(test.listing_known.mean()), 3)
    summary['mcap_coverage_pct'] = round(float(test.mcap.notna().mean()), 3)
    summary['split_excluded_stock_months'] = int((~feats.live & feats.split_window).sum())
    summary['live_asof'] = str(last.date()); summary['live_candidates'] = int(len(live))
    summary['notes'] = ('Basket members excluded from scoring. Thresholds = basket quantile at each '
                        'month-end. Outcome = max high in month vs prior month close.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print('RESULT', json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
