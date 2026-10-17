#!/usr/bin/env python3
"""
Daily signal log for the spike rule over the last N months.

Rule (step 2 of spike_profile.py), checked after each close:
  market cap >= --min-cap, 20d median $ volume >= --min-liquidity, price >= --min-price,
  no split in last 120 sessions, >= 120 sessions history,
  60-day volatility >= --min-vol60, ATR14 / price >= --min-atr
Signal at close of day t -> buy at OPEN of day t+1.

Prints one line per day: number of stocks passing, how many are NEW (did not pass the
day before), and the symbols. Also tracks each NEW signal: entry price (next open),
best gain so far, and return at the latest close (or exit at +target / max days).

Outputs in --out:
  daily_counts.csv    date, passing, new, symbols
  signals_detail.csv  every new signal with outcome so far
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

SPLIT_K = [2,3,4,5,8,10,15,20,25,30,40,50,100]


def detect_splits(close, vol):
    ratio = close / close.shift(1)
    pre = vol.rolling(10, min_periods=5).median().shift(1)
    post = vol[::-1].rolling(10, min_periods=5).median()[::-1]
    vr = post / pre
    flag = pd.DataFrame(False, index=close.index, columns=close.columns)
    for k in SPLIT_K:
        flag |= (((ratio / k - 1).abs().le(0.04) & vr.le(3.0 / k)) |
                 ((ratio * k - 1).abs().le(0.04) & vr.ge(k / 3.0))).fillna(False)
    return flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', default='/data/fund.csv')
    ap.add_argument('--out', default='/data/spike_profile')
    ap.add_argument('--months', type=int, default=2)
    ap.add_argument('--since', help='start date YYYY-MM-DD (overrides --months)')
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--quiet', action='store_true', help='do not print the day-by-day table')
    ap.add_argument('--min-cap', type=float, default=1e9)
    ap.add_argument('--min-liquidity', type=float, default=5e6)
    ap.add_argument('--min-price', type=float, default=5.0)
    ap.add_argument('--min-vol60', type=float, default=0.6418)
    ap.add_argument('--min-atr', type=float, default=0.07154)
    ap.add_argument('--target', type=float, default=100)
    ap.add_argument('--max-days', type=int, default=42)
    ap.add_argument('--open-col', default='o'); ap.add_argument('--high-col', default='h')
    ap.add_argument('--low-col', default='l'); ap.add_argument('--close-col', default='c')
    ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    eng = create_engine(os.environ['DATABASE_URL'])
    with eng.connect() as cn:
        last_db = pd.to_datetime(pd.read_sql(text('SELECT MAX(session_date) AS d FROM market_candles_1d'), cn).d[0])
        start_sig = pd.Timestamp(a.since) if a.since else last_db - pd.DateOffset(months=a.months)
        load_from = (start_sig - pd.Timedelta(days=270)).strftime('%Y-%m-%d')
        q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS o, {a.high_col} AS h,
                     {a.low_col} AS l, {a.close_col} AS c, {a.volume_col} AS v
                     FROM market_candles_1d WHERE session_date >= :s''')
        d = pd.concat(pd.read_sql(q, cn, params={'s': load_from}, chunksize=500_000), ignore_index=True)
    d = d[~d.symbol.astype(str).str.endswith('.CA')]
    d['date'] = pd.to_datetime(d.date)
    for k in 'ohlcv':
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date'])
    q90 = (d.c * d.v).groupby(d.symbol).quantile(0.9)
    d = d[d.symbol.isin(q90[q90 >= a.min_liquidity].index)]

    P = {k: d.pivot(index='date', columns='symbol', values=k) for k in 'ohlcv'}
    C = P['c']; dates, cols = C.index, C.columns
    O, H, L, V = (P[k].reindex(index=dates, columns=cols) for k in 'ohlv')
    cnt = C.notna().sum(axis=1)
    full = cnt[cnt >= 0.8 * cnt.tail(30).median()].index
    last_full = full.max()

    fund = pd.read_csv(a.fundamentals)
    fund['asof_date'] = pd.to_datetime(fund.asof_date, errors='coerce')
    sh = (fund.dropna(subset=['asof_date']).sort_values('asof_date')
              .drop_duplicates(['symbol', 'asof_date'], keep='last')
              .pivot(index='asof_date', columns='symbol', values='shares_outstanding'))
    sh = sh.reindex(sh.index.union(dates)).sort_index().ffill().reindex(index=dates, columns=cols)

    split = detect_splits(C, V)
    r = C.pct_change(fill_method=None).mask(split)
    mcap = sh * C
    med_dv20 = (C * V).rolling(20, min_periods=15).median()
    vol60 = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr = tr.rolling(14).mean() / C
    hist = C.notna().cumsum()
    split_recent = split.rolling(120, min_periods=1).max().astype(bool)
    rule = (mcap.ge(a.min_cap) & med_dv20.ge(a.min_liquidity) & C.ge(a.min_price) & ~split_recent
            & hist.ge(120) & vol60.ge(a.min_vol60) & atr.ge(a.min_atr))
    new = rule & ~rule.shift(1, fill_value=False)

    days = [t for t in full if t >= start_sig]
    rows, det = [], []
    pos = {t: i for i, t in enumerate(dates)}
    busy_until = {}                      # symbol -> index of exit day (no re-buy while holding)
    for t in days:
        syms = cols[rule.loc[t].values]; nsyms = cols[new.loc[t].values]
        rows.append({'date': t.date(), 'passing': len(syms), 'new': len(nsyms),
                     'new_symbols': ' '.join(sorted(nsyms)), 'all_symbols': ' '.join(sorted(syms))})
        i = pos[t]
        for s in nsyms:
            if i + 1 >= len(dates):
                det.append({'signal_date': t.date(), 'symbol': s, 'status': 'buy at next open'}); continue
            j = cols.get_loc(s); e = i + 1; o0 = O.iat[e, j]
            if not np.isfinite(o0): continue
            if busy_until.get(s, -1) >= e: continue      # already holding this stock
            last = min(e + a.max_days - 1, pos[last_full])
            hs = H.iloc[e:last + 1, j].values
            hit = np.flatnonzero(hs >= o0 * (1 + a.target / 100))
            if len(hit):
                status, ret, xd = f'hit +{a.target:g}%', a.target, dates[e + hit[0]].date()
            elif e + a.max_days - 1 <= pos[last_full]:
                status, ret, xd = f'closed after {a.max_days} days', (C.iat[last, j] / o0 - 1) * 100, dates[last].date()
            else:
                status, ret, xd = 'still open', (C.iat[pos[last_full], j] / o0 - 1) * 100, None
            busy_until[s] = pos[pd.Timestamp(xd)] if xd is not None else 10**9
            det.append({'signal_date': t.date(), 'symbol': s, 'entry_date': dates[e].date(),
                        'entry_open': round(o0, 2), 'best_gain_pct': round((np.nanmax(hs) / o0 - 1) * 100, 1),
                        'worst_pct': round((np.nanmin(L.iloc[e:last + 1, j].values) / o0 - 1) * 100, 1),
                        'return_pct': round(ret, 1), 'status': status, 'exit_date': xd})
    R = pd.DataFrame(rows); D = pd.DataFrame(det)
    R.to_csv(out / 'daily_counts.csv', index=False); D.to_csv(out / 'signals_detail.csv', index=False)

    print(f'آخر يوم كامل في البيانات: {last_full.date()}  (آخر تاريخ في القاعدة {last_db.date()})\n')
    if not a.quiet:
        print('التاريخ      | عدّى الشروط | جديد | الأسهم الجديدة')
        for x in R.itertuples():
            print(f'{x.date} | {x.passing:>10} | {x.new:>4} | {x.new_symbols}')
    if len(D) and 'return_pct' in D:
        dd = D.dropna(subset=['return_pct'])
        print(f'\nإشارات جديدة: {len(D)} | متوسط أحسن صعود {dd.best_gain_pct.mean():.1f}% | '
              f'متوسط أسوأ نزول {dd.worst_pct.mean():.1f}% | متوسط العائد لحد دلوقتي {dd.return_pct.mean():.1f}% | '
              f'كسبانة {(dd.return_pct > 0).mean() * 100:.0f}% | وصلت +{a.target:g}%: {(dd.status.str.startswith("hit")).sum()}')
    if len(D) and 'return_pct' in D:
        c = D[D.status.notna() & (D.status != 'still open') & (D.status != 'buy at next open')].copy()
        c['net'] = c.return_pct - a.cost
        c['month'] = pd.to_datetime(c.signal_date).dt.to_period('M').astype(str)
        R['month'] = pd.to_datetime(R.date).dt.to_period('M').astype(str)
        sig_per_day = R.groupby('month').agg(avg_passing=('passing', 'mean'), avg_new=('new', 'mean'))
        m = c.groupby('month').agg(closed=('net', 'size'), mean_net=('net', 'mean'), median_net=('net', 'median'),
                                   win_pct=('net', lambda x: (x > 0).mean() * 100),
                                   hit_target=('status', lambda x: x.str.startswith('hit').sum()),
                                   worst=('net', 'min'), best=('net', 'max'))
        m = sig_per_day.join(m, how='left').round(1)
        print(f'\n--- شهر بشهر (الصفقات اللي خلصت فقط، بعد تكلفة {a.cost:g}%) ---')
        print('الشهر    | متوسط الأسهم/يوم | جديد/يوم | خلصت | متوسط | الوسيط | كسبانة% | وصلت الهدف | أسوأ | أحسن')
        for k, x in m.iterrows():
            print(f'{k} | {x.avg_passing:>15} | {x.avg_new:>8} | {x.closed if pd.notna(x.closed) else 0:>5} | '
                  f'{x.mean_net} | {x.median_net} | {x.win_pct} | {x.hit_target} | {x.worst} | {x.best}')
        print(f'\nكل الصفقات اللي خلصت: {len(c)} | متوسط بعد التكلفة {c.net.mean():.1f}% | '
              f'كسبانة {(c.net > 0).mean() * 100:.0f}% | وصلت الهدف {c.status.str.startswith("hit").sum()} | '
              f'أسوأ صفقة {c.net.min():.1f}%')
    print(f'\nالتفاصيل: {out / "signals_detail.csv"}')


if __name__ == '__main__':
    main()
