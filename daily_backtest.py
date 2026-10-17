#!/usr/bin/env python3
"""
DAILY version of the fingerprint strategy.

Every trading day, after the close, using only data up to that close:
  match = eligible  (price >= --min-price, 20d median $ volume >= --min-liquidity,
                     no split in last 120 sessions, >= 40 valid returns in 60d)
        & turnover (20d median $vol / market cap) >= basket q-quantile that day
        & beta to basket index (60d)               >= basket q-quantile that day
        & volatility (60d)                         >= basket q-quantile that day
        & market cap between --cap-min and --cap-max
        & correlation with basket index (60d)      >= --min-corr
Entry variants (signal at close of day t, buy at OPEN of day t+1):
  state : stock matches today, not already held, not exited in last --cooldown days
  new   : stock matches today but did NOT match yesterday
  surge : stock matches today AND 5d/60d $-volume ratio crosses above --surge today
Exit: take-profit +T%, stop -S%, else sell at close after H trading days.
  Same-day stop+target -> stop assumed first. Gaps handled at the open.
One position per symbol at a time. Basket members are not traded (picked with hindsight).
Trades with a split inside the holding window are dropped (prices would be wrong).

Outputs in --out:
  daily_grid.csv          every entry x target x stop x hold combination
  daily_best.txt          ranked report (chosen on first half, checked on second half)
  daily_trades_best.csv   trade list of the best combination
  live_daily_signals.csv  signals at the last complete day (buy at next open)
"""
import argparse, itertools, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASKET = ['RGTI','QBTS','IONQ','SOUN','BBAI','RKLB','ASTS','LUNR','ACHR',
          'OKLO','SMR','IREN','CIFR','HIMS']
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


def sim_symbol(sig, o, h, l, c, split, T, S, H, cooldown):
    """Sequential trades for one symbol. sig[t] True = signal at close t -> buy open t+1."""
    n = len(sig); trades = []; t = 0; next_ok = 0
    idx = np.flatnonzero(sig)
    for s in idx:
        if s < next_ok or s + 1 >= n:
            continue
        e = s + 1; o0 = o[e]
        if not np.isfinite(o0) or o0 <= 0:
            continue
        sp, tp = o0 * (1 - S / 100), o0 * (1 + T / 100)
        last = min(e + H - 1, n - 1)
        ret, xd = None, last
        for d in range(e, last + 1):
            op, hi, lo = o[d], h[d], l[d]
            if not (np.isfinite(op) and np.isfinite(hi) and np.isfinite(lo)):
                continue
            if d > e and op <= sp: ret, xd = op / o0 - 1, d; break
            if d > e and op >= tp: ret, xd = op / o0 - 1, d; break
            if lo <= sp: ret, xd = -S / 100, d; break
            if hi >= tp: ret, xd = T / 100, d; break
        if ret is None:
            cl = c[last]
            if not np.isfinite(cl):
                continue
            ret = cl / o0 - 1
        if split[e:xd + 1].any():
            next_ok = xd + 1; continue
        trades.append((s, e, xd, ret * 100))
        next_ok = xd + 1 + cooldown
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', default='/data/fund.csv')
    ap.add_argument('--out', default='/data/fingerprint_daily')
    ap.add_argument('--start', default='2024-04-01', help='first signal date')
    ap.add_argument('--min-liquidity', type=float, default=4_500_000)
    ap.add_argument('--min-price', type=float, default=2.0)
    ap.add_argument('--cap-min', type=float, default=1e9)
    ap.add_argument('--cap-max', type=float, default=40e9)
    ap.add_argument('--quantile', type=float, default=0.10)
    ap.add_argument('--min-corr', type=float, default=0.55)
    ap.add_argument('--surge', type=float, default=2.0)
    ap.add_argument('--cooldown', type=int, default=5)
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--min-trades', type=int, default=100)
    ap.add_argument('--open-col', default='o'); ap.add_argument('--high-col', default='h')
    ap.add_argument('--low-col', default='l'); ap.add_argument('--close-col', default='c')
    ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    fund = pd.read_csv(a.fundamentals)
    fund['asof_date'] = pd.to_datetime(fund.asof_date, errors='coerce')
    if 'shares_outstanding' not in fund.columns:
        raise SystemExit('fund.csv needs shares_outstanding (made by build_fund_csv.py)')

    load_from = (pd.Timestamp(a.start) - pd.Timedelta(days=260)).strftime('%Y-%m-%d')
    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS o, {a.high_col} AS h,
                 {a.low_col} AS l, {a.close_col} AS c, {a.volume_col} AS v
                 FROM market_candles_1d WHERE session_date >= :s''')
    parts = []
    with eng.connect() as cn:
        for ch in pd.read_sql(q, cn, params={'s': load_from}, chunksize=500_000):
            parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    d = d[~d.symbol.astype(str).str.endswith('.CA')]
    d['date'] = pd.to_datetime(d.date)
    for k in 'ohlcv':
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date'])
    # Memory guard: drop symbols that can never pass the liquidity rule
    med = (d.c * d.v).groupby(d.symbol).quantile(0.9)
    keep_syms = set(med[med >= a.min_liquidity].index) | set(BASKET)
    d = d[d.symbol.isin(keep_syms)]
    print('candles', len(d), 'symbols kept', d.symbol.nunique(), flush=True)

    P = {k: d.pivot(index='date', columns='symbol', values=k) for k in 'ohlcv'}
    C = P['c']; cols = C.columns; dates = C.index
    for k in 'ohlv':
        P[k] = P[k].reindex(index=dates, columns=cols)
    cnt = C.notna().sum(axis=1)
    last_full = cnt[cnt >= 0.8 * cnt.tail(30).median()].index.max()

    split = detect_splits(C, P['v'])
    r = C.pct_change(fill_method=None).mask(split)
    dv = C * P['v']
    med_dv20 = dv.rolling(20, min_periods=15).median()

    # Point-in-time shares outstanding (filed date), forward-filled per day
    sh = (fund.dropna(subset=['asof_date']).sort_values('asof_date')
              .drop_duplicates(['symbol', 'asof_date'], keep='last')
              .pivot(index='asof_date', columns='symbol', values='shares_outstanding'))
    sh = sh.reindex(sh.index.union(dates)).sort_index().ffill().reindex(index=dates, columns=cols)
    mcap = sh * C
    turnover = med_dv20 / mcap * 100

    # Basket index; members are measured against the index WITHOUT themselves
    bm = [b for b in BASKET if b in cols]
    b_sum = r[bm].sum(axis=1, min_count=1); b_cnt = r[bm].notna().sum(axis=1)
    bidx = b_sum / b_cnt
    W, MP = 60, 40
    mx = bidx.rolling(W, min_periods=MP).mean(); vx = bidx.rolling(W, min_periods=MP).var()
    valid = r.notna()
    my = r.rolling(W, min_periods=MP).mean()
    mxy = r.mul(bidx, axis=0).rolling(W, min_periods=MP).mean()
    nobs = valid.rolling(W, min_periods=1).sum()
    cov = (mxy - my.mul(mx, axis=0)) * nobs / (nobs - 1)
    beta = cov.div(vx, axis=0)
    sy = r.rolling(W, min_periods=MP).std()
    corr = cov.div(sy.mul(np.sqrt(vx), axis=0))
    vol60 = sy * np.sqrt(252)
    for m in bm:                                   # ex-self for basket members
        ix = (b_sum - r[m].fillna(0)) / (b_cnt - r[m].notna())
        cv = r[m].rolling(W, min_periods=MP).cov(ix)
        beta[m] = cv / ix.rolling(W, min_periods=MP).var()
        corr[m] = r[m].rolling(W, min_periods=MP).corr(ix)
    surge = dv.rolling(5).mean() / dv.rolling(60, min_periods=40).median()

    thr_t = turnover[bm].quantile(a.quantile, axis=1)
    thr_b = beta[bm].quantile(a.quantile, axis=1)
    thr_v = vol60[bm].quantile(a.quantile, axis=1)
    split_recent = split.rolling(120, min_periods=1).max().astype(bool)
    eligible = (C.ge(a.min_price) & med_dv20.ge(a.min_liquidity) & ~split_recent & nobs.ge(40))
    match = (eligible & turnover.ge(thr_t, axis=0) & beta.ge(thr_b, axis=0) & vol60.ge(thr_v, axis=0)
             & mcap.ge(a.cap_min) & mcap.le(a.cap_max) & corr.ge(a.min_corr))
    match.loc[dates < pd.Timestamp(a.start)] = False
    match.loc[dates > last_full] = False
    match[bm] = False
    sig_new = match & ~match.shift(1, fill_value=False)
    sig_surge = match & surge.ge(a.surge) & ~surge.shift(1).ge(a.surge)
    ds = pd.Series(dates, index=dates)
    is_month_end = ds.groupby(dates.to_period('M')).transform('max').eq(ds)
    is_week_end = ds.groupby(dates.to_period('W')).transform('max').eq(ds)
    SIG = {'state': match, 'state_cd20': match, 'new': sig_new, 'surge': sig_surge,
           'month_end': match.mul(is_month_end.values, axis=0),
           'week_end': match.mul(is_week_end.values, axis=0)}
    print('match stock-days', int(match.values.sum()), '| new', int(sig_new.values.sum()),
          '| surge', int(sig_surge.values.sum()), '| last complete day', last_full.date(), flush=True)

    # Live signals at last complete day
    t_ = last_full
    live = pd.DataFrame({'symbol': cols, 'match': match.loc[t_].values, 'new': sig_new.loc[t_].values,
                         'surge_signal': sig_surge.loc[t_].values, 'close': C.loc[t_].values,
                         'corr_basket60': corr.loc[t_].values, 'beta_basket': beta.loc[t_].values,
                         'turnover_pct': turnover.loc[t_].values, 'vol60': vol60.loc[t_].values,
                         'mcap': mcap.loc[t_].values, 'surge_5_60': surge.loc[t_].values})
    live = live[live.match].sort_values(['surge_signal', 'new', 'corr_basket60'], ascending=False)
    live.insert(0, 'signal_date', t_.date())
    live.to_csv(out / 'live_daily_signals.csv', index=False)

    O, Hh, L, Cc = (P['o'].values, P['h'].values, P['l'].values, C.values)
    SP = split.values
    month_of = dates.to_period('M').astype(str)
    sig_dates = dates[(dates >= pd.Timestamp(a.start)) & (dates <= last_full)]
    cut = sig_dates[len(sig_dates) // 2]

    rows, best_trades = [], {}
    grid = list(itertools.product(SIG, [20, 30, 50], [8, 10, 15], [5, 10, 21]))
    for i, (ename, T, S, H) in enumerate(grid, 1):
        sig = SIG[ename].values
        cd = {'state': a.cooldown, 'state_cd20': 20}.get(ename, 0)
        tr = []
        for j in np.flatnonzero(sig.any(axis=0)):
            for s, e, x, ret in sim_symbol(sig[:, j], O[:, j], Hh[:, j], L[:, j], Cc[:, j], SP[:, j], T, S, H, cd):
                tr.append((cols[j], dates[s], dates[e], dates[x], ret - a.cost))
        t = pd.DataFrame(tr, columns=['symbol', 'signal_date', 'entry_date', 'exit_date', 'ret'])
        if len(t) < (30 if ename == 'surge' else a.min_trades):
            continue
        t['month'] = t.entry_date.dt.to_period('M').astype(str)
        m = t.groupby('month').ret.mean()
        h1, h2 = t[t.signal_date < cut], t[t.signal_date >= cut]
        rows.append({'entry': ename, 'target': T, 'stop': S, 'max_days': H, 'trades': len(t),
                     'trades_per_month': round(len(t) / max(1, len(m)), 1),
                     'win_pct': round((t.ret > 0).mean() * 100, 1), 'mean_trade': round(t.ret.mean(), 2),
                     'median_trade': round(t.ret.median(), 2),
                     'avg_days_held': round(np.busday_count(t.entry_date.values.astype('datetime64[D]'),
                                            t.exit_date.values.astype('datetime64[D]')).mean() + 1, 1),
                     'months_positive': int((m > 0).sum()), 'months': len(m), 'worst_month': round(m.min(), 2),
                     'first_half_mean': round(h1.ret.mean(), 2), 'first_half_trades': len(h1),
                     'second_half_mean': round(h2.ret.mean(), 2), 'second_half_trades': len(h2)})
        best_trades[(ename, T, S, H)] = t
        if i % 9 == 0:
            print(f'  {i}/{len(grid)} combinations', flush=True)

    G = pd.DataFrame(rows)
    G.to_csv(out / 'daily_grid.csv', index=False)
    G['rank_first_half'] = G.first_half_mean.rank(ascending=False)
    top = G.sort_values('first_half_mean', ascending=False)
    lines = [f'Signals {sig_dates[0].date()} -> {last_full.date()} | first half < {cut.date()} <= second half',
             f'Cost {a.cost}% per trade. Ranked by FIRST half; SECOND half is the honest check.', '']
    for e in SIG:
        g = top[top.entry == e].head(5)
        lines.append(f'===== entry: {e} =====')
        lines.append(g.drop(columns='rank_first_half').to_string(index=False))
        lines.append('')
    same = G[(G.target == 30) & (G.stop == 10) & (G.max_days == 21)]
    lines.append('===== same exit as the monthly test (+30% / -10% / 21 days), by entry type =====')
    lines.append(same.drop(columns='rank_first_half').to_string(index=False))
    lines.append('')
    both = G[(G.first_half_mean > 0) & (G.second_half_mean > 0)].copy()
    both['min_half'] = both[['first_half_mean', 'second_half_mean']].min(axis=1)
    lines.append('===== positive in BOTH halves, ranked by the weaker half =====')
    lines.append(both.sort_values('min_half', ascending=False).head(10).drop(columns='rank_first_half').to_string(index=False))
    if len(both):
        b = both.sort_values('min_half', ascending=False).iloc[0]
        key = (b.entry, int(b.target), int(b.stop), int(b.max_days))
        best_trades[key].to_csv(out / 'daily_trades_best.csv', index=False)
        lines.append(f'\nBest robust combination: entry={key[0]} target=+{key[1]}% stop=-{key[2]}% max_days={key[3]}')
        sel = live[live['new'] if key[0] == 'new' else live['surge_signal'] if key[0] == 'surge' else live['match']]  # month/week_end: use on those days only
        lines.append(f'Signals on {t_.date()} for that entry type (buy at next open): '
                     + (', '.join(sel.symbol.head(30)) or 'none'))
    txt = '\n'.join(lines)
    (out / 'daily_best.txt').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
