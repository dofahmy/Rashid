#!/usr/bin/env python3
"""
Sector-wave LAGGARD test.

Idea: when 1-3 stocks in an industry have already jumped (leaders), buy the
industry's large liquid stocks that have NOT moved yet (laggards), expecting the
wave to reach them.

Every day after the close (known data only):
  leaders(group)  = liquid members with 20-day return >= --leader-gain (several levels tested)
  laggard signal  = stock is eligible (cap/liquidity/price/no split),
                    its own 20-day return is between -15% and LAG_MAX,
                    and its group has >= K leaders today
  baseline        = SAME condition on the stock itself, but its group has ZERO leaders
                    (tells us whether the wave adds anything)
Buy at next open; exits tested: target / stop / max days. One position per stock.
Rules and exits are compared on the period BEFORE --test-start; the period from
--test-start on (2026) is the honest check.

Needs spike_profile.py in the same folder (re-uses its helpers).
Outputs (in --out): laggard_grid.csv, laggard_report.txt, laggard_live.csv
"""
import argparse, itertools, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from spike_profile import detect_splits, load_groups, sim


def run(sig, M, cut, cost, T, S, H):
    O, Hh, L, C, SP, dates = M
    tr = []
    for j in np.flatnonzero(sig.any(axis=0)):
        for s_, e_, x_, ret in sim(sig[:, j], O[:, j], Hh[:, j], L[:, j], C[:, j], SP[:, j], T, S, H):
            tr.append((dates[s_], j, ret - cost))
    t = pd.DataFrame(tr, columns=['signal_date', 'j', 'ret'])
    if t.empty:
        return None
    m = t.groupby(t.signal_date.dt.to_period('M')).ret.mean()
    a, b = t[t.signal_date < cut], t[t.signal_date >= cut]
    return {'trades': len(t), 'win_pct': round((t.ret > 0).mean() * 100, 1), 'mean': round(t.ret.mean(), 2),
            'median': round(t.ret.median(), 2), 'months_pos': f'{int((m > 0).sum())}/{len(m)}',
            'worst_month': round(m.min(), 1),
            'train_n': len(a), 'train_mean': round(a.ret.mean(), 2) if len(a) else np.nan,
            'test_n': len(b), 'test_mean': round(b.ret.mean(), 2) if len(b) else np.nan,
            'test_win_pct': round((b.ret > 0).mean() * 100, 1) if len(b) else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', default='/data/fund.csv')
    ap.add_argument('--out', default='/data/laggard_test')
    ap.add_argument('--test-start', default='2026-01-01')
    ap.add_argument('--min-cap', type=float, default=1e9)
    ap.add_argument('--min-liquidity', type=float, default=5e6)
    ap.add_argument('--min-price', type=float, default=5.0)
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--open-col', default='o'); ap.add_argument('--high-col', default='h')
    ap.add_argument('--low-col', default='l'); ap.add_argument('--close-col', default='c')
    ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(a.test_start)

    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS o, {a.high_col} AS h,
                 {a.low_col} AS l, {a.close_col} AS c, {a.volume_col} AS v FROM market_candles_1d''')
    with eng.connect() as cn:
        d = pd.concat(pd.read_sql(q, cn, chunksize=500_000), ignore_index=True)
    d = d[~d.symbol.astype(str).str.endswith('.CA')]
    d['date'] = pd.to_datetime(d.date)
    for k in 'ohlcv':
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date'])
    q90 = (d.c * d.v).groupby(d.symbol).quantile(0.9)
    d = d[d.symbol.isin(q90[q90 >= a.min_liquidity / 2].index)]
    P = {k: d.pivot(index='date', columns='symbol', values=k) for k in 'ohlcv'}
    C = P['c']; dates, cols = C.index, C.columns
    O, H, L, V = (P[k].reindex(index=dates, columns=cols) for k in 'ohlv')
    cnt = C.notna().sum(axis=1)
    last_full = cnt[cnt >= 0.8 * cnt.tail(30).median()].index.max()
    print('candles', len(d), '| symbols', len(cols), '| last complete day', last_full.date(), flush=True)

    fund = pd.read_csv(a.fundamentals)
    fund['asof_date'] = pd.to_datetime(fund.asof_date, errors='coerce')
    sh = (fund.dropna(subset=['asof_date']).sort_values('asof_date')
              .drop_duplicates(['symbol', 'asof_date'], keep='last')
              .pivot(index='asof_date', columns='symbol', values='shares_outstanding'))
    sh = sh.reindex(sh.index.union(dates)).sort_index().ffill().reindex(index=dates, columns=cols)
    mcap = sh * C

    split = detect_splits(C, V)
    split20 = split.rolling(20, min_periods=1).max().astype(bool)
    ret20 = (C / C.shift(20) - 1).mask(split20)
    med_dv20 = (C * V).rolling(20, min_periods=15).median()
    r = C.pct_change(fill_method=None).mask(split)
    vol60 = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    hist = C.notna().cumsum()
    split_recent = split.rolling(120, min_periods=1).max().astype(bool)
    liquid = med_dv20.ge(a.min_liquidity) & C.ge(a.min_price)
    eligible = liquid & mcap.ge(a.min_cap) & ~split_recent & hist.ge(120)
    in_range = (pd.Series(dates, index=dates) <= last_full).values[:, None]

    grp, sector_of = load_groups(list(cols))
    real = grp[~grp.str.startswith('S:unknown') & (grp != 'unknown')]
    groups = {g: [cols.get_loc(m) for m in mem] for g, mem in real.groupby(real).groups.items()}
    print('industry groups used:', len(groups), flush=True)

    def leader_count(gain):
        lead = (liquid & ret20.ge(gain)).values.astype(np.int16)
        cntm = np.zeros(C.shape, dtype=np.int16)
        for g, idx in groups.items():
            cntm[:, idx] = lead[:, idx].sum(axis=1, keepdims=True)
        return cntm

    M = (O.values, H.values, L.values, C.values, split.values, dates)
    exits = [(100, 99, 42), (50, 99, 42), (50, 25, 42), (30, 15, 21), (100, 40, 42)]
    in_group = np.zeros(len(cols), dtype=bool)
    for idx in groups.values():
        in_group[idx] = True

    rows = []
    # reference: every eligible stock, no condition
    base_all = (eligible.values & in_range & in_group[None, :])
    for (T, S, Hd) in exits:
        res = run(base_all, M, cut, a.cost, T, S, Hd)
        if res:
            rows.append({'variant': 'كل الأسهم الكبيرة (مرجع)', 'leader_gain': '-', 'min_leaders': '-',
                         'lag_max': '-', 'vol_filter': '-', 'target': T, 'stop': 'none' if S == 99 else S,
                         'max_days': Hd, **res})
    print('reference done', flush=True)

    vol_ok = vol60.ge(0.5).values
    for gain in (0.3, 0.5):
        LC = leader_count(gain)
        for lag_max, vf in itertools.product((0.05, 0.15), (False, True)):
            own = (eligible.values & in_range & in_group[None, :] & ret20.le(lag_max).values
                   & ret20.ge(-0.15).values & (vol_ok if vf else True))
            for k in (1, 2, 3):
                wave = own & (LC >= k)
                for (T, S, Hd) in exits:
                    res = run(wave, M, cut, a.cost, T, S, Hd)
                    if res:
                        rows.append({'variant': 'متأخر في صناعة طالعة', 'leader_gain': gain, 'min_leaders': k,
                                     'lag_max': lag_max, 'vol_filter': vf, 'target': T,
                                     'stop': 'none' if S == 99 else S, 'max_days': Hd, **res})
            calm = own & (LC == 0)
            for (T, S, Hd) in exits:
                res = run(calm, M, cut, a.cost, T, S, Hd)
                if res:
                    rows.append({'variant': 'نفس الشرط وصناعته هادية (مقارنة)', 'leader_gain': gain, 'min_leaders': 0,
                                 'lag_max': lag_max, 'vol_filter': vf, 'target': T,
                                 'stop': 'none' if S == 99 else S, 'max_days': Hd, **res})
            print(f'  leader +{int(gain*100)}% lag<={int(lag_max*100)}% vol_filter={vf} done', flush=True)

    G = pd.DataFrame(rows)
    G.to_csv(out / 'laggard_grid.csv', index=False)

    # pair each wave variant with its calm twin (same gain/lag/vol/exit)
    key = ['leader_gain', 'lag_max', 'vol_filter', 'target', 'stop', 'max_days']
    W = G[G.variant == 'متأخر في صناعة طالعة'].copy()
    Q = G[G.variant == 'نفس الشرط وصناعته هادية (مقارنة)'][key + ['train_mean', 'test_mean']] \
        .rename(columns={'train_mean': 'calm_train', 'test_mean': 'calm_test'})
    W = W.merge(Q, on=key, how='left')
    W['edge_train'] = (W.train_mean - W.calm_train).round(2)
    W['edge_test'] = (W.test_mean - W.calm_test).round(2)
    W = W[(W.train_n >= 50) & (W.test_n >= 30)]
    pick = W.sort_values('train_mean', ascending=False).head(1)
    show = ['leader_gain', 'min_leaders', 'lag_max', 'vol_filter', 'target', 'stop', 'max_days', 'trades',
            'win_pct', 'train_n', 'train_mean', 'calm_train', 'edge_train', 'test_n', 'test_mean', 'calm_test',
            'edge_test', 'test_win_pct', 'months_pos', 'worst_month']

    lines = [f'التقسيم: استنتاج قبل {cut.date()} | اختبار من {cut.date()} لحد {last_full.date()} | تكلفة {a.cost}% للصفقة',
             'train = فترة الاستنتاج، test = فترة الاختبار، calm = نفس الأسهم وصناعتها هادية، edge = الفرق', '',
             '--- المرجع: كل الأسهم الكبيرة السائلة من غير شروط ---',
             G[G.variant.str.startswith('كل')][['target', 'stop', 'max_days', 'trades', 'train_mean', 'test_mean',
                                                'test_win_pct']].to_string(index=False), '',
             '--- أحسن ١٥ توليفة للمتأخرين (مرتبة بفترة الاستنتاج) ---',
             W.sort_values('train_mean', ascending=False).head(15)[show].to_string(index=False), '',
             '--- متوسط الميزة على كل التوليفات (المتأخر في صناعة طالعة ناقص نفس السهم وصناعته هادية) ---',
             f'فترة الاستنتاج: {W.edge_train.mean():.2f}% | فترة الاختبار: {W.edge_test.mean():.2f}% | '
             f'التوليفات اللي الميزة فيها موجبة في الفترتين: {int(((W.edge_train > 0) & (W.edge_test > 0)).sum())} من {len(W)}']
    if len(pick):
        p = pick.iloc[0]
        lines.append(f'\nالاختيار على فترة الاستنتاج بس: قادة +{int(p.leader_gain*100)}%، على الأقل {int(p.min_leaders)} قائد، '
                     f'المتأخر طالع ≤ {int(p.lag_max*100)}%، فلتر تذبذب={p.vol_filter}، هدف +{p.target}% وقف {p.stop} '
                     f'{p.max_days} يوم -> استنتاج {p.train_mean}% | اختبار {p.test_mean}% (هادي {p.calm_test}%)')
        # live list for the picked variant
        LC = leader_count(p.leader_gain)
        t = last_full; i = dates.get_loc(t)
        own_t = (eligible.loc[t].values & in_group & (ret20.loc[t].values <= p.lag_max)
                 & (ret20.loc[t].values >= -0.15) & ((vol60.loc[t].values >= 0.5) if p.vol_filter else True))
        sel = own_t & (LC[i] >= p.min_leaders)
        live = pd.DataFrame({'symbol': cols[sel], 'industry': grp.reindex(cols[sel]).values,
                             'ret20_pct': (ret20.loc[t].values[sel] * 100).round(1),
                             'leaders_in_industry': LC[i][sel]})
        lead_now = (liquid.loc[t] & ret20.loc[t].ge(p.leader_gain))
        live['leaders'] = [' '.join(sorted(c for c in grp[grp == g].index if lead_now.get(c, False)))
                           for g in live.industry]
        live = live.sort_values(['leaders_in_industry', 'industry'], ascending=False)
        live.insert(0, 'signal_date', t.date())
        live.to_csv(out / 'laggard_live.csv', index=False)
        lines.append(f'\nإشارات {t.date()} (شراء افتتاح اليوم التالي): {len(live)} سهم')
        for g, x in live.groupby('industry', sort=False):
            lines.append(f'  {g} | القادة: {x.leaders.iloc[0]} | المتأخرين: {" ".join(x.symbol)}')
    txt = '\n'.join(lines)
    (out / 'laggard_report.txt').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
