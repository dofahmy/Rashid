#!/usr/bin/env python3
"""
HOW BIG are the moves after a signal? Numbers for long / short / options traders.

For each combination of conditions (same definitions as combo_test.py) and for a
baseline (any large liquid stock on any day), measured from the signal-day CLOSE:
  - absolute move |return| after 5 / 10 / 21 / 42 trading days (median, 75th, 90th pct)
  - max rise (from highs) and max fall (from lows) within 21 days
  - share of signals that touched +-20% / +-30% / +-50% within 21 days, and BOTH +20% and -20%
  - realized volatility over the next 21 days vs the trailing 60-day volatility
  - straddle proxy: average |21-day move| divided by the price of an at-the-money
    21-day straddle estimated from trailing volatility (0.8 x vol x sqrt(21/252)).
    > 1 means the stock moved MORE than its own recent volatility implied.
    (Real option prices usually sit ABOVE trailing volatility for these stocks,
     so check against real implied volatility before trading options.)
Signals = first day a stock passes the combination. Train = before --test-start, test = from it.

Needs combo_test.py and spike_profile.py in the same folder.
Output: /data/movement_stats/movement_report.txt and movement_stats.csv
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from spike_profile import detect_splits, load_groups
from combo_test import COND, TURNOVER_MAX

DEFAULT_COMBOS = ['base', '1', '2', '1+2', '1+2+5', '4+5+6', '2+4+5+6', '1+2+4+5+6']


def fwd_roll(df, w, how):
    x = df[::-1].rolling(w, min_periods=1)
    x = x.max() if how == 'max' else x.min()
    return x[::-1].shift(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', default='/data/fund.csv')
    ap.add_argument('--out', default='/data/movement_stats')
    ap.add_argument('--combos', default=','.join(DEFAULT_COMBOS), help='comma list, e.g. base,1,1+2,4+5+6')
    ap.add_argument('--test-start', default='2026-01-01')
    ap.add_argument('--min-cap', type=float, default=1e9)
    ap.add_argument('--min-liquidity', type=float, default=5e6)
    ap.add_argument('--min-price', type=float, default=5.0)
    ap.add_argument('--high-col', default='h'); ap.add_argument('--low-col', default='l')
    ap.add_argument('--close-col', default='c'); ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(a.test_start)

    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.high_col} AS h, {a.low_col} AS l,
                 {a.close_col} AS c, {a.volume_col} AS v FROM market_candles_1d''')
    with eng.connect() as cn:
        d = pd.concat(pd.read_sql(q, cn, chunksize=500_000), ignore_index=True)
    d = d[~d.symbol.astype(str).str.endswith('.CA')]
    d['date'] = pd.to_datetime(d.date)
    for k in 'hlcv':
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date'])
    q90 = (d.c * d.v).groupby(d.symbol).quantile(0.9)
    d = d[d.symbol.isin(q90[q90 >= a.min_liquidity].index)]
    P = {k: d.pivot(index='date', columns='symbol', values=k) for k in 'hlcv'}
    C = P['c']; dates, cols = C.index, C.columns
    H, L, V = (P[k].reindex(index=dates, columns=cols) for k in 'hlv')
    cnt = C.notna().sum(axis=1)
    last_full = cnt[cnt >= 0.8 * cnt.tail(30).median()].index.max()
    pos_last = dates.get_loc(last_full)
    print('candles', len(d), '| symbols', len(cols), '| last complete day', last_full.date(), flush=True)

    fund = pd.read_csv(a.fundamentals)
    fund['asof_date'] = pd.to_datetime(fund.asof_date, errors='coerce')
    sh = (fund.dropna(subset=['asof_date']).sort_values('asof_date')
              .drop_duplicates(['symbol', 'asof_date'], keep='last')
              .pivot(index='asof_date', columns='symbol', values='shares_outstanding'))
    sh = sh.reindex(sh.index.union(dates)).sort_index().ffill().reindex(index=dates, columns=cols)
    mcap = sh * C

    split = detect_splits(C, V)
    r = C.pct_change(fill_method=None).mask(split)
    med_dv20 = (C * V).rolling(20, min_periods=15).median()
    hist = C.notna().cumsum()
    split_recent = split.rolling(120, min_periods=1).max().astype(bool)
    base = (mcap.ge(a.min_cap) & med_dv20.ge(a.min_liquidity) & C.ge(a.min_price) & ~split_recent
            & hist.ge(120) & (med_dv20 / mcap * 100).le(TURNOVER_MAX))

    F = {}
    F['vol60'] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    F['atr14_pct'] = tr.rolling(14).mean() / C
    F['from_low60'] = C / L.rolling(60, min_periods=40).min() - 1
    F['min_ret_20'] = r.rolling(20, min_periods=15).min()
    F['max_ret_20'] = r.rolling(20, min_periods=15).max()
    need3 = any('3' in c.split('+') for c in a.combos.split(','))
    if need3:
        grp, _ = load_groups(list(cols))
        gd = np.full(C.shape, np.nan)
        for g, members in grp.groupby(grp).groups.items():
            idx = [cols.get_loc(m) for m in members]
            lvl = (1 + r.iloc[:, idx].mean(axis=1).fillna(0)).cumprod()
            gd[:, idx] = (lvl / lvl.rolling(50).mean() - 1).values[:, None]
        F['grp_dist_sma50'] = pd.DataFrame(gd, index=dates, columns=cols)
    passes = {}
    for i, (_, f, op, thr) in COND.items():
        if f in F:
            passes[i] = ((F[f] >= thr) if op == '>=' else (F[f] <= thr)).fillna(False).values

    # forward measures (from signal close)
    M = {}
    for k in (5, 10, 21, 42):
        M[f'abs{k}'] = ((C.shift(-k) / C - 1).abs() * 100).values
    M['ret21'] = ((C.shift(-21) / C - 1) * 100).values
    M['up21'] = ((fwd_roll(H, 21, 'max') / C - 1) * 100).values
    M['dn21'] = ((fwd_roll(L, 21, 'min') / C - 1) * 100).values
    M['rv21'] = (r[::-1].rolling(21, min_periods=15).std()[::-1].shift(-1) * np.sqrt(252) * 100).values
    M['vol60'] = (F['vol60'] * 100).values
    M['straddle'] = (0.8 * F['vol60'] * np.sqrt(21 / 252) * 100).values
    split42 = fwd_roll(split.astype(float), 42, 'max').fillna(1).astype(bool).values
    ok = (np.arange(len(dates)) <= pos_last - 42)[:, None] & ~split42 & np.isfinite(M['abs42'])
    is_test = (dates >= cut)[:, None]
    B = base.values

    def stats(mask):
        s = mask & ok
        n = int(s.sum())
        if n == 0:
            return {'n': 0}
        g = {k: v[s] for k, v in M.items()}
        o = {'n': n}
        for k in (5, 10, 21, 42):
            x = g[f'abs{k}']
            o[f'abs{k}_med'] = round(float(np.nanmedian(x)), 1)
            o[f'abs{k}_p75'] = round(float(np.nanpercentile(x, 75)), 1)
            o[f'abs{k}_p90'] = round(float(np.nanpercentile(x, 90)), 1)
        o['up21_med'] = round(float(np.nanmedian(g['up21'])), 1)
        o['dn21_med'] = round(float(np.nanmedian(g['dn21'])), 1)
        for t in (20, 30, 50):
            o[f'touch_pm{t}'] = round(float(((g['up21'] >= t) | (g['dn21'] <= -t)).mean() * 100), 1)
            o[f'touch_up{t}'] = round(float((g['up21'] >= t).mean() * 100), 1)
            o[f'touch_dn{t}'] = round(float((g['dn21'] <= -t).mean() * 100), 1)
        o['touch_both20'] = round(float(((g['up21'] >= 20) & (g['dn21'] <= -20)).mean() * 100), 1)
        o['ret21_mean'] = round(float(np.nanmean(g['ret21'])), 2)
        o['ret21_med'] = round(float(np.nanmedian(g['ret21'])), 2)
        o['vol60_med'] = round(float(np.nanmedian(g['vol60'])), 0)
        o['rv21_med'] = round(float(np.nanmedian(g['rv21'])), 0)
        o['rv_over_vol60'] = round(float(np.nanmedian(g['rv21'] / g['vol60'])), 2)
        o['straddle_ratio'] = round(float(np.nanmean(g['abs21']) / np.nanmean(g['straddle'])), 2)
        return o

    rows = []
    for c in a.combos.split(','):
        c = c.strip()
        if c == 'base':
            m = B.copy(); label = 'أي سهم كبير (كل الأيام)'
            sig = m
        else:
            ids = [int(x) for x in c.split('+')]
            m = B.copy()
            for i in ids:
                m &= passes[i]
            prev = np.vstack([np.zeros((1, m.shape[1]), bool), m[:-1]])
            sig = m & ~prev; label = c
        for per, part in (('كل الفترة', np.ones_like(is_test)), ('قبل ٢٠٢٦', ~is_test), ('٢٠٢٦', is_test)):
            rows.append({'combo': label, 'period': per, **stats(sig & part)})
        print('done', label, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / 'movement_stats.csv', index=False)

    A = R[R.period == 'كل الفترة']; T26 = R[R.period == '٢٠٢٦']
    L_ = ['كل الأرقام بالنسبة المئوية من سعر إغلاق يوم الإشارة. الوسيط = نص الحالات فوقه ونصها تحته.',
          'الشروط: ' + ' | '.join(f'{i}= {v[0]}' for i, v in COND.items()), '',
          '=== ١) حجم الحركة في أي اتجاه (|العائد|): وسيط / ٧٥٪ من الحالات أقل من / ٩٠٪ أقل من ===',
          'التوليفة | إشارات | بعد ٥ أيام | بعد ١٠ | بعد ٢١ | بعد ٤٢']
    for x in A.itertuples():
        L_.append(f'{x.combo} | {x.n} | {x.abs5_med}/{x.abs5_p75}/{x.abs5_p90} | {x.abs10_med}/{x.abs10_p75}/{x.abs10_p90} | '
                  f'{x.abs21_med}/{x.abs21_p75}/{x.abs21_p90} | {x.abs42_med}/{x.abs42_p75}/{x.abs42_p90}')
    L_ += ['', '=== ٢) خلال ٢١ يوم: وسيط أقصى طلوع وأقصى نزول، ونسبة اللي لمس الحد (فوق أو تحت / فوق / تحت) ===',
           'التوليفة | أقصى طلوع | أقصى نزول | ±٢٠٪ (فوق/تحت) | ±٣٠٪ (فوق/تحت) | ±٥٠٪ (فوق/تحت) | لمس +٢٠ و −٢٠ الاتنين']
    for x in A.itertuples():
        L_.append(f'{x.combo} | +{x.up21_med}% | {x.dn21_med}% | {x.touch_pm20}% ({x.touch_up20}/{x.touch_dn20}) | '
                  f'{x.touch_pm30}% ({x.touch_up30}/{x.touch_dn30}) | {x.touch_pm50}% ({x.touch_up50}/{x.touch_dn50}) | {x.touch_both20}%')
    L_ += ['', '=== ٣) للأوبشنز: التذبذب قبل وبعد الإشارة (سنوي ٪)، ونسبة الحركة لسعر الستراددل التقريبي ===',
           'التوليفة | تذبذب ٦٠ يوم قبل | تذبذب ٢١ يوم بعد | بعد÷قبل | الحركة÷سعر الستراددل | متوسط عائد ٢١ يوم | وسيط']
    for x in A.itertuples():
        L_.append(f'{x.combo} | {x.vol60_med:.0f}% | {x.rv21_med:.0f}% | {x.rv_over_vol60} | {x.straddle_ratio} | '
                  f'{x.ret21_mean}% | {x.ret21_med}%')
    L_ += ['', '=== ٤) نفس الأرقام الأساسية في ٢٠٢٦ لوحدها (الشروط مشافتهاش) ===',
           'التوليفة | إشارات | وسيط الحركة ٢١ يوم | لمس ±٣٠٪ | لمس +٣٠ / −٣٠ | بعد÷قبل | الحركة÷الستراددل']
    for x in T26.itertuples():
        if x.n:
            L_.append(f'{x.combo} | {x.n} | {x.abs21_med}% | {x.touch_pm30}% | {x.touch_up30}/{x.touch_dn30} | '
                      f'{x.rv_over_vol60} | {x.straddle_ratio}')
    L_.append(f'\nالتفاصيل: {out/"movement_stats.csv"}')
    txt = '\n'.join(L_)
    (out / 'movement_report.txt').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
