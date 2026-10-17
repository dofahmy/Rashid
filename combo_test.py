#!/usr/bin/env python3
"""
Test every combination (3..6 of 6) of the spike conditions, with the base filter and
condition 7 always on.

Base (always): market cap >= 1B, 20d median $ volume >= 5M, price >= 5,
               no split in last 120 sessions, >= 120 sessions history
Cond 7 (always): 20d median $ volume / market cap <= 13.39%
Conditions combined:
  1  60-day volatility            >= 0.6786
  2  ATR14 / price                >= 0.06805
  3  industry index vs 50d avg    >= -0.01814
  4  price vs 60-day low          >= +49.44%
  5  worst daily move (20d)       <= -8.03%
  6  best daily move (20d)        >= +15.24%

A "signal" = first day a stock passes the combination (it did not pass the day before).
For each signal: did the HIGH reach +100% (and +50%) above the signal-day close within
the next 21 trading days, and after how many days. Signals with a split inside that
window, or without 21 future days of data, are not scored.
Train = before --test-start, test = from --test-start (2026).

Outputs in --out: combo_results.csv, combo_report.txt, combo_hits.csv
"""
import argparse, itertools, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from spike_profile import detect_splits, load_groups

COND = {
    1: ('تذبذب ٦٠ يوم ≥ ٦٨٪', 'vol60', '>=', 0.6786),
    2: ('المدى الحقيقي ≥ ٦٫٨٪ من السعر', 'atr14_pct', '>=', 0.06805),
    3: ('الصناعة مش تحت متوسط ٥٠ بأكتر من ١٫٨٪', 'grp_dist_sma50', '>=', -0.01814),
    4: ('فوق قاع ٦٠ يوم بـ٤٩٪+', 'from_low60', '>=', 0.4944),
    5: ('أكبر نزول يومي ٢٠ يوم ≤ −٨٪', 'min_ret_20', '<=', -0.08032),
    6: ('أكبر صعود يومي ٢٠ يوم ≥ +١٥٪', 'max_ret_20', '>=', 0.1524),
}
TURNOVER_MAX = 13.39


def first_hit_days(Hv, Cv, gain, window):
    """days (1..window) until HIGH >= close*(1+gain); 0 = never."""
    n = Hv.shape[0]
    out = np.zeros(Hv.shape, dtype=np.int16)
    tgt = Cv * (1 + gain)
    for k in range(window, 0, -1):            # reverse so the smallest k wins
        hk = np.full(Hv.shape, np.nan)
        hk[:n - k] = Hv[k:]
        out = np.where(hk >= tgt, k, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', default='/data/fund.csv')
    ap.add_argument('--out', default='/data/combo_test')
    ap.add_argument('--test-start', default='2026-01-01')
    ap.add_argument('--window', type=int, default=21)
    ap.add_argument('--min-cap', type=float, default=1e9)
    ap.add_argument('--min-liquidity', type=float, default=5e6)
    ap.add_argument('--min-price', type=float, default=5.0)
    ap.add_argument('--min-k', type=int, default=3, help='smallest number of conditions in a combination')
    ap.add_argument('--hold', type=int, default=42, help='holding period in trading days (no stop, no target)')
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--open-col', default='o')
    ap.add_argument('--high-col', default='h'); ap.add_argument('--low-col', default='l')
    ap.add_argument('--close-col', default='c'); ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(a.test_start)

    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS o, {a.high_col} AS h, {a.low_col} AS l,
                 {a.close_col} AS c, {a.volume_col} AS v FROM market_candles_1d''')
    with eng.connect() as cn:
        d = pd.concat(pd.read_sql(q, cn, chunksize=500_000), ignore_index=True)
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
    base = (mcap.ge(a.min_cap) & med_dv20.ge(a.min_liquidity) & C.ge(a.min_price) & ~split_recent & hist.ge(120))
    turnover = med_dv20 / mcap * 100
    base &= turnover.le(TURNOVER_MAX)

    F = {}
    F['vol60'] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    F['atr14_pct'] = tr.rolling(14).mean() / C
    F['from_low60'] = C / L.rolling(60, min_periods=40).min() - 1
    F['min_ret_20'] = r.rolling(20, min_periods=15).min()
    F['max_ret_20'] = r.rolling(20, min_periods=15).max()
    grp, _ = load_groups(list(cols))
    gd = np.full(C.shape, np.nan)
    for g, members in grp.groupby(grp).groups.items():
        idx = [cols.get_loc(m) for m in members]
        lvl = (1 + r.iloc[:, idx].mean(axis=1).fillna(0)).cumprod()
        gd[:, idx] = (lvl / lvl.rolling(50).mean() - 1).values[:, None]
    F['grp_dist_sma50'] = pd.DataFrame(gd, index=dates, columns=cols)
    passes = {}
    for i, (_, f, op, thr) in COND.items():
        passes[i] = ((F[f] >= thr) if op == '>=' else (F[f] <= thr)).fillna(False).values

    # labels
    Hv, Cv = H.values, C.values
    d100 = first_hit_days(Hv, Cv, 1.0, a.window)
    d50 = first_hit_days(Hv, Cv, 0.5, a.window)
    split_ahead = split.astype(float)[::-1].rolling(a.window, min_periods=1).max()[::-1].shift(-1).fillna(1).astype(bool).values
    rows_ok = (np.arange(len(dates)) <= pos_last - a.window)[:, None]
    scorable = rows_ok & ~split_ahead
    is_test = (dates >= cut)[:, None]
    B = base.values
    # hold-for-N-days return: buy next open, sell at the close of the N-th trading day, minus cost
    Hn = a.hold
    entry = O.shift(-1).values; exitp = C.shift(-Hn).values
    retN = (exitp / entry - 1) * 100 - a.cost
    split_holdN = split.astype(float)[::-1].rolling(Hn, min_periods=1).max()[::-1].shift(-1).fillna(1).astype(bool).values
    okN = (np.arange(len(dates)) <= pos_last - Hn)[:, None] & ~split_holdN & np.isfinite(retN)

    def hold_stats(mask):
        x = retN[mask & okN]
        if len(x) == 0:
            return dict(n=0, mean=np.nan, median=np.nan, win=np.nan, up50=np.nan, dn50=np.nan, best=np.nan, worst=np.nan)
        return dict(n=len(x), mean=round(float(x.mean()), 2), median=round(float(np.median(x)), 2),
                    win=round(float((x > 0).mean() * 100), 1), up50=round(float((x >= 50).mean() * 100), 1),
                    dn50=round(float((x <= -50).mean() * 100), 1), best=round(float(x.max()), 1),
                    worst=round(float(x.min()), 1))

    res, hit_rows = [], []
    combos = [c for k in range(a.min_k, 7) for c in itertools.combinations(range(1, 7), k)]
    for combo in combos:
        m = B.copy()
        for i in combo:
            m &= passes[i]
        prev = np.vstack([np.zeros((1, m.shape[1]), bool), m[:-1]])
        new = m & ~prev
        sc = new & scorable
        row = {'combo': '+'.join(map(str, combo)), 'n_conditions': len(combo),
               'stock_days': int(m[rows_ok[:, 0]].sum()),
               'distinct_stocks': int(m[rows_ok[:, 0]].any(axis=0).sum())}
        for name, part in (('all', np.ones_like(is_test)), ('train', ~is_test), ('test', is_test)):
            s = sc & part
            n = int(s.sum()); h = s & (d100 > 0); h5 = s & (d50 > 0)
            row[f'{name}_signals'] = n
            row[f'{name}_hits100'] = int(h.sum())
            row[f'{name}_rate100_pct'] = round(h.sum() / n * 100, 2) if n else np.nan
            row[f'{name}_hits50'] = int(h5.sum())
            row[f'{name}_rate50_pct'] = round(h5.sum() / n * 100, 2) if n else np.nan
            if name == 'all':
                dd = d100[h]; dd5 = d50[h5]
                row['median_days_to_100'] = float(np.median(dd)) if len(dd) else np.nan
                row['median_days_to_50'] = float(np.median(dd5)) if len(dd5) else np.nan
                ii, jj = np.nonzero(h)
                for x, y in zip(ii, jj):
                    hit_rows.append({'combo': row['combo'], 'signal_date': dates[x].date(), 'symbol': cols[y],
                                     'days_to_100': int(d100[x, y]), 'period': 'test' if dates[x] >= cut else 'train'})
        for name, part in (('all', np.ones_like(is_test)), ('train', ~is_test), ('test', is_test)):
            for k_, v_ in hold_stats(new & part).items():
                row[f'hold_{name}_{k_}'] = v_
        res.append(row)
    R = pd.DataFrame(res)
    bench = {nm: hold_stats(B & pt) for nm, pt in (('all', np.ones_like(is_test)), ('train', ~is_test), ('test', is_test))}
    # baseline: base + cond 7 only
    nb = B & scorable                       # every base stock-day (a random day in a random big stock)
    base_rate = (nb & (d100 > 0)).sum() / max(1, nb.sum()) * 100
    base_rate_t = (nb & is_test & (d100 > 0)).sum() / max(1, (nb & is_test).sum()) * 100
    R['test_lift'] = (R.test_rate100_pct / base_rate_t).round(1) if base_rate_t else np.nan
    R = R.sort_values(['test_rate100_pct', 'all_rate100_pct'], ascending=False)
    R.to_csv(out / 'combo_results.csv', index=False)
    pd.DataFrame(hit_rows).to_csv(out / 'combo_hits.csv', index=False)

    lines = ['الأساس ثابت: قيمة سوقية ≥ مليار، سيولة ≥ ٥ مليون، سعر ≥ ٥، بدون تقسيمات، + شرط ٧ (التداول ≤ ١٣٫٣٩٪ من القيمة السوقية)',
             'الشروط: ' + ' | '.join(f'{i}= {v[0]}' for i, v in COND.items()),
             f'الإشارة = أول يوم السهم يعدّي التوليفة | الطفرة = أعلى سعر يوصل +١٠٠٪ (أو +٥٠٪) خلال {a.window} يوم تداول',
             f'الاستنتاج قبل {cut.date()} | الاختبار من {cut.date()} | أي سهم كبير في أي يوم: نسبة الطفرة {base_rate:.2f}% (في ٢٠٢٦ {base_rate_t:.2f}%)', '',
             'التوليفة | أسهم مختلفة | إشارات | طفرات +100% | النسبة | وسيط الأيام | طفرات +50% | النسبة | ٢٠٢٦: إشارات / طفرات / النسبة / أضعاف الأساس']
    for x in R.itertuples():
        lines.append(f'{x.combo:12} | {x.distinct_stocks:>5} | {x.all_signals:>6} | {x.all_hits100:>4} | {x.all_rate100_pct}% | '
                     f'{x.median_days_to_100} | {x.all_hits50:>4} | {x.all_rate50_pct}% | '
                     f'{x.test_signals} / {x.test_hits100} / {x.test_rate100_pct}% / ×{x.test_lift}')
    R2 = R.sort_values('hold_test_mean', ascending=False)
    lines += ['', f'===== الأداء بعد {Hn} يوم تداول لكل الإشارات (شراء افتتاح اليوم التالي، بيع إغلاق اليوم {Hn}، من غير وقف ولا هدف، بعد تكلفة {a.cost:g}%) =====',
              f'المرجع: أي سهم كبير في أي يوم -> كل الفترة: متوسط {bench["all"]["mean"]}% وسيط {bench["all"]["median"]}% كسبان {bench["all"]["win"]}% | '
              f'٢٠٢٦: متوسط {bench["test"]["mean"]}% وسيط {bench["test"]["median"]}% كسبان {bench["test"]["win"]}%',
              'التوليفة | صفقات | متوسط | وسيط | كسبان% | طلع+50% | نزل-50% | أحسن | أسوأ || قبل ٢٠٢٦: متوسط | ٢٠٢٦: صفقات / متوسط / وسيط / كسبان% / الفرق عن السوق']
    for x in R2.itertuples():
        lines.append(f'{x.combo:12} | {x.hold_all_n:>5} | {x.hold_all_mean}% | {x.hold_all_median}% | {x.hold_all_win}% | '
                     f'{x.hold_all_up50}% | {x.hold_all_dn50}% | {x.hold_all_best}% | {x.hold_all_worst}% || '
                     f'{x.hold_train_mean}% | {x.hold_test_n} / {x.hold_test_mean}% / {x.hold_test_median}% / {x.hold_test_win}% / '
                     f'{round(x.hold_test_mean - bench["test"]["mean"], 2) if pd.notna(x.hold_test_mean) else "nan"}%')
    lines.append(f'\nالتفاصيل: {out/"combo_results.csv"} | كل طفرة باسم السهم وتاريخها: {out/"combo_hits.csv"}')
    txt = '\n'.join(lines)
    (out / 'combo_report.txt').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
