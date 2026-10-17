#!/usr/bin/env python3
"""
Greedy filter search on the fingerprint-flagged trades.

1. Rebuild the 582 trades (flagged.csv) and score each with the exit rule
   (default: sell all at +30%, stop -10%, entry = first open of the month).
2. Compute ~20 extra indicators per trade using ONLY data up to the as-of close.
3. Compute the same indicators for the 14 basket stocks at every month-end.
   A filter is allowed only if the basket "agrees" with it: at least
   --basket-keep of basket stock-months pass it.
4. Forward selection: at each step add the single filter (indicator, >= or <=,
   threshold) that most improves total profit of kept trades in the SELECTION
   half (first half of months). Each step is also scored on the HOLDOUT half
   (second half), which is never used to choose filters.
5. Stop when no filter improves the selection half by --min-gain, or when fewer
   than --min-trades would remain. Recommended step = best holdout result.

Outputs in --out:
  trades_with_indicators.csv   every trade, all indicators, outcome
  basket_indicators.csv        indicators of the 14 at each month-end
  filter_steps.csv             each step: filter, trades/winners/losers kept, both halves
  filter_steps.txt             readable report
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASKET = ['RGTI','QBTS','IONQ','SOUN','BBAI','RKLB','ASTS','LUNR','ACHR',
          'OKLO','SMR','IREN','CIFR','HIMS']

NAMES_AR = {
    'ret_5': 'عائد آخر ٥ أيام', 'ret_20': 'عائد آخر ٢٠ يوم', 'ret_60': 'عائد آخر ٦٠ يوم',
    'dist_sma20': 'البعد عن متوسط ٢٠ يوم', 'dist_sma50': 'البعد عن متوسط ٥٠ يوم',
    'rsi14': 'مؤشر القوة النسبية ١٤', 'atr14_pct': 'متوسط المدى الحقيقي ١٤ كنسبة من السعر',
    'from_high120': 'البعد عن أعلى سعر ١٢٠ يوم', 'from_low20': 'البعد عن أقل سعر ٢٠ يوم',
    'days_since_high60': 'أيام من أعلى سعر ٦٠ يوم', 'bb_width20': 'عرض نطاق بولينجر ٢٠',
    'vol_ratio_20_60': 'تذبذب ٢٠ يوم ÷ تذبذب ٦٠ يوم', 'dv_trend_20_60': 'قيمة تداول ٢٠ ÷ ٦٠ يوم',
    'surge_5_60': 'قيمة تداول ٥ أيام ÷ ٦٠ يوم', 'up_days_20': 'نسبة أيام الصعود من ٢٠',
    'close_loc_5': 'مكان الإغلاق في مدى اليوم (متوسط ٥ أيام)', 'max_ret_20': 'أكبر صعود يومي في ٢٠ يوم',
    'min_ret_20': 'أكبر هبوط يومي في ٢٠ يوم', 'corr_basket60': 'الارتباط بالسلة ٦٠ يوم',
    'basket_ret20': 'عائد السلة آخر ٢٠ يوم', 'basket_dist_sma50': 'بعد السلة عن متوسط ٥٠ يوم',
    'log_price': 'لوغاريتم السعر', 'turnover_pct': 'نسبة التداول من القيمة السوقية',
    'beta_basket': 'معامل الحركة مع السلة', 'vol60': 'تذبذب ٦٠ يوم', 'mcap_log': 'لوغاريتم القيمة السوقية',
}


def simulate(days, target, stop):
    o0 = days.open.iloc[0]
    if not np.isfinite(o0) or o0 <= 0:
        return np.nan
    sp = o0 * (1 - stop / 100); tp = o0 * (1 + target / 100)
    for op, hi, lo, cl in days[['open', 'high', 'low', 'close']].itertuples(index=False):
        if op <= sp: return (op / o0 - 1) * 100
        if op >= tp: return (op / o0 - 1) * 100
        if lo <= sp: return -stop                      # stop assumed first (conservative)
        if hi >= tp: return target
    return (days.close.iloc[-1] / o0 - 1) * 100


def indicators(g, bidx):
    """g: one symbol's daily frame indexed by date (open, high, low, close, volume)."""
    c, h, l, v = g.close, g.high, g.low, g.volume
    r = c.pct_change(fill_method=None)
    dv = c * v
    f = pd.DataFrame(index=g.index)
    f['ret_5'] = c / c.shift(5) - 1
    f['ret_20'] = c / c.shift(20) - 1
    f['ret_60'] = c / c.shift(60) - 1
    f['dist_sma20'] = c / c.rolling(20).mean() - 1
    f['dist_sma50'] = c / c.rolling(50).mean() - 1
    up = r.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    f['rsi14'] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f['atr14_pct'] = tr.rolling(14).mean() / c
    f['from_high120'] = c / h.rolling(120, min_periods=60).max() - 1
    f['from_low20'] = c / l.rolling(20).min() - 1
    f['days_since_high60'] = h.rolling(60).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)
    f['bb_width20'] = 4 * c.rolling(20).std() / c.rolling(20).mean()
    f['vol_ratio_20_60'] = r.rolling(20).std() / r.rolling(60).std()
    f['dv_trend_20_60'] = dv.rolling(20).median() / dv.rolling(60).median()
    f['surge_5_60'] = dv.rolling(5).mean() / dv.rolling(60).median()
    f['up_days_20'] = (r > 0).rolling(20).mean()
    f['close_loc_5'] = ((c - l) / (h - l).replace(0, np.nan)).rolling(5).mean()
    f['max_ret_20'] = r.rolling(20).max()
    f['min_ret_20'] = r.rolling(20).min()
    br = bidx.reindex(g.index)
    f['corr_basket60'] = r.rolling(60, min_periods=40).corr(br)
    f['log_price'] = np.log(c)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', default='/data/fingerprint_audit', help='folder with flagged.csv and features_all.csv')
    ap.add_argument('--target', type=float, default=30)
    ap.add_argument('--stop', type=float, default=10)
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--basket-keep', type=float, default=0.70,
                    help='a filter must keep at least this share of basket stock-months')
    ap.add_argument('--min-trades', type=int, default=120, help='never keep fewer trades than this (all months)')
    ap.add_argument('--min-gain', type=float, default=0.03,
                    help='stop when best filter improves selection-half profit by less than this share')
    ap.add_argument('--max-steps', type=int, default=8)
    ap.add_argument('--open-col', default='o'); ap.add_argument('--high-col', default='h')
    ap.add_argument('--low-col', default='l'); ap.add_argument('--close-col', default='c')
    ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.audit)

    fl = pd.read_csv(out / 'flagged.csv')
    fa = pd.read_csv(out / 'features_all.csv', usecols=lambda c: c in {
        'symbol', 'month', 'asof', 'in_basket', 'live', 'turnover_pct', 'beta_basket', 'vol60', 'mcap'})
    for df in (fl, fa):
        df['asof'] = pd.to_datetime(df['asof'])
    bk = fa[(fa.in_basket == True) & (fa.live == False)].copy()

    syms = sorted(set(fl.symbol) | set(BASKET))
    start = (fl['asof'].min() - pd.Timedelta(days=260)).strftime('%Y-%m-%d')
    end = (pd.to_datetime(fl.month.max() + '-01') + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS open, {a.high_col} AS high,
                 {a.low_col} AS low, {a.close_col} AS close, {a.volume_col} AS volume
                 FROM market_candles_1d WHERE session_date BETWEEN :s AND :e''')
    with eng.connect() as c:
        d = pd.read_sql(q, c, params={'s': start, 'e': end})
    d = d[d.symbol.isin(syms)].copy()
    d['date'] = pd.to_datetime(d.date)
    for k in ['open', 'high', 'low', 'close', 'volume']:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date']).sort_values(['symbol', 'date'])
    by = {s: g.set_index('date') for s, g in d.groupby('symbol')}
    print('trades', len(fl), 'symbols loaded', len(by), flush=True)

    # Basket index (equal-weight daily return) and its trend
    bc = pd.DataFrame({s: by[s].close for s in BASKET if s in by})
    bidx = bc.pct_change(fill_method=None).clip(-0.5, 1.0).mean(axis=1)
    blvl = (1 + bidx.fillna(0)).cumprod()
    b_ret20 = blvl / blvl.shift(20) - 1
    b_dist50 = blvl / blvl.rolling(50).mean() - 1

    ind_cache = {s: indicators(by[s], bidx) for s in syms if s in by}

    def attach(rows):
        recs = []
        for r in rows.itertuples():
            f = ind_cache.get(r.symbol)
            if f is None: recs.append({}); continue
            sub = f.loc[:r.asof]
            recs.append(sub.iloc[-1].to_dict() if len(sub) else {})
        x = pd.DataFrame(recs, index=rows.index)
        x['basket_ret20'] = rows['asof'].map(lambda t: b_ret20.loc[:t].iloc[-1]).values
        x['basket_dist_sma50'] = rows['asof'].map(lambda t: b_dist50.loc[:t].iloc[-1]).values
        return x

    # Trades: outcome + indicators
    res = []
    for r in fl.itertuples():
        g = by.get(r.symbol)
        ms = pd.Timestamp(r.month + '-01'); me = ms + pd.offsets.MonthEnd(0)
        days = None if g is None else g.loc[(g.index >= ms) & (g.index <= me)].dropna(subset=['open', 'high', 'low', 'close'])
        res.append(np.nan if days is None or len(days) < 3 else simulate(days, a.target, a.stop) - a.cost)
    fl['ret'] = res
    fl = fl.dropna(subset=['ret']).copy()
    X = attach(fl)
    for k in ['turnover_pct', 'beta_basket', 'vol60']:
        X[k] = fl[k]
    X['mcap_log'] = np.log(fl['mcap'])
    trades = pd.concat([fl[['symbol', 'month', 'asof', 'ret']], X], axis=1)
    trades['win'] = trades.ret > 0
    trades.to_csv(out / 'trades_with_indicators.csv', index=False)

    BX = attach(bk)
    for k in ['turnover_pct', 'beta_basket', 'vol60']:
        BX[k] = bk[k]
    BX['mcap_log'] = np.log(bk['mcap'])
    basket = pd.concat([bk[['symbol', 'month', 'asof']], BX], axis=1)
    basket.to_csv(out / 'basket_indicators.csv', index=False)

    feats = [c for c in NAMES_AR if c in trades.columns]
    months = sorted(trades.month.unique()); cut = months[len(months) // 2]
    sel = trades.month < cut; hold = ~sel

    def stats(mask):
        t = trades[mask]
        s1, s2 = t[t.month < cut], t[t.month >= cut]
        m = t.groupby('month').ret.mean()
        return {'trades': len(t), 'winners': int(t.win.sum()), 'losers': int((~t.win).sum()),
                'win_pct': round(t.win.mean() * 100, 1) if len(t) else np.nan,
                'mean_trade': round(t.ret.mean(), 2), 'total_profit': round(t.ret.sum(), 1),
                'months_positive': int((m > 0).sum()), 'months': len(m),
                'worst_month': round(m.min(), 1) if len(m) else np.nan,
                'sel_mean': round(s1.ret.mean(), 2), 'sel_total': round(s1.ret.sum(), 1),
                'hold_mean': round(s2.ret.mean(), 2), 'hold_total': round(s2.ret.sum(), 1),
                'hold_trades': len(s2)}

    keep = pd.Series(True, index=trades.index)
    bkeep = pd.Series(True, index=basket.index)
    base = stats(keep)
    steps = [{'step': 0, 'filter': 'بدون فلاتر إضافية', 'basket_pass_pct': 100.0,
              'winners_removed': 0, 'losers_removed': 0, **base}]
    used = set()
    for step in range(1, a.max_steps + 1):
        cur_sel = trades.ret[keep & sel].sum()
        best = None
        for f in feats:
            if f in used: continue
            x = trades[f]; xb = basket[f]
            qs = x[keep & sel].quantile(np.arange(0.05, 0.96, 0.05)).unique()
            for thr in qs:
                for op in ('>=', '<='):
                    m = (x >= thr) if op == '>=' else (x <= thr)
                    mb = (xb >= thr) if op == '>=' else (xb <= thr)
                    bpass = mb[bkeep].mean()
                    if not np.isfinite(bpass) or bpass < a.basket_keep: continue
                    new = keep & m.fillna(False)
                    if new.sum() < a.min_trades: continue
                    gain = trades.ret[new & sel].sum() - cur_sel
                    if best is None or gain > best[0]:
                        best = (gain, f, op, thr, new, bkeep & mb.fillna(False), bpass)
        if best is None or best[0] <= abs(cur_sel) * a.min_gain or best[0] <= 0:
            break
        gain, f, op, thr, new, newb, bpass = best
        removed = keep & ~new
        st = stats(new)
        steps.append({'step': step, 'filter': f'{NAMES_AR.get(f, f)} {op} {thr:.4g}  [{f}]',
                      'basket_pass_pct': round(bpass * 100, 1),
                      'winners_removed': int((removed & trades.win).sum()),
                      'losers_removed': int((removed & ~trades.win).sum()), **st})
        keep, bkeep = new, newb; used.add(f)
        print(f'step {step}: {f} {op} {thr:.4g}  trades={st["trades"]} sel_mean={st["sel_mean"]} '
              f'hold_mean={st["hold_mean"]}', flush=True)

    S = pd.DataFrame(steps)
    S.to_csv(out / 'filter_steps.csv', index=False)
    best_step = int(S.loc[S.hold_mean.idxmax(), 'step'])
    lines = [f'قاعدة الخروج: بيع الكل عند +{a.target:g}% ووقف -{a.stop:g}% وتكلفة {a.cost:g}%',
             f'نص الاختيار: الشهور قبل {cut} | نص التأكيد: من {cut} وبعدها', '']
    for r in S.itertuples():
        lines.append(f'خطوة {r.step}: {r.filter}')
        lines.append(f'   الصفقات {r.trades} (كسبانة {r.winners} / خسرانة {r.losers})  نسبة الكسب {r.win_pct}%  '
                     f'| اتشال: كسبانة {r.winners_removed} / خسرانة {r.losers_removed}  | السلة بتعدّي {r.basket_pass_pct}%')
        lines.append(f'   متوسط الصفقة: الكل {r.mean_trade}%  | نص الاختيار {r.sel_mean}%  | نص التأكيد {r.hold_mean}% '
                     f'({r.hold_trades} صفقة)  | شهور كسبانة {r.months_positive}/{r.months}  | أسوأ شهر {r.worst_month}%')
    lines.append('')
    lines.append(f'أحسن خطوة على نص التأكيد: {best_step}  (اللي بعدها مش بيحسّن على بيانات مشافهاش)')
    txt = '\n'.join(lines)
    (out / 'filter_steps.txt').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
