#!/usr/bin/env python3
"""
Profile stocks that DOUBLED within a month, build daily screening rules from what
they looked like the day BEFORE, and test the rules every day after the close.

Universe each day (known at that close):
  market cap >= --min-cap, 20d median $ volume >= --min-liquidity, price >= --min-price,
  no split in the last 120 sessions, >= 120 sessions of history.
Label (future, used only for scoring):
  spike = max HIGH over the next --window sessions >= close * (1 + --gain/100)
  (and no split inside that window, so the move is real)
Events = first spike day of each run per stock (then skip --window sessions).

Steps
 1. events.csv                 every doubling event (symbol, day before, cap, max gain)
 2. feature_compare.csv        each indicator: events vs all other stock-days, separation score
 3. Greedy rules (chosen on the FIRST half of days, checked on the SECOND half):
    each step adds the filter that raises the spike rate the most while keeping
    >= --keep-step of the spikes still selected. Stops when the gain is small.
 4. rule_steps.txt / .csv      spike rate, lift, recall at every step, both halves
 5. trade test of the chosen rule set: buy next open, take-profit / stop / max days
 6. live_signals.csv           stocks passing the rules at the last complete day
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

SPLIT_K = [2,3,4,5,8,10,15,20,25,30,40,50,100]
AR = {
    'ret_5': 'عائد ٥ أيام', 'ret_20': 'عائد ٢٠ يوم', 'ret_60': 'عائد ٦٠ يوم', 'ret_120': 'عائد ١٢٠ يوم',
    'dist_sma20': 'البعد عن متوسط ٢٠', 'dist_sma50': 'البعد عن متوسط ٥٠', 'dist_sma100': 'البعد عن متوسط ١٠٠',
    'rsi14': 'القوة النسبية ١٤', 'atr14_pct': 'المدى الحقيقي ١٤ ÷ السعر', 'from_high120': 'البعد عن قمة ١٢٠ يوم',
    'from_low60': 'البعد عن قاع ٦٠ يوم', 'days_since_high120': 'أيام من قمة ١٢٠ يوم', 'bb_width20': 'عرض بولينجر ٢٠',
    'vol20': 'تذبذب ٢٠ يوم', 'vol60': 'تذبذب ٦٠ يوم', 'vol_ratio_20_60': 'تذبذب ٢٠ ÷ ٦٠',
    'dv_trend_20_60': 'قيمة تداول ٢٠ ÷ ٦٠', 'surge_5_60': 'قيمة تداول ٥ ÷ ٦٠', 'surge_1_20': 'تداول اليوم ÷ متوسط ٢٠',
    'up_days_20': 'نسبة أيام الصعود ٢٠', 'close_loc_5': 'مكان الإغلاق في مدى اليوم ٥', 'max_ret_20': 'أكبر صعود يومي ٢٠',
    'min_ret_20': 'أكبر هبوط يومي ٢٠', 'turnover_pct': 'نسبة التداول من القيمة السوقية',
    'mcap_log': 'لوغاريتم القيمة السوقية', 'log_price': 'لوغاريتم السعر', 'beta_mkt60': 'الحركة مع السوق ٦٠',
    'corr_mkt60': 'الارتباط بالسوق ٦٠', 'rel_ret_20': 'العائد ٢٠ يوم ناقص السوق',
}


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


def fwd_max(h, w):
    """max of h over the next w rows (t+1..t+w)."""
    return h[::-1].rolling(w, min_periods=1).max()[::-1].shift(-1)


def sim(sig, o, h, l, c, sp, T, S, H):
    n = len(sig); out = []; nxt = 0
    for s in np.flatnonzero(sig):
        if s < nxt or s + 1 >= n: continue
        e = s + 1; o0 = o[e]
        if not np.isfinite(o0) or o0 <= 0: continue
        stp, tgt = o0 * (1 - S / 100), o0 * (1 + T / 100)
        last = min(e + H - 1, n - 1); ret = None; xd = last
        for d in range(e, last + 1):
            if not (np.isfinite(o[d]) and np.isfinite(h[d]) and np.isfinite(l[d])): continue
            if d > e and o[d] <= stp: ret, xd = o[d] / o0 - 1, d; break
            if d > e and o[d] >= tgt: ret, xd = o[d] / o0 - 1, d; break
            if l[d] <= stp: ret, xd = -S / 100, d; break
            if h[d] >= tgt: ret, xd = T / 100, d; break
        if ret is None:
            if not np.isfinite(c[last]): continue
            ret = c[last] / o0 - 1
        nxt = xd + 1 + H
        if sp[e:xd + 1].any(): continue
        out.append((s, e, xd, ret * 100))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamentals', default='/data/fund.csv')
    ap.add_argument('--out', default='/data/spike_profile')
    ap.add_argument('--min-cap', type=float, default=1e9, help='"large" = market cap at least this')
    ap.add_argument('--min-liquidity', type=float, default=5e6)
    ap.add_argument('--min-price', type=float, default=5.0)
    ap.add_argument('--gain', type=float, default=100, help='spike = +gain%% within window')
    ap.add_argument('--window', type=int, default=21, help='trading days (~1 month)')
    ap.add_argument('--keep-step', type=float, default=0.80, help='each filter must keep this share of spikes')
    ap.add_argument('--min-recall', type=float, default=0.25, help='never keep fewer than this share of all spikes')
    ap.add_argument('--min-lift-gain', type=float, default=0.10, help='stop when lift improves less than 10%%')
    ap.add_argument('--max-steps', type=int, default=8)
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--open-col', default='o'); ap.add_argument('--high-col', default='h')
    ap.add_argument('--low-col', default='l'); ap.add_argument('--close-col', default='c')
    ap.add_argument('--volume-col', default='v')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    fund = pd.read_csv(a.fundamentals)
    fund['asof_date'] = pd.to_datetime(fund.asof_date, errors='coerce')

    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS o, {a.high_col} AS h,
                 {a.low_col} AS l, {a.close_col} AS c, {a.volume_col} AS v FROM market_candles_1d''')
    parts = []
    with eng.connect() as cn:
        for ch in pd.read_sql(q, cn, chunksize=500_000):
            parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    d = d[~d.symbol.astype(str).str.endswith('.CA')]
    d['date'] = pd.to_datetime(d.date)
    for k in 'ohlcv':
        d[k] = pd.to_numeric(d[k], errors='coerce').astype('float64')
    d = d.drop_duplicates(['symbol', 'date'])
    q90 = (d.c * d.v).groupby(d.symbol).quantile(0.9)
    d = d[d.symbol.isin(q90[q90 >= a.min_liquidity].index)]
    print('candles', len(d), 'symbols kept', d.symbol.nunique(), flush=True)

    P = {k: d.pivot(index='date', columns='symbol', values=k) for k in 'ohlcv'}
    C = P['c']; dates, cols = C.index, C.columns
    for k in 'ohlv':
        P[k] = P[k].reindex(index=dates, columns=cols)
    O, Hh, L, V = P['o'], P['h'], P['l'], P['v']
    cnt = C.notna().sum(axis=1)
    last_full = cnt[cnt >= 0.8 * cnt.tail(30).median()].index.max()

    split = detect_splits(C, V)
    r = C.pct_change(fill_method=None).mask(split)
    dv = C * V
    sh = (fund.dropna(subset=['asof_date']).sort_values('asof_date')
              .drop_duplicates(['symbol', 'asof_date'], keep='last')
              .pivot(index='asof_date', columns='symbol', values='shares_outstanding'))
    sh = sh.reindex(sh.index.union(dates)).sort_index().ffill().reindex(index=dates, columns=cols)
    mcap = sh * C
    med_dv20 = dv.rolling(20, min_periods=15).median()
    hist = C.notna().cumsum()
    split_recent = split.rolling(120, min_periods=1).max().astype(bool)
    eligible = (mcap.ge(a.min_cap) & med_dv20.ge(a.min_liquidity) & C.ge(a.min_price)
                & ~split_recent & hist.ge(120))

    # ---------- label ----------
    fmax = fwd_max(Hh, a.window)
    split_ahead = fwd_max(split.astype(float), a.window).fillna(0).astype(bool)
    pos_last = dates.get_loc(last_full)
    row_ok = np.arange(len(dates)) <= pos_last - a.window       # full future window available
    has_future = pd.DataFrame(np.broadcast_to(row_ok[:, None], C.shape), index=dates, columns=cols)
    gain = (fmax / C - 1) * 100
    spike = gain.ge(a.gain) & ~split_ahead
    labeled = eligible & has_future & ~split_ahead

    # ---------- indicators (all known at the close of t) ----------
    mkt = r.where(eligible).mean(axis=1)
    F = {}
    for k_ in [5, 20, 60, 120]:
        F[f'ret_{k_}'] = C / C.shift(k_) - 1
    for k_ in [20, 50, 100]:
        F[f'dist_sma{k_}'] = C / C.rolling(k_).mean() - 1
    up = r.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    F['rsi14'] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    pc = C.shift(1)
    tr = np.maximum(Hh - L, np.maximum((Hh - pc).abs(), (L - pc).abs()))
    F['atr14_pct'] = tr.rolling(14).mean() / C
    hi120 = Hh.rolling(120, min_periods=60).max()
    F['from_high120'] = C / hi120 - 1
    F['from_low60'] = C / L.rolling(60, min_periods=40).min() - 1
    F['days_since_high120'] = Hh.rolling(120, min_periods=60).apply(lambda x: len(x) - 1 - np.nanargmax(x), raw=True)
    F['bb_width20'] = 4 * C.rolling(20).std() / C.rolling(20).mean()
    F['vol20'] = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    F['vol60'] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    F['vol_ratio_20_60'] = F['vol20'] / F['vol60']
    F['dv_trend_20_60'] = med_dv20 / dv.rolling(60, min_periods=40).median()
    F['surge_5_60'] = dv.rolling(5).mean() / dv.rolling(60, min_periods=40).median()
    F['surge_1_20'] = dv / dv.rolling(20, min_periods=15).mean().shift(1)
    F['up_days_20'] = (r > 0).astype(float).where(r.notna()).rolling(20, min_periods=15).mean()
    F['close_loc_5'] = ((C - L) / (Hh - L).replace(0, np.nan)).rolling(5).mean()
    F['max_ret_20'] = r.rolling(20, min_periods=15).max()
    F['min_ret_20'] = r.rolling(20, min_periods=15).min()
    F['turnover_pct'] = med_dv20 / mcap * 100
    F['mcap_log'] = np.log10(mcap)
    F['log_price'] = np.log10(C)
    mx = mkt.rolling(60, min_periods=40).mean(); vx = mkt.rolling(60, min_periods=40).var()
    my = r.rolling(60, min_periods=40).mean()
    mxy = r.mul(mkt, axis=0).rolling(60, min_periods=40).mean()
    cov = mxy - my.mul(mx, axis=0)
    F['beta_mkt60'] = cov.div(vx, axis=0)
    F['corr_mkt60'] = cov.div((F['vol60'] / np.sqrt(252)).mul(np.sqrt(vx), axis=0))
    mlvl = (1 + mkt.fillna(0)).cumprod()
    F['rel_ret_20'] = F['ret_20'].sub(mlvl / mlvl.shift(20) - 1, axis=0)
    feats = list(F)
    for k_ in feats:
        F[k_] = F[k_].astype('float32')

    # ---------- long table of labeled stock-days ----------
    mask = labeled.values
    ii, jj = np.nonzero(mask)
    T = pd.DataFrame({'date': dates[ii], 'symbol': cols[jj], 'spike': spike.values[ii, jj],
                      'gain_fwd': gain.values[ii, jj], 'mcap': mcap.values[ii, jj]})
    for f in feats:
        T[f] = F[f].values[ii, jj].astype('float32')
    T = T.dropna(subset=feats, thresh=len(feats) - 3)
    days = np.sort(T.date.unique()); cut = days[len(days) // 2]
    T['half'] = np.where(T.date < cut, 1, 2)
    print(f'labeled stock-days {len(T)} | spike days {int(T.spike.sum())} '
          f'| base rate {T.spike.mean():.4%} | split day {pd.Timestamp(cut).date()}', flush=True)

    # Events: first spike day per run
    ev = []
    for s, g in T[T.spike].sort_values('date').groupby('symbol'):
        last = None
        for dt, row in zip(g.date, g.itertuples()):
            if last is None or dates.get_loc(dt) - dates.get_loc(last) > a.window:
                ev.append(row); last = dt
    E = pd.DataFrame(ev).drop(columns='Index')
    E.sort_values('gain_fwd', ascending=False).to_csv(out / 'events.csv', index=False)
    print(f'doubling events (first day of each run): {len(E)} in {E.symbol.nunique()} stocks', flush=True)

    # ---------- feature comparison ----------
    rows = []
    ctrl = T[~T.spike]
    for f in feats:
        x1, x0 = E[f].dropna(), ctrl[f].dropna()
        if len(x1) < 5: continue
        # AUC via ranks: probability an event value > a control value
        allv = np.concatenate([x1.values, x0.sample(min(len(x0), 200000), random_state=0).values])
        rk = pd.Series(allv).rank().values
        auc = (rk[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * (len(allv) - len(x1)))
        rows.append({'feature': f, 'name_ar': AR.get(f, f), 'events_median': x1.median(),
                     'others_median': x0.median(), 'events_p25': x1.quantile(.25), 'events_p75': x1.quantile(.75),
                     'separation_auc': auc, 'strength': abs(auc - 0.5) * 2})
    FC = pd.DataFrame(rows).sort_values('strength', ascending=False)
    FC.to_csv(out / 'feature_compare.csv', index=False)

    # ---------- greedy rules on stock-days (first half), checked on second half ----------
    keep = pd.Series(True, index=T.index)
    s1 = T.half == 1
    tot_sp1 = T.spike[s1].sum()

    def st(m):
        o = {}
        for h in (1, 2):
            sub = T[m & (T.half == h)]; allh = T[T.half == h]
            rate = sub.spike.mean() if len(sub) else np.nan
            o[f'h{h}_days'] = len(sub); o[f'h{h}_spikes'] = int(sub.spike.sum())
            o[f'h{h}_rate_pct'] = round(rate * 100, 3) if len(sub) else np.nan
            o[f'h{h}_lift'] = round(rate / allh.spike.mean(), 2) if len(sub) else np.nan
            o[f'h{h}_recall_pct'] = round(sub.spike.sum() / allh.spike.sum() * 100, 1)
            o[f'h{h}_fwd50_pct'] = round((sub.gain_fwd >= 50).mean() * 100, 1) if len(sub) else np.nan
        return o

    steps = [{'step': 0, 'rule': 'كل الأسهم الكبيرة السائلة', **st(keep)}]
    used = set(); chosen = []
    for step in range(1, a.max_steps + 1):
        cur = T[keep & s1]; cur_rate = cur.spike.mean(); cur_sp = cur.spike.sum()
        best = None
        for f in feats:
            if f in used: continue
            ev_vals = cur.loc[cur.spike, f].dropna()
            if len(ev_vals) < 10: continue
            for qv in np.arange(0.05, 0.51, 0.05):
                for op, thr in (('>=', ev_vals.quantile(qv)), ('<=', ev_vals.quantile(1 - qv))):
                    m = (T[f] >= thr) if op == '>=' else (T[f] <= thr)
                    new = keep & m.fillna(False)
                    sub = T[new & s1]
                    if sub.spike.sum() < a.keep_step * cur_sp: continue
                    if sub.spike.sum() < a.min_recall * tot_sp1: continue
                    rate = sub.spike.mean()
                    if best is None or rate > best[0]:
                        best = (rate, f, op, thr, new)
        if best is None or best[0] < cur_rate * (1 + a.min_lift_gain):
            break
        rate, f, op, thr, new = best
        keep = new; used.add(f); chosen.append((f, op, float(thr)))
        steps.append({'step': step, 'rule': f'{AR.get(f, f)} {op} {thr:.4g}  [{f}]', **st(keep)})
        print(f'step {step}: {f} {op} {thr:.4g} | h1 rate {rate:.3%} | h2 {steps[-1]["h2_rate_pct"]}% '
              f'lift {steps[-1]["h2_lift"]}', flush=True)
    S = pd.DataFrame(steps); S.to_csv(out / 'rule_steps.csv', index=False)
    S2 = S[S.h2_spikes >= 5]
    best_step = int(S2.loc[S2.h2_lift.idxmax(), 'step']) if len(S2) else 0

    # ---------- rebuild chosen rules on the full daily grid + trade test + live ----------
    sel = eligible.copy()
    for f, op, thr in chosen[:best_step]:
        sel &= (F[f] >= thr) if op == '>=' else (F[f] <= thr)
    sel_eval = sel & (pd.Series(dates, index=dates) <= last_full).values[:, None]
    trades_rows = []
    for (Tg, Sg, Hd) in [(50, 15, 21), (100, 20, 21), (30, 10, 21), (50, 10, 10)]:
        tr_ = []
        for j in np.flatnonzero(sel_eval.values.any(axis=0)):
            for s_, e_, x_, ret in sim(sel_eval.values[:, j], O.values[:, j], Hh.values[:, j], L.values[:, j],
                                       C.values[:, j], split.values[:, j], Tg, Sg, Hd):
                tr_.append((dates[s_], ret - a.cost))
        t = pd.DataFrame(tr_, columns=['signal_date', 'ret'])
        if t.empty: continue
        h1, h2 = t[t.signal_date < cut], t[t.signal_date >= cut]
        trades_rows.append({'target': Tg, 'stop': Sg, 'max_days': Hd, 'trades': len(t),
                            'win_pct': round((t.ret > 0).mean() * 100, 1), 'mean_trade': round(t.ret.mean(), 2),
                            'h1_trades': len(h1), 'h1_mean': round(h1.ret.mean(), 2),
                            'h2_trades': len(h2), 'h2_mean': round(h2.ret.mean(), 2)})
    TR = pd.DataFrame(trades_rows); TR.to_csv(out / 'rule_trades.csv', index=False)

    lv = sel.loc[last_full]
    live = pd.DataFrame({'symbol': cols[lv.values]})
    live.insert(0, 'signal_date', last_full.date())
    for f in feats:
        live[f] = F[f].loc[last_full, lv].values
    live['mcap'] = mcap.loc[last_full, lv].values
    live.to_csv(out / 'live_signals.csv', index=False)

    # ---------- report ----------
    L_ = [f'الأسهم: قيمة سوقية ≥ {a.min_cap/1e9:g} مليار، سيولة ≥ {a.min_liquidity/1e6:g} مليون/يوم، سعر ≥ {a.min_price:g}، بدون تقسيمات',
          f'الطفرة: +{a.gain:g}% خلال {a.window} يوم تداول | النص الأول قبل {pd.Timestamp(cut).date()}',
          f'عدد الطفرات: {len(E)} في {E.symbol.nunique()} سهم', '',
          '--- أكتر ١٢ مؤشر بيميز أسهم الطفرة (قبلها بيوم) ---']
    for r_ in FC.head(12).itertuples():
        L_.append(f'{r_.name_ar}: الطفرات {r_.events_median:.3g} مقابل الباقي {r_.others_median:.3g} | قوة التمييز {r_.strength:.2f}')
    L_.append('\n--- خطوات الشروط ---')
    for r_ in S.itertuples():
        L_.append(f'خطوة {r_.step}: {r_.rule}')
        L_.append(f'   النص الأول: نسبة الطفرة {r_.h1_rate_pct}% (×{r_.h1_lift}) يلقط {r_.h1_recall_pct}% منها | '
                  f'النص التاني: {r_.h2_rate_pct}% (×{r_.h2_lift}) يلقط {r_.h2_recall_pct}% | '
                  f'طلع +50%: {r_.h2_fwd50_pct}% | أيام سهم: {r_.h2_days}')
    L_.append(f'\nالخطوة المختارة (أحسن على النص التاني): {best_step}')
    L_.append('\n--- تداول بالشروط المختارة (شراء افتتاح اليوم التالي) ---')
    L_.append(TR.to_string(index=False) if len(TR) else 'لا صفقات')
    L_.append(f'\nإشارات {last_full.date()} (شراء افتتاح اليوم التالي): ' + (', '.join(live.symbol) or 'لا يوجد'))
    txt = '\n'.join(L_)
    (out / 'rule_steps.txt').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
