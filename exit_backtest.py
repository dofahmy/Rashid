#!/usr/bin/env python3
"""
Day-by-day exit backtest for the fingerprint-flagged trades (flagged.csv).

For every flagged stock-month:
  entry  = OPEN of the first trading day of the month (signal is known at prior close)
  exits  = staged take-profit targets + optional stop-loss (+ optional move stop to
           break-even after the first target), remainder sold at the month's last close
  cost   = --cost %% per trade (round trip)

Same-day ambiguity (low hits the stop AND high hits a target on the same day): the
stop is assumed to come FIRST (conservative). Gaps are handled at the open: an open
below the stop exits there, an open above a target fills there.

Also tests a basket-trend filter known at the as-of close:
  regime_up = equal-weight basket index above its 50-day average

Outputs in --out:
  exit_grid.csv     every plan x stop x breakeven, all trades and regime_up only
  exit_best.txt     top plans, with first-half / second-half check
"""
import argparse, itertools, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASKET = ['RGTI','QBTS','IONQ','SOUN','BBAI','RKLB','ASTS','LUNR','ACHR',
          'OKLO','SMR','IREN','CIFR','HIMS']
PLANS = {
    'hold':              [],
    '30@10 30@20 20@30': [(.3, 10), (.3, 20), (.2, 30)],
    '40@10 30@20':       [(.4, 10), (.3, 20)],
    '50@10 25@20 25@30': [(.5, 10), (.25, 20), (.25, 30)],
    '25@10 25@20 25@30 25@50': [(.25, 10), (.25, 20), (.25, 30), (.25, 50)],
    '33@20 33@30 34@50': [(.33, 20), (.33, 30), (.34, 50)],
    '50@15 50@30':       [(.5, 15), (.5, 30)],
    '100@20':            [(1.0, 20)],
    '100@30':            [(1.0, 30)],
}
STOPS = [None, 8, 10, 15, 20, 25]


def simulate(days, plan, stop, breakeven):
    """days: DataFrame (open, high, low, close) for one trade, chronological."""
    o0 = days.open.iloc[0]
    if not np.isfinite(o0) or o0 <= 0:
        return np.nan
    stop_px = o0 * (1 - stop / 100) if stop else -np.inf
    targets = [(w, o0 * (1 + t / 100)) for w, t in plan]
    filled = [False] * len(targets)
    rest, pnl = 1.0, 0.0
    for op, hi, lo, cl in days[['open', 'high', 'low', 'close']].itertuples(index=False):
        if rest <= 1e-9:
            break
        # 1) gap at the open
        if op <= stop_px:
            pnl += rest * (op / o0 - 1); rest = 0; break
        for k, (w, px) in enumerate(targets):
            if not filled[k] and op >= px:
                pnl += w * (op / o0 - 1); rest -= w; filled[k] = True
                if breakeven: stop_px = max(stop_px, o0)
        # 2) intraday: stop first (conservative), then targets
        if lo <= stop_px:
            pnl += rest * (stop_px / o0 - 1); rest = 0; break
        for k, (w, px) in enumerate(targets):
            if not filled[k] and hi >= px:
                pnl += w * (px / o0 - 1); rest -= w; filled[k] = True
                if breakeven: stop_px = max(stop_px, o0)
    if rest > 1e-9:
        pnl += rest * (days.close.iloc[-1] / o0 - 1)
    return pnl * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flagged', default='/data/fingerprint_audit/flagged.csv')
    ap.add_argument('--out', default='/data/fingerprint_audit')
    ap.add_argument('--cost', type=float, default=1.0)
    ap.add_argument('--open-col', default='o'); ap.add_argument('--high-col', default='h')
    ap.add_argument('--low-col', default='l'); ap.add_argument('--close-col', default='c')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    fl = pd.read_csv(a.flagged)
    fl['asof'] = pd.to_datetime(fl['asof'])
    fl['m_start'] = pd.to_datetime(fl.month + '-01')
    fl['m_end'] = fl.m_start + pd.offsets.MonthEnd(0)
    syms = sorted(set(fl.symbol) | set(BASKET))
    start = (fl['asof'].min() - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
    end = fl.m_end.max().strftime('%Y-%m-%d')

    eng = create_engine(os.environ['DATABASE_URL'])
    q = text(f'''SELECT symbol, session_date AS date, {a.open_col} AS open, {a.high_col} AS high,
                 {a.low_col} AS low, {a.close_col} AS close FROM market_candles_1d
                 WHERE session_date BETWEEN :s AND :e''')
    with eng.connect() as c:
        d = pd.read_sql(q, c, params={'s': start, 'e': end})
    d = d[d.symbol.isin(syms)].copy()
    d['date'] = pd.to_datetime(d.date)
    for k in ['open', 'high', 'low', 'close']:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.drop_duplicates(['symbol', 'date']).sort_values(['symbol', 'date'])
    print('trades', len(fl), 'candles loaded', len(d), flush=True)

    # Basket regime at each as-of date (known at that close)
    bc = d[d.symbol.isin(BASKET)].pivot(index='date', columns='symbol', values='close')
    br = bc.pct_change(fill_method=None).clip(-0.5, 1.0).mean(axis=1).fillna(0)
    lvl = (1 + br).cumprod()
    regime = (lvl > lvl.rolling(50, min_periods=40).mean())
    fl['regime_up'] = fl['asof'].map(lambda t: bool(regime.loc[:t].iloc[-1]) if len(regime.loc[:t]) else np.nan)

    by_sym = {s: g.set_index('date') for s, g in d.groupby('symbol')}
    trades = []
    for r in fl.itertuples():
        g = by_sym.get(r.symbol)
        if g is None:
            continue
        days = g.loc[(g.index >= r.m_start) & (g.index <= r.m_end)].dropna(subset=['open', 'high', 'low', 'close'])
        if len(days) < 3:
            continue
        trades.append((r.Index, days))
    print('trades with daily data', len(trades), flush=True)

    rows = []
    for (pname, plan), stop, be in itertools.product(PLANS.items(), STOPS, [False, True]):
        if be and (not plan or stop is None):
            continue
        res = pd.Series({i: simulate(days, plan, stop, be) for i, days in trades}) - a.cost
        t = fl.loc[res.index].assign(ret=res.values).dropna(subset=['ret'])
        months = sorted(t.month.unique()); cut = months[len(months) // 2]
        for scope, s in [('all', t), ('regime_up', t[t.regime_up == True])]:
            if s.empty:
                continue
            m = s.groupby('month').ret.mean()
            rows.append({'plan': pname, 'stop_pct': stop or 0, 'breakeven': be, 'scope': scope,
                         'trades': len(s), 'mean_trade': s.ret.mean(), 'median_trade': s.ret.median(),
                         'win_trades_pct': (s.ret > 0).mean() * 100,
                         'months': len(m), 'months_positive': int((m > 0).sum()),
                         'worst_month': m.min(), 'avg_month': m.mean(),
                         'first_half_mean': s[s.month < cut].ret.mean(),
                         'second_half_mean': s[s.month >= cut].ret.mean()})
    grid = pd.DataFrame(rows).round(2)
    grid.to_csv(out / 'exit_grid.csv', index=False)

    lines = []
    for scope in ['all', 'regime_up']:
        g = grid[grid.scope == scope]
        robust = g[(g.first_half_mean > 0) & (g.second_half_mean > 0)]
        lines.append(f'===== {scope}  (trades: {int(g.trades.max())}) =====')
        lines.append('-- hold, no stop (baseline):')
        lines.append(g[(g.plan == 'hold') & (g.stop_pct == 0)].to_string(index=False))
        lines.append('-- top 12 by average month, positive in BOTH halves:')
        lines.append(robust.sort_values(['avg_month'], ascending=False).head(12).to_string(index=False))
    pos = fl.regime_up.mean()
    lines.append(f'\nshare of flagged trades taken in regime_up months: {pos:.0%}')
    txt = '\n'.join(lines)
    (out / 'exit_best.txt').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
