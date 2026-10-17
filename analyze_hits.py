#!/usr/bin/env python3
"""
Look at the signals that hit +100% (from daily_signals.py output):
  - sector / industry of each (from public NASDAQ/NYSE/AMEX symbol lists on GitHub)
  - hit rate per sector vs all signals in that sector (is a sector over-represented?)
  - timing: did they move together (same weeks), or spread out?

Input : /data/spike_profile/signals_detail.csv
Output: /data/spike_profile/hits_detail.csv, hits_by_sector.csv, hits_by_week.csv
"""
import argparse, json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

LISTS = ['https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/{e}/{e}_full_tickers.json'
         .format(e=e) for e in ('nasdaq', 'nyse', 'amex')]


def load_sectors():
    rows = []
    for u in LISTS:
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                rows += json.loads(r.read().decode())
        except Exception as ex:
            print('could not download', u, ex)
    if not rows:
        return pd.DataFrame(columns=['symbol', 'name', 'sector', 'industry'])
    s = pd.DataFrame(rows)[['symbol', 'name', 'sector', 'industry']]
    s['symbol'] = s.symbol.str.upper().str.replace('/', '-', regex=False)
    return s.drop_duplicates('symbol')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default='/data/spike_profile')
    a = ap.parse_args()
    f = Path(a.folder)
    d = pd.read_csv(f / 'signals_detail.csv')
    d = d.dropna(subset=['status'])
    sec = load_sectors()
    d = d.merge(sec, on='symbol', how='left')
    d['sector'] = d.sector.replace('', np.nan).fillna('غير معروف')
    d['industry'] = d.industry.replace('', np.nan).fillna('غير معروف')
    d['hit'] = d.status.str.startswith('hit')
    for c in ['signal_date', 'entry_date', 'exit_date']:
        d[c] = pd.to_datetime(d[c], errors='coerce')
    done = d[d.status != 'still open']
    h = d[d.hit].copy()
    h['days_to_hit'] = [np.busday_count(e.date(), x.date()) + 1 if pd.notna(x) else np.nan
                        for e, x in zip(h.entry_date, h.exit_date)]
    h = h.sort_values('entry_date')
    h[['symbol', 'name', 'sector', 'industry', 'signal_date', 'entry_date', 'exit_date', 'days_to_hit',
       'entry_open', 'best_gain_pct', 'worst_pct']].to_csv(f / 'hits_detail.csv', index=False)

    print(f'عدد اللي وصلت +100%: {len(h)}  من {len(done)} صفقة خلصت  ({len(h)/max(1,len(done))*100:.1f}%)\n')
    print('--- تفاصيل كل سهم (مترتب بتاريخ الدخول) ---')
    for r in h.itertuples():
        nm = (r.name or '')[:28] if isinstance(r.name, str) else ''
        print(f'{r.symbol:6} | {str(r.entry_date.date())} -> {str(r.exit_date.date())} ({int(r.days_to_hit):>2} يوم) | '
              f'أسوأ نزول قبلها {r.worst_pct:>6}% | {r.sector} / {r.industry} | {nm}')

    # sector concentration vs base
    base = done.groupby('sector').size().rename('signals_done')
    hs = h.groupby('sector').size().rename('hits')
    S = pd.concat([base, hs], axis=1).fillna(0).astype(int)
    S['hit_rate_pct'] = (S.hits / S.signals_done.replace(0, np.nan) * 100).round(1)
    S['share_of_hits_pct'] = (S.hits / max(1, len(h)) * 100).round(1)
    S['share_of_signals_pct'] = (S.signals_done / max(1, len(done)) * 100).round(1)
    S = S.sort_values('hits', ascending=False)
    S.to_csv(f / 'hits_by_sector.csv')
    overall = len(h) / max(1, len(done)) * 100
    print(f'\n--- القطاعات (نسبة النجاح العامة {overall:.1f}%) ---')
    print('القطاع | إشارات خلصت | وصلت +100% | نسبة النجاح | نصيبه من الطفرات | نصيبه من الإشارات')
    for k, r in S.iterrows():
        print(f'{k} | {int(r.signals_done)} | {int(r.hits)} | {r.hit_rate_pct}% | {r.share_of_hits_pct}% | {r.share_of_signals_pct}%')

    ind = h.groupby('industry').size().sort_values(ascending=False)
    print('\n--- الصناعات اللي فيها أكتر من طفرة ---')
    for k, v in ind[ind > 1].items():
        syms = ' '.join(h[h.industry == k].symbol)
        print(f'{k}: {v}  ({syms})')

    # timing
    h['entry_week'] = h.entry_date.dt.to_period('W').astype(str)
    h['hit_week'] = h.exit_date.dt.to_period('W').astype(str)
    W = h.groupby('entry_week').agg(n=('symbol', 'size'), symbols=('symbol', ' '.join),
                                    sectors=('sector', lambda x: ', '.join(sorted(set(x)))))
    W.to_csv(f / 'hits_by_week.csv')
    print('\n--- أسابيع الدخول اللي فيها طفرات ---')
    for k, r in W.iterrows():
        print(f'{k}: {r.n} | {r.symbols} | {r.sectors}')
    hm = h.groupby(h.entry_date.dt.to_period('M')).size()
    print('\nالطفرات حسب شهر الدخول: ' + ' | '.join(f'{k}: {v}' for k, v in hm.items()))
    top3 = W.n.sort_values(ascending=False).head(3).sum()
    print(f'أكبر ٣ أسابيع فيهم {top3} من {len(h)} طفرة ({top3/max(1,len(h))*100:.0f}%)')
    print(f'\nالملفات: {f/"hits_detail.csv"} , {f/"hits_by_sector.csv"} , {f/"hits_by_week.csv"}')


if __name__ == '__main__':
    main()
