"""Rank DODZ stock movements using the latest saved US 1H close (read only)."""
import argparse
import csv
from pathlib import Path
from sqlalchemy import text
from core import database

TRADES = 'dodz_hourly_trades'
BARS = 'us_all_seven_hourly_bars'


def run(output_dir, top=100):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    query = text(f"""
        WITH symbols AS (SELECT DISTINCT symbol FROM {TRADES}),
        last_bars AS (
          SELECT s.symbol, b.close AS last_close, b.bar_time AS asof_utc
          FROM symbols s
          LEFT JOIN LATERAL (
            SELECT close, bar_time FROM {BARS} b
            WHERE b.symbol=s.symbol ORDER BY bar_time DESC LIMIT 1
          ) b ON TRUE
        )
        SELECT t.symbol, t.signal_utc, t.squaring_factor, t.factors,
               t.entry_price, t.gap_1h, t.touches, t.status, t.exit_reason,
               t.realized_return_pct, t.floating_return_pct,
               t.last_price, t.data_end_utc, lb.last_close, lb.asof_utc
        FROM {TRADES} t LEFT JOIN last_bars lb ON lb.symbol=t.symbol
        ORDER BY t.signal_utc DESC, t.symbol
    """)
    with database()() as session:
        records = session.execute(query).mappings().all()
    entries = []
    for r in records:
        entry = float(r['entry_price'] or 0)
        last = r['last_close'] if r['last_close'] is not None else r['last_price']
        if entry <= 0 or last is None:
            continue
        last = float(last)
        change = (last / entry - 1) * 100
        parts = sorted(set(int(x.strip()) for x in (r['factors'] or str(r['squaring_factor'])).split(',') if x.strip()))
        factors = ','.join(map(str, parts))
        asof = r['asof_utc'] if r['last_close'] is not None else r['data_end_utc']
        entries.append({
            'السهم': r['symbol'], 'تاريخ الإشارة UTC': str(r['signal_utc']),
            'كل أرقام النظام': factors, 'عدد العوامل': len(parts),
            'رقم النظام الأساسي': r['squaring_factor'],
            'Repeat Price': round(entry, 6), 'آخر إغلاق محفوظ': round(last, 6),
            'تاريخ آخر إغلاق UTC': str(asof), 'التغير السعري %': round(change, 4),
            'مقدار الحركة %': round(abs(change), 4),
            'Gap': r['gap_1h'], 'Touches': r['touches'],
            'حالة دودز': r['status'], 'سبب الخروج': r['exit_reason'] or '',
            'عائد دودز %': round(float(r['realized_return_pct'] if r['status'] == 'EXIT' else r['floating_return_pct']), 4)
                if (r['realized_return_pct'] if r['status'] == 'EXIT' else r['floating_return_pct']) is not None else '',
            'مصدر السعر': 'hourly_bars' if r['last_close'] is not None else 'saved_scan_snapshot',
        })
    if not entries:
        raise RuntimeError('لا توجد صفقات لها أسعار قابلة للحساب في قاعدة البيانات.')
    cols = list(entries[0].keys())
    def export(name, data):
        path = out / name
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(data)
        return path
    biggest_up = sorted((x for x in entries if x['التغير السعري %'] > 0), key=lambda r: -r['التغير السعري %'])
    biggest_down = sorted((x for x in entries if x['التغير السعري %'] < 0), key=lambda r: r['التغير السعري %'])
    biggest_abs = sorted(entries, key=lambda r: -r['مقدار الحركة %'])
    sync = [r for r in biggest_abs if r['عدد العوامل'] >= 2]
    files = [export('dodz_all_biggest_rises.csv',biggest_up),
             export('dodz_all_biggest_falls.csv',biggest_down),
             export('dodz_all_biggest_moves.csv',biggest_abs),
             export('dodz_sync_biggest_moves.csv',sync)]
    print(f'DODZ: {len(entries)} signals, simultaneous factors signals: {len(sync)}')
    for heading, values in [('أكبر 20 صعود', biggest_up), ('أكبر 20 هبوط', biggest_down), ('أكبر 20 حركة بين التزامنات', sync)]:
        print('\n'+heading)
        print(f"{'Symbol':<10} {'All factors':<17} {'Change %':>12} {'Signal UTC':<20} {'As of UTC':<20}")
        for row in values[:min(top,20)]:
            print(f"{row['السهم']:<10} {row['كل أرقام النظام']:<17} {row['التغير السعري %']:>+11.2f}% {row['تاريخ الإشارة UTC']:<20} {row['تاريخ آخر إغلاق UTC']:<20}")
    print('\nملفات CSV:')
    for file in files:
        print(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/dodz-big-moves')
    parser.add_argument('--top', type=int, default=20)
    a = parser.parse_args()
    run(a.output_dir, a.top)
