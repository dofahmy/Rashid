"""Largest post-signal DODZ moves with peak price/time, 1H bar count and Gap.

Read-only: existing dodz_hourly_trades + us_all_seven_hourly_bars in PostgreSQL.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from sqlalchemy import text
from core import database

TRADES = 'dodz_hourly_trades'
BARS = 'us_all_seven_hourly_bars'


def run(output_dir: str, top: int = 100) -> None:
    top = int(top)
    if not 1 <= top <= 3786:
        raise ValueError('--top must be between 1 and 3786')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # First rank DODZ signals by *latest-close rise*, for compatibility with
    # the previous big-moves report. Then calculate the true post-signal high
    # and number of saved hourly candles until that high for every selected row.
    query = text(f"""
      WITH chosen AS (
        SELECT t.symbol, t.signal_utc, t.squaring_factor, t.factors,
               t.entry_price, t.gap_1h, t.touches, t.deadline_bars,
               t.status, t.exit_reason, t.exit_utc,
               t.realized_return_pct, t.floating_return_pct,
               last_bar.close AS current_close, last_bar.bar_time AS current_utc,
               (last_bar.close / NULLIF(t.entry_price,0) - 1) * 100 AS current_gain_pct
        FROM {TRADES} t
        LEFT JOIN LATERAL (
          SELECT b.close, b.bar_time FROM {BARS} b
          WHERE b.symbol = t.symbol ORDER BY b.bar_time DESC LIMIT 1
        ) last_bar ON TRUE
        WHERE t.entry_price > 0
      ), top_signals AS (
        SELECT * FROM chosen
        WHERE current_close IS NOT NULL AND current_utc > signal_utc
        ORDER BY current_gain_pct DESC, symbol ASC, signal_utc ASC
        LIMIT :top
      )
      SELECT t.*,
             peak.high AS peak_high, peak.bar_time AS peak_utc,
             bar_count.bars_to_peak AS bars_to_peak,
             (peak.high / NULLIF(t.entry_price,0) - 1) * 100 AS peak_gain_pct
      FROM top_signals t
      LEFT JOIN LATERAL (
        SELECT b.high,b.bar_time FROM {BARS} b
        WHERE b.symbol=t.symbol AND b.bar_time>t.signal_utc
          AND b.bar_time<=t.current_utc AND b.high IS NOT NULL
        ORDER BY b.high DESC, b.bar_time ASC LIMIT 1
      ) peak ON TRUE
      LEFT JOIN LATERAL (
        SELECT COUNT(*)::integer AS bars_to_peak FROM {BARS} b
        WHERE b.symbol=t.symbol AND b.bar_time>t.signal_utc AND b.bar_time<=peak.bar_time
      ) bar_count ON TRUE
      ORDER BY t.current_gain_pct DESC,t.symbol,t.signal_utc
    """)
    with database()() as session:
        results = session.execute(query, {'top': top}).mappings().all()
    cols = ['السهم','تاريخ الإشارة UTC','كل أرقام النظام','رقم النظام الأساسي','Gap','Touches',
            'Repeat Price','أعلى سعر بعد الإشارة','أكبر صعود بعد الإشارة %',
            'عدد شموع 1H حتى أعلى سعر','توقيت القمة UTC',
            'آخر إغلاق محفوظ','أداء السهم حتى آخر إغلاق %','تاريخ آخر إغلاق UTC',
            'مهلة دودز بالشموع','حالة دودز','سبب خروج دودز','عائد دودز %']
    data = []
    for r in results:
        factor_str = ','.join(map(str,sorted(set(int(v.strip()) for v in (r['factors'] or str(r['squaring_factor'])).split(',') if v.strip()))))
        rr = r['realized_return_pct'] if r['status']=='EXIT' else r['floating_return_pct']
        data.append({
            'السهم':r['symbol'],'تاريخ الإشارة UTC':str(r['signal_utc']),
            'كل أرقام النظام':factor_str,'رقم النظام الأساسي':r['squaring_factor'],
            'Gap':r['gap_1h'],'Touches':r['touches'],
            'Repeat Price':round(float(r['entry_price']),6),
            'أعلى سعر بعد الإشارة':round(float(r['peak_high']),6) if r['peak_high'] is not None else '',
            'أكبر صعود بعد الإشارة %':round(float(r['peak_gain_pct']),4) if r['peak_gain_pct'] is not None else '',
            'عدد شموع 1H حتى أعلى سعر':r['bars_to_peak'] if r['peak_utc'] is not None else '',
            'توقيت القمة UTC':str(r['peak_utc']) if r['peak_utc'] else '',
            'آخر إغلاق محفوظ':round(float(r['current_close']),6),
            'أداء السهم حتى آخر إغلاق %':round(float(r['current_gain_pct']),4),
            'تاريخ آخر إغلاق UTC':str(r['current_utc']),
            'مهلة دودز بالشموع':r['deadline_bars'],
            'حالة دودز':r['status'],'سبب خروج دودز':r['exit_reason'] or '',
            'عائد دودز %':round(float(rr),4) if rr is not None else ''
        })
    def save(name, records):
        path=out/name
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=cols)
            w.writeheader();w.writerows(records)
        return path
    by_latest=save('dodz_top_rises_full_details.csv',data)
    by_peak=save('dodz_top_rises_sorted_by_peak.csv',sorted(data,key=lambda x: x['أكبر صعود بعد الإشارة %'] if x['أكبر صعود بعد الإشارة %']!='' else float('-inf'),reverse=True))
    print(f'Analyzed {len(data)} selected DODZ signals (ranked by latest-close rise).')
    print(f"{'Symbol':<9} {'Factors':<13} {'Gap':>4} {'Peak %':>12} {'Bars':>6} {'Latest %':>12}")
    for x in data[:min(40,top)]:
        print(f"{x['السهم']:<9} {x['كل أرقام النظام']:<13} {x['Gap']:>4} {x['أكبر صعود بعد الإشارة %']:>11}% {x['عدد شموع 1H حتى أعلى سعر']:>6} {x['أداء السهم حتى آخر إغلاق %']:>11}%")
    print(f'Latest-rise ranking: {by_latest}\nPeak ranking within selected signals: {by_peak}')
    print('NOTE: Highest high is measured from post-signal candles until last saved hourly bar, even after DODZ exit. Bars count excludes signal candle.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',default='/tmp/dodz-peak-moves')
    p.add_argument('--top',type=int,default=100)
    a=p.parse_args();run(a.output_dir,a.top)
