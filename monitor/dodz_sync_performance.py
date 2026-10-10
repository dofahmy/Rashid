"""Read-only DODZ simultaneous-factor stock performance report.

Run: python -m monitor.dodz_sync_performance --output-dir /tmp/dodz-sync-performance
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from sqlalchemy import text
from core import database

TRADES = 'dodz_hourly_trades'
BARS = 'us_all_seven_hourly_bars'


def normalize_factors(value):
    try:
        factors = sorted(set(int(x.strip()) for x in str(value or '').split(',') if x.strip()))
        return tuple(factors)
    except ValueError:
        return ()


def iso(value):
    return value.isoformat(sep=' ', timespec='minutes') if isinstance(value, datetime) else (str(value) if value is not None else '')


def pc(value):
    return round(float(value), 4) if value is not None else ''


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Latest saved candle per symbol, whether or not its DODZ trade has exited.
    # Trades are read-only; no recalculation or modification of historical exit results.
    query = text(f"""
        WITH sync_trades AS (
            SELECT symbol, signal_utc, factors, squaring_factor, entry_price,
                   status, exit_reason, exit_utc, exit_price,
                   realized_return_pct, floating_return_pct,
                   last_price, data_end_utc
            FROM {TRADES}
            WHERE factors LIKE '%,%'
        ), symbols AS (
            SELECT DISTINCT symbol FROM sync_trades
        ), latest_candles AS (
            SELECT u.symbol, last_bar.close AS last_close,
                   last_bar.bar_time AS last_bar_utc
            FROM symbols u
            LEFT JOIN LATERAL (
                SELECT b.close, b.bar_time FROM {BARS} b
                WHERE b.symbol = u.symbol
                ORDER BY b.bar_time DESC LIMIT 1
            ) last_bar ON TRUE
        )
        SELECT t.*, b.last_close, b.last_bar_utc
        FROM sync_trades t
        LEFT JOIN latest_candles b ON b.symbol = t.symbol
        ORDER BY t.signal_utc DESC, t.symbol
    """)
    with database()() as session:
        rows = session.execute(query).mappings().all()

    detail = []
    summary = defaultdict(lambda: dict(count=0, targets=0, timed=0, opened=0,
                                       valid_current=0, sum_current=0.0, positive_current=0,
                                       negative_current=0, missing_current=0,
                                       closed_count=0, sum_closed=0.0))
    for rec in rows:
        factors = normalize_factors(rec['factors'])
        if len(factors) < 2:
            continue
        combo = ','.join(str(n) for n in factors)
        entry = float(rec['entry_price']) if rec['entry_price'] is not None else 0
        bar_price = rec['last_close']
        # Avoid showing historical prices as live quotes: attach actual as-of date.
        current_price = float(bar_price) if bar_price is not None else None
        asof = rec['last_bar_utc']
        source = 'hourly_bars'
        if current_price is None:
            # Fallback only; this is the old stored scan snapshot, not a fresh quote.
            current_price = float(rec['last_price']) if rec['last_price'] is not None else None
            asof = rec['data_end_utc']
            source = 'saved_scan_snapshot' if current_price is not None else 'not_available'
        change = 100 * (current_price / entry - 1) if entry > 0 and current_price is not None else None
        reason = rec['exit_reason'] or ''
        status = rec['status'] or ''
        stat = summary[combo]
        stat['count'] += 1
        if reason == 'TARGET_3':
            stat['targets'] += 1
        elif reason == 'TIME':
            stat['timed'] += 1
        elif status == 'OPEN':
            stat['opened'] += 1
        if change is None:
            stat['missing_current'] += 1
        else:
            stat['valid_current'] += 1
            stat['sum_current'] += change
            stat['positive_current'] += change > 0
            stat['negative_current'] += change < 0
        ret = rec['realized_return_pct'] if status == 'EXIT' else rec['floating_return_pct']
        if ret is not None:
            stat['closed_count'] += 1
            stat['sum_closed'] += float(ret)
        detail.append({
            'العوامل المتزامنة':combo,'السهم':rec['symbol'],
            'تاريخ الإشارة UTC':iso(rec['signal_utc']),
            'Repeat Price':pc(entry),'الحالة':status,
            'سبب الخروج':reason,'تاريخ خروج دودز UTC':iso(rec['exit_utc']),
            'سعر خروج دودز':pc(rec['exit_price']),
            'عائد دودز المحقق/العائم %':pc(ret),
            'آخر إغلاق محفوظ':pc(current_price),
            'تاريخ آخر إغلاق UTC':iso(asof),
            'مصدر آخر إغلاق':source,
            'أداء السهم منذ الإشارة حتى آخر إغلاق %':pc(change),
        })
    columns_detail = [
        'العوامل المتزامنة','السهم','تاريخ الإشارة UTC','Repeat Price','الحالة',
        'سبب الخروج','تاريخ خروج دودز UTC','سعر خروج دودز',
        'عائد دودز المحقق/العائم %','آخر إغلاق محفوظ','تاريخ آخر إغلاق UTC',
        'مصدر آخر إغلاق','أداء السهم منذ الإشارة حتى آخر إغلاق %'
    ]
    details_csv=output/'dodz_sync_trades_performance.csv'
    with details_csv.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns_detail)
        writer.writeheader();writer.writerows(detail)

    group_csv=output/'dodz_sync_groups_performance.csv'
    cols=['العوامل المتزامنة','التوصيات','قفلت على +3%','نسبة +3%','إغلاق زمني',
          'مفتوحة','متوسط أداء السهم حتى آخر إغلاق %','رابحة منذ الإشارة',
          'خاسرة منذ الإشارة','بدون سعر محفوظ','متوسط عائد دودز المحقق/العائم %']
    group_rows=[]
    for combo, s in sorted(summary.items(),key=lambda row:(-row[1]['count'],row[0])):
        group_rows.append({'العوامل المتزامنة':combo,'التوصيات':s['count'],
            'قفلت على +3%':s['targets'],
            'نسبة +3%':pc(100*s['targets']/s['count']),
            'إغلاق زمني':s['timed'],'مفتوحة':s['opened'],
            'متوسط أداء السهم حتى آخر إغلاق %':pc(s['sum_current']/s['valid_current']) if s['valid_current'] else '',
            'رابحة منذ الإشارة':s['positive_current'],
            'خاسرة منذ الإشارة':s['negative_current'],
            'بدون سعر محفوظ':s['missing_current'],
            'متوسط عائد دودز المحقق/العائم %':pc(s['sum_closed']/s['closed_count']) if s['closed_count'] else ''})
    with group_csv.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=cols)
        writer.writeheader();writer.writerows(group_rows)
    print('DODZ | أداء كل توصيات التزامن حتى آخر إغلاق محفوظ بالقاعدة')
    print(f'توصيات التزامن: {len(detail)} | عدد مجموعات التزامن: {len(group_rows)}')
    print(f"{'العوامل':>16} {'التوصيات':>10} {'هدف 3%':>9} {'متوسط السهم %':>16} {'متوسط دودز %':>16}")
    for row in group_rows:
        print(f"{row['العوامل المتزامنة']:>16} {row['التوصيات']:>10} {row['قفلت على +3%']:>9} {str(row['متوسط أداء السهم حتى آخر إغلاق %']):>16} {str(row['متوسط عائد دودز المحقق/العائم %']):>16}")
    print('التفصيلي:',details_csv)
    print('ملخص التزامنات:',group_csv)
    print('تنبيه: الأسعار هي آخر إغلاق محفوظ بالقاعدة، ليست بالضرورة سعر السوق اللحظي.')
    return detail,group_rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='/tmp/dodz-sync-performance')
    args=parser.parse_args()
    run(args.output_dir)

if __name__=='__main__':main()
