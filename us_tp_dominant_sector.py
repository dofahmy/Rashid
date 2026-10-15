#!/usr/bin/env python3
"""Read-only: test dominant sector of each (day, signal_type, anchor_type, status) group."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from sqlalchemy import text
from core import database


def metrics(rows):
    closed = [r for r in rows if r['outcome'] in ('TP', 'TIME_EXIT')]
    returns = [float(r['return_pct']) for r in closed if r.get('return_pct') not in ('', None)]
    tp = sum(r['outcome'] == 'TP' for r in closed)
    return {
        'signals': len(rows), 'closed': len(closed), 'open': len(rows)-len(closed),
        'tp': tp, 'time_exits': len(closed)-tp,
        'tp_rate_pct': round(tp / len(closed) * 100, 2) if closed else None,
        'avg_return_pct': round(mean(returns), 4) if returns else None,
        'worst_return_pct': round(min(returns), 4) if returns else None,
    }


def write_csv(p, rows, fields):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trades', required=True, help='Backtest _trades.csv file')
    ap.add_argument('--out', default='/tmp/us_tp_dominant')
    ap.add_argument('--date', help='Filter one signal day (YYYY-MM-DD)')
    args = ap.parse_args()
    with open(args.trades, newline='', encoding='utf-8-sig') as f:
        trades = list(csv.DictReader(f))
    if args.date:
        trades = [r for r in trades if r['day'] == args.date]
    if not trades:
        raise SystemExit('No trade rows found. Check --trades and --date.')
    unique = {r['symbol'] for r in trades}
    lookup = {}
    db = database()
    with db() as s:
        for row in s.execute(text('SELECT symbol,sector,status FROM us_tp_sector_lookup')).mappings():
            if row['symbol'] in unique:
                sector = (row['sector'] or '').strip()
                if sector and sector.lower() not in ('unknown', 'n/a', 'none', 'null'):
                    lookup[row['symbol']] = sector
    groups = defaultdict(list)
    for r in trades:
        key = tuple(r[k] for k in ('day', 'signal_type', 'anchor_type', 'status'))
        groups[key].append(r)
    selected, matched_all, group_rows, skipped_groups = [], [], [], []
    for key, rows in sorted(groups.items()):
        symbols = {r['symbol'] for r in rows}
        counts = Counter(lookup[sym] for sym in symbols if sym in lookup)
        unknown = len(symbols) - sum(counts.values())
        if not counts:
            skipped_groups.append((key, 'no classified sectors'))
            continue
        top = max(counts.values())
        winners = sorted(k for k,v in counts.items() if v == top)
        winners_set = set(winners)
        chosen = [r for r in rows if lookup.get(r['symbol']) in winners_set]
        selected.extend(chosen)
        matched_all.extend(rows)
        m = metrics(chosen)
        m_all = metrics(rows)
        group_rows.append({
            'day': key[0], 'signal_type': key[1], 'anchor_type': key[2], 'status': key[3],
            'total_stocks': len(symbols), 'classified_stocks': len(symbols) - unknown,
            'unknown_stocks': unknown, 'sector_count': len(counts),
            'dominant_sectors': ' | '.join(winners), 'dominant_stocks': top,
            'selected_stocks': len({r['symbol'] for r in chosen}),
            'selected_tp_rate_pct': m['tp_rate_pct'], 'selected_avg_return_pct': m['avg_return_pct'],
            'all_tp_rate_pct': m_all['tp_rate_pct'], 'all_avg_return_pct': m_all['avg_return_pct'],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out.with_name(out.name + '_groups.csv'), group_rows, list(group_rows[0]) if group_rows else ['day'])
    if selected:
        write_csv(out.with_name(out.name + '_trades.csv'), selected, list(selected[0]))
    print('TRADE ROWS:', len(trades), '| UNIQUE SYMBOLS:', len(unique), '| CLASSIFIED:', len(set(unique) & set(lookup)))
    print('GROUPS:', len(groups), '| COMPARED GROUPS:', len(group_rows), '| SKIPPED WITHOUT SECTORS:', len(skipped_groups))
    print('ALL TRADES (same groups):', metrics(matched_all))
    print('DOMINANT SECTOR TRADES (ties included):', metrics(selected))
    print('GROUP CSV:', out.with_name(out.name + '_groups.csv'))
    if selected:
        print('TRADE CSV:', out.with_name(out.name + '_trades.csv'))
    if any(x['unknown_stocks'] for x in group_rows):
        print('WARNING: Some stocks lack classifications. Dominance may be wrong for those groups.')
    print('READ-ONLY: The database was not changed. Current sector classifications; historical sectors may differ.')

if __name__ == '__main__':
    main()
