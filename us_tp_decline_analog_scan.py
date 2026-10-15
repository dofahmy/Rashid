#!/usr/bin/env python3
"""Research-only S&P 500 decline analogue finder: daily prior-20 close return.
No database writes, orders, or Telegram alerts.
"""
import argparse
import csv
import math
from pathlib import Path

# Frozen previously qualified Time-Price dates (>=9 distinct stocks, >=3 sectors).
QUALIFIED = '''2026-10-14 2026-10-01 2026-09-14 2026-05-15 2026-03-20 2026-03-19 2026-03-13 2026-03-09 2026-03-06 2026-03-03 2026-02-23 2026-02-12 2026-02-05 2026-02-04 2026-02-03 2026-01-29 2025-11-21 2025-11-17 2025-11-14 2025-11-07 2025-11-04 2025-10-30 2025-10-22 2025-10-16 2025-10-10 2025-08-01 2025-07-31 2025-07-30 2025-04-09 2025-04-08 2025-04-07 2025-04-04 2025-04-03 2025-03-31 2025-03-28 2025-03-13 2025-03-11 2025-03-10 2025-03-04 2025-03-03 2025-02-28 2025-02-25 2025-02-24 2025-02-21 2025-02-20 2025-02-03 2025-01-27 2025-01-13 2025-01-10 2024-12-19 2024-12-18 2024-12-17 2024-12-13 2024-11-15 2024-11-06 2024-10-31 2024-09-10 2024-09-09 2024-09-05 2024-08-05 2024-08-02 2024-08-01 2024-04-12 2024-02-13 2024-02-05 2024-01-17 2023-10-20 2023-10-19 2023-10-03 2023-10-02 2023-09-27 2023-09-21 2023-09-07 2023-08-17 2023-07-06 2023-03-15 2023-03-13 2023-03-10 2022-10-13 2022-09-26 2022-09-23 2022-09-22 2022-09-20 2022-09-19 2022-09-01 2022-08-29'''.split()
QUALIFIED_SET = set(QUALIFIED) - {'2024-12-18'}


def load_index(symbol, start, end):
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as e:
        raise SystemExit('Missing dependency. Run: pip install yfinance pandas') from e
    # Always download adequate lookback for first selected date.
    earliest = (pd.Timestamp(start) - pd.Timedelta(days=75)).strftime('%Y-%m-%d')
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if end else None
    raw = yf.download(symbol, start=earliest, end=end_exclusive, interval='1d',
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise SystemExit('No index data downloaded; check Yahoo access and symbol.')
    # Handle yfinance multi-index variations.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    missing = {'Open', 'High', 'Low', 'Close'} - set(raw.columns)
    if missing:
        raise SystemExit(f'Missing OHLC columns: {missing}')
    raw = raw[['Open', 'High', 'Low', 'Close']].dropna().sort_index().copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw


def pct(value, base):
    return 100.0 * (float(value) / float(base) - 1.0)


def r(value):
    if value is None or not math.isfinite(float(value)):
        return ''
    return round(float(value), 4)


def csvout(file, records):
    if not records:
        print('NO ROWS:', file)
        return
    with open(file, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('CSV:', file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2022-08-01', help='First day to search, YYYY-MM-DD')
    p.add_argument('--end', default=None, help='Last day to search, YYYY-MM-DD; default latest available')
    p.add_argument('--symbol', default='^GSPC')
    p.add_argument('--out', default='/tmp/sp500_decline_analogs')
    p.add_argument('--cluster-sessions', type=int, default=3, help='Max index sessions from first hit in one event')
    p.add_argument('--aug-tolerance', type=float, default=.75)
    p.add_argument('--april-tolerance', type=float, default=.75)
    p.add_argument('--april-range-low', type=float, default=-12.0641)
    p.add_argument('--april-range-high', type=float, default=-9.8371)
    args = p.parse_args()
    if args.cluster_sessions < 0:
        p.error('--cluster-sessions must be >=0')
    df = load_index(args.symbol, args.start, args.end)
    dates = df.index.strftime('%Y-%m-%d').tolist()
    search_start = next((i for i, x in enumerate(dates) if x >= args.start), len(df))
    search_end = len(df) if args.end is None else next((i for i, x in enumerate(dates) if x > args.end), len(df))
    anchors = {'AUG_2024': -6.9358, 'APR_04_2025': -12.0641, 'APR_08_2025': -10.5760}

    rows = []
    for i in range(max(search_start, 20), search_end):
        previous = pct(df['Close'].iloc[i], df['Close'].iloc[i-20])
        models = []
        if abs(previous - anchors['AUG_2024']) <= args.aug_tolerance:
            models.append('AUG_2024')
        if args.april_range_low <= previous <= args.april_range_high or abs(previous - anchors['APR_04_2025']) <= args.april_tolerance:
            models.append('APR_04_2025')
        if abs(previous - anchors['APR_08_2025']) <= args.april_tolerance:
            models.append('APR_08_2025')
        if not models:
            continue
        mainmodel = min(models, key=lambda name: abs(previous - anchors[name]))
        base = float(df['Close'].iloc[i])
        item = {'date':dates[i], 'index_session':i, 'prior20_pct':r(previous),
                'models_matched':'+'.join(models), 'nearest_model':mainmodel,
                'diff_nearest_pp':r(abs(previous-anchors[mainmodel])),
                'signal_close':r(base), 'is_qualified_tp_signal':int(dates[i] in QUALIFIED_SET)}
        for h in (20,60):
            future = df.iloc[i+1:min(i+1+h, len(df))]
            item[f'observed_{h}'] = len(future)
            item[f'complete_{h}'] = int(len(future)==h)
            if len(future):
                hi_idx = future['High'].idxmax()
                lo_idx = future['Low'].idxmin()
                item[f'max_rise_{h}_pct'] = r(pct(future.loc[hi_idx,'High'], base))
                item[f'max_fall_{h}_pct'] = r(pct(future.loc[lo_idx,'Low'], base))
                item[f'peak_{h}_date'] = hi_idx.strftime('%Y-%m-%d')
                item[f'trough_{h}_date'] = lo_idx.strftime('%Y-%m-%d')
                item[f'close_return_{h}_pct'] = r(pct(future['Close'].iloc[-1], base)) if len(future)==h else ''
            else:
                for key in (f'max_rise_{h}_pct',f'max_fall_{h}_pct',f'peak_{h}_date',f'trough_{h}_date',f'close_return_{h}_pct'):
                    item[key] = ''
        rows.append(item)

    # Group every hit within <= N trading sessions of FIRST hit in the cluster,
    # not previous hit, to prevent a chain of hits turning into a multi-month event.
    groups = []
    for row in rows:
        if not groups or row['index_session'] - groups[-1][0]['index_session'] > args.cluster_sessions:
            groups.append([row])
        else:
            groups[-1].append(row)
    events = []
    for group in groups:
        rep = min(group, key=lambda x: float(x['prior20_pct']))
        matched_models = sorted(set(m for x in group for m in x['models_matched'].split('+')))
        # If any date is a qualified TP signal, show that distinctly rather than inferring
        # it from proximity to a selected representative date.
        qualified_dates = [x['date'] for x in group if x['is_qualified_tp_signal']]
        ev = {'event_start':group[0]['date'], 'event_end':group[-1]['date'],
              'matching_sessions':len(group), 'event_models':'+'.join(matched_models),
              'representative_date':rep['date'], 'prior20_pct':rep['prior20_pct'],
              'tp_signal_same_matching_day':int(bool(qualified_dates)),
              'tp_signal_dates':','.join(qualified_dates)}
        for col in ('observed_20','complete_20','max_rise_20_pct','max_fall_20_pct',
                    'peak_20_date','trough_20_date','close_return_20_pct',
                    'observed_60','complete_60','max_rise_60_pct','max_fall_60_pct',
                    'peak_60_date','trough_60_date','close_return_60_pct'):
            ev[col] = rep[col]
        events.append(ev)

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csvout(f'{prefix}_days.csv', rows)
    csvout(f'{prefix}_events.csv', events)
    print(f'INDEX={args.symbol} bars={len(df)} from={dates[0]} to={dates[-1]}')
    print(f'MATCHED_DAYS={len(rows)} DISTINCT_EVENTS={len(events)} qualified_TP_matched_days={sum(x["is_qualified_tp_signal"] for x in rows)}')
    print('NOTE: APRIL anchor references overlap; matches are unioned before clustering. April 4/7/8 are the SAME market episode for this exploratory analysis.')
    print('EVENT START | END | REP DATE | PRE20 | MODELS | TP QUALIFIED DATE | MAX RISE 20 | MAX FALL 20 | MAX RISE 60 | MAX FALL 60 | 60 COMPLETE')
    for e in events:
        print(' | '.join(str(e[k]) for k in ('event_start','event_end','representative_date','prior20_pct','event_models','tp_signal_dates','max_rise_20_pct','max_fall_20_pct','max_rise_60_pct','max_fall_60_pct','complete_60')))
    print('READ ONLY. No DB modification; no alerts; current/future HIGH and LOW are hindsight diagnostics, not simulated fills.')
    print('CAUTION: shared 20-session drawdown amounts alone do not prove timing alpha. Test against comparable index days and non-overlapping out-of-sample events.')

if __name__ == '__main__':
    main()
