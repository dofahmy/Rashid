#!/usr/bin/env python3
"""
Build fund.csv (point-in-time shares outstanding) from SEC EDGAR company facts.

Output columns: symbol, asof_date, shares_outstanding
  asof_date = the date the filing was PUBLISHED (filed), so the number is only
  used after the market could actually know it (no look-ahead).

listing_date is not available from SEC; find_similar_before_spike.py falls back
to the first candle date in your database.

Usage:
  # symbols taken from your candles table (recommended)
  DATABASE_URL=... python3 build_fund_csv.py --email you@example.com --out fund.csv
  # or from a text file with one symbol per line
  python3 build_fund_csv.py --email you@example.com --symbols symbols.txt --out fund.csv

SEC asks for a real contact e-mail in the User-Agent and max 10 requests/second.
Re-running resumes from the cache folder, so an interrupted run is not lost.
"""
import argparse, json, os, time
from pathlib import Path
import urllib.request, urllib.error
import pandas as pd

TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
CONCEPTS = [('dei', 'EntityCommonStockSharesOutstanding'),       # cover page, every 10-Q/10-K/20-F
            ('us-gaap', 'CommonStockSharesOutstanding')]          # balance sheet, fallback


def get(url, ua, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': ua, 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 503):
                time.sleep(2 * (k + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (k + 1))
    return None


def shares_rows(facts):
    out = []
    for ns, concept in CONCEPTS:
        units = facts.get('facts', {}).get(ns, {}).get(concept, {}).get('units', {})
        for x in units.get('shares', []):
            if x.get('val') and x.get('filed'):
                out.append({'accn': x.get('accn'), 'filed': x['filed'], 'end': x.get('end'),
                            'val': float(x['val']), 'src': concept})
        if out:                      # prefer dei; use us-gaap only if dei is empty
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).drop_duplicates(['accn', 'end', 'val'])
    # Several share classes in one filing (Class A + Class B) -> add them up
    g = df.groupby(['accn', 'filed'], as_index=False).agg(val=('val', 'sum'), end=('end', 'max'))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--email', required=True, help='your contact e-mail (SEC requires it)')
    ap.add_argument('--symbols', help='text file, one symbol per line (else read from DATABASE_URL)')
    ap.add_argument('--out', default='fund.csv')
    ap.add_argument('--cache', default='sec_cache')
    ap.add_argument('--since', default='2022-01-01', help='drop filings published before this date')
    a = ap.parse_args()
    ua = f'fingerprint-research {a.email}'
    cache = Path(a.cache); cache.mkdir(exist_ok=True)

    if a.symbols:
        syms = [s.strip().upper() for s in open(a.symbols) if s.strip()]
    else:
        from sqlalchemy import create_engine, text
        eng = create_engine(os.environ['DATABASE_URL'])
        with eng.connect() as c:
            syms = pd.read_sql(text("SELECT DISTINCT symbol FROM market_candles_1d "
                                    "WHERE session_date >= '2024-01-01'"), c).symbol.astype(str).tolist()
    syms = sorted({s for s in syms if not s.endswith('.CA')})
    print('symbols requested:', len(syms), flush=True)

    tk = get(TICKERS_URL, ua)
    if not tk:
        raise SystemExit('could not download SEC ticker list - check internet / e-mail')
    cik_of = {v['ticker'].upper().replace('.', '-'): int(v['cik_str']) for v in tk.values()}
    norm = lambda s: s.upper().replace('.', '-').replace('/', '-')
    found = [(s, cik_of[norm(s)]) for s in syms if norm(s) in cik_of]
    print('matched to SEC:', len(found), 'not found:', len(syms) - len(found), flush=True)

    rows, no_data = [], []
    for i, (sym, cik) in enumerate(found, 1):
        f = cache / f'{cik}.json'
        if f.exists():
            facts = json.loads(f.read_text())
        else:
            facts = get(FACTS_URL.format(cik=cik), ua) or {}
            f.write_text(json.dumps(facts))
            time.sleep(0.12)                       # stay under SEC's 10 req/s
        g = shares_rows(facts)
        if g.empty:
            no_data.append(sym)
        else:
            g['symbol'] = sym
            rows.append(g)
        if i % 200 == 0:
            print(f'  {i}/{len(found)}', flush=True)

    if not rows:
        raise SystemExit('no share data collected')
    df = pd.concat(rows, ignore_index=True)
    df = df[df.filed >= a.since]
    df = (df.sort_values(['symbol', 'filed', 'end'])
            .drop_duplicates(['symbol', 'filed'], keep='last')
            .rename(columns={'filed': 'asof_date', 'val': 'shares_outstanding'}))
    df[['symbol', 'asof_date', 'shares_outstanding']].to_csv(a.out, index=False)
    Path(a.out).with_suffix('.missing.txt').write_text(
        '\n'.join(sorted(set(syms) - {s for s, _ in found}) + no_data))
    print(f'wrote {a.out}: {len(df)} rows, {df.symbol.nunique()} symbols. '
          f'No SEC data for {len(no_data)}; list in {Path(a.out).with_suffix(".missing.txt")}')


if __name__ == '__main__':
    main()
