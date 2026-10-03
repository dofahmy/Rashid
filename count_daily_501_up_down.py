#!/usr/bin/env python3
import csv
from pathlib import Path
from sqlalchemy import MetaData, Table, select
from core import database

INPUT = "daily_rule_all_dates.csv"

p = Path(INPUT)
if not p.exists():
    raise SystemExit(f"Missing {INPUT}. Run test_daily_rule_all_501_dates.py first.")

with p.open(encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

DB = database()
with DB() as s:
    md = MetaData()
    daily = Table("market_candles_1d", md, autoload_with=s.get_bind())

up = down = flat = usable = 0
returns = []

for n, r in enumerate(rows, 1):
    sym = r["symbol"]
    dt = r["signal_date"]

    with DB() as s:
        bars = list(s.execute(
            select(daily.c.session_date, daily.c.adj_c)
            .where(daily.c.symbol == sym)
            .order_by(daily.c.session_date)
        ).all())

    clean = [(str(d), float(c)) for d, c in bars if c is not None and float(c) > 0]
    dates = [d for d, _ in clean]

    try:
        i = dates.index(dt)
    except ValueError:
        continue

    if i + 21 >= len(clean):
        continue

    start = clean[i][1]
    end = clean[i + 21][1]
    ret = 100 * (end / start - 1)
    returns.append(ret)
    usable += 1

    if ret > 0.0001:
        up += 1
    elif ret < -0.0001:
        down += 1
    else:
        flat += 1

print("\n=== 21-SESSION FINAL DIRECTION ===")
print(f"Usable setups: {usable}")
print(f"Finished UP after 21 sessions:   {up}/{usable} = {100*up/usable:.2f}%")
print(f"Finished DOWN after 21 sessions: {down}/{usable} = {100*down/usable:.2f}%")
print(f"Finished FLAT:                   {flat}/{usable} = {100*flat/usable:.2f}%")
print(f"Average 21d close return: {sum(returns)/len(returns):.2f}%")
print(f"Median 21d close return: {sorted(returns)[len(returns)//2]:.2f}%")
