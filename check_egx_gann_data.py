#!/usr/bin/env python3
from sqlalchemy import text
from core import database

DB=database()
with DB() as s:
    rows=s.execute(text("""
        SELECT
          COUNT(*) AS candles,
          COUNT(DISTINCT symbol) AS symbols,
          MIN(session_date) AS first_date,
          MAX(session_date) AS last_date
        FROM market_candles_1d
        WHERE symbol ILIKE '%.CA'
    """)).mappings().one()

    print("EGX Gann candles:", rows["candles"])
    print("EGX symbols:", rows["symbols"])
    print("First date:", rows["first_date"])
    print("Last date:", rows["last_date"])

    sample=s.execute(text("""
        SELECT symbol, COUNT(*) AS candles, MAX(session_date) AS last_date
        FROM market_candles_1d
        WHERE symbol ILIKE '%.CA'
        GROUP BY symbol
        ORDER BY symbol
        LIMIT 15
    """)).mappings().all()

    print("\\nSample:")
    for r in sample:
        print(r["symbol"], r["candles"], r["last_date"])
