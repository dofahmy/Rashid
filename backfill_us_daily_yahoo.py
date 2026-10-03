#!/usr/bin/env python3
"""
Rajih — US DAILY historical data backfill into a SEPARATE database table.

Purpose
-------
Download daily OHLCV history for the existing US stock universe from Yahoo
and store it locally before we decide any daily strategy.

IMPORTANT
---------
- Does NOT touch market_candles_15m.
- Creates/uses a new table: market_candles_1d
- Existing daily rows are preserved; duplicate symbol/date rows are skipped.
- Uses Stock.feed_symbol for Yahoo and stores under the project's canonical symbol.
- Stores Yahoo adjusted close too, for later research.
- Checkpoint/resume supported.
- Default history = 3 years. Change with --years if desired.

Default run:
    python backfill_us_daily_yahoo.py --fresh

Small test first:
    python backfill_us_daily_yahoo.py --limit 20 --fresh

Resume after interruption:
    python backfill_us_daily_yahoo.py

Verify only:
    python backfill_us_daily_yahoo.py --verify-only

Examples:
    python backfill_us_daily_yahoo.py --years 5 --fresh
    python backfill_us_daily_yahoo.py --concurrency 6 --fresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import quote

import aiohttp
from sqlalchemy import (
    BigInteger, Column, Float, String, Text,
    Index, select, func
)

from core import Base, database, now
from monitor.models import Stock

UTC = timezone.utc
NY = ZoneInfo("America/New_York")

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


class DailyCandle(Base):
    __tablename__ = "market_candles_1d"

    symbol = Column(String(40), primary_key=True)
    session_date = Column(String(10), primary_key=True)  # YYYY-MM-DD in New York
    ts = Column(BigInteger, nullable=False, index=True)

    o = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    l = Column(Float, nullable=False)
    c = Column(Float, nullable=False)
    v = Column(Float, nullable=False)
    adj_c = Column(Float)

    feed_symbol = Column(String(60), nullable=False, default="")
    source = Column(String(40), nullable=False, default="Yahoo public chart")
    retrieved_at = Column(String(40), nullable=False, default=now)
    metadata_json = Column(Text, nullable=False, default="{}")

    __table_args__ = (
        Index("ix_market_candles_1d_symbol_ts", "symbol", "ts"),
    )


def finite_num(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def period_bounds(years: float):
    end = datetime.now(UTC) + timedelta(days=1)
    start = end - timedelta(days=int(round(years * 365.25)))
    return int(start.timestamp()), int(end.timestamp())


def checkpoint_load(path: Path, years: float, fresh: bool):
    base = {
        "version": 1,
        "years": float(years),
        "next_index": 0,
        "completed": 0,
        "added_total": 0,
        "updated_total": 0,
        "empty": 0,
        "errors": {},
    }
    if fresh or not path.exists():
        return base

    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        if float(old.get("years", -1)) == float(years):
            base.update(old)
            print(f"RESUME: symbol index {base['next_index']}")
        else:
            print("Checkpoint years differ; starting a new checkpoint.")
    except Exception:
        print("WARNING: unreadable checkpoint; starting fresh.")
    return base


def checkpoint_save(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_chart(payload, canonical_symbol, feed_symbol):
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise RuntimeError("provider_invalid_chart")

    if chart.get("error"):
        err = chart["error"]
        raise RuntimeError(
            f"provider_chart_error:{err.get('code')}:{err.get('description')}"
        )

    results = chart.get("result")
    if not results:
        return []

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = ((result.get("indicators") or {}).get("quote") or [{}])
    q = quote_rows[0] if quote_rows else {}

    opens = q.get("open") or []
    highs = q.get("high") or []
    lows = q.get("low") or []
    closes = q.get("close") or []
    volumes = q.get("volume") or []

    adj_rows = ((result.get("indicators") or {}).get("adjclose") or [{}])
    adj = (adj_rows[0].get("adjclose") or []) if adj_rows else []

    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}

    rows = {}
    n = len(timestamps)
    for i in range(n):
        ts = int(timestamps[i])
        o = finite_num(opens[i] if i < len(opens) else None)
        h = finite_num(highs[i] if i < len(highs) else None)
        l = finite_num(lows[i] if i < len(lows) else None)
        c = finite_num(closes[i] if i < len(closes) else None)
        v = finite_num(volumes[i] if i < len(volumes) else 0)
        ac = finite_num(adj[i] if i < len(adj) else None)

        if None in (o, h, l, c):
            continue
        if min(o, h, l, c) <= 0:
            continue
        if not (l <= o <= h and l <= c <= h):
            continue

        # Yahoo daily timestamps identify the US trading session.
        session_date = datetime.fromtimestamp(ts, NY).date().isoformat()

        rows[session_date] = {
            "symbol": canonical_symbol,
            "session_date": session_date,
            "ts": ts,
            "o": o,
            "h": h,
            "l": l,
            "c": c,
            "v": max(0.0, v or 0.0),
            "adj_c": ac,
            "feed_symbol": feed_symbol,
            "source": "Yahoo public chart",
            "retrieved_at": now(),
            "metadata_json": json.dumps(
                {
                    "currency": meta.get("currency"),
                    "exchangeName": meta.get("exchangeName"),
                    "instrumentType": meta.get("instrumentType"),
                    "dataGranularity": meta.get("dataGranularity"),
                    "timezone": meta.get("timezone"),
                    "gmtoffset": meta.get("gmtoffset"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }

    return [rows[k] for k in sorted(rows)]


class YahooDailyFetcher:
    def __init__(self, concurrency=8, timeout=35):
        self.sem = asyncio.Semaphore(max(1, int(concurrency)))
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/154.0 Safari/537.36",
                "Accept": "application/json,text/plain,*/*",
            },
            connector=aiohttp.TCPConnector(limit=max(12, self.sem._value * 2)),
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def fetch(self, canonical_symbol, feed_symbol, period1, period2, retries=6):
        url = YAHOO_URL.format(symbol=quote(feed_symbol, safe=""))
        params = {
            "interval": "1d",
            "period1": str(period1),
            "period2": str(period2),
            "includePrePost": "false",
            "events": "div,splits",
        }

        async with self.sem:
            last = None
            for attempt in range(1, retries + 1):
                try:
                    async with self.session.get(url, params=params) as resp:
                        text = await resp.text()

                        if resp.status in (429, 500, 502, 503, 504):
                            last = f"http_{resp.status}"
                            if attempt < retries:
                                wait = min(60, 2 ** attempt) + random.random()
                                await asyncio.sleep(wait)
                                continue
                            raise RuntimeError(last)

                        if resp.status != 200:
                            raise RuntimeError(f"http_{resp.status}:{text[:180]}")

                        try:
                            payload = json.loads(text)
                        except Exception:
                            raise RuntimeError("provider_bad_json")

                        rows = parse_chart(payload, canonical_symbol, feed_symbol)
                        return rows, None

                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                    last = f"{type(exc).__name__}:{exc}"
                    if attempt < retries:
                        await asyncio.sleep(min(30, 2 ** (attempt - 1)) + random.random())
                    else:
                        return [], last

            return [], last or "unknown_error"


def upsert_symbol(DB, canonical_symbol, rows):
    """
    Insert only missing dates. Existing rows are not overwritten.
    Returns (added, existing_count).
    """
    if not rows:
        return 0, 0

    dates = [r["session_date"] for r in rows]
    with DB() as s:
        existing = set(
            s.execute(
                select(DailyCandle.session_date).where(
                    DailyCandle.symbol == canonical_symbol,
                    DailyCandle.session_date.in_(dates),
                )
            ).scalars().all()
        )

        added = 0
        for r in rows:
            if r["session_date"] in existing:
                continue
            s.add(DailyCandle(**r))
            added += 1

        if added:
            s.commit()

        return added, len(existing)


def verify(DB):
    with DB() as s:
        total_rows = int(s.scalar(select(func.count()).select_from(DailyCandle)) or 0)
        symbols = int(
            s.scalar(select(func.count(func.distinct(DailyCandle.symbol)))) or 0
        )
        mn, mx = s.execute(
            select(func.min(DailyCandle.session_date), func.max(DailyCandle.session_date))
        ).one()

        counts = s.execute(
            select(DailyCandle.symbol, func.count())
            .group_by(DailyCandle.symbol)
            .order_by(func.count().desc())
            .limit(10)
        ).all()

        enough_200 = int(
            s.scalar(
                select(func.count()).select_from(
                    select(DailyCandle.symbol)
                    .group_by(DailyCandle.symbol)
                    .having(func.count() >= 200)
                    .subquery()
                )
            ) or 0
        )

    print("\n=== DAILY DATA VERIFICATION ===")
    print(f"Table: market_candles_1d")
    print(f"Rows: {total_rows:,}")
    print(f"Symbols with daily data: {symbols:,}")
    print(f"Symbols with >=200 daily bars: {enough_200:,}")
    print(f"Global date range: {mn or 'NONE'} -> {mx or 'NONE'}")
    if counts:
        print("Largest histories:")
        for sym, cnt in counts:
            print(f"  {sym}: {int(cnt):,} bars")


async def run(args):
    DB = database()  # Base.metadata.create_all() creates market_candles_1d too.

    if args.verify_only:
        verify(DB)
        return

    period1, period2 = period_bounds(args.years)

    with DB() as s:
        stocks = list(
            s.execute(
                select(Stock.symbol, Stock.feed_symbol)
                .where(Stock.market == "US")
                .order_by(Stock.symbol)
            ).all()
        )

    if args.limit and args.limit > 0:
        stocks = stocks[:args.limit]

    cp_path = Path(args.checkpoint)
    cp = checkpoint_load(cp_path, args.years, args.fresh)
    start_index = int(cp.get("next_index", 0))
    start_index = min(start_index, len(stocks))

    print("\nRajih — US DAILY Yahoo historical backfill")
    print("Destination table: market_candles_1d")
    print("15m table is NOT modified.")
    print(f"History requested: ~{args.years:g} years")
    print(
        "Requested UTC period: "
        f"{datetime.fromtimestamp(period1, UTC).date()} -> "
        f"{datetime.fromtimestamp(period2, UTC).date()}"
    )
    print(f"US symbols: {len(stocks)}")
    print(f"Starting at index: {start_index}")
    print(f"Concurrency: {args.concurrency}\n")

    started = time.time()

    async with YahooDailyFetcher(args.concurrency, args.timeout) as fetcher:
        for base in range(start_index, len(stocks), args.batch_size):
            batch = stocks[base:base + args.batch_size]

            jobs = [
                fetcher.fetch(
                    canonical_symbol=symbol,
                    feed_symbol=(feed_symbol or symbol),
                    period1=period1,
                    period2=period2,
                    retries=args.retries,
                )
                for symbol, feed_symbol in batch
            ]

            results = await asyncio.gather(*jobs)

            for (symbol, feed_symbol), (rows, err) in zip(batch, results):
                if err:
                    cp["errors"][symbol] = err
                    print(f"ERROR {symbol} [{feed_symbol}]: {err}")
                elif not rows:
                    cp["empty"] = int(cp.get("empty", 0)) + 1
                    cp["errors"].pop(symbol, None)
                    print(f"EMPTY {symbol} [{feed_symbol}]")
                else:
                    try:
                        added, existed = upsert_symbol(DB, symbol, rows)
                        cp["added_total"] = int(cp.get("added_total", 0)) + added
                        cp["completed"] = int(cp.get("completed", 0)) + 1
                        cp["errors"].pop(symbol, None)

                        if args.verbose:
                            print(
                                f"OK {symbol:<8} feed={feed_symbol:<12} "
                                f"fetched={len(rows):4d} added={added:4d} "
                                f"existing={existed:4d}"
                            )
                    except Exception as exc:
                        cp["errors"][symbol] = f"db:{type(exc).__name__}:{exc}"
                        print(f"DB ERROR {symbol}: {type(exc).__name__}: {exc}")

            cp["next_index"] = min(base + len(batch), len(stocks))
            checkpoint_save(cp_path, cp)

            elapsed = time.time() - started
            print(
                f"Progress {cp['next_index']}/{len(stocks)} | "
                f"added_total={cp['added_total']:,} | "
                f"errors={len(cp['errors'])} | "
                f"elapsed={elapsed:.1f}s | checkpoint saved"
            )

            if args.pause_between_batches > 0:
                await asyncio.sleep(args.pause_between_batches)

    print("\n=== BACKFILL COMPLETE ===")
    print(f"Added rows this checkpoint: {cp['added_total']:,}")
    print(f"Errors currently recorded: {len(cp['errors'])}")
    print(f"Checkpoint: {cp_path}")
    verify(DB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all US symbols")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=35.0)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--pause-between-batches", type=float, default=0.6)
    ap.add_argument(
        "--checkpoint",
        default="backfill_us_daily_yahoo_checkpoint.json",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.years <= 0:
        raise SystemExit("--years must be > 0")
    if args.concurrency <= 0 or args.batch_size <= 0:
        raise SystemExit("--concurrency and --batch-size must be > 0")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
