# monitor/sp500_seven_data.py
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from sqlalchemy import text

from core import database, Setting

STATUS_KEY = "sp500_seven_refresh_status_v1"

DAILY_TABLE = "sp500_seven_daily"
CONSTIT_TABLE = "sp500_seven_constituents"
INDEX_TABLE = "sp500_seven_index"


def ensure_tables():
    DB = database()
    ddl = [
        f"""
        CREATE TABLE IF NOT EXISTS {CONSTIT_TABLE} (
            symbol TEXT PRIMARY KEY,
            feed_symbol TEXT NOT NULL,
            company TEXT,
            sector TEXT,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {DAILY_TABLE} (
            symbol TEXT NOT NULL,
            session_date DATE NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION NOT NULL,
            adj_close DOUBLE PRECISION,
            volume BIGINT,
            source TEXT,
            retrieved_at TIMESTAMP,
            PRIMARY KEY(symbol, session_date)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {INDEX_TABLE} (
            session_date DATE PRIMARY KEY,
            close DOUBLE PRECISION NOT NULL,
            volume BIGINT,
            source TEXT,
            retrieved_at TIMESTAMP
        )
        """,
    ]
    with DB.begin() as s:
        for q in ddl:
            s.execute(text(q))
        for col in ('open','high','low'):
            s.execute(text(f"ALTER TABLE {DAILY_TABLE} ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"))


def set_status(status):
    DB = database()
    payload = json.dumps(status, ensure_ascii=False, default=str)
    with DB.begin() as s:
        row = s.get(Setting, STATUS_KEY)
        if row is None:
            s.add(Setting(key=STATUS_KEY, value=payload))
        else:
            row.value = payload


def get_status(session):
    row = session.get(Setting, STATUS_KEY)
    if row is None or not row.value:
        return None
    try:
        return json.loads(row.value)
    except Exception:
        return None


def fetch_sp500_constituents():
    """
    Pull the current S&P 500 constituent table from Wikipedia.
    Note: the index can contain slightly more than 500 ticker rows because
    a few companies have multiple share classes.
    """
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if not tables:
        raise RuntimeError("Could not read S&P 500 constituent table.")

    df = tables[0].copy()
    required = {"Symbol", "Security", "GICS Sector"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Unexpected S&P 500 table columns: {list(df.columns)}")

    out = pd.DataFrame({
        "symbol": df["Symbol"].astype(str).str.strip().str.upper(),
        "company": df["Security"].astype(str).str.strip(),
        "sector": df["GICS Sector"].astype(str).str.strip(),
    })
    out["feed_symbol"] = out["symbol"].str.replace(".", "-", regex=False)
    return out.drop_duplicates("symbol").reset_index(drop=True)


def _upsert_constituents(df):
    DB = database()
    now = datetime.now(timezone.utc)
    with DB.begin() as s:
        s.execute(text(f"UPDATE {CONSTIT_TABLE} SET active=FALSE"))
        q = text(f"""
            INSERT INTO {CONSTIT_TABLE}
                (symbol, feed_symbol, company, sector, active, updated_at)
            VALUES
                (:symbol, :feed_symbol, :company, :sector, TRUE, :updated_at)
            ON CONFLICT(symbol) DO UPDATE SET
                feed_symbol=excluded.feed_symbol,
                company=excluded.company,
                sector=excluded.sector,
                active=TRUE,
                updated_at=excluded.updated_at
        """)
        for r in df.to_dict("records"):
            r["updated_at"] = now
            s.execute(q, r)


def _existing_minmax():
    DB = database()
    with DB() as s:
        rows = s.execute(text(f"""
            SELECT symbol, MIN(session_date), MAX(session_date), COUNT(*)
            FROM {DAILY_TABLE}
            GROUP BY symbol
        """)).all()
    return {str(sym): (mn, mx, int(n)) for sym, mn, mx, n in rows}


def _extract_yf(raw, feed_symbol, multi):
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if multi:
        level0 = list(raw.columns.get_level_values(0))
        if feed_symbol not in level0:
            return pd.DataFrame()
        x = raw[feed_symbol].copy()
    else:
        x = raw.copy()

    if x.empty:
        return x

    x = x.reset_index()
    cols = {str(c).lower().replace(" ", "_"): c for c in x.columns}
    dc = cols.get("date") or cols.get("datetime") or x.columns[0]

    def ser(name):
        c = cols.get(name)
        if c is None:
            return pd.Series(index=x.index, dtype=float)
        return pd.to_numeric(x[c], errors="coerce")

    return pd.DataFrame({
        "date": pd.to_datetime(x[dc], errors="coerce"),
        "open": ser("open"),
        "high": ser("high"),
        "low": ser("low"),
        "close": ser("close"),
        "adj_close": ser("adj_close"),
        "volume": ser("volume"),
    }).dropna(subset=["date", "close"])


def _download_batch(feed_symbols, start, end):
    import yfinance as yf
    return yf.download(
        tickers=" ".join(feed_symbols),
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
    )


def _write_daily(symbol, df, source="YAHOO_SP500_SEVEN"):
    if df is None or df.empty:
        return 0
    DB = database()
    now = datetime.now(timezone.utc)
    q = text(f"""
        INSERT INTO {DAILY_TABLE}
            (symbol, session_date, open, high, low, close, adj_close, volume, source, retrieved_at)
        VALUES
            (:symbol, :session_date, :open, :high, :low, :close, :adj_close, :volume, :source, :retrieved_at)
        ON CONFLICT(symbol, session_date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            adj_close=excluded.adj_close,
            volume=excluded.volume,
            source=excluded.source,
            retrieved_at=excluded.retrieved_at
    """)
    n = 0
    with DB.begin() as s:
        for _, r in df.iterrows():
            close = float(r["close"])
            if not pd.notna(close) or close <= 0:
                continue
            adj = float(r["adj_close"]) if pd.notna(r["adj_close"]) else close
            o = float(r["open"]) if pd.notna(r.get("open")) else None
            h = float(r["high"]) if pd.notna(r.get("high")) else None
            l = float(r["low"]) if pd.notna(r.get("low")) else None
            vol = int(round(float(r["volume"]))) if pd.notna(r["volume"]) else None
            s.execute(q, {
                "symbol": symbol,
                "session_date": pd.Timestamp(r["date"]).date(),
                "open": o,
                "high": h,
                "low": l,
                "close": close,
                "adj_close": adj,
                "volume": vol,
                "source": source,
                "retrieved_at": now,
            })
            n += 1
    return n


def _refresh_index(start, end):
    import yfinance as yf
    raw = yf.download(
        "^GSPC", start=start, end=end, interval="1d",
        auto_adjust=False, actions=False, progress=False
    )
    if raw is None or len(raw) == 0:
        return 0
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    now = datetime.now(timezone.utc)
    DB = database()
    q = text(f"""
        INSERT INTO {INDEX_TABLE}
            (session_date, close, volume, source, retrieved_at)
        VALUES
            (:session_date, :close, :volume, 'YAHOO_^GSPC', :retrieved_at)
        ON CONFLICT(session_date) DO UPDATE SET
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source,
            retrieved_at=excluded.retrieved_at
    """)
    n = 0
    with DB.begin() as s:
        for _, r in raw.iterrows():
            c = r.get("Close")
            if c is None or not pd.notna(c):
                continue
            v = r.get("Volume")
            s.execute(q, {
                "session_date": pd.Timestamp(r["Date"]).date(),
                "close": float(c),
                "volume": int(round(float(v))) if v is not None and pd.notna(v) else None,
                "retrieved_at": now,
            })
            n += 1
    return n


def refresh_sp500_data(force_full=False):
    """
    First load:
      downloads history from SP500_HISTORY_START (default 2000-01-01).

    Later loads:
      downloads a rolling recent window and upserts it.

    This keeps the page fast because analysis reads PostgreSQL, not Yahoo live.
    """
    ensure_tables()
    started = datetime.now(timezone.utc)
    set_status({
        "status": "running",
        "started_at_utc": started.isoformat(),
        "message": "جارٍ تحديث قائمة S&P 500 والأسعار...",
    })

    constituents = fetch_sp500_constituents()
    _upsert_constituents(constituents)
    existing = _existing_minmax()

    full_start = os.getenv("SP500_HISTORY_START", "2000-01-01")
    recent_days = int(os.getenv("SP500_REFRESH_DAYS", "14"))
    today = datetime.now(timezone.utc).date()
    end = (today + timedelta(days=1)).isoformat()

    total_rows = 0
    touched = 0
    errors = []

    # Group symbols by needed start date.
    groups = {}
    for r in constituents.to_dict("records"):
        sym = r["symbol"]
        feed = r["feed_symbol"]
        if force_full or sym not in existing or existing[sym][2] < 200:
            start = full_start
        else:
            start = (today - timedelta(days=max(7, recent_days))).isoformat()
        groups.setdefault(start, []).append((sym, feed))

    for start, items in groups.items():
        for off in range(0, len(items), 25):
            batch = items[off:off+25]
            feeds = [x[1] for x in batch]
            try:
                raw = _download_batch(feeds, start, end)
            except Exception as exc:
                errors.append(f"{start} batch {off}: {type(exc).__name__}: {exc}")
                continue

            multi = isinstance(raw.columns, pd.MultiIndex)
            for sym, feed in batch:
                try:
                    x = _extract_yf(raw, feed, multi)
                    n = _write_daily(sym, x)
                    total_rows += n
                    touched += int(n > 0)
                except Exception as exc:
                    errors.append(f"{sym}: {type(exc).__name__}: {exc}")

            print(
                f"S&P500 refresh start={start} "
                f"{min(off+25,len(items))}/{len(items)} "
                f"symbols_touched={touched} rows={total_rows}",
                flush=True
            )

    # Index history must span the same stored start.
    try:
        index_rows = _refresh_index(full_start if force_full else (today - timedelta(days=max(7,recent_days))).isoformat(), end)
        # If index table empty, backfill full history.
        DB = database()
        with DB() as s:
            cnt = int(s.scalar(text(f"SELECT COUNT(*) FROM {INDEX_TABLE}")) or 0)
        if cnt < 200:
            index_rows += _refresh_index(full_start, end)
    except Exception as exc:
        index_rows = 0
        errors.append(f"^GSPC: {type(exc).__name__}: {exc}")

    result = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "message": "اكتمل تحديث بيانات S&P 500.",
        "constituents": len(constituents),
        "symbols_touched": touched,
        "rows_upserted": total_rows,
        "index_rows": index_rows,
        "history_start": full_start,
        "errors": errors[:30],
    }
    set_status(result)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    print(json.dumps(refresh_sp500_data(force_full=args.full), ensure_ascii=False, indent=2))
