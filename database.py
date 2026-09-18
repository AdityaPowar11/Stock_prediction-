import os
import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path(os.getenv("STOCK_DB_PATH", "stock_prediction.db"))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                news_sentiment REAL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news (
                url TEXT PRIMARY KEY,
                date TEXT,
                title TEXT NOT NULL,
                news_sentiment REAL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def upsert_market_data(df):
    if df is None or df.empty:
        return

    required = ["Date", "Open", "High", "Low", "Close", "Volume", "news_sentiment"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    records = []
    for row in df[required].itertuples(index=False, name=None):
        date, open_, high, low, close, volume, sentiment = row
        records.append(
            (
                pd.to_datetime(date).date().isoformat(),
                float(open_),
                float(high),
                float(low),
                float(close),
                float(volume),
                float(sentiment) if pd.notna(sentiment) else 0.0,
            )
        )

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO market_data
                (date, open, high, low, close, volume, news_sentiment, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                news_sentiment=excluded.news_sentiment,
                updated_at=CURRENT_TIMESTAMP
            """,
            records,
        )


def upsert_news(df):
    if df is None or df.empty:
        return

    required = ["date", "title", "url", "news_sentiment"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing news columns: {missing}")

    records = []
    for row in df[required].itertuples(index=False, name=None):
        date, title, url, sentiment = row
        if pd.isna(url) or not str(url).strip():
            continue
        records.append(
            (
                str(url).strip(),
                str(date) if pd.notna(date) else "",
                str(title) if pd.notna(title) else "Untitled",
                float(sentiment) if pd.notna(sentiment) else 0.0,
            )
        )

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO news (url, date, title, news_sentiment, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
                date=excluded.date,
                title=excluded.title,
                news_sentiment=excluded.news_sentiment,
                updated_at=CURRENT_TIMESTAMP
            """,
            records,
        )


def load_market_data():
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                date AS Date,
                open AS Open,
                high AS High,
                low AS Low,
                close AS Close,
                volume AS Volume,
                news_sentiment
            FROM market_data
            ORDER BY date
            """,
            conn,
        )


def load_latest_news(limit=5):
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT date, title, url, news_sentiment
            FROM news
            ORDER BY rowid DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def database_is_empty():
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()
        return row[0] == 0


def seed_from_csv():
    """One-time bootstrap for an existing checkout that already has CSV data."""
    if not database_is_empty():
        return

    market_path = Path("nifty_index_with_sentiment.csv")
    news_path = Path("news_df.csv")

    if market_path.exists():
        market_df = pd.read_csv(market_path)
        upsert_market_data(market_df)

    if news_path.exists():
        news_df = pd.read_csv(news_path)
        upsert_news(news_df)


def set_metadata(key, value):
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_metadata (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
            """,
            (key, str(value)),
        )
