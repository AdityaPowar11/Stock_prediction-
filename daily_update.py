"""
Reliable daily data collection for the NIFTY 50 dashboard.

Market data is the required source of truth. News collection is best-effort:
a broken news page must not prevent the market row from being persisted.
"""
import time
import pandas as pd
import yfinance as yf

from database import init_db, upsert_market_data, upsert_news, set_metadata
from stock_prediction import news_scraper, get_sentiment_score

MARKET_RETRIES = 3
NEWS_RETRIES = 3


def fetch_market_data():
    last_error = None
    for attempt in range(1, MARKET_RETRIES + 1):
        try:
            stock = yf.Ticker("^NSEI")
            data = stock.history(period="1d", auto_adjust=False).reset_index()
            if data.empty:
                raise RuntimeError("yfinance returned no NIFTY 50 rows.")
            data["Date"] = pd.to_datetime(data["Date"]).dt.date
            result = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            result["news_sentiment"] = 0.0
            if result["Close"].isna().all():
                raise RuntimeError("NIFTY 50 close price is empty.")
            return result
        except Exception as exc:
            last_error = exc
            print(f"Market attempt {attempt}/{MARKET_RETRIES} failed: {exc}")
            if attempt < MARKET_RETRIES:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Market update failed after {MARKET_RETRIES} attempts: {last_error}")


def fetch_news_data():
    last_error = None
    for attempt in range(1, NEWS_RETRIES + 1):
        try:
            news = news_scraper()
            if news is None:
                raise RuntimeError("News scraper returned None.")
            return news
        except Exception as exc:
            last_error = exc
            print(f"News attempt {attempt}/{NEWS_RETRIES} failed: {exc}")
            if attempt < NEWS_RETRIES:
                time.sleep(5 * attempt)
    print(f"News collection skipped after {NEWS_RETRIES} attempts: {last_error}")
    return pd.DataFrame(columns=["date", "title", "url", "news_sentiment"])


def main():
    init_db()

    market_df = fetch_market_data()

    # Persist market data immediately. News failures cannot block this.
    upsert_market_data(market_df)

    news_df = fetch_news_data()
    sentiment = 0.0

    if not news_df.empty:
        scores = [get_sentiment_score(url) for url in news_df["url"]]
        news_df["news_sentiment"] = scores
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            sentiment = sum(valid_scores) / len(valid_scores)
        upsert_news(news_df)

    # Store sentiment against the market row after both collectors finish.
    market_df["news_sentiment"] = sentiment
    upsert_market_data(market_df)

    latest_date = str(market_df["Date"].max())
    set_metadata("last_market_update", pd.Timestamp.utcnow().isoformat())
    set_metadata("last_market_date", latest_date)
    set_metadata(
        "last_news_update",
        pd.Timestamp.utcnow().isoformat() if not news_df.empty else "news collection failed",
    )
    set_metadata("last_news_rows", str(len(news_df)))

    print(
        f"SUCCESS: market rows={len(market_df)}, "
        f"latest_date={latest_date}, news rows={len(news_df)}"
    )


if __name__ == "__main__":
    main()
