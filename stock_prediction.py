import warnings

import nltk
import pandas as pd
import yfinance as yf
from newspaper import Article
from textblob import TextBlob

from database import init_db, load_market_data, seed_from_csv, upsert_market_data, upsert_news, set_metadata

warnings.filterwarnings("ignore")
init_db()
nltk.download("punkt", quiet=True)


def get_sentiment_score(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return TextBlob(article.text).polarity
    except Exception as exc:
        print(f"Error processing URL {url}: {exc}")
        return None


def news_scraper():
    from bs4 import BeautifulSoup
    import requests

    url = "https://www.moneycontrol.com/indian-indices/NIFTY-50-9.html"
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    dates = soup.find_all("div", class_="date_block")
    matches = soup.find_all("div", class_="news_block")
    data = []

    for match, date in zip(matches, dates):
        anchor = match.find("a")
        if anchor is not None and date is not None:
            title = anchor.get_text(strip=True)
            news_url = anchor.get("href", "")
            news_date = date.get_text(strip=True)

            if news_url:
                data.append(
                    {"date": news_date, "title": title, "url": news_url}
                )

    return pd.DataFrame(data, columns=["date", "title", "url"])


def update_database():
    """Fetch the latest market/news data and persist it in SQLite."""
    stock = yf.Ticker("^NSEI")
    stock_data = stock.history(period="1d").reset_index()

    if stock_data.empty:
        raise RuntimeError("No NIFTY 50 market data was returned.")

    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date
    market_df = stock_data[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].copy()

    news_df = news_scraper()

    if not news_df.empty:
        scores = [get_sentiment_score(url) for url in news_df["url"]]
        valid_scores = [score for score in scores if score is not None]
        average_sentiment = (
            sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        )
        news_df["news_sentiment"] = scores
        market_df["news_sentiment"] = average_sentiment
    else:
        news_df["news_sentiment"] = pd.Series(dtype=float)
        market_df["news_sentiment"] = 0.0

    upsert_market_data(market_df)

    if not news_df.empty:
        upsert_news(news_df)

    set_metadata("last_market_update", pd.Timestamp.utcnow().isoformat())
    return market_df, news_df


def get_data():
    """Load persisted data, bootstrapping the database from CSV on first run."""
    seed_from_csv()
    return load_market_data()


def update_and_predict():
    import os
    import numpy as np
    from tensorflow.keras.models import load_model

    model_path = "nifty_price_prediction_model (1).h5"

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None

    df = get_data()

    if len(df) < 10:
        print("Insufficient data for prediction. Need at least 10 data points.")
        return None

    model = load_model(model_path)
    features = ["Open", "High", "Low", "Volume", "news_sentiment"]
    X_new = df[features].values[-10:]
    X_new = X_new.reshape(1, 10, len(features))

    try:
        y_pred = model.predict(X_new, verbose=0)

        max_close = df["Close"].max()
        min_close = df["Close"].min()
        predicted_price = y_pred[0][0] * (max_close - min_close) + min_close

        print(f"Predicted Close Price: ₹{predicted_price:,.2f}")
        return float(predicted_price)
    except Exception as exc:
        print(f"Error in prediction: {exc}")
        return None
