import pandas as pd
import yfinance as yf
from textblob import TextBlob
from newspaper import Article
import nltk
import datetime
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')

# Function to get sentiment score for a given URL
def get_sentiment_score(url):
    try:
        # Extract article
        article = Article(url)
        article.download()
        article.parse()

        # Sentiment analysis
        analysis = TextBlob(article.text)
        polarity = analysis.polarity

        return polarity

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

# News Scraper function to generate URLs
def news_scraper():
    from bs4 import BeautifulSoup
    import requests

    url = 'https://www.moneycontrol.com/indian-indices/NIFTY-50-9.html'
    web = requests.get(url)
    content = web.text
    soup = BeautifulSoup(content, 'lxml')
    dates = soup.find_all('div', class_='date_block')
    matches = soup.find_all('div', class_='news_block')
    data = []

    for match, date in zip(matches, dates):
        anchor = match.find('a')
        if anchor is not None and date is not None:
            title = anchor.get_text(strip=True)
            news_url = anchor.get('href', '')
            news_date = date.get_text(strip=True)

            data.append({'date': news_date, 'title': title, 'url': news_url})

    news_df = pd.DataFrame(data)
    return news_df

# Fetch stock data
stock = yf.Ticker("^NSEI")
stock_data = stock.history(period="1d")
stock_data = stock_data.reset_index()
stock_data['Date'] = stock_data['Date'].dt.date

# Extract required columns
df2 = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Get the news URLs
news_df = news_scraper()
urls = news_df['url'].tolist()

# Calculate sentiment scores for each URL
sentiment_scores = [get_sentiment_score(url) for url in urls]

# Calculate average sentiment score
if sentiment_scores:
    average_sentiment_score = sum([score for score in sentiment_scores if score is not None]) / len(sentiment_scores)
else:
    average_sentiment_score = 0

# Add sentiment score to DataFrames
df2['news_sentiment'] = average_sentiment_score
news_df['news_sentiment'] = average_sentiment_score

# Handle `nifty_index_with_sentiment.csv`
try:
    df = pd.read_csv("nifty_index_with_sentiment.csv")
    df = pd.concat([df, df2], ignore_index=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
except FileNotFoundError:
    df = df2
df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
df.to_csv("nifty_index_with_sentiment.csv", index=False)

# Handle `news_df.csv`
try:
    news_df1 = pd.read_csv("news_df.csv")
    news_df1 = pd.concat([news_df1, news_df], ignore_index=True)
    news_df1.drop_duplicates(subset=["url"], keep="last", inplace=True)
except FileNotFoundError:
    news_df1 = news_df

news_df1.to_csv("news_df.csv", index=False)

# print(df)

def update_and_predict():
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model
    import os


    model_path = 'nifty_price_prediction_model (1).h5'
    

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None

    # Load the saved model
    model = load_model(model_path)

    # Define feature columns
    features = ['Open', 'High', 'Low', 'Volume', 'news_sentiment']

    # Ensure 'df' is populated with the latest data
    if len(df) < 10:
        print("Insufficient data for prediction. Need at least 10 data points.")
        return None

    # Extract the latest 10 records for prediction
    X_new = df[features].values[-10:]

    # Reshape to 3D format: (1, timesteps, features)
    X_new = X_new.reshape(1, 10, len(features))

    # Get min and max values for scaling back the prediction
    max_close = df['Close'].max()
    min_close = df['Close'].min()

    # Predict using the model
    try:
        y_pred = model.predict(X_new)
        # Scale the prediction back to actual price range
        predicted_price = y_pred[0][0] * (max_close - min_close) + min_close
        predicted_price =predicted_price/10
        print(f"Predicted Close Price: {(predicted_price):,.2f}")
        return (predicted_price)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

prediction = update_and_predict()
print(f"The predicted price is: {prediction}")

