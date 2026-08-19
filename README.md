# 📈 NIFTY 50 Stock Prediction & News Sentiment Analysis


<p align="center">
  <b>A market intelligence dashboard that combines NIFTY 50 historical data with financial-news sentiment to support next-price prediction.</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-limitations">Limitations</a>
</p>

---

## 🔍 Overview

Financial markets are influenced by more than historical prices. News, market narratives, and investor sentiment can affect short-term movement too.

This project explores that idea by combining:

- 📊 **NIFTY 50 OHLCV data** collected with `yfinance`
- 📰 **Financial news articles** scraped from Moneycontrol
- 🧠 **News sentiment** calculated with TextBlob
- 🤖 **LSTM-based time-series prediction** using TensorFlow/Keras
- 📈 **Interactive Streamlit dashboard** for market analysis and predictions

The application brings these pieces together into one workflow so users can inspect market history, sentiment trends, recent news, and model predictions from a single interface.

> **Important:** This is an educational machine-learning project, not a financial advisory system. Market prediction is inherently uncertain and model outputs should not be treated as trading advice.

---

## ✨ Features

### 📊 Market Analysis
- Interactive NIFTY 50 candlestick chart
- Open, High, Low, Close and Volume analysis
- Date-range filtering
- Latest market metrics displayed in dashboard cards
- Interactive Plotly visualizations

### 📰 News & Sentiment
- Financial-news collection from Moneycontrol
- Article text extraction with `newspaper3k`
- Sentiment polarity scoring with TextBlob
- Sentiment trend visualization
- Latest news list with article links

### 🔮 Prediction Engine
- TensorFlow/Keras LSTM model for sequence-based prediction
- Uses the latest **10 observations** as the prediction window
- Features include:
  - Open
  - High
  - Low
  - Volume
  - News sentiment
- Loads the trained model from the repository and generates a prediction through the Streamlit app

### 🎨 Dashboard Experience
- Streamlit-based web interface
- Three focused sections:
  - **Market Analysis**
  - **News & Sentiment**
  - **Predictions**
- Responsive wide layout
- Plotly-powered interactive charts

---

## 🧠 How It Works

```text
                 ┌─────────────────────┐
                 │   NIFTY 50 Data     │
                 │      yFinance       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Financial News      │
                 │    Moneycontrol     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Article Extraction  │
                 │    newspaper3k      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Sentiment Analysis  │
                 │      TextBlob       │
                 └──────────┬──────────┘
                            │
                            ▼
          ┌──────────────────────────────────┐
          │ Historical + Sentiment Features │
          └────────────────┬─────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │    LSTM Model       │
                 │    TensorFlow       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Price Prediction    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 └─────────────────────┘
```

---

## 🏗️ Architecture

### 1. Data Collection

The application retrieves NIFTY 50 market data through `yfinance` and collects financial-news metadata from Moneycontrol.

### 2. Sentiment Processing

Each article is downloaded and parsed. TextBlob computes a polarity score that is used as the project's news-sentiment feature.

### 3. Data Preparation

The project stores processed historical data in:

```text
nifty_index_with_sentiment.csv
```

News records are stored in:

```text
news_df.csv
```

### 4. Sequence Prediction

The prediction pipeline selects the latest 10 records and builds a 3D LSTM input shaped as:

```text
(samples, timesteps, features)
```

For the current implementation:

```text
1 × 10 × 5
```

### 5. Visualization

Streamlit presents the processed information through interactive charts, metrics, news cards, and a prediction workflow.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Market Data | yFinance |
| Deep Learning | TensorFlow / Keras |
| NLP | NLTK, TextBlob |
| Article Extraction | newspaper3k |
| Web Scraping | Requests, BeautifulSoup |
| Visualization | Plotly |
| Model Format | H5 / Keras model |

---

## 📁 Project Structure

```text
Stock_prediction-/
│
├── app.py
├── stock_prediction.py
├── requirements.txt
├── nifty_index_with_sentiment.csv
├── news_df.csv
├── nifty_price_prediction_model (1).h5
└── README.md
```

### File responsibilities

**`app.py`**

Main Streamlit application. It controls the dashboard layout, charts, filters, news display, and prediction interface.

**`stock_prediction.py`**

Handles data collection, article scraping, sentiment analysis, CSV updates, and LSTM prediction logic.

**`requirements.txt`**

Pinned Python dependencies required by the project.

**`nifty_index_with_sentiment.csv`**

Historical NIFTY 50 market data enriched with sentiment information.

**`news_df.csv`**

Collected news metadata and sentiment information.

**`nifty_price_prediction_model (1).h5`**

Saved TensorFlow/Keras model used by the prediction pipeline.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AdityaPowar11/Stock_prediction-.git
cd Stock_prediction-
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🖥️ Dashboard Walkthrough

### Market Analysis

Use the date filter to explore historical NIFTY 50 performance. The dashboard provides key metrics and an interactive candlestick + volume view.

### News & Sentiment

Review sentiment movement alongside the latest scraped financial headlines. Each article can be opened directly from the dashboard.

### Predictions

Use **Generate New Prediction** to run the prediction function with the latest available feature sequence and display the resulting market movement estimate.

---

## 📌 Model Input Features

The current prediction pipeline uses five features:

```text
Open
High
Low
Volume
news_sentiment
```

The model receives the latest 10 observations from these features and returns a predicted close-price value.

---

## 📊 Example Output Flow

```text
Historical Market Data
        │
        ├── OHLCV trends
        ├── Volume behavior
        └── Date filtering

Financial News
        │
        ├── Article extraction
        ├── Sentiment polarity
        └── Sentiment trend

             ↓

          LSTM Model
             ↓

       Predicted Price
             ↓

      Streamlit Dashboard
```

---

## ⚠️ Limitations & Practical Considerations

This project is a research and learning implementation. A few things are especially important when interpreting results:

- Stock prices are influenced by many variables that are not included in this model.
- A sentiment polarity score is a simplified representation of financial news.
- Scraped websites can change their HTML structure, which may break the news pipeline.
- Historical performance does not guarantee future performance.
- Prediction quality depends heavily on the training data, preprocessing, and model calibration.
- The current pipeline should be evaluated with proper out-of-sample testing before being considered for real trading decisions.

---

## 🔭 Future Improvements

Potential next steps for the project include:

- [ ] Add technical indicators such as RSI, MACD, EMA, and Bollinger Bands
- [ ] Compare LSTM against GRU, XGBoost, Random Forest, and Transformer-based models
- [ ] Use a dedicated financial sentiment model such as FinBERT
- [ ] Add proper train / validation / test time-series splits
- [ ] Report MAE, RMSE, MAPE, and directional accuracy
- [ ] Add backtesting and benchmark comparisons
- [ ] Improve prediction confidence reporting
- [ ] Add automated model retraining
- [ ] Deploy the Streamlit dashboard to a cloud platform
- [ ] Add stronger error handling for unavailable news sources and market data

---

## 🎯 Project Goal

The main goal of this project is to demonstrate how **time-series forecasting + NLP sentiment analysis + interactive data visualization** can be combined into a practical financial analytics application.

Rather than relying only on historical prices, the project experiments with incorporating market news sentiment into the prediction pipeline.

---

## 👤 Author

**Aditya Powar**

B.Tech CSE (Data Science) | Machine Learning | Data Science | Python

GitHub: [@AdityaPowar11](https://github.com/AdityaPowar11)

---

## ⭐ Support the Project

If you find this project useful for learning or experimentation, consider giving the repository a ⭐ on GitHub.

---

<div align="center">

### 📈 Learn from the data. Understand the news. Experiment with prediction.

</div>
