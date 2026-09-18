import datetime
import warnings
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from database import load_latest_news
from stock_prediction import get_data, update_and_predict, update_database

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NIFTY 50 Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: rgb(78, 203, 212);
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("📊 Controls & Filters")

try:
    df = get_data()
    if df.empty:
        st.sidebar.warning("No historical data available. Use Update Data below.")
        df_filtered = None
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        min_date = df["Date"].min()
        max_date = df["Date"].max()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=30), max_date),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (
                (df["Date"].dt.date >= start_date)
                & (df["Date"].dt.date <= end_date)
            )
            df_filtered = df.loc[mask]
        else:
            df_filtered = df

except Exception as exc:
    st.sidebar.error(f"Database error: {exc}")
    df_filtered = None

st.sidebar.divider()
if st.sidebar.button("🔄 Update Market & News Data"):
    with st.spinner("Fetching the latest NIFTY 50 and news data..."):
        try:
            update_database()
            st.success("Database updated successfully.")
            st.rerun()
        except Exception as exc:
            st.error(f"Update failed: {exc}")

st.title("📈 NIFTY 50 Stock Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Market Analysis", "📰 News & Sentiment", "🔮 Predictions"]
)

with tab1:
    if df_filtered is not None and not df_filtered.empty:
        col1, col2, col3, col4 = st.columns(4)
        latest_data = df_filtered.iloc[-1]

        with col1:
            day_change = (
                (latest_data["Close"] - latest_data["Open"])
                / latest_data["Open"]
                * 100
            )
            st.metric(
                "Latest Close",
                f"₹{latest_data['Close']:,.2f}",
                f"{day_change:,.2f}%",
            )

        with col2:
            st.metric("Volume", f"{latest_data['Volume']:,.0f}")

        with col3:
            st.metric("Day's High", f"₹{latest_data['High']:,.2f}")

        with col4:
            st.metric("Day's Low", f"₹{latest_data['Low']:,.2f}")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price", "Volume"),
            row_width=[0.7, 0.3],
        )

        fig.add_trace(
            go.Candlestick(
                x=df_filtered["Date"],
                open=df_filtered["Open"],
                high=df_filtered["High"],
                low=df_filtered["Low"],
                close=df_filtered["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df_filtered["Date"],
                y=df_filtered["Volume"],
                name="Volume",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="NIFTY 50 Price Movement",
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=800,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        if df_filtered is not None and not df_filtered.empty:
            sentiment_fig = go.Figure()
            sentiment_fig.add_trace(
                go.Scatter(
                    x=df_filtered["Date"],
                    y=df_filtered["news_sentiment"],
                    mode="lines+markers",
                    name="Sentiment Score",
                )
            )
            sentiment_fig.update_layout(
                title="News Sentiment Analysis Trend",
                yaxis_title="Sentiment Score",
                xaxis_title="Date",
                template="plotly_white",
            )
            st.plotly_chart(sentiment_fig, use_container_width=True)

    with col2:
        st.subheader("Latest News")
        news_df = load_latest_news(limit=5)

        if news_df.empty:
            st.info("No news data available. Update the database first.")
        else:
            for _, row in news_df.iterrows():
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>{row['title']}</h4>
                        <p>Date: {row['date']}</p>
                        <p>Sentiment: {row['news_sentiment']:.2f}</p>
                        <a href="{row['url']}" target="_blank">Read More</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab3:
    st.subheader("Stock Price Prediction")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Generate New Prediction", key="predict_button"):
            with st.spinner("Calculating prediction..."):
                try:
                    prediction = update_and_predict()

                    if prediction is None:
                        st.error("Prediction could not be generated.")
                    elif df_filtered is not None and not df_filtered.empty:
                        last_close = df_filtered["Close"].iloc[-1]
                        change = ((prediction - last_close) / last_close) * 100
                        st.metric(
                            "Predicted Close",
                            f"₹{prediction:,.2f}",
                        )
                        st.metric("Predicted Change", f"{change:,.2f}%")
                except Exception as exc:
                    st.error(f"Error generating prediction: {exc}")

    with col2:
        if df_filtered is not None and not df_filtered.empty:
            st.subheader("Recent Market Data")
            st.line_chart(df_filtered[["Close"]].tail(10))

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Data is persisted in a local SQLite database.")
with col2:
    st.markdown(
        "Last checked: "
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
