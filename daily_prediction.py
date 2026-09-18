from datetime import datetime, timedelta

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from database import init_db, load_market_data, record_prediction, update_prediction_actuals
from stock_prediction import update_database

MODEL_PATH = "models/best_model.keras"
FEATURE_SCALER_PATH = "models/best_features.pkl"
TARGET_SCALER_PATH = "models/best_target.pkl"


def predict():
    init_db()
    update_database()
    update_prediction_actuals()

    df = load_market_data().sort_values("Date")
    if len(df) < 25:
        raise RuntimeError("Not enough market data for a daily prediction.")

    model = load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    features = [
        "Open", "High", "Low", "Close", "Volume", "news_sentiment",
        "return_1d", "volatility_10d", "range_pct", "volume_change"
    ]

    work = df.copy()
    work["return_1d"] = work["Close"].pct_change()
    work["volatility_10d"] = work["return_1d"].rolling(10).std()
    work["range_pct"] = (work["High"] - work["Low"]) / work["Close"]
    work["volume_change"] = work["Volume"].pct_change()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if len(work) < 20:
        raise RuntimeError("Not enough clean observations for prediction.")

    X = work[features].iloc[-20:].values.astype(np.float32)
    X = feature_scaler.transform(X).reshape(1, 20, len(features))

    pred_scaled = model.predict(X, verbose=0).ravel()[0]
    predicted_close = target_scaler.inverse_transform([[pred_scaled]])[0][0]

    last_date = work["Date"].iloc[-1]
    prediction_date = (last_date + timedelta(days=1)).date()

    with open("models/CURRENT_MODEL.txt", encoding="utf-8") as f:
        model_name, model_version = f.read().strip().split(",")

    record_prediction(
        prediction_date,
        float(predicted_close),
        model_name,
        model_version,
    )

    print(
        f"{prediction_date}: predicted NIFTY 50 close = "
        f"{predicted_close:,.2f} using {model_name}/{model_version}"
    )


if __name__ == "__main__":
    predict()
