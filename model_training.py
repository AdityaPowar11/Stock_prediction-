import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from database import get_connection, init_db, load_market_data, record_model_metric

SEQUENCE_LENGTH = 20
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["Open", "High", "Low", "Close", "Volume", "news_sentiment"]
TARGET = "Close"


def prepare_data(df):
    df = df.copy().sort_values("Date")
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = df["Volume"].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    features = FEATURES + ["return_1d", "volatility_10d", "range_pct", "volume_change"]
    X, y, dates = [], [], []

    for i in range(SEQUENCE_LENGTH, len(df) - 1):
        X.append(df[features].iloc[i-SEQUENCE_LENGTH:i].values)
        y.append(df["Close"].iloc[i + 1])
        dates.append(df["Date"].iloc[i + 1])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), dates, features


def build_model(name, input_shape):
    if name == "lstm":
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
    elif name == "gru":
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
    elif name == "cnn_lstm":
        model = Sequential([
            Conv1D(32, 3, activation="relu", input_shape=input_shape),
            LSTM(48),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
    else:
        raise ValueError(f"Unknown model: {name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def scale_sequences(X_train, X_val, y_train, y_val):
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def scale_x(X):
        shape = X.shape
        return feature_scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)

    target_scaler = StandardScaler()
    target_scaler.fit(y_train.reshape(-1, 1))

    return (
        scale_x(X_train),
        scale_x(X_val),
        target_scaler.transform(y_train.reshape(-1, 1)).ravel(),
        target_scaler.transform(y_val.reshape(-1, 1)).ravel(),
        feature_scaler,
        target_scaler,
    )


def train_and_select():
    init_db()
    df = load_market_data()

    if len(df) < 300:
        raise RuntimeError("At least 300 market observations are required for reliable training.")

    X, y, dates, features = prepare_data(df)
    split = int(len(X) * 0.8)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_s, X_val_s, y_train_s, y_val_s, feature_scaler, target_scaler = scale_sequences(
        X_train, X_val, y_train, y_val
    )

    results = []
    for name in ["lstm", "gru", "cnn_lstm"]:
        tf.keras.backend.clear_session()
        model = build_model(name, X_train_s.shape[1:])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ]

        model.fit(
            X_train_s,
            y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=80,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        pred_scaled = model.predict(X_val_s, verbose=0).ravel()
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))

        actual_direction = np.sign(np.diff(np.r_[y_train[-1], y_val]))
        predicted_direction = np.sign(np.diff(np.r_[y_train[-1], pred]))
        directional_accuracy = np.mean(actual_direction == predicted_direction)

        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model.save(f"{MODEL_DIR}/{name}_{version}.keras")
        joblib.dump(feature_scaler, f"{MODEL_DIR}/{name}_{version}_features.pkl")
        joblib.dump(target_scaler, f"{MODEL_DIR}/{name}_{version}_target.pkl")

        record_model_metric(name, version, mae, rmse, directional_accuracy)
        results.append((name, version, mae, rmse, directional_accuracy))

    # Select by validation MAE, with directional accuracy as a secondary signal.
    results.sort(key=lambda x: (x[2], -x[4]))
    best_name, best_version, best_mae, best_rmse, best_direction = results[0]

    with open(f"{MODEL_DIR}/CURRENT_MODEL.txt", "w", encoding="utf-8") as f:
        f.write(f"{best_name},{best_version}\n")

    # Copy the selected artifacts to stable paths used by daily prediction.
    import shutil
    shutil.copyfile(
        f"{MODEL_DIR}/{best_name}_{best_version}.keras",
        f"{MODEL_DIR}/best_model.keras",
    )
    shutil.copyfile(
        f"{MODEL_DIR}/{best_name}_{best_version}_features.pkl",
        f"{MODEL_DIR}/best_features.pkl",
    )
    shutil.copyfile(
        f"{MODEL_DIR}/{best_name}_{best_version}_target.pkl",
        f"{MODEL_DIR}/best_target.pkl",
    )

    print(
        f"Selected {best_name} {best_version} | "
        f"MAE={best_mae:.4f} RMSE={best_rmse:.4f} "
        f"DirectionalAccuracy={best_direction:.4f}"
    )


if __name__ == "__main__":
    train_and_select()
