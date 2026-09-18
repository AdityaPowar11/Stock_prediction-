from pathlib import Path
import json
import shutil
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from database import load_market_data

SEQ_LEN = 20
MODEL_DIR = Path("models")
FEATURES = ["Open","High","Low","Close","Volume","news_sentiment","return_1d","volatility_10d","range_pct","volume_change"]

def feature_frame():
    df = load_market_data().copy()
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["volume_change"] = df["Volume"].pct_change()
    return df.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)

def sequences(df):
    xs=StandardScaler(); ys=StandardScaler()
    Xs=xs.fit_transform(df[FEATURES]); y=ys.fit_transform(df[["Close"]]).ravel()
    X=[]; Y=[]
    for i in range(SEQ_LEN, len(df)-1):
        X.append(Xs[i-SEQ_LEN:i]); Y.append(y[i+1])
    return np.asarray(X),np.asarray(Y),xs,ys

def models(shape):
    return {
      "lstm": Sequential([LSTM(64,input_shape=shape),Dropout(.2),Dense(32,activation="relu"),Dense(1)]),
      "gru": Sequential([GRU(64,input_shape=shape),Dropout(.2),Dense(32,activation="relu"),Dense(1)]),
      "cnn_lstm": Sequential([Conv1D(32,3,activation="relu",input_shape=shape),MaxPooling1D(2),LSTM(48),Dense(1)])
    }

def train():
    df=feature_frame(); X,y,xs,ys=sequences(df)
    if len(X)<80: raise RuntimeError("Not enough history to train reliably.")
    cut=int(len(X)*.8); Xtr,Xv=X[:cut],X[cut:]; ytr,yv=y[:cut],y[cut:]
    MODEL_DIR.mkdir(exist_ok=True)
    results=[]
    for name,m in models(X.shape[1:]).items():
        m.compile(optimizer=Adam(1e-3),loss="huber")
        m.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=60,batch_size=32,verbose=0,
              callbacks=[EarlyStopping(patience=8,restore_best_weights=True),ReduceLROnPlateau(patience=4)])
        pred=ys.inverse_transform(m.predict(Xv,verbose=0)).ravel()
        actual=ys.inverse_transform(yv.reshape(-1,1)).ravel()
        mae=mean_absolute_error(actual,pred); rmse=mean_squared_error(actual,pred)**.5
        direction=float(np.mean(np.sign(np.diff(actual))==np.sign(np.diff(pred)))) if len(actual)>1 else 0
        path=MODEL_DIR/f"{name}.keras"; m.save(path)
        results.append((mae,-direction,name,path,rmse,direction))
    best=min(results)
    shutil.copy2(best[3],MODEL_DIR/"best_model.keras")
    joblib.dump(xs,MODEL_DIR/"best_features.pkl"); joblib.dump(ys,MODEL_DIR/"best_target.pkl")
    (MODEL_DIR/"CURRENT_MODEL.txt").write_text(best[2])
    metrics={"model":best[2],"mae":best[0],"rmse":best[4],"directional_accuracy":best[5]}
    (MODEL_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(metrics)

if __name__=="__main__": train()
