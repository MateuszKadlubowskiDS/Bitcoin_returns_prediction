# === 1. Imports and settings ===
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import ta  # technical analysis library

LOOKBACK = 60
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# === 2. Load BTC OHLCV data ===
btc_path = r"data\processed\btc_usd_daily.csv"

btc = pd.read_csv(
    btc_path,
    parse_dates=['Date']
)
btc = btc.sort_values('Date').set_index('Date')
print(btc.head())


# === 3a. Feature engineering: variables from previous models ===
eps = 1e-9

# Lower wick ratio
btc['lower_wick_ratio'] = (np.minimum(btc['Open'], btc['Close']) - btc['Low']) / (btc['High'] - btc['Low'] + eps)

# Gap open (log)
btc['gap_open_log'] = np.log(btc['Open']) - np.log(btc['Close'].shift(1))

# Volume features: 20-day z-score
k = 20
btc['vol_ma_20'] = btc['Volume'].rolling(k).mean()
btc['vol_std_20'] = btc['Volume'].rolling(k).std()
btc['z_vol_20'] = (btc['Volume'] - btc['vol_ma_20']) / (btc['vol_std_20'] + eps)

# Momentum (20 days)
btc['momentum_20'] = np.log(btc['Close']) - np.log(btc['Close'].shift(20))

# Log return (feature + shifted target)
btc['log_ret'] = np.log(btc['Close'] / btc['Close'].shift(1))
btc['target'] = btc['log_ret'].shift(-1)


# === 3b. Feature engineering: technical indicators ===

# RSI 14
btc['RSI14'] = ta.momentum.RSIIndicator(btc['Close'], window=14).rsi()

# EMA14 and flag
btc['EMA14'] = ta.trend.EMAIndicator(btc['Close'], window=14).ema_indicator()
btc['EMA14_flag'] = (btc['Close'] > btc['EMA14']).astype(int)

# SMA120 and flag + distance
btc['SMA120'] = ta.trend.SMAIndicator(btc['Close'], window=120).sma_indicator()
btc['SMA120_flag'] = (btc['Close'] > btc['SMA120']).astype(int)
btc['dist_SMA120'] = (btc['Close'] / btc['SMA120']) - 1

# MACD (12,26,9)
macd = ta.trend.MACD(btc['Close'], window_slow=26, window_fast=12, window_sign=9)
btc['MACD_hist'] = macd.macd_diff()
btc['MACD_cross'] = (macd.macd() > macd.macd_signal()).astype(int)

# ADX14
adx = ta.trend.ADXIndicator(btc['High'], btc['Low'], btc['Close'], window=14)
btc['ADX14'] = adx.adx()

# Donchian breakout (20 days)
btc['donchian20_high'] = btc['High'].rolling(20).max()
btc['donchian20_low'] = btc['Low'].rolling(20).min()
btc['Donchian20_flag'] = ((btc['Close'] > btc['donchian20_high']).astype(int) -
                          (btc['Close'] < btc['donchian20_low']).astype(int))

# Remove NaN values
btc = btc.dropna()


# === 4. Build time sequences ===
features = [
    'log_ret',
    'RSI14','EMA14_flag','SMA120_flag','dist_SMA120',
    'MACD_hist','MACD_cross','ADX14','Donchian20_flag'
]

X, y, dates = [], [], []
for i in range(LOOKBACK, len(btc)):
    window = btc[features].values[i-LOOKBACK:i]
    target = btc['target'].values[i]
    X.append(window)
    y.append(target)
    dates.append(btc.index[i])

X = np.array(X)
y = np.array(y)
dates = pd.DatetimeIndex(dates)

print(X.shape, y.shape)


# === 5. Train/validation/test split ===
train_end = pd.to_datetime("2021-12-31")
val_end   = pd.to_datetime("2022-12-31")

m_train = dates <= train_end
m_val   = (dates > train_end) & (dates <= val_end)
m_test  = dates > val_end

X_train, y_train = X[m_train], y[m_train]
X_val, y_val     = X[m_val], y[m_val]
X_test, y_test   = X[m_test], y[m_test]

print(X_train.shape, X_val.shape, X_test.shape)


# === 6. Scaling ===
scalers = []
X_train_s, X_val_s, X_test_s = [], [], []
for j in range(X.shape[2]):
    sc = StandardScaler()
    f_train = X_train[:,:,j]
    f_val   = X_val[:,:,j]
    f_test  = X_test[:,:,j]
    sc.fit(f_train)
    X_train_s.append(sc.transform(f_train))
    X_val_s.append(sc.transform(f_val))
    X_test_s.append(sc.transform(f_test))
    scalers.append(sc)

X_train_s = np.stack(X_train_s, axis=2)
X_val_s   = np.stack(X_val_s, axis=2)
X_test_s  = np.stack(X_test_s, axis=2)

print(X_train_s.shape, X_val_s.shape, X_test_s.shape)


# === 7. LSTM model (all features) ===
model = Sequential([
    LSTM(64, input_shape=(LOOKBACK, X.shape[2])),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# Evaluation on test set
y_pred = model.predict(X_test_s).reshape(-1)

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
r2 = r2_score(y_test, y_pred)
hit = (np.sign(y_test) == np.sign(y_pred)).mean()
ic = spearmanr(y_test, y_pred).correlation

print("Full LSTM (Model 5a features):", {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2,
    "HitRate": hit,
    "IC": ic
})