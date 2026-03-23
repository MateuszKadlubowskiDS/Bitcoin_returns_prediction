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

LOOKBACK = 60
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# === 2. Load BTC + crypto + macro data ===
btc_path = r"data\processed\btc_usd_daily.csv"
crypto_feat_path = r"data\processed\btc_crypto_features.csv"
macro_path = r"data\processed\macro_daily.csv"

btc = pd.read_csv(btc_path, parse_dates=['Date']).sort_values('Date').set_index('Date')
crypto = pd.read_csv(crypto_feat_path, parse_dates=['Date']).set_index('Date')
macro = pd.read_csv(macro_path, parse_dates=['Date']).set_index('Date')

# Clean macro data (string → float)
for col in macro.columns:
    macro[col] = (
        macro[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', '', regex=False)
        .replace(["", "nan"], np.nan)
        .astype(float)
    )

# Align macro data to BTC start date
macro = macro[macro.index >= "2014-09-17"]

# Merge datasets
btc = btc.join(crypto, how='left')
btc = btc.join(macro, how='left')


# === 3a. Feature engineering: variables from previous models ===
eps = 1e-9

# Volume z-score
k = 20
btc['vol_ma_20'] = btc['Volume'].rolling(k).mean()
btc['vol_std_20'] = btc['Volume'].rolling(k).std()
btc['z_vol_20'] = (btc['Volume'] - btc['vol_ma_20']) / (btc['vol_std_20'] + eps)

# Lower wick ratio
btc['lower_wick_ratio'] = (np.minimum(btc['Open'], btc['Close']) - btc['Low']) / (btc['High'] - btc['Low'] + eps)

# Gap open (log)
btc['gap_open_log'] = np.log(btc['Open']) - np.log(btc['Close'].shift(1))

# Momentum 20
btc['momentum_20'] = np.log(btc['Close']) - np.log(btc['Close'].shift(20))

# SMA120 distance
btc['SMA120'] = btc['Close'].rolling(120).mean()
btc['dist_SMA120'] = (btc['Close'] / btc['SMA120']) - 1

# MACD cross
ema12 = btc['Close'].ewm(span=12, adjust=False).mean()
ema26 = btc['Close'].ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
btc['MACD_cross'] = (macd > signal).astype(int)

# Log return + target
btc['log_ret'] = np.log(btc['Close'] / btc['Close'].shift(1))
btc['target'] = btc['log_ret'].shift(-1)


# === 3b. Feature engineering: macro variables ===
btc['ret_SPX'] = np.log(btc['SPX'] / btc['SPX'].shift(1))
btc['ret_DXY'] = np.log(btc['DXY'] / btc['DXY'].shift(1))
btc['ret_Gold'] = np.log(btc['Gold'] / btc['Gold'].shift(1))
btc['ret_VIX'] = np.log(btc['VIX'] / btc['VIX'].shift(1))
btc['ret_US10Y'] = np.log(btc['US10Y'] / btc['US10Y'].shift(1))
btc['ret_BCM'] = np.log(btc['BCM'] / btc['BCM'].shift(1))
btc['ret_Oil'] = np.log(btc['Oil'] / btc['Oil'].shift(1))
# CPI, M2, FEDFUNDS remain in levels

btc = btc.dropna()


# === 4. Build time sequences ===
features = [
    'log_ret',
    'ret_SPX','ret_DXY','ret_Gold','ret_VIX','ret_US10Y',
    'ret_BCM','ret_Oil','CPI','M2','FEDFUNDS'
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


# === 7. LSTM full model ===
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

# Evaluation
y_pred = model.predict(X_test_s).reshape(-1)

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
r2 = r2_score(y_test, y_pred)
hit = (np.sign(y_test) == np.sign(y_pred)).mean()
ic = spearmanr(y_test, y_pred).correlation

print("Full LSTM (Model 6 - all macro):", {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2,
    "HitRate": hit,
    "IC": ic
})
