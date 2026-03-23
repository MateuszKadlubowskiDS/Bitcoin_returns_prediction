import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Settings
LOOKBACK = 60
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# === 1. Load BTC OHLCV data ===
# Use full path and parse_dates to ensure Date column is datetime

btc_path = r"data\processed\btc_usd_daily.csv"

btc = pd.read_csv(
    btc_path,
    parse_dates=['Date']
)

# Set Date as index
btc = btc.sort_values('Date').set_index('Date')

print(btc.index)
print(btc.head())


# === 3. Compute log returns from Close price ===
# Target: predict log return at time t using previous 60 days

btc['Close'] = btc['Close'].astype(float)
btc['log_ret'] = np.log(btc['Close'] / btc['Close'].shift(1))
btc = btc.dropna()

btc[['Close', 'log_ret']].head()


# === 4. Build time sequences (lookback=60) ===
X, y, dates = [], [], []

for i in range(LOOKBACK, len(btc)):
    window = btc['log_ret'].values[i-LOOKBACK:i]
    target = btc['log_ret'].values[i]
    X.append(window)
    y.append(target)
    dates.append(btc.index[i])

X = np.array(X).reshape(-1, LOOKBACK, 1)
y = np.array(y)
dates = pd.DatetimeIndex(dates)

print("Date range:", dates.min(), "→", dates.max())
print("Shapes:", X.shape, y.shape)


# === 5. Train/validation/test split ===
# train: until 2021-12-31
# val: year 2022
# test: from 2023-01-01

assert isinstance(dates, pd.DatetimeIndex), "dates is not a DatetimeIndex!"

train_end = pd.to_datetime("2021-12-31")
val_end   = pd.to_datetime("2022-12-31")

m_train = dates <= train_end
m_val   = (dates > train_end) & (dates <= val_end)
m_test  = dates > val_end

X_train, y_train = X[m_train], y[m_train]
X_val, y_val     = X[m_val], y[m_val]
X_test, y_test   = X[m_test], y[m_test]

print("Date range:", dates.min(), "→", dates.max())
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# === 6. Scaling ===
# Standardization (mean=0, std=1) fitted on train only,
# then applied to val and test

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, LOOKBACK)).reshape(X_train.shape)
X_val   = scaler.transform(X_val.reshape(-1, LOOKBACK)).reshape(X_val.shape)
X_test  = scaler.transform(X_test.reshape(-1, LOOKBACK)).reshape(X_test.shape)


# === 7. Build and train LSTM model ===
# Architecture: LSTM(64) → Dropout(0.3) → Dense(1)

model = Sequential([
    LSTM(64, input_shape=(LOOKBACK, 1)),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[es],
    verbose=1
)


# === 8. Evaluation on test set ===
y_pred = model.predict(X_test).reshape(-1)

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
hit = (np.sign(y_test) == np.sign(y_pred)).mean()

print({"MAE": mae, "RMSE": rmse, "HitRate": hit})


# === 9. Prediction vs actual plot ===
plt.figure(figsize=(12,5))
plt.plot(dates[m_test], y_test, label="True")
plt.plot(dates[m_test], y_pred, label="Predicted")
plt.legend()
plt.title("BTC — test set predictions (log-returns)")
plt.show()


# === Price reconstruction from log-returns (1-step ahead) ===

true_prices = btc.loc[dates[m_test], "Close"].values

P_pred_1step = true_prices[:-1] * np.exp(y_pred[1:])

plt.figure(figsize=(12,6))
plt.plot(dates[m_test], true_prices, label="True Close", color="black")
plt.plot(dates[m_test][1:], P_pred_1step, label="Predicted Close (1-step)", color="red", alpha=0.5)
plt.legend()
plt.title("BTC — price reconstruction (1-step ahead from log-returns)")
plt.tight_layout()
plt.show()


# === Bias analysis and comparison with random baseline ===

mean_true = y_test.mean()
mean_pred = y_pred.mean()
print("Mean log-return (true):", mean_true)
print("Mean log-return (pred):", mean_pred)

mae_model = mean_absolute_error(y_test, y_pred)
rmse_model = math.sqrt(mean_squared_error(y_test, y_pred))
print("Model -> MAE:", mae_model, " RMSE:", rmse_model)

# Random baseline (sampling from empirical distribution)
np.random.seed(42)
y_rand = np.random.choice(y_test, size=len(y_test), replace=True)

mae_rand = mean_absolute_error(y_test, y_rand)
rmse_rand = math.sqrt(mean_squared_error(y_test, y_rand))
print("Random baseline -> MAE:", mae_rand, " RMSE:", rmse_rand)

# Improvement vs baseline
improv_mae = 100 * (1 - mae_model / mae_rand)
improv_rmse = 100 * (1 - rmse_model / rmse_rand)

print(f"Improvement vs baseline: MAE {improv_mae:.2f}% | RMSE {improv_rmse:.2f}%")