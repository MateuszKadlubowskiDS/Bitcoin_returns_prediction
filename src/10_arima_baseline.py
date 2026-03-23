# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
file_path = r"data\processed\btc_usd_daily.csv"

df = pd.read_csv(file_path, parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Keep only closing prices
df = df[["Close"]]
df.head()

# === Preprocessing – log returns ===
df["log_close"] = np.log(df["Close"])
df["log_return"] = df["log_close"].diff()

# Remove NaN from differencing
df = df.dropna()

df[["log_close", "log_return"]].plot(subplots=True, figsize=(12,6))
plt.show()

# === ADF stationarity test (Augmented Dickey-Fuller) ===
result = adfuller(df["log_return"])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.2e}")

if result[1] < 0.05:
    print("Stationary series (p < 0.05)")
else:
    print("Non-stationary series (p >= 0.05)")

# === Train/test split ===
train_size = int(len(df) * 0.8)
train, test = df["log_return"][:train_size], df["log_return"][train_size:]
print(f"Train: {len(train)}, Test: {len(test)}")

# === Auto ARIMA (p,d,q selection) ===
model_auto = auto_arima(
    train,
    seasonal=False,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)
print(model_auto.summary())

# === Fit ARIMA model ===
model = sm.tsa.ARIMA(train, order=model_auto.order)
model_fit = model.fit()
print(model_fit.summary())

# === Forecast and evaluation ===
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

plt.figure(figsize=(12,6))
plt.plot(train, label="Train")
plt.plot(test, label="Test", color="orange")
plt.plot(forecast, label="Forecast", color="red")
plt.legend()
plt.show()

# Evaluation metrics
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("RMSE:", rmse)

# Directional accuracy (hit rate)
hit_rate = np.mean(np.sign(test) == np.sign(forecast))
print("Hit Rate:", hit_rate)

# === Rolling forecast (optional, more realistic evaluation) ===
history = list(train)
predictions = []

for t in range(len(test)):
    model = sm.tsa.ARIMA(history, order=model_auto.order)
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test.iloc[t])

rolling_forecast = pd.Series(predictions, index=test.index)

plt.figure(figsize=(12,6))
plt.plot(test, label="Test")
plt.plot(rolling_forecast, label="Rolling Forecast", color="red")
plt.legend()
plt.show()

mae_roll = mean_absolute_error(test, rolling_forecast)
rmse_roll = mean_squared_error(test, rolling_forecast, squared=False)
print("Rolling MAE:", mae_roll)
print("Rolling RMSE:", rmse_roll)
