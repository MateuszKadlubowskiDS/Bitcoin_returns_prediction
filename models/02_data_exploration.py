import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 1. Load data ===
btc = pd.read_csv("data/processed/btc_usd_daily.csv")
print(btc.head())

# Format float display
pd.set_option("display.float_format", "{:,.2f}".format)

# Ensure Date column is datetime
btc["Date"] = pd.to_datetime(btc["Date"])

# === 1a. Descriptive statistics for Close price ===
print("=== Close price statistics ===")
print(btc["Close"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).round(2))

# === 1b. Descriptive statistics for Volume ===
print("\n=== Volume statistics ===")
print(btc["Volume"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).round(2))

# === 1c. General dataset overview ===
print("\n=== Dataset overview ===")
print(btc.describe(include="all").round(2))

# === 2a. Daily volatility (Open → Close) ===
btc["Volatility_OC"] = (btc["Close"] - btc["Open"]).abs()
btc["Volatility_OC_pct"] = (btc["Close"] - btc["Open"]).abs() / btc["Open"] * 100

# === 2b. High-Low spread ===
btc["Spread_HL"] = btc["High"] - btc["Low"]
btc["Spread_HL_pct"] = (btc["High"] - btc["Low"]) / btc["Low"] * 100

# === 2c. Volatility statistics ===
print("=== Volatility (Open→Close, USD) ===")
print(btc["Volatility_OC"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).round(2))

print("\n=== Volatility (Open→Close, %) ===")
print(btc["Volatility_OC_pct"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).round(2))

print("\n=== Spread (High→Low, USD) ===")
print(btc["Spread_HL"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).round(2))

print("\n=== Spread (High→Low, %) ===")
print(btc["Spread_HL_pct"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).round(2))

# Halving dates
halvings = [
    pd.to_datetime("2016-07-09"),
    pd.to_datetime("2020-05-11"),
    pd.to_datetime("2024-04-20"),
    btc["Date"].max()
]

# Create 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
axes = axes.flatten()

start = btc["Date"].min()

for i, halving in enumerate(halvings):
    subset = btc[(btc["Date"] >= start) & (btc["Date"] < halving)]
    
    axes[i].plot(subset["Date"], subset["Close"], label=f"BTC price (cycle {i+1})", color="tab:blue")
    axes[i].set_title(f"Cycle {i+1}: {start.date()} → {halving.date()}")
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Price (USD)")
    axes[i].legend(fontsize=8)
    
    ymin, ymax = subset["Close"].min(), subset["Close"].max()
    axes[i].set_ylim(ymin * 0.9, ymax * 1.1)
    
    start = halving

plt.tight_layout()
plt.show()

# Halving cycles (log scale)
halvings = [
    pd.to_datetime("2016-07-09"),
    pd.to_datetime("2020-05-11"),
    pd.to_datetime("2024-04-20"),
    btc["Date"].max()
]

start = btc["Date"].min()

plt.figure(figsize=(12, 6))

for i, halving in enumerate(halvings):
    subset = btc[(btc["Date"] >= start) & (btc["Date"] < halving)]
    plt.plot(subset["Date"], subset["Close"], label=f"Cycle {i+1}")
    start = halving

plt.yscale("log")
plt.title("BTC price – halving cycles (log scale)")
plt.xlabel("Date")
plt.ylabel("Price (USD, log)")
plt.legend()
plt.show()

# Bear market bottom dates
bottom_dates = [
    pd.to_datetime("2015-01-14"),
    pd.to_datetime("2018-12-15"),
    pd.to_datetime("2022-11-21")
]

# Halving dates
halving_dates = [
    pd.to_datetime("2016-07-09"),
    pd.to_datetime("2020-05-11"),
    pd.to_datetime("2024-04-20")
]

plt.figure(figsize=(12, 6))

for i, b_date in enumerate(bottom_dates):
    if i < len(bottom_dates) - 1:
        subset = btc[(btc["Date"] >= b_date) & (btc["Date"] < bottom_dates[i+1])].copy()
    else:
        subset = btc[btc["Date"] >= b_date].copy()
    
    if subset.empty:
        continue
    
    subset["Days_Since_Bottom"] = (subset["Date"] - b_date).dt.days
    subset["Norm_Close"] = subset["Close"] / subset["Close"].iloc[0] * 100
    
    plt.plot(
        subset["Days_Since_Bottom"], 
        subset["Norm_Close"], 
        label=f"Cycle from bottom {b_date.date()}"
    )
    
    if i < len(halving_dates):
        h_date = halving_dates[i]
        if h_date in subset["Date"].values:
            h_day = (h_date - b_date).days
            h_price = subset.loc[subset["Date"] == h_date, "Norm_Close"].values[0]
            plt.scatter(h_day, h_price, color="red", zorder=5, s=10, label="_nolegend_")

plt.yscale("log")
plt.title("BTC – cycles from bear market bottoms (normalized, log scale)")
plt.xlabel("Days since bottom")
plt.ylabel("Normalized price (log, 100 = bottom)")
plt.legend()
plt.show()

# Extended version with ATH markers
plt.figure(figsize=(12, 6))

for i, b_date in enumerate(bottom_dates):
    if i < len(bottom_dates) - 1:
        subset = btc[(btc["Date"] >= b_date) & (btc["Date"] < bottom_dates[i+1])].copy()
    else:
        subset = btc[btc["Date"] >= b_date].copy()
    
    if subset.empty:
        continue
    
    subset["Days_Since_Bottom"] = (subset["Date"] - b_date).dt.days
    subset["Norm_Close"] = subset["Close"] / subset["Close"].iloc[0] * 100
    
    plt.plot(
        subset["Days_Since_Bottom"], 
        subset["Norm_Close"], 
        label=f"Cycle from bottom {b_date.date()}"
    )
    
    if i < len(halving_dates):
        h_date = halving_dates[i]
        if h_date in subset["Date"].values:
            h_day = (h_date - b_date).days
            h_price = subset.loc[subset["Date"] == h_date, "Norm_Close"].values[0]
            plt.scatter(h_day, h_price, color="red", zorder=5, s=20, label="_nolegend_")
    
    if i < len(bottom_dates) - 1:
        peak_idx = subset["Close"].idxmax()
        peak_day = subset.loc[peak_idx, "Days_Since_Bottom"]
        plt.axvline(x=peak_day, color="black", linestyle="--", alpha=0.7)

plt.yscale("log")
plt.title("BTC – cycles from local bottoms")
plt.xlabel("Days since cycle bottom")
plt.ylabel("Normalized price (log, 100 = bottom)")
plt.legend()
plt.show()