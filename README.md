# Bitcoin Return Prediction (LSTM vs ARIMA)

Master thesis project (SGH, Big Data) focused on predicting daily Bitcoin returns.

## Overview

This project investigates whether daily Bitcoin log-returns can be predicted using:

* classical time series models (ARIMA)
* deep learning models (LSTM)

Additionally, it evaluates whether **technical indicators** and **macroeconomic variables** improve predictive performance.

The repository contains a **simplified version** of the full thesis, focusing on:

* data exploration
* baseline models
* selected feature sets (technical + macro)

---

## Data

The analysis is based on daily data (2014–2025):

* Bitcoin OHLCV data
* Technical indicators (RSI, MACD, volatility measures)
* Macroeconomic variables:

  * S&P500
  * VIX
  * US10Y
  * DXY
  * Commodities (BCOM)

Target variable:

* **daily log returns of Bitcoin**

---

## Models

### 1. ARIMA (baseline)

* Automatically selected: ARIMA(0,0,0)
* Equivalent to predicting the mean (white noise process)
* Serves as a **benchmark**

---

### 2. LSTM

Several LSTM models were tested with different feature sets:

* baseline (returns only)
* technical indicators
* macroeconomic variables

---

## Methodology

* Strict **time-based split**:

  * train: until 2021
  * validation: 2022
  * test: 2023+

* No data leakage (scaling and feature selection done only on train set)

* Evaluation metrics:

  * MAE
  * RMSE
  * Directional Accuracy (hit rate)

---

## Results

| Model            | MAE     | RMSE    | Hit Rate |
| ---------------- | ------- | ------- | -------- |
| ARIMA            | 0.01727 | 0.02465 | 0.496    |
| LSTM (baseline)  | ~0.0177 | ~0.0252 | ~0.49    |
| LSTM (technical) | ~0.0173 | ~0.0248 | ~0.51    |
| LSTM (macro)     | ~0.0177 | ~0.0251 | ~0.51    |

### Key findings:

* Predicting daily Bitcoin returns is **extremely difficult**
* LSTM **does not significantly outperform** ARIMA
* Technical and macro variables provide **only marginal improvement**
* Directional accuracy remains close to **random (~50%)**

---

## Conclusion

The results support the **weak form of market efficiency**:

* Bitcoin daily returns behave close to a random process
* Complex models do not provide meaningful predictive advantage
* Macroeconomic variables have **limited short-term impact**

---

## Project Structure

src/

* data_extraction.py → data download and preprocessing
* data_exploration.py → exploratory data analysis
* arima_baseline.py → ARIMA model
* lstm_baseline.py → LSTM baseline
* lstm_technical_indicators.py → LSTM with technical features
* lstm_macro_features.py → LSTM with macro variables

data/

* btc_usd_daily.csv → main dataset
* macro_daily.csv → macroeconomic variables
* btc_crypto_features.csv → additional crypto features

---

## Pipeline

1. Data extraction and preprocessing
2. Feature engineering (technical indicators, macro variables)
3. Model training (ARIMA, LSTM)
4. Evaluation on test set

---

## Notes

* This repository contains a **simplified version** of the full thesis
* Additional models and experiments are not included
* Data stored in repository is cleaned and prepared by the author

---

## Tech Stack

* Python
* pandas, numpy
* scikit-learn
* TensorFlow / Keras
* statsmodels

---

## Author

Mateusz Kadłubowski
SGH Warsaw School of Economics
