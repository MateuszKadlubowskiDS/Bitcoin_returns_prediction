import yfinance as yf
import pandas as pd

# File paths (relative for GitHub)
raw_path = "data/raw/btc_usd_daily_raw.csv"
processed_path = "data/processed/btc_usd_daily.csv"

# Download BTC data
df_raw = yf.download("BTC-USD", start="2010-01-01", interval="1d")

# Save raw data
df_raw.to_csv(raw_path)

# Process data
df = df_raw.reset_index()
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Round price columns
df[["Open", "High", "Low", "Close"]] = df[
    ["Open", "High", "Low", "Close"]
].round(2)

# Save processed data
df.to_csv(processed_path, index=False)

# Preview
print(df.head())
