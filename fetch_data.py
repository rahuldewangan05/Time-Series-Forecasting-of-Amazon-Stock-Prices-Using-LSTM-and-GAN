
import sys
sys.path.append("/opt/.manus/.sandbox-runtime")
from data_api import ApiClient
import pandas as pd
import json

client = ApiClient()

# Fetch 10 years of daily Amazon stock data
stock_data = client.call_api(
    "YahooFinance/get_stock_chart",
    query={
        "symbol": "AMZN",
        "range": "10y",
        "interval": "1d",
        "includeAdjustedClose": True,
    },
)

# Check if data was retrieved successfully
if stock_data and stock_data["chart"]["result"]:
    result = stock_data["chart"]["result"][0]
    timestamps = result["timestamp"]
    indicators = result["indicators"]
    quotes = indicators["quote"][0]
    adjclose = indicators["adjclose"][0]["adjclose"]

    # Create a DataFrame
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps, unit="s"),
        "Open": quotes["open"],
        "High": quotes["high"],
        "Low": quotes["low"],
        "Close": quotes["close"],
        "Volume": quotes["volume"],
        "Adj Close": adjclose,
    })

    # Set Timestamp as index
    df.set_index("Timestamp", inplace=True)

    # Save data to CSV
    df.to_csv("amazon_stock_data_10y.csv")
    print("Amazon stock data saved to amazon_stock_data_10y.csv")
else:
    print("Failed to retrieve stock data.")
    if stock_data and stock_data["chart"]["error"]:
        print(f'Error: {stock_data["chart"]["error"]}')


