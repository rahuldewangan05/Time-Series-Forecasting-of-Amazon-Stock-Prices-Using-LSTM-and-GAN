# Import necessary libraries
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load the data
df = pd.read_csv("amazon_stock_data_10y.csv", index_col="Timestamp", parse_dates=True)

# --- 1. Purpose of Time Series Analysis & Descriptive Techniques ---
print("--- Descriptive Statistics ---")
print(df.describe())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Handle potential missing values (e.g., forward fill)
df.ffill(inplace=True)
print("\n--- Missing Values After Forward Fill ---")
print(df.isnull().sum())

# --- 2. Time Series Plots (Line Chart) ---
fig_close = go.Figure()
fig_close.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price"))
fig_close.update_layout(
    title="Amazon (AMZN) Closing Price Over 10 Years",
    xaxis_title="Date",
    yaxis_title="Closing Price (USD)",
    template="plotly_white"
)
fig_close.write_html("plot_closing_price.html")
print("\nGenerated plot_closing_price.html")

fig_volume = go.Figure()
fig_volume.add_trace(go.Scatter(x=df.index, y=df["Volume"], mode="lines", name="Volume"))
fig_volume.update_layout(
    title="Amazon (AMZN) Trading Volume Over 10 Years",
    xaxis_title="Date",
    yaxis_title="Volume",
    template="plotly_white"
)
fig_volume.write_html("plot_volume.html")
print("Generated plot_volume.html")

# --- 3. Visualizing Multidimensional Time Series / Multiple Time Series ---
fig_ohlc = go.Figure()
fig_ohlc.add_trace(go.Scatter(x=df.index, y=df["Open"], mode="lines", name="Open"))
fig_ohlc.add_trace(go.Scatter(x=df.index, y=df["High"], mode="lines", name="High"))
fig_ohlc.add_trace(go.Scatter(x=df.index, y=df["Low"], mode="lines", name="Low"))
fig_ohlc.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
fig_ohlc.update_layout(
    title="Amazon (AMZN) OHLC Prices Over 10 Years",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white"
)
fig_ohlc.write_html("plot_ohlc.html")
print("Generated plot_ohlc.html")

# --- 4. Histograms (e.g., Daily Returns) ---
# Calculate daily returns
df["Daily Return"] = df["Adj Close"].pct_change()

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=df["Daily Return"].dropna(), name="Daily Returns"))
fig_hist.update_layout(
    title="Histogram of Amazon (AMZN) Daily Returns Over 10 Years",
    xaxis_title="Daily Return",
    yaxis_title="Frequency",
    template="plotly_white"
)
fig_hist.write_html("plot_daily_returns_histogram.html")
print("Generated plot_daily_returns_histogram.html")

# Save the processed data with returns for later use
df.to_csv("amazon_stock_data_processed.csv")
print("\nProcessed data saved to amazon_stock_data_processed.csv")


