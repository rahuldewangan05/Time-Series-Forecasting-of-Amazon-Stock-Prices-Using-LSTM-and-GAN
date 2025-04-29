# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load the processed data
df = pd.read_csv("amazon_stock_data_processed.csv", index_col="Timestamp", parse_dates=True)

print("--- Testing for Stationarity ---")

# --- 1. Stationarity and Second-order stationarity ---
# Function to test stationarity using ADF and KPSS tests
def test_stationarity(series, series_name):
    print(f"\n--- Stationarity Tests for {series_name} ---")
    
    # ADF Test
    print("\nAugmented Dickey-Fuller (ADF) Test:")
    adf_result = adfuller(series.dropna())
    adf_output = pd.Series(
        adf_result[0:4],
        index=['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations Used']
    )
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    print(adf_output)
    
    if adf_result[1] <= 0.05:
        print("Conclusion: Series is STATIONARY (reject H0)")
    else:
        print("Conclusion: Series is NON-STATIONARY (fail to reject H0)")
    
    # KPSS Test
    print("\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:")
    kpss_result = kpss(series.dropna(), regression='c')
    kpss_output = pd.Series(
        kpss_result[0:3],
        index=['KPSS Test Statistic', 'p-value', '# Lags Used']
    )
    for key, value in kpss_result[3].items():
        kpss_output[f'Critical Value ({key})'] = value
    print(kpss_output)
    
    if kpss_result[1] <= 0.05:
        print("Conclusion: Series is NON-STATIONARY (reject H0)")
    else:
        print("Conclusion: Series is STATIONARY (fail to reject H0)")

# Test stationarity of closing prices
test_stationarity(df['Close'], 'Closing Prices')

# Test stationarity of daily returns
test_stationarity(df['Daily Return'].dropna(), 'Daily Returns')

# --- 2. Transformations and Trend Identification ---
# Log transformation
df['Log_Close'] = np.log(df['Close'])
print("\n--- Log Transformation ---")
print(df['Log_Close'].describe())

# Test stationarity of log prices
test_stationarity(df['Log_Close'], 'Log Closing Prices')

# Differencing (first difference of log prices)
df['Log_Diff'] = df['Log_Close'].diff()
print("\n--- First Differencing of Log Prices ---")
print(df['Log_Diff'].describe())

# Test stationarity of differenced log prices
test_stationarity(df['Log_Diff'].dropna(), 'Differenced Log Closing Prices')

# --- 3. Autocorrelation and Moving Average ---
# Create ACF and PACF plots for differenced log prices
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(df['Log_Diff'].dropna(), lags=40, ax=plt.gca(), title='ACF - Differenced Log Prices')
plt.subplot(122)
plot_pacf(df['Log_Diff'].dropna(), lags=40, ax=plt.gca(), title='PACF - Differenced Log Prices')
plt.tight_layout()
plt.savefig('acf_pacf_plots.png')
print("\nGenerated ACF and PACF plots: acf_pacf_plots.png")

# Calculate moving averages
df['MA_7'] = df['Close'].rolling(window=7).mean()  # 7-day moving average
df['MA_30'] = df['Close'].rolling(window=30).mean()  # 30-day moving average
df['MA_90'] = df['Close'].rolling(window=90).mean()  # 90-day moving average

# Plot original series with moving averages
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA_7'], mode='lines', name='7-day MA'))
fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA_30'], mode='lines', name='30-day MA'))
fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA_90'], mode='lines', name='90-day MA'))
fig_ma.update_layout(
    title='Amazon (AMZN) Closing Price with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    template='plotly_white'
)
fig_ma.write_html('plot_moving_averages.html')
print("Generated plot_moving_averages.html")

# --- 4. Visualize transformations ---
# Plot original, log, and differenced series
fig_trans = make_subplots(rows=3, cols=1, 
                         subplot_titles=('Original Closing Prices', 
                                         'Log Transformed Prices', 
                                         'Differenced Log Prices'))

fig_trans.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'),
                   row=1, col=1)
fig_trans.add_trace(go.Scatter(x=df.index, y=df['Log_Close'], mode='lines', name='Log Close'),
                   row=2, col=1)
fig_trans.add_trace(go.Scatter(x=df.index, y=df['Log_Diff'], mode='lines', name='Diff Log Close'),
                   row=3, col=1)

fig_trans.update_layout(height=900, title_text='Amazon Stock Price Transformations', template='plotly_white')
fig_trans.write_html('plot_transformations.html')
print("Generated plot_transformations.html")

# Save the transformed data for later use
df.to_csv('amazon_stock_data_transformed.csv')
print("\nTransformed data saved to amazon_stock_data_transformed.csv")
