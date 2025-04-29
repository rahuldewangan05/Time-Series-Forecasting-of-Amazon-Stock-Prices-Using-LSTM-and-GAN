# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# Load the transformed data
df = pd.read_csv("amazon_stock_data_transformed.csv", index_col="Timestamp", parse_dates=True)

print("--- Implementing Traditional Time Series Models ---")

# Define a function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    # Ensure y_pred is a numpy array or has compatible index
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.reindex(y_true.index).fillna(0) # Align index, fill missing if any
    
    # Ensure lengths match after alignment
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {model_name} Performance Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²): {r2:.6f}")
    
    return mse, rmse, mae, r2

# --- 1. White Noise Model ---
print("\n--- White Noise Model ---")
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Use differenced log prices for modeling
train_diff = train_data["Log_Diff"].dropna()
test_diff = test_data["Log_Diff"].dropna()

# White noise model predicts the mean of the training data
white_noise_pred = np.full(len(test_diff), train_diff.mean())
white_noise_metrics = evaluate_model(test_diff, white_noise_pred, "White Noise Model")

# --- 2. Random Walk Model ---
print("\n--- Random Walk Model ---")
random_walk_pred = np.full(len(test_diff), train_diff.iloc[-1])
random_walk_metrics = evaluate_model(test_diff, random_walk_pred, "Random Walk Model")

# --- 3. Autoregressive (AR) Process ---
print("\n--- Autoregressive (AR) Process ---")
plt.figure(figsize=(10, 6))
plot_pacf(train_diff, lags=40, method="ywm", alpha=0.05)
plt.title("Partial Autocorrelation Function (PACF) for Differenced Log Prices")
plt.savefig("pacf_for_ar_order.png")
print("Generated PACF plot for AR order selection: pacf_for_ar_order.png")

p = 5
ar_model = sm.tsa.AutoReg(train_diff, lags=p, old_names=False)
ar_results = ar_model.fit()
print("\nAR Model Summary:")
print(ar_results.summary().tables[1])

ar_pred = ar_results.predict(start=len(train_diff), end=len(train_diff)+len(test_diff)-1)
ar_metrics = evaluate_model(test_diff, ar_pred, "AR Model")

# --- 4. ARIMA Model ---
print("\n--- ARIMA Model ---")
best_aic = float("inf")
best_order = None
best_arima_model = None

p_values = range(0, 3)
d_values = [1]
q_values = range(0, 3)

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(train_data["Log_Close"], order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    best_arima_model = results
                print(f"ARIMA({p},{d},{q}) - AIC: {results.aic:.4f}")
            except:
                continue

print(f"\nBest ARIMA Model: ARIMA{best_order} with AIC: {best_aic:.4f}")
print(best_arima_model.summary().tables[1])

arima_forecast = best_arima_model.forecast(steps=len(test_data))
arima_metrics = evaluate_model(test_data["Log_Close"], arima_forecast, "ARIMA Model")

# --- 5. SARIMA Model ---
print("\n--- SARIMA Model ---")
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(train_diff, lags=40, alpha=0.05)
plt.title("ACF for Differenced Log Prices")
plt.subplot(212)
plot_pacf(train_diff, lags=40, method="ywm", alpha=0.05)
plt.title("PACF for Differenced Log Prices")
plt.tight_layout()
plt.savefig("acf_pacf_for_sarima.png")
print("Generated ACF/PACF plots for SARIMA parameter selection: acf_pacf_for_sarima.png")

seasonal_period = 5
best_aic = float("inf")
best_order_s = None
best_seasonal_order = None
best_sarima_model = None

P_values = range(0, 2)
D_values = [0, 1]
Q_values = range(0, 2)

# Use best ARIMA order as base for SARIMA
base_p, base_d, base_q = best_order

for P in P_values:
    for D in D_values:
        for Q in Q_values:
            try:
                model = SARIMAX(
                    train_data["Log_Close"],
                    order=(base_p, base_d, base_q),
                    seasonal_order=(P, D, Q, seasonal_period)
                )
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order_s = (base_p, base_d, base_q)
                    best_seasonal_order = (P, D, Q, seasonal_period)
                    best_sarima_model = results
                print(f"SARIMA({base_p},{base_d},{base_q})({P},{D},{Q},{seasonal_period}) - AIC: {results.aic:.4f}")
            except:
                continue

print(f"\nBest SARIMA Model: SARIMA{best_order_s}{best_seasonal_order} with AIC: {best_aic:.4f}")
sarima_forecast = None
sarima_metrics = [np.nan] * 4
if best_sarima_model is not None:
    print(best_sarima_model.summary().tables[1])
    sarima_forecast = best_sarima_model.forecast(steps=len(test_data))
    sarima_metrics = evaluate_model(test_data["Log_Close"], sarima_forecast, "SARIMA Model")

# --- 6. ARMA Model ---
print("\n--- ARMA Model ---")
best_aic = float("inf")
best_order_arma = None
best_arma_model = None

p_values = range(0, 4)
q_values = range(0, 4)

for p in p_values:
    for q in q_values:
        if p == 0 and q == 0:
            continue
        try:
            model = ARIMA(train_diff, order=(p, 0, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order_arma = (p, 0, q)
                best_arma_model = results
            print(f"ARMA({p},{q}) - AIC: {results.aic:.4f}")
        except:
            continue

print(f"\nBest ARMA Model: ARMA{best_order_arma} with AIC: {best_aic:.4f}")
print(best_arma_model.summary().tables[1])

arma_forecast = best_arma_model.forecast(steps=len(test_diff))
arma_metrics = evaluate_model(test_diff, arma_forecast, "ARMA Model")

# --- 7. Visualize Model Predictions ---
# Ensure all prediction arrays are numpy arrays of the correct length
pred_len = len(test_diff)

# Convert predictions to numpy arrays and ensure correct length
white_noise_pred_np = np.array(white_noise_pred)[:pred_len]
random_walk_pred_np = np.array(random_walk_pred)[:pred_len]
ar_pred_np = np.array(ar_pred)[:pred_len]
arma_forecast_np = np.array(arma_forecast)[:pred_len]

# Create DataFrame using test_diff index
predictions_df = pd.DataFrame({
    "Actual": test_diff.values, # Use values to avoid index issues
    "White Noise": white_noise_pred_np,
    "Random Walk": random_walk_pred_np,
    "AR": ar_pred_np,
    "ARMA": arma_forecast_np
}, index=test_diff.index) # Explicitly set the index

# Plot predictions for differenced data
fig = go.Figure()
fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df["Actual"], mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df["AR"], mode="lines", name="AR Model"))
fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df["ARMA"], mode="lines", name="ARMA Model"))
fig.update_layout(
    title="Comparison of Model Predictions for Differenced Log Prices",
    xaxis_title="Date",
    yaxis_title="Differenced Log Price",
    template="plotly_white"
)
fig.write_html("plot_model_predictions_diff.html")
print("\nGenerated plot_model_predictions_diff.html")

# Plot predictions for level data (ARIMA/SARIMA)
fig_levels = go.Figure()
fig_levels.add_trace(go.Scatter(x=test_data.index, y=test_data["Log_Close"], mode="lines", name="Actual"))
fig_levels.add_trace(go.Scatter(x=arima_forecast.index, y=arima_forecast.values, mode="lines", name="ARIMA"))
if sarima_forecast is not None:
    fig_levels.add_trace(go.Scatter(x=sarima_forecast.index, y=sarima_forecast.values, mode="lines", name="SARIMA"))
fig_levels.update_layout(
    title="Comparison of ARIMA and SARIMA Predictions for Log Prices",
    xaxis_title="Date",
    yaxis_title="Log Price",
    template="plotly_white"
)
fig_levels.write_html("plot_arima_sarima_predictions.html")
print("Generated plot_arima_sarima_predictions.html")

# --- 8. Save Model Results ---
models = ["White Noise", "Random Walk", "AR", "ARMA", "ARIMA", "SARIMA"]
metrics = ["MSE", "RMSE", "MAE", "R²"]

metrics_data = []
metrics_data.append(white_noise_metrics)
metrics_data.append(random_walk_metrics)
metrics_data.append(ar_metrics)
metrics_data.append(arma_metrics)
metrics_data.append(arima_metrics)
metrics_data.append(sarima_metrics)

metrics_df = pd.DataFrame(metrics_data, index=models, columns=metrics)
print("\n--- Model Performance Comparison ---")
print(metrics_df)

metrics_df.to_csv("model_performance_metrics.csv")
print("\nModel performance metrics saved to model_performance_metrics.csv")

predictions_df.to_csv("model_predictions.csv")
print("Model predictions saved to model_predictions.csv")
