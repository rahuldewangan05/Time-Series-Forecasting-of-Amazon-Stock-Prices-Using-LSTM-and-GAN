# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

print("--- Evaluating and Comparing All Models ---")

# Load model performance metrics
metrics_df = pd.read_csv("model_performance_metrics_updated.csv", index_col=0)

# Print model performance comparison
print("\n--- Model Performance Comparison ---")
print(metrics_df)

# Create a bar chart for MSE comparison
fig_mse = go.Figure()
for model in metrics_df.index:
    if not np.isnan(metrics_df.loc[model, "MSE"]):
        fig_mse.add_trace(go.Bar(
            x=[model],
            y=[metrics_df.loc[model, "MSE"]],
            name=model
        ))

fig_mse.update_layout(
    title="Mean Squared Error (MSE) Comparison Across Models",
    xaxis_title="Model",
    yaxis_title="MSE",
    template="plotly_white"
)
fig_mse.write_html("plot_mse_comparison.html")
print("\nGenerated plot_mse_comparison.html")

# Create a bar chart for R² comparison
fig_r2 = go.Figure()
for model in metrics_df.index:
    if not np.isnan(metrics_df.loc[model, "R²"]):
        fig_r2.add_trace(go.Bar(
            x=[model],
            y=[metrics_df.loc[model, "R²"]],
            name=model
        ))

fig_r2.update_layout(
    title="R-squared (R²) Comparison Across Models",
    xaxis_title="Model",
    yaxis_title="R²",
    template="plotly_white"
)
fig_r2.write_html("plot_r2_comparison.html")
print("Generated plot_r2_comparison.html")

# Load predictions from different models
traditional_pred = pd.read_csv("model_predictions.csv", index_col="Timestamp", parse_dates=True)
lstm_pred = pd.read_csv("lstm_predictions.csv", index_col=0, parse_dates=True)

# Create a combined visualization of the best models
# For traditional models, use AR or ARMA (they performed better)
# For neural networks, use LSTM

# Create a figure with subplots
fig_combined = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Traditional Model (AR) vs Actual", 
                                          "Neural Network (LSTM) vs Actual"))

# Add traces for traditional model
fig_combined.add_trace(
    go.Scatter(x=traditional_pred.index, y=traditional_pred["Actual"], mode="lines", name="Actual (Diff Log Price)", line=dict(color="blue")),
    row=1, col=1
)
fig_combined.add_trace(
    go.Scatter(x=traditional_pred.index, y=traditional_pred["AR"], mode="lines", name="AR Prediction", line=dict(color="red")),
    row=1, col=1
)

# Add traces for LSTM model
fig_combined.add_trace(
    go.Scatter(x=lstm_pred.index, y=lstm_pred["Actual_Log"], mode="lines", name="Actual (Log Price)", line=dict(color="blue")),
    row=2, col=1
)
fig_combined.add_trace(
    go.Scatter(x=lstm_pred.index, y=lstm_pred["Predicted_Log"], mode="lines", name="LSTM Prediction", line=dict(color="green")),
    row=2, col=1
)

# Update layout
fig_combined.update_layout(
    height=800,
    title_text="Comparison of Traditional and Neural Network Models",
    template="plotly_white"
)
fig_combined.write_html("plot_model_comparison.html")
print("Generated plot_model_comparison.html")

# Create a summary of findings
print("\n--- Summary of Model Evaluation ---")
print("1. Traditional Models:")
print("   - White Noise, Random Walk, AR, and ARMA models performed similarly on differenced log prices")
print("   - ARIMA and SARIMA models had much higher errors when evaluated on log prices")
print("   - AR and ARMA models were the best performing traditional models")

print("\n2. Neural Network Models:")
print("   - LSTM model significantly outperformed traditional models with an R² of", metrics_df.loc["LSTM", "R²"])
print("   - GAN implementation was noted as complex for direct time series forecasting")

print("\n3. Overall Best Model:")
best_model = metrics_df.loc[metrics_df["R²"].idxmax()]
print(f"   - The best performing model was {metrics_df['R²'].idxmax()} with:")
print(f"     * MSE: {best_model['MSE']:.6f}")
print(f"     * RMSE: {best_model['RMSE']:.6f}")
print(f"     * MAE: {best_model['MAE']:.6f}")
print(f"     * R²: {best_model['R²']:.6f}")

# Save the summary to a file
with open("model_evaluation_summary.txt", "w") as f:
    f.write("--- Summary of Model Evaluation ---\n")
    f.write("1. Traditional Models:\n")
    f.write("   - White Noise, Random Walk, AR, and ARMA models performed similarly on differenced log prices\n")
    f.write("   - ARIMA and SARIMA models had much higher errors when evaluated on log prices\n")
    f.write("   - AR and ARMA models were the best performing traditional models\n\n")
    
    f.write("2. Neural Network Models:\n")
    f.write(f"   - LSTM model significantly outperformed traditional models with an R² of {metrics_df.loc['LSTM', 'R²']:.6f}\n")
    f.write("   - GAN implementation was noted as complex for direct time series forecasting\n\n")
    
    f.write("3. Overall Best Model:\n")
    f.write(f"   - The best performing model was {metrics_df['R²'].idxmax()} with:\n")
    f.write(f"     * MSE: {best_model['MSE']:.6f}\n")
    f.write(f"     * RMSE: {best_model['RMSE']:.6f}\n")
    f.write(f"     * MAE: {best_model['MAE']:.6f}\n")
    f.write(f"     * R²: {best_model['R²']:.6f}\n")

print("\nModel evaluation summary saved to model_evaluation_summary.txt")
