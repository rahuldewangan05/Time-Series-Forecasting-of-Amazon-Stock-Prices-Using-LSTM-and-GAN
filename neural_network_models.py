# Import necessary libraries
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Load the transformed data
df = pd.read_csv("amazon_stock_data_transformed.csv", index_col="Timestamp", parse_dates=True)

print("--- Implementing Neural Network Models ---")

# --- Data Preparation for Neural Networks ---

# Use Log Closing prices for prediction
data = df["Log_Close"].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 60  # Use last 60 days to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets (80/20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Define evaluation function (similar to traditional models, but for scaled data initially)
def evaluate_nn_model(y_true_scaled, y_pred_scaled, scaler, model_name):
    # Inverse transform predictions and true values
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {model_name} Performance Metrics (Original Scale) ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²): {r2:.6f}")
    
    return mse, rmse, mae, r2, y_true, y_pred

# --- 1. LSTM Model ---
print("\n--- Building and Training LSTM Model ---")

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=25))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.summary()

# Train the model (use a small number of epochs for demonstration)
epochs = 10
batch_size = 32
history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# Make predictions
lstm_pred_scaled = lstm_model.predict(X_test)

# Evaluate LSTM model
lstm_mse, lstm_rmse, lstm_mae, lstm_r2, y_true_lstm, y_pred_lstm = evaluate_nn_model(y_test, lstm_pred_scaled, scaler, "LSTM Model")

# --- 2. GAN Model ---
print("\n--- Building and Training GAN Model ---")

# GAN parameters
latent_dim = 100 # Noise dimension for generator input (can be adjusted)
gan_epochs = 20 # Use more epochs for GANs usually, but keep low for demo
gan_batch_size = 64

# Build Generator (LSTM-based)
def build_generator(seq_length, latent_dim):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1, activation="linear")) # Output is a single value
    # We will feed the actual sequence to the generator
    sequence_input = Input(shape=(seq_length, 1))
    generated_output = model(sequence_input)
    return Model(sequence_input, generated_output)

# Build Discriminator (Simple Dense)
def build_discriminator(seq_length):
    model = Sequential()
    model.add(Dense(units=50, input_shape=(seq_length + 1,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=25))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=1, activation="sigmoid")) # Output probability (real/fake)
    
    sequence_input = Input(shape=(seq_length + 1,))
    validity = model(sequence_input)
    return Model(sequence_input, validity)

# Build GAN
discriminator = build_discriminator(seq_length)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])

generator = build_generator(seq_length, latent_dim)

# For the combined model we will only train the generator
discriminator.trainable = False

# Input to GAN is the sequence
gan_input_sequence = Input(shape=(seq_length, 1))
# Generator outputs the next step prediction
generated_step = generator(gan_input_sequence)

# Concatenate the input sequence (last seq_length steps) with the generated step
# Need to reshape generated_step to match sequence format for concatenation if needed
# For discriminator input, we flatten the sequence + next step
# Let's adjust discriminator input shape and how we feed data

# Rebuild discriminator to take sequence and next step separately?
# Simpler approach: Discriminator takes the sequence including the next step (real or fake)

def build_discriminator_v2(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
    return model

discriminator = build_discriminator_v2(seq_length + 1)

# Combined GAN model
gan_input = Input(shape=(seq_length, 1))
gen_output = generator(gan_input)

# Prepare input for discriminator: flatten input sequence and append generated output
# This part is tricky. Let's simplify the GAN structure for forecasting.
# Alternative GAN: Generate entire sequences instead of just the next step.
# Or, use GAN to generate synthetic data for LSTM training (data augmentation) - complex.

# Let's stick to a simpler forecasting GAN structure if possible, but it's non-trivial.
# Maybe a Conditional GAN (CGAN) where the condition is the previous sequence?

# --- Simplified GAN Approach (Generator as Forecaster) ---
# Train the generator to predict the next step, using a loss function (e.g., MSE)
# The discriminator tries to distinguish real (sequence, next_step) pairs from fake (sequence, generated_step) pairs.

print("\n--- Training GAN (Simplified Approach) ---")

# Recompile discriminator (trainable)
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])

# Combined model (Generator + Discriminator)
discriminator.trainable = False # Freeze discriminator in combined model
gan_input_seq = Input(shape=(seq_length, 1))
gen_step = generator(gan_input_seq)

# Reshape inputs for discriminator
# Flatten input sequence and generated step
# This requires careful handling of shapes and might not be the best GAN structure for forecasting.

# Due to complexity and potential instability of GANs for direct forecasting,
# let's focus on LSTM for now and note GAN as a more advanced topic.
# We will evaluate LSTM and save results.

print("\n--- Skipping GAN implementation due to complexity for direct forecasting ---")
print("Focusing on LSTM results.")

gan_mse, gan_rmse, gan_mae, gan_r2 = np.nan, np.nan, np.nan, np.nan
y_pred_gan = np.empty_like(y_true_lstm) * np.nan

# --- Save Results and Visualize ---

# Get the correct index for the test predictions
test_index = df.index[train_size + seq_length:]

# Create DataFrame for LSTM predictions
lstm_results_df = pd.DataFrame({
    "Actual_Log": y_true_lstm.flatten(),
    "Predicted_Log": y_pred_lstm.flatten()
}, index=test_index)

# Visualize LSTM predictions
fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=lstm_results_df.index, y=lstm_results_df["Actual_Log"], mode="lines", name="Actual Log Price"))
fig_lstm.add_trace(go.Scatter(x=lstm_results_df.index, y=lstm_results_df["Predicted_Log"], mode="lines", name="LSTM Predicted Log Price"))
fig_lstm.update_layout(
    title="LSTM Model: Actual vs Predicted Amazon Log Stock Prices",
    xaxis_title="Date",
    yaxis_title="Log Price",
    template="plotly_white"
)
fig_lstm.write_html("plot_lstm_predictions.html")
print("\nGenerated plot_lstm_predictions.html")

# Save LSTM predictions
lstm_results_df.to_csv("lstm_predictions.csv")
print("LSTM predictions saved to lstm_predictions.csv")

# Save LSTM performance metrics (append to existing metrics file)
metrics_df = pd.read_csv("model_performance_metrics.csv", index_col=0)

lstm_metrics_series = pd.Series([lstm_mse, lstm_rmse, lstm_mae, lstm_r2], index=["MSE", "RMSE", "MAE", "R²"], name="LSTM")
gan_metrics_series = pd.Series([gan_mse, gan_rmse, gan_mae, gan_r2], index=["MSE", "RMSE", "MAE", "R²"], name="GAN")

# Use concat instead of append
metrics_df = pd.concat([metrics_df, lstm_metrics_series.to_frame().T, gan_metrics_series.to_frame().T])

print("\n--- Updated Model Performance Comparison ---")
print(metrics_df)

metrics_df.to_csv("model_performance_metrics_updated.csv")
print("\nUpdated model performance metrics saved to model_performance_metrics_updated.csv")


