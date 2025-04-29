import pandas as pd
import numpy as np

# Load the transformed data
df = pd.read_csv('amazon_stock_data_transformed.csv', index_col='Timestamp', parse_dates=True)

# Define train/test split
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Use differenced log prices for modeling
train_diff = train_data['Log_Diff'].dropna()
test_diff = test_data['Log_Diff'].dropna()

# Print shapes for debugging
print("Shape of test_diff:", test_diff.shape)
print("Type of test_diff:", type(test_diff))
print("Index of test_diff:", type(test_diff.index))

# Create dummy predictions of same length
white_noise_pred = np.full(len(test_diff), train_diff.mean())
print("Shape of white_noise_pred:", white_noise_pred.shape)
print("Type of white_noise_pred:", type(white_noise_pred))

# Try creating a simple DataFrame
simple_df = pd.DataFrame({
    'Actual': test_diff,
    'White_Noise': white_noise_pred
})

print("Successfully created simple DataFrame with shape:", simple_df.shape)
