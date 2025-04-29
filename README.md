# Time Series Forecasting of Amazon Stock Prices Using Neural Networks (LSTM & GAN)

This project explores various time series analysis and forecasting techniques to predict Amazon (AMZN) stock prices. It covers fundamental concepts and applies both traditional statistical models and modern deep learning approaches, specifically Long Short-Term Memory (LSTM) networks and discusses Generative Adversarial Networks (GANs).

## Project Overview

The primary goal is to analyze 10 years of historical Amazon stock data and build predictive models. The project follows a structured approach, covering key topics typically found in a time series analysis curriculum:

1.  **Data Acquisition:** Fetching 10 years of daily AMZN stock data.
2.  **Exploratory Data Analysis (EDA):**
    *   Understanding the purpose of time series analysis.
    *   Applying descriptive techniques (summary statistics).
    *   Visualizing the data using time series plots (line charts for price and volume).
    *   Visualizing multidimensional aspects (OHLC charts).
    *   Analyzing the distribution of returns using histograms.
3.  **Stationarity:**
    *   Understanding the concept of stationarity (including second-order stationarity).
    *   Performing statistical tests for stationarity: Augmented Dickey-Fuller (ADF) Test and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test.
    *   Applying transformations (logarithm, differencing) to achieve stationarity.
    *   Identifying trends through visualization and differencing.
4.  **Autocorrelation:**
    *   Analyzing Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots (Correlograms) to understand dependencies.
    *   Understanding the concept of moving averages and their use in smoothing and trend identification.
5.  **Traditional Time Series Models:**
    *   **White Noise & Random Walk:** Establishing baseline models.
    *   **Autoregressive (AR) Processes:** Fitting an AR model using Yule-Walker equations.
    *   **ARMA & ARIMA Models:** Combining AR and MA components, including integration (differencing).
    *   **SARIMA Model:** Incorporating seasonality into the ARIMA framework.
    *   Understanding concepts like invertibility, general linear process, and the Wold decomposition theorem (implicitly through model structures).
6.  **Neural Network Models:**
    *   **LSTM:** Implementing an LSTM network for time series forecasting, leveraging its ability to capture long-range dependencies.
    *   **GAN:** Discussing the concept and potential application of Generative Adversarial Networks for time series forecasting, acknowledging implementation complexities.
7.  **Forecasting & Evaluation:**
    *   Generating forecasts using the implemented models.
    *   Evaluating model performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
    *   Comparing the effectiveness of different models.

## Technologies Used

*   **Python 3.10+**
*   **Libraries:**
    *   `pandas` for data manipulation and analysis.
    *   `numpy` for numerical operations.
    *   `yfinance` for fetching stock data.
    *   `statsmodels` for traditional time series models (AR, ARIMA, SARIMA) and statistical tests (ADF, KPSS).
    *   `scikit-learn` for data preprocessing (MinMaxScaler) and evaluation metrics.
    *   `tensorflow` & `keras` for building and training the LSTM model.
    *   `plotly` for interactive visualizations.
    *   `matplotlib` for static plots (ACF/PACF).
*   **Jupyter Notebook** for code execution, visualization, and documentation.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Fetch the data (if not already included):**
    Run the `fetch_data.py` script (or the equivalent cell in the notebook) to download the latest 10 years of AMZN stock data.
    ```bash
    python fetch_data.py
    ```
5.  **Run the Jupyter Notebook:**
    Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
    Or open the `Time_Series_Forecasting_Amazon_Stock.ipynb` file directly in VS Code with the Jupyter extension installed.
6.  **Execute the cells:** Run the notebook cells sequentially to perform the analysis, train the models, and view the results.

## File Structure

```
amazon_stock_project/
├── README.md                         # This file
├── Time_Series_Forecasting_Amazon_Stock.ipynb # Main Jupyter Notebook with analysis and models
├── fetch_data.py                     # Script to download stock data (using yfinance)
├── requirements.txt                  # List of required Python packages
├── amazon_stock_data_10y.csv         # Raw stock data (generated by fetch_data.py)
├── amazon_stock_data_processed.csv   # Data after initial processing (EDA)
├── amazon_stock_data_transformed.csv # Data after stationarity transformations
├── model_performance_metrics.csv     # Performance metrics for traditional models
├── model_performance_metrics_updated.csv # Performance metrics including LSTM
├── model_predictions.csv             # Predictions from traditional models
├── lstm_predictions.csv              # Predictions from LSTM model
├── model_evaluation_summary.txt      # Text summary of model comparison
├── *.png                             # Saved ACF/PACF plots
└── *.html                            # Saved Plotly visualizations
```

## Results Summary

The analysis demonstrated that while traditional models like AR and ARMA provide baseline forecasts on the stationary (differenced log) series, the LSTM model significantly outperformed them when forecasting the actual (log) stock prices, achieving a much higher R-squared value. This highlights the capability of deep learning models to capture complex patterns in financial time series data. The implementation of GANs was discussed but deemed complex for direct forecasting in this context.

Refer to the `model_evaluation_summary.txt` file and the final sections of the Jupyter Notebook for detailed performance metrics and visualizations.
