
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from src.forecasting_script import forecast_sarimax
from src.advanced_forecasting_script import forecast_lstm
from src.config import BACKTEST_YEARS

def evaluate_models(enriched_df):
    """Performs a backtest on forecasting models to evaluate performance.

    This function splits the historical data into a training and a testing set.
    It trains both the SARIMAX and LSTM models on the training set, generates
    forecasts for the test set period, and then compares these forecasts against
    the actual historical values.

    Args:
        enriched_df (pd.DataFrame): The complete, enriched DataFrame with a
                                    'Year' column and all features.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary of evaluation metrics (MAE and RMSE) for
                       both models.
               - pd.DataFrame: A DataFrame comparing the actual values to the
                               forecasts from both models for the test period.
               Returns None if there is not enough data to perform a backtest.
    """
    logging.info("Starting model evaluation backtest...")

    # --- 1. Split Data ---
    # Use the last N years for testing and the rest for training
    split_point = enriched_df['Year'].max() - pd.DateOffset(years=BACKTEST_YEARS)
    train_df = enriched_df[enriched_df['Year'] <= split_point]
    test_df = enriched_df[enriched_df['Year'] > split_point]

    if len(test_df) < BACKTEST_YEARS:
        logging.warning(f"Not enough data for a full {BACKTEST_YEARS}-year backtest. Skipping evaluation.")
        return None

    actual_values = test_df.set_index('Year')['Value']

    # --- 2. Evaluate SARIMAX ---
    logging.info("Evaluating SARIMAX model...")
    sarimax_forecast = forecast_sarimax(train_df)
    sarimax_pred = sarimax_forecast['mean']

    # --- 3. Evaluate LSTM ---
    logging.info("Evaluating LSTM model...")
    lstm_forecast = forecast_lstm(train_df)
    lstm_pred = lstm_forecast['mean']

    # --- 4. Calculate Metrics ---
    metrics = {
        'SARIMAX_MAE': mean_absolute_error(actual_values, sarimax_pred),
        'SARIMAX_RMSE': np.sqrt(mean_squared_error(actual_values, sarimax_pred)),
        'LSTM_MAE': mean_absolute_error(actual_values, lstm_pred),
        'LSTM_RMSE': np.sqrt(mean_squared_error(actual_values, lstm_pred)),
    }

    logging.info(f"Model Evaluation Metrics:\n{metrics}")
    
    # --- 5. Create a comparison DataFrame ---
    results_df = pd.DataFrame({
        'Actual': actual_values,
        'SARIMAX_Forecast': sarimax_pred,
        'LSTM_Forecast': lstm_pred
    })
    results_df['SARIMAX_Error'] = results_df['Actual'] - results_df['SARIMAX_Forecast']
    results_df['LSTM_Error'] = results_df['Actual'] - results_df['LSTM_Forecast']

    logging.info(f"Backtest Results:\n{results_df}")

    return metrics, results_df
