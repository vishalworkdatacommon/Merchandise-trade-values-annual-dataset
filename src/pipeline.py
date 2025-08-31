
import os
import pandas as pd
from data_processing_script import process_trade_data
from data_cleaning_script import clean_and_treat_outliers
from data_integration_script import integrate_external_data
from forecasting_script import forecast_sarimax
from advanced_forecasting_script import forecast_lstm

def run_analysis_pipeline(reporter, partner, product, country_code, progress=gr.Progress()):
    """
    Runs the full end-to-end analysis pipeline for a given selection.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    raw_csv_path = os.path.join(data_dir, 'merchandise_values_annual_input.csv')
    
    # Define temporary file paths for this specific run
    processed_path = os.path.join(data_dir, f"processed_{reporter}_{partner}_{product}.csv")

    progress(0.1, desc="Step 1/5: Filtering raw data...")
    processed_df = process_trade_data(raw_csv_path, processed_path, reporter, partner, product)
    if processed_df.empty:
        return None, None

    progress(0.3, desc="Step 2/5: Cleaning and treating outliers...")
    cleaned_df = clean_and_treat_outliers(processed_df)

    progress(0.5, desc="Step 3/5: Fetching GDP data...")
    enriched_df = integrate_external_data(cleaned_df, country_code)

    progress(0.7, desc="Step 4/5: Training SARIMAX model...")
    sarimax_forecast = forecast_sarimax(enriched_df)

    progress(0.9, desc="Step 5/5: Training LSTM model...")
    lstm_forecast = forecast_lstm(enriched_df)
    
    # Combine forecasts
    combined_df = sarimax_forecast[['mean']].rename(columns={'mean': 'SARIMAX_Forecast'})
    combined_df['LSTM_Forecast'] = lstm_forecast['mean']

    return combined_df, None # No backtest data for on-the-fly runs

if __name__ == "__main__":
    # Example run for testing
    run_analysis_pipeline("China", "World", "Total merchandise", "CHN")
