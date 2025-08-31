import os
from data_processing_script import process_trade_data
from data_cleaning_script import clean_and_treat_outliers
from data_integration_script import integrate_external_data
from forecasting_script import forecast_trade_data as forecast_sarimax
from advanced_forecasting_script import forecast_with_lstm
from backtesting_script import run_backtest

def run_full_pipeline():
    """
    Runs the entire data pipeline from processing to forecasting.
    This ensures all necessary data files are generated.
    """
    print("--- Running Full Data and Modeling Pipeline ---")

    # Define all file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    raw_csv_path = os.path.join(data_dir, 'merchandise_values_annual_input.csv')
    processed_csv_path = os.path.join(data_dir, 'processed_china_exports.csv')
    cleaned_csv_path = os.path.join(data_dir, 'china_exports_cleaned.csv')
    enriched_csv_path = os.path.join(data_dir, 'china_exports_enriched.csv')
    sarimax_forecast_path = os.path.join(data_dir, 'china_exports_forecast.csv')
    lstm_model_path = os.path.join(data_dir, 'lstm_model.h5')
    lstm_forecast_path = os.path.join(data_dir, 'china_exports_forecast_lstm.csv')
    backtest_results_path = os.path.join(data_dir, 'backtest_results.csv')

    # --- Execute Pipeline ---
    print("\nStep 1: Processing raw data...")
    process_trade_data(raw_csv_path, processed_csv_path)
    
    print("\nStep 2: Cleaning data and handling outliers...")
    clean_and_treat_outliers(processed_csv_path, cleaned_csv_path)
    
    print("\nStep 3: Integrating external GDP data...")
    integrate_external_data(cleaned_csv_path, enriched_csv_path)
    
    print("\nStep 4: Running SARIMAX forecast...")
    forecast_sarimax(enriched_csv_path, sarimax_forecast_path)
        
    print("\nStep 5: Running LSTM forecast...")
    forecast_with_lstm(enriched_csv_path, lstm_model_path, lstm_forecast_path)
        
    print("\nStep 6: Running backtest for model evaluation...")
    run_backtest(enriched_csv_path, backtest_results_path)

    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    run_full_pipeline()
