import os
from data_processing_script import process_trade_data
from data_cleaning_script import clean_and_treat_outliers
from forecasting_script import forecast_trade_data as forecast_sarimax
from advanced_forecasting_script import forecast_with_lstm
from backtesting_script import run_backtest
import app

def main():
    """
    Runs the entire data pipeline from processing to launching the web app.
    This ensures that all necessary data files are generated before the app starts.
    """
    print("--- Starting Data Pipeline ---")

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
    # Note: We skip the download script as the raw data is assumed to be present.
    
    if not os.path.exists(processed_csv_path):
        process_trade_data(raw_csv_path, processed_csv_path)
    
    if not os.path.exists(cleaned_csv_path):
        clean_and_treat_outliers(processed_csv_path, cleaned_csv_path)
        
    # The data integration script is now part of the main flow
    # and does not need to be called separately.
    
    if not os.path.exists(sarimax_forecast_path):
        forecast_sarimax(enriched_csv_path, sarimax_forecast_path)
        
    if not os.path.exists(lstm_forecast_path):
        forecast_with_lstm(enriched_csv_path, lstm_model_path, lstm_forecast_path)
        
    if not os.path.exists(backtest_results_path):
        run_backtest(enriched_csv_path, backtest_results_path)

    print("\n--- Data Pipeline Complete. Launching Web Application ---")
    
    # Now, launch the app
    app.main()


if __name__ == "__main__":
    main()
