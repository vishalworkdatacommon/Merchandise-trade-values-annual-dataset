
import os
import pandas as pd
import gradio as gr
import json
from src.data_processing_script import process_trade_data
from src.data_cleaning_script import clean_and_treat_outliers
from src.data_integration_script import integrate_external_data
from src.forecasting_script import forecast_sarimax
from src.advanced_forecasting_script import forecast_lstm

def run_analysis_pipeline(reporter, partner, product, country_code, progress=gr.Progress()):
    """
    Runs the full end-to-end analysis pipeline, with a caching layer to speed up repeated queries.
    """
    # --- Caching Logic ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    cache_dir = os.path.join(data_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique filename for the cache based on the query
    cache_filename = f"cache_{reporter}_{partner}_{product}.json"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"--- Loading result from cache: {cache_filename} ---")
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        
        # Convert the cached forecast back to a DataFrame
        forecast_df = pd.read_json(cached_data['forecast'], orient='split')
        return forecast_df, None

    # --- If not in cache, run the full pipeline ---
    try:
        raw_csv_path = os.path.join(data_dir, 'merchandise_values_annual_input.csv')
        processed_path = os.path.join(data_dir, f"processed_temp.csv")

        progress(0.1, desc="Step 1/5: Filtering raw data...")
        processed_df = process_trade_data(raw_csv_path, processed_path, reporter, partner, product)
        if processed_df.empty:
            return None, "No data found for the selected combination."

        MIN_YEARS = 15
        if len(processed_df) < MIN_YEARS:
            return None, f"Not enough historical data ({len(processed_df)} years) to generate a reliable forecast. A minimum of {MIN_YEARS} years is required."

        progress(0.3, desc="Step 2/5: Cleaning data...")
        cleaned_df = clean_and_treat_outliers(processed_df)

        progress(0.5, desc="Step 3/5: Fetching GDP data...")
        enriched_df = integrate_external_data(cleaned_df, country_code)

        progress(0.7, desc="Step 4/5: Training SARIMAX model...")
        sarimax_forecast = forecast_sarimax(enriched_df)

        progress(0.9, desc="Step 5/5: Training LSTM model...")
        lstm_forecast = forecast_lstm(enriched_df)
        
        combined_df = sarimax_forecast[['mean']].rename(columns={'mean': 'SARIMAX_Forecast'})
        combined_df['LSTM_Forecast'] = lstm_forecast['mean']

        # --- Save to Cache ---
        print(f"--- Saving result to cache: {cache_filename} ---")
        cache_data = {
            'forecast': combined_df.to_json(orient='split')
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        return combined_df, None

    except Exception as e:
        print(f"An error occurred in the pipeline: {e}")
        return None, f"An unexpected error occurred during the analysis: {e}."
