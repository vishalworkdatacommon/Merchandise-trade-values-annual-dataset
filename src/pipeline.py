import os
import pandas as pd
import gradio as gr
from src.comtrade_api import get_comtrade_data
from src.data_cleaning_script import clean_and_treat_outliers
from src.data_integration_script import integrate_external_data
from src.forecasting_script import forecast_sarimax
from src.advanced_forecasting_script import forecast_lstm

def run_analysis_pipeline(reporter_id, partner_id, product_id, country_code, progress=gr.Progress()):
    """
    Runs the full end-to-end analysis pipeline using live API data.
    """
    try:
        progress(0.1, desc="Step 1/5: Fetching live data from UN Comtrade API...")
        live_df = get_comtrade_data(reporter_id, partner_id, product_id)
        
        if live_df.empty:
            return None, "No data was returned from the UN Comtrade API for this selection. This can happen with rare trade combinations. Please try another."

        MIN_YEARS = 15
        if len(live_df) < MIN_YEARS:
            return None, f"Not enough historical data to generate a reliable forecast. A minimum of {MIN_YEARS} years of data is required, but only {len(live_df)} were found for this selection."

        progress(0.3, desc="Step 2/5: Cleaning and treating outliers...")
        cleaned_df = clean_and_treat_outliers(live_df)

        progress(0.5, desc="Step 3/5: Fetching GDP data...")
        enriched_df = integrate_external_data(cleaned_df, country_code)

        progress(0.7, desc="Step 4/5: Training SARIMAX model...")
        sarimax_forecast = forecast_sarimax(enriched_df)

        progress(0.9, desc="Step 5/5: Training LSTM model...")
        lstm_forecast = forecast_lstm(enriched_df)
        
        combined_df = sarimax_forecast[['mean']].rename(columns={'mean': 'SARIMAX_Forecast'})
        combined_df['LSTM_Forecast'] = lstm_forecast['mean']

        return combined_df, None

    except Exception as e:
        print(f"An error occurred in the pipeline: {e}")
        return None, f"An unexpected error occurred during the analysis: {e}. Please try a different combination."