import os
import pandas as pd
import gradio as gr
import logging
from src.comtrade_api import get_comtrade_data
from src.data_cleaning_script import clean_and_treat_outliers
from src.data_integration_script import integrate_external_data
from src.forecasting_script import forecast_sarimax
from src.advanced_forecasting_script import forecast_lstm
from src.model_evaluation import evaluate_models
from src.config import MIN_YEARS_FOR_FORECAST

def run_analysis_pipeline(reporter_id, partner_id, product_id, country_code, progress=gr.Progress()):
    """
    Runs the full end-to-end analysis pipeline using live API data.
    """
    try:
        # Step 1: Fetch Data
        progress(0.1, desc="Step 1/6: Fetching live data...")
        live_df = get_comtrade_data(reporter_id, partner_id, product_id)
        if live_df.empty:
            return None, None, "No data returned from the API. Please try another selection."
        if len(live_df) < MIN_YEARS_FOR_FORECAST:
            return None, None, f"Not enough data for a reliable forecast. Found {len(live_df)} years, need {MIN_YEARS_FOR_FORECAST}."

        # Step 2: Clean Data
        progress(0.2, desc="Step 2/6: Cleaning data...")
        cleaned_df = clean_and_treat_outliers(live_df)

        # Step 3: Enrich Data
        progress(0.3, desc="Step 3/6: Enriching data with GDP...")
        enriched_df = integrate_external_data(cleaned_df, country_code)
        enriched_df['Year'] = pd.to_datetime(enriched_df['Year'])

        # Step 4: Evaluate Models
        progress(0.5, desc="Step 4/6: Evaluating models...")
        evaluation_results = evaluate_models(enriched_df)
        if evaluation_results:
            metrics, backtest_df = evaluation_results
            logging.info(f"Model evaluation metrics: {metrics}")
        else:
            backtest_df = pd.DataFrame() # Empty df if no evaluation

        # Step 5: Generate Future Forecasts
        progress(0.7, desc="Step 5/6: Training SARIMAX model...")
        sarimax_forecast = forecast_sarimax(enriched_df)
        progress(0.9, desc="Step 6/6: Training LSTM model...")
        lstm_forecast = forecast_lstm(enriched_df)
        
        combined_df = sarimax_forecast[['mean']].rename(columns={'mean': 'SARIMAX_Forecast'})
        combined_df['LSTM_Forecast'] = lstm_forecast['mean']

        logging.info("Analysis pipeline completed successfully.")
        return combined_df, backtest_df, None

    except Exception as e:
        logging.exception("An error occurred in the pipeline.")
        return None, None, f"An unexpected error occurred: {e}"