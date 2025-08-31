import pandas as pd
import gradio as gr
import os
from transformers import pipeline
import torch
import sys

# --- 0. Setup and Data Generation ---
# This block ensures that all necessary data files are present before launching the app.
# If the forecast files don't exist, it runs the full data pipeline.
def setup_application():
    """Checks for necessary files and runs the data pipeline if they are missing."""
    print("--- Verifying Application Setup ---")
    
    # Define paths to the final output files the app needs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    sarimax_csv_path = os.path.join(data_dir, 'china_exports_forecast.csv')
    lstm_csv_path = os.path.join(data_dir, 'china_exports_forecast_lstm.csv')
    backtest_csv_path = os.path.join(data_dir, 'backtest_results.csv')

    # Check if all necessary files exist
    if not all(os.path.exists(p) for p in [sarimax_csv_path, lstm_csv_path, backtest_csv_path]):
        print("One or more data files are missing. Running the full data pipeline...")
        try:
            from setup_pipeline import run_full_pipeline
            run_full_pipeline()
            print("Data pipeline completed successfully.")
        except Exception as e:
            print(f"An error occurred during the data pipeline setup: {e}")
            # Exit if the pipeline fails, as the app cannot run.
            sys.exit(1)
    else:
        print("All necessary data files found. Proceeding to launch.")

# Run the setup process
setup_application()


# --- 1. Initialize the Generative AI Model ---
print("Initializing Generative AI pipeline...")
try:
    generator = pipeline(
        'text-generation', 
        model='google/gemma-2b-it', 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN") # Use the token from secrets
    )
    print("Generative AI pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading LLM: {e}. Using a placeholder function.")
    def generator(prompt, **kwargs):
        return [{"generated_text": "Generative AI model could not be loaded. This is a placeholder."}]

# --- 2. Load Forecast & Backtest Data ---
def load_data(sarimax_path, lstm_path, backtest_path):
    """Loads all necessary data files."""
    try:
        sarimax_df = pd.read_csv(sarimax_path, index_col='Year', parse_dates=True)
        lstm_df = pd.read_csv(lstm_path, index_col='Year', parse_dates=True)
        backtest_df = pd.read_csv(backtest_path)
        
        combined_df = sarimax_df[['mean']].rename(columns={'mean': 'SARIMAX_Forecast'})
        combined_df['LSTM_Forecast'] = lstm_df['mean']
        
        return combined_df, backtest_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data files: {e}. Please run all prerequisite scripts.")

# --- 3. Define Core Logic ---
def get_trade_forecast(year, forecast_df, backtest_df):
    """
    Retrieves forecasts and uses a Gen AI model to generate a dynamic analysis.
    """
    if not year:
        return "Please enter a year to get a forecast."
        
    try:
        year = int(year)
        target_date = pd.to_datetime(f"{year}-01-01")
        
        if target_date in forecast_df.index:
            data = forecast_df.loc[target_date]
            sarimax_val = data['SARIMAX_Forecast'] / 1e6
            lstm_val = data['LSTM_Forecast'] / 1e6
            
            prompt = f"""
            <start_of_turn>user
            You are an expert economic analyst providing a forecast summary for an executive.
            
            **Context:**
            - We have two predictive models for China's total merchandise exports.
            - A backtest on historical data (2010-2024) showed that the SARIMAX model was more accurate, with a Mean Absolute Error of $345B vs. the LSTM's $843B.
            
            **Forecast for {year}:**
            - **SARIMAX (Statistical Model):** ${sarimax_val:.2f} trillion USD
            - **LSTM (Deep Learning Model):** ${lstm_val:.2f} trillion USD
            
            **Your Task:**
            Write a concise, professional analysis. Start by stating the most likely forecast based on the more historically accurate model. Then, briefly compare the two model outputs and provide a concluding sentence on the overall economic outlook implied by the forecast. Do not repeat the backtesting error numbers.
            <end_of_turn>
            <start_of_turn>model
            """

            print(f"\nGenerating analysis for year {year}...")
            outputs = generator(prompt, max_new_tokens=256)
            generated_text = outputs[0]['generated_text'].split('<start_of_turn>model\n')[-1]
            print("Analysis generated successfully.")
            
            return generated_text

        else:
            min_year, max_year = forecast_df.index.min().year, forecast_df.index.max().year
            return f"Sorry, I don't have a forecast for {year}. Please choose a year between {min_year} and {max_year}."

    except (ValueError, TypeError):
        return "Invalid input. Please enter a valid year (e.g., 2025)."

# --- 4. Setup and Launch the App ---
def main():
    """Main function to launch the Gradio app."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    sarimax_csv_path = os.path.join(data_dir, 'china_exports_forecast.csv')
    lstm_csv_path = os.path.join(data_dir, 'china_exports_forecast_lstm.csv')
    backtest_csv_path = os.path.join(data_dir, 'backtest_results.csv')
    
    forecast_df, backtest_df = load_data(sarimax_csv_path, lstm_csv_path, backtest_csv_path)

    iface = gr.Interface(
        fn=lambda year: get_trade_forecast(year, forecast_df, backtest_df),
        inputs=gr.Textbox(label="Enter a Year (e.g., 2025)", placeholder="2025"),
        outputs=gr.Markdown(label="Generative AI Forecast Analysis"),
        title="Gen AI-Powered Global Trade Forecaster",
        description="Compare forecasts from statistical and deep learning models, with a dynamic analysis generated by an LLM.",
        article="Built with Python, Statsmodels, TensorFlow, Gradio, and Hugging Face Transformers.",
        allow_flagging='never'
    )

    print("Launching Gradio web application...")
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()