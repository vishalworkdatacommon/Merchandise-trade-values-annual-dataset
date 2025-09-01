import gradio as gr
import pandas as pd
import os
from transformers import pipeline
import torch
import sys
import json
import logging
from src.logging_config import setup_logging

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_analysis_pipeline
from config import (
    LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    REPORTERS_JSON_PATH,
    COMMODITIES_JSON_PATH,
    COUNTRY_CODE_MAP,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
)

# --- 0. Setup Logging ---
setup_logging()

# --- 1. Initialize the Generative AI Model ---
logging.info("Initializing Generative AI pipeline...")
try:
    generator = pipeline(
        'text-generation', 
        model=LLM_MODEL, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN")
    )
    logging.info("Generative AI pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Error loading LLM: {e}. Using a placeholder function.")
    def generator(prompt, **kwargs):
        return [{"generated_text": "Generative AI model could not be loaded. This is a placeholder."}]

# --- 2. Load Data for Dropdowns ---
def get_dropdown_choices():
    """Loads the metadata from local JSON files to populate the dropdowns."""
    try:
        with open(REPORTERS_JSON_PATH, 'r') as f:
            reporters = json.load(f)
        with open(COMMODITIES_JSON_PATH, 'r') as f:
            commodities = json.load(f)
            
        partners = [{"id": "0", "text": "World"}] + reporters
        
        reporter_choices = [(r['text'], r['id']) for r in reporters if r['text'] != 'World']
        partner_choices = [(p['text'], p['id']) for p in partners]
        commodity_choices = [(c['text'], c['id']) for c in commodities]
        
        return reporter_choices, partner_choices, commodity_choices
    except FileNotFoundError as e:
        logging.error(f"Could not load dropdown choices: {e}")
        return [], [], []

# --- 3. Define Core Logic ---
def generate_analysis(reporter_id, partner_id, product_id, progress=gr.Progress()):
    """
    Main function for the Gradio interface. Runs the pipeline and generates AI analysis.
    """
    # Clear previous outputs
    empty_df = pd.DataFrame()
    initial_outputs = [empty_df, empty_df, ""]

    if not all([reporter_id, partner_id, product_id]):
        logging.warning("User did not make a selection for all dropdowns.")
        return initial_outputs + ["Please make a selection for all dropdowns."]

    country_code = COUNTRY_CODE_MAP.get(reporter_id, "WLD")

    forecast_df, backtest_df, error_message = run_analysis_pipeline(reporter_id, partner_id, product_id, country_code, progress)

    if error_message:
        logging.error(f"Analysis failed: {error_message}")
        return initial_outputs + [f"**Analysis Failed**\n\n{error_message}"]

    if forecast_df is None:
        logging.error("An unknown error occurred in the pipeline.")
        return initial_outputs + ["An unknown error occurred."]

    prompt = f"""
    <start_of_turn>user
    You are an expert economic analyst. Provide a forecast summary for the trade relationship based on the following data.
    
    **Forecasts for the next 5 years:**
    {forecast_df.to_string()}

    **Backtest Results (performance on last 5 years of historical data):**
    {backtest_df.to_string()}

    **Your Task:**
    Write a concise, professional analysis. Start by mentioning which model performed better in the backtest (lower error). Then, summarize the future forecast, referencing the better-performing model. Conclude with the overall economic outlook implied by the numbers.
    <end_of_turn>
    <start_of_turn>model
    """

    progress(1.0, desc="Generating AI Analysis...")
    outputs = generator(prompt, max_new_tokens=LLM_MAX_NEW_TOKENS)
    generated_text = outputs[0]['generated_text'].split('<start_of_turn>model\n')[-1]
    
    return forecast_df, backtest_df, generated_text, ""

# --- 4. Setup and Launch the App ---
if __name__ == "__main__":
    reporter_choices, partner_choices, commodity_choices = get_dropdown_choices()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📈 Gen AI-Powered Global Trade Forecaster")
        gr.Markdown("Select a country, partner, and product to generate a 5-year forecast and model performance backtest.")
        
        with gr.Row():
            reporter_dd = gr.Dropdown(reporter_choices, label="Reporter (Exporting Country)", value="842") # Default USA
            partner_dd = gr.Dropdown(partner_choices, label="Partner (Importing Country/Region)", value="0") # Default World
            product_dd = gr.Dropdown(commodity_choices, label="Product Category", value="87") # Default Vehicles
        
        submit_btn = gr.Button("Generate Forecast and Analysis", variant="primary")
        
        with gr.Accordion("Help / About", open=False):
            gr.Markdown("""
            - **Forecasted Values:** The predicted trade values for the next 5 years from two different models (SARIMAX and LSTM).
            - **Model Backtest Results:** How the models performed when forecasting the *last 5 years* of historical data. This helps gauge which model is more reliable. 'Error' is the difference between the actual and forecasted value.
            - **Generative AI Analysis:** An AI-generated summary of the results.
            """)

        error_box = gr.Markdown(value="", visible=False)

        forecast_output = gr.DataFrame(label="Forecasted Values (Next 5 Years)")
        backtest_output = gr.DataFrame(label="Model Backtest Results (Last 5 Years)")
        analysis_output = gr.Markdown()

        def submit_logic(reporter_id, partner_id, product_id):
            forecast_df, backtest_df, analysis, error_msg = generate_analysis(reporter_id, partner_id, product_id)
            error_visibility = bool(error_msg)
            return {
                forecast_output: forecast_df,
                backtest_output: backtest_df,
                analysis_output: analysis,
                error_box: gr.update(value=error_msg, visible=error_visibility)
            }

        submit_btn.click(
            fn=submit_logic,
            inputs=[reporter_dd, partner_dd, product_dd],
            outputs=[forecast_output, backtest_output, analysis_output, error_box]
        )

    logging.info("Launching Gradio web application...")
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
