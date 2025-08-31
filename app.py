import gradio as gr
import pandas as pd
import os
from transformers import pipeline
import torch
import sys
import json

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_analysis_pipeline

# --- 1. Initialize the Generative AI Model ---
print("Initializing Generative AI pipeline...")
try:
    generator = pipeline(
        'text-generation', 
        model='google/gemma-2b-it', 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN")
    )
    print("Generative AI pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading LLM: {e}. Using a placeholder function.")
    def generator(prompt, **kwargs):
        return [{"generated_text": "Generative AI model could not be loaded. This is a placeholder."}]

# --- 2. Load Data for Dropdowns ---
def get_dropdown_choices():
    """Loads the metadata from local JSON files to populate the dropdowns."""
    try:
        with open('data/reporters.json', 'r') as f:
            reporters = json.load(f)
        with open('data/commodities.json', 'r') as f:
            commodities = json.load(f)
            
        # "World" is a valid partner, but not a reporter.
        partners = [{"id": "0", "text": "World"}] + reporters
        
        reporter_choices = [(r['text'], r['id']) for r in reporters if r['text'] != 'World']
        partner_choices = [(p['text'], p['id']) for p in partners]
        commodity_choices = [(c['text'], c['id']) for c in commodities]
        
        return reporter_choices, partner_choices, commodity_choices
    except FileNotFoundError:
        return [], [], []

# --- 3. Define Core Logic ---
def generate_analysis(reporter_id, partner_id, product_id, progress=gr.Progress()):
    """
    Main function for the Gradio interface. Runs the pipeline and generates AI analysis.
    """
    if not all([reporter_id, partner_id, product_id]):
        return None, "Please make a selection for all dropdowns."

    country_code_map = {"842": "USA", "156": "CHN", "276": "DEU", "392": "JPN", "356": "IND"}
    country_code = country_code_map.get(reporter_id, "WLD")

    forecast_df, error_message = run_analysis_pipeline(reporter_id, partner_id, product_id, country_code, progress)

    if error_message:
        return None, f"**Analysis Failed**\n\n{error_message}"

    if forecast_df is None:
        return None, "An unknown error occurred."

    prompt = f"""
    <start_of_turn>user
    You are an expert economic analyst. Provide a forecast summary for the trade relationship based on the following data.
    
    **Forecasts for the next 5 years:**
    {forecast_df.to_string()}

    **Your Task:**
    Write a concise, professional analysis based on the provided forecasts. Compare the outputs of the two models (SARIMAX and LSTM) and provide a concluding sentence on the overall economic outlook implied by the numbers.
    <end_of_turn>
    <start_of_turn>model
    """

    progress(1.0, desc="Generating AI Analysis...")
    outputs = generator(prompt, max_new_tokens=512)
    generated_text = outputs[0]['generated_text'].split('<start_of_turn>model\n')[-1]
    
    return forecast_df, generated_text

# --- 4. Setup and Launch the App ---
if __name__ == "__main__":
    reporter_choices, partner_choices, commodity_choices = get_dropdown_choices()

    with gr.Blocks() as demo:
        gr.Markdown("# Gen AI-Powered Global Trade Forecaster")
        gr.Markdown("Select an exporting country, an importing partner, and a product category to generate a 5-year forecast using live data from the UN Comtrade API.")
        
        with gr.Row():
            reporter_dd = gr.Dropdown(reporter_choices, label="Reporter (Exporting Country)", value="842") # Default to USA
            partner_dd = gr.Dropdown(partner_choices, label="Partner (Importing Country/Region)", value="0") # Default to World
            product_dd = gr.Dropdown(commodity_choices, label="Product Category", value="87") # Default to Vehicles
        
        submit_btn = gr.Button("Generate Forecast and Analysis")
        
        forecast_output = gr.DataFrame(label="Forecasted Values (Next 5 Years)")
        analysis_output = gr.Markdown(label="Generative AI Analysis")

        submit_btn.click(
            fn=generate_analysis,
            inputs=[reporter_dd, partner_dd, product_dd],
            outputs=[forecast_output, analysis_output]
        )

    print("Launching Gradio web application...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
