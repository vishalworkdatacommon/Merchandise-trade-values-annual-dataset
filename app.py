import gradio as gr
import pandas as pd
import os
from transformers import pipeline
import torch
import sys

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
    """Loads the raw data to populate the dropdown menus."""
    try:
        df = pd.read_csv('data/merchandise_values_annual_input.csv', usecols=['Reporter', 'Partner', 'Product'], encoding='latin1')
        reporters = sorted(df['Reporter'].unique().tolist())
        partners = sorted(df['Partner'].unique().tolist())
        products = sorted(df['Product'].unique().tolist())
        return reporters, partners, products
    except FileNotFoundError:
        return [], [], []

# --- 3. Define Core Logic ---
def generate_analysis(reporter, partner, product, progress=gr.Progress()):
    """
    Main function for the Gradio interface. Runs the pipeline and generates AI analysis.
    """
    if not all([reporter, partner, product]):
        return "Please make a selection for all dropdowns.", ""

    # A simple mapping for country name to country code for the GDP API
    # This would be more robust with a proper country code library
    country_code_map = {"China": "CHN", "United States": "USA", "Germany": "DEU", "Japan": "JPN"}
    country_code = country_code_map.get(reporter, "WLD") # Default to World if not found

    forecast_df, _ = run_analysis_pipeline(reporter, partner, product, country_code, progress)

    if forecast_df is None:
        return "No data found for the selected combination. Please try another.", ""

    # --- Create a Dynamic Prompt for the LLM ---
    prompt = f"""
    <start_of_turn>user
    You are an expert economic analyst. Provide a forecast summary for the following trade relationship:
    - **Exporting Country:** {reporter}
    - **Importing Country/Region:** {partner}
    - **Product Category:** {product}

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
    reporters, partners, products = get_dropdown_choices()

    with gr.Blocks() as demo:
        gr.Markdown("# Gen AI-Powered Global Trade Forecaster")
        gr.Markdown("Select an exporting country, an importing partner, and a product category to generate a 5-year forecast and a dynamic AI-powered analysis.")
        
        with gr.Row():
            reporter_dd = gr.Dropdown(reporters, label="Reporter (Exporting Country)", value="China")
            partner_dd = gr.Dropdown(partners, label="Partner (Importing Country/Region)", value="World")
            product_dd = gr.Dropdown(products, label="Product Category", value="Total merchandise")
        
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
