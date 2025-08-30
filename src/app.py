
import pandas as pd
import gradio as gr
import os

# --- 1. Load Forecast Data ---
def load_forecast_data(file_path):
    """Loads the pre-generated forecast data from the CSV file."""
    try:
        df = pd.read_csv(file_path, index_col='Year')
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
        # This is a critical error for the app, so we raise it
        raise FileNotFoundError(f"Error: Forecast file not found at {file_path}. Please run the forecasting script first.")

# --- 2. Define Core Logic ---
def get_trade_forecast(year):
    """
    The main function for the Gradio interface.
    Takes a year as input, retrieves the forecast, and formats a response.
    """
    if not year:
        return "Please enter a year to get a forecast."
        
    try:
        year = int(year)
        target_date = pd.to_datetime(f"{year}-01-01")
        
        if target_date in forecast_df.index:
            data = forecast_df.loc[target_date]
            # --- 3. Generate Natural Language Response ---
            mean_value = data['mean'] / 1e6  # Convert to trillions
            ci_lower = data['mean_ci_lower'] / 1e6
            ci_upper = data['mean_ci_upper'] / 1e6

            response = (
                f"## 📈 Trade Forecast Analysis for {year}\n\n"
                f"Our model forecasts that the total merchandise export value for China in **{year}** will be approximately **${mean_value:.2f} trillion USD**.\n\n"
                f"We are 95% confident that the actual value will fall between **${ci_lower:.2f} trillion** and **${ci_upper:.2f} trillion USD**.\n\n"
                f"---\n"
                f"*Disclaimer: This is a simplified forecast based on historical data and does not account for unforeseen economic events."
            )
            return response
        else:
            return f"Sorry, I don't have a forecast for the year {year}. I can only provide forecasts between {forecast_df.index.min().year} and {forecast_df.index.max().year}."

    except (ValueError, TypeError):
        return "Invalid input. Please enter a valid year (e.g., 2025)."

# --- 4. Setup and Launch the App ---
if __name__ == "__main__":
    # Define the path to the forecast data
    # We assume the script is run from the root of the project
    forecast_csv_path = os.path.join('data', 'china_exports_forecast.csv')
    
    # Load the data when the app starts
    forecast_df = load_forecast_data(forecast_csv_path)

    # Create the Gradio interface
    iface = gr.Interface(
        fn=get_trade_forecast,
        inputs=gr.Textbox(label="Enter a Year (e.g., 2025)", placeholder="2025"),
        outputs=gr.Markdown(label="Forecast Analysis"),
        title="AI-Powered Global Trade Forecaster",
        description="This app forecasts China's total annual merchandise export value. Enter a year from the forecast period to see the prediction.",
        article="Built with Python, Statsmodels, and Gradio. A project to demonstrate a full data science workflow.",
        allow_flagging='never'
    )

    # Launch the web server
    print("Launching Gradio web application...")
    iface.launch(server_name="0.0.0.0", server_port=7860)
