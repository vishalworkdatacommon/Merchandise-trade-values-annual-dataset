
import pandas as pd
import statsmodels.api as sm
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def forecast_trade_data(input_path, output_path):
    """
    Loads processed trade data, builds a SARIMAX forecasting model,
    and saves the forecast to a CSV file.

    Args:
        input_path (str): Path to the processed CSV file.
        output_path (str): Path to save the forecast CSV file.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # --- Data Preparation ---
    print("Aggregating and preparing time-series data...")
    df_agg = df.set_index('Year')
    df_agg.index = pd.to_datetime(df_agg.index) # No format needed, pandas will infer
    df_agg = df_agg.asfreq('YS')

    # --- Model Training ---
    print("\nTraining the SARIMAX forecasting model...")
    # The (p,d,q) order is chosen based on the clear trend in the data.
    # d=2 is used for a strong, almost exponential trend.
    # p=1 and q=1 are common starting points.
    model = sm.tsa.statespace.SARIMAX(
        df_agg['Value'],
        order=(1, 2, 1), # Using d=2 to handle the strong trend
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("Model training complete.")
    print(results.summary())

    # --- Forecasting ---
    print("\nGenerating forecast for the next 5 years...")
    forecast_steps = 5
    forecast = results.get_forecast(steps=forecast_steps)
    
    forecast_df = forecast.summary_frame()
    forecast_df.index.name = 'Year'
    
    forecast_df.to_csv(output_path)
    
    print(f"Forecast successfully saved to {output_path}")
    print("\n--- Forecasted Values (Next 5 Years) ---")
    print(forecast_df)
    print("----------------------------------------")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    processed_csv_path = os.path.join(script_dir, 'data', 'china_exports_cleaned.csv')
    forecast_output_path = os.path.join(script_dir, 'data', 'china_exports_forecast.csv')
    
    forecast_trade_data(processed_csv_path, forecast_output_path)
