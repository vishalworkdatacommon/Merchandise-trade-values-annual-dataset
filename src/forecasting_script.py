import pandas as pd
import statsmodels.api as sm
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def forecast_trade_data(input_path, output_path):
    """
    Loads enriched trade data, builds a SARIMAX forecasting model with an external regressor,
    and saves the forecast to a CSV file.

    Args:
        input_path (str): Path to the enriched CSV file.
        output_path (str): Path to save the forecast CSV file.
    """
    print(f"Loading and preparing enriched data from {input_path}...")
    df = pd.read_csv(input_path, index_col='Year', parse_dates=True)
    df = df.asfreq('YS')
    df.dropna(inplace=True)

    print("\n--- Training the SARIMAX model with external regressor (GDP)...")
    
    endog = df['Value']
    exog = df[['GDP_USD']]

    model = sm.tsa.statespace.SARIMAX(
        endog=endog,
        exog=exog,
        order=(1, 2, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("Model training complete.")
    print(results.summary())

    print("\nGenerating forecast with external regressor...")
    future_gdp = [exog['GDP_USD'].iloc[-1] * (1.04)**i for i in range(1, 6)]
    exog_forecast = pd.DataFrame({'GDP_USD': future_gdp}, index=pd.date_range(start='2024-01-01', periods=5, freq='YS'))
    
    forecast = results.get_forecast(steps=5, exog=exog_forecast)
    
    forecast_df = forecast.summary_frame()
    forecast_df.index.name = 'Year'
    
    forecast_df.to_csv(output_path)
    
    print(f"Forecast successfully saved to {output_path}")
    print("\n--- Forecasted Values (Next 5 Years) ---")
    print(forecast_df)
    print("----------------------------------------")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    enriched_csv_path = os.path.join(script_dir, '..', 'data', 'china_exports_enriched.csv')
    forecast_output_path = os.path.join(script_dir, '..', 'data', 'china_exports_forecast.csv')
    
    forecast_trade_data(enriched_csv_path, forecast_output_path)