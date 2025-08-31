
import pandas as pd
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings("ignore")

def forecast_sarimax(input_df):
    """
    Builds a SARIMAX forecasting model and returns the forecast.
    """
    print("Training SARIMAX model...")
    
    df = input_df.set_index('Year')
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('YS')
    df.dropna(inplace=True)

    endog = df['Value']
    exog = df[['GDP_USD']]

    model = sm.tsa.statespace.SARIMAX(
        endog=endog,
        exog=exog,
        order=(1, 2, 1),
    ).fit(disp=False)

    print("Generating SARIMAX forecast...")
    future_gdp = [exog['GDP_USD'].iloc[-1] * (1.04)**i for i in range(1, 6)]
    exog_forecast = pd.DataFrame({'GDP_USD': future_gdp}, index=pd.date_range(start=df.index.max() + pd.DateOffset(years=1), periods=5, freq='YS'))
    
    forecast = model.get_forecast(steps=5, exog=exog_forecast)
    
    forecast_df = forecast.summary_frame()
    forecast_df.index.name = 'Year'
    
    return forecast_df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    enriched_csv_path = os.path.join(data_dir, 'china_exports_enriched.csv')
    forecast_output_path = os.path.join(data_dir, 'china_exports_forecast.csv')
    
    df = pd.read_csv(enriched_csv_path)
    forecast_df = forecast_sarimax(df)
    forecast_df.to_csv(forecast_output_path)
    print(f"Forecast saved to {forecast_output_path}")
