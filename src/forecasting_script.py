
import pandas as pd
import statsmodels.api as sm
import os
import warnings
import logging
from src.config import SARIMAX_ORDER, FORECAST_STEPS, GDP_GROWTH_ASSUMPTION

warnings.filterwarnings("ignore")

def forecast_sarimax(input_df):
    """Builds and trains a SARIMAX model to generate a multi-year forecast.

    This function uses the statsmodels library to create a Seasonal AutoRegressive
    Integrated Moving Average with eXogenous regressors (SARIMAX) model. It uses
    the trade 'Value' as the endogenous variable and 'GDP_USD' as the exogenous
    variable.

    Args:
        input_df (pd.DataFrame): The enriched DataFrame containing 'Year', 'Value',
                                 and 'GDP_USD' columns.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast for the next `FORECAST_STEPS`
                      years. Includes the mean forecast, and confidence intervals.
    """
    logging.info("Training SARIMAX model...")
    
    df = input_df.set_index('Year')
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('YS')
    df.dropna(inplace=True)

    endog = df['Value']
    exog = df[['GDP_USD']]

    model = sm.tsa.statespace.SARIMAX(
        endog=endog,
        exog=exog,
        order=SARIMAX_ORDER,
    ).fit(disp=False)

    logging.info("Generating SARIMAX forecast...")
    future_gdp = [exog['GDP_USD'].iloc[-1] * (GDP_GROWTH_ASSUMPTION)**i for i in range(1, FORECAST_STEPS + 1)]
    exog_forecast = pd.DataFrame({'GDP_USD': future_gdp}, index=pd.date_range(start=df.index.max() + pd.DateOffset(years=1), periods=FORECAST_STEPS, freq='YS'))
    
    forecast = model.get_forecast(steps=FORECAST_STEPS, exog=exog_forecast)
    
    forecast_df = forecast.summary_frame()
    forecast_df.index.name = 'Year'
    
    logging.info("SARIMAX forecast generated successfully.")
    return forecast_df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    enriched_csv_path = os.path.join(data_dir, 'china_exports_enriched.csv')
    forecast_output_path = os.path.join(data_dir, 'china_exports_forecast.csv')
    
    df = pd.read_csv(enriched_csv_path)
    forecast_df = forecast_sarimax(df)
    forecast_df.to_csv(forecast_output_path)
    print(f"Forecast saved to {forecast_output_path}")
