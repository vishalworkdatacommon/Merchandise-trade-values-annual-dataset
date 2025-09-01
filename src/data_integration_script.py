
import pandas as pd
import wbgapi as wb
import os
import logging
from src.config import WB_INDICATOR, WB_START_YEAR, WB_END_YEAR

def integrate_external_data(input_df, country_code="CHN"):
    """Fetches and integrates World Bank GDP data with the trade data.

    This function queries the World Bank API (wbgapi) for annual GDP data
    for a specified country. It then merges this data with the cleaned trade
    data DataFrame.

    Args:
        input_df (pd.DataFrame): The cleaned trade data, indexed by year.
        country_code (str): The ISO 3-letter country code for which to fetch GDP data.
                            Defaults to "CHN".

    Returns:
        pd.DataFrame: An enriched DataFrame containing both the original trade 'Value'
                      and the new 'GDP_USD' feature. Returns the original DataFrame
                      with a 'GDP_USD' column of zeros if the API call fails.
    """
    logging.info(f"Fetching GDP data for {country_code}...")
    
    trade_df = input_df.set_index('Year')
    trade_df.index = pd.to_datetime(trade_df.index)

    try:
        gdp_df = wb.data.DataFrame(
            WB_INDICATOR,
            country_code,
            time=range(WB_START_YEAR, WB_END_YEAR)
        ).transpose()
        
        gdp_df.index = pd.to_datetime(gdp_df.index.str.replace('YR', ''), format='%Y')
        gdp_df.index.name = 'Year'
        gdp_df.rename(columns={country_code: 'GDP_USD'}, inplace=True)
        gdp_df['GDP_USD'] = gdp_df['GDP_USD'] / 1e6
        
    except Exception as e:
        logging.error(f"Could not fetch GDP data: {e}. Proceeding without it.")
        trade_df['GDP_USD'] = 0 # Return a column of zeros if API fails
        return trade_df.reset_index()

    enriched_df = trade_df.join(gdp_df, how='left')
    enriched_df['GDP_USD'].fillna(method='ffill', inplace=True)
    enriched_df.dropna(inplace=True)

    logging.info("Successfully integrated external data.")
    return enriched_df.reset_index()

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    cleaned_csv_path = os.path.join(data_dir, 'china_exports_cleaned.csv')
    enriched_output_path = os.path.join(data_dir, 'china_exports_enriched.csv')
    
    df = pd.read_csv(cleaned_csv_path)
    enriched_df = integrate_external_data(df)
    enriched_df.to_csv(enriched_output_path, index=False)
    print(f"Enriched data saved to {enriched_output_path}")
