import pandas as pd
import os
import logging
import sys
import comtradeapicall

# Adjust path for standalone execution and imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_comtrade_data(reporter_id, partner_id, product_id):
    """Fetches and processes annual trade data from the UN Comtrade public API
    using the comtradeapicall package's preview function.

    This function queries the UN Comtrade API for a specific trade flow,
    processes the JSON response into a pandas DataFrame, and formats it
    for use in the forecasting pipeline.

    Args:
        reporter_id (str): The Comtrade code for the reporting country.
        partner_id (str): The Comtrade code for the partner country/region (e.g., "0" for World).
        product_id (str): The Comtrade Harmonized System (HS) code for the product.

    Returns:
        pd.DataFrame: A DataFrame containing the formatted trade data with columns
                      ['Year', 'Reporter', 'Partner', 'Product', 'Value'].
                      Returns an empty DataFrame if the API call fails or returns no data.
    """
    logging.info(f"Fetching data from UN Comtrade API for reporter:{reporter_id}, partner:{partner_id}, product:{product_id}")

    try:
        # Using the public preview function which does not require a key
        df = comtradeapicall.previewFinalData(
            typeCode='C',
            freqCode='A', # Annual data
            clCode='HS',
            period='recent', # Fetch recent years as per public API limitations
            reporterCode=reporter_id,
            cmdCode=product_id,
            flowCode='M', # Imports
            partnerCode=partner_id,
            partner2Code=None,
            customsCode=None,
            motCode=None,
            format_output='JSON',
            includeDesc=True,
            maxRecords=5000 # Increase max records to get more history
        )
        
        if df is None or df.empty:
            logging.warning("No data returned from the API for this selection.")
            return pd.DataFrame()

        # Select and rename columns to match the project's existing structure
        df = df[['period', 'reporterDesc', 'partnerDesc', 'cmdDesc', 'primaryValue']]
        df.rename(columns={
            'period': 'Year',
            'reporterDesc': 'Reporter',
            'partnerDesc': 'Partner',
            'cmdDesc': 'Product',
            'primaryValue': 'Value'
        }, inplace=True)
        
        # Convert value to millions for consistency
        df['Value'] = df['Value'] / 1e6
        
        logging.info(f"Successfully fetched and processed {len(df)} rows of data.")
        return df

    except Exception as e:
        logging.error(f"An error occurred while calling the Comtrade API: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage: Fetch data for USA (842) importing Cars (8703) from the World (0)
    test_df = get_comtrade_data(reporter_id="842", partner_id="0", product_id="8703")
    if not test_df.empty:
        print("\n--- Test API Fetch Successful ---")
        print(test_df.head())
        print("---------------------------------")
    else:
        print("\n--- Test API Fetch Failed ---")
