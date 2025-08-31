import requests
import pandas as pd
import os

def get_comtrade_data(reporter_id, partner_id, product_id):
    """
    Fetches annual trade data from the UN Comtrade public API using the correct endpoint and parameters.
    """
    # This is the correct, documented public API endpoint structure
    base_url = f"https://comtradeapi.un.org/public/v1/get/C/A/HS"
    
    params = {
        "reporterCode": reporter_id,
        "partnerCode": partner_id,
        "cmdCode": product_id,
        "period": "recent",
        "flowCode": "M", # M for Imports, X for Exports
        "includeDesc": "true"
    }
    
    print(f"Fetching data from UN Comtrade Public API with correct parameters...")
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or not data.get('data'):
            print("No data returned from the API for this selection.")
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])
        
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
        
        print(f"Successfully fetched and processed {len(df)} rows of data.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the Comtrade API: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage: Fetch data for USA (842) importing Cars (8703) from the World (0)
    test_df = get_comtrade_data(reporter_id="842", partner_id="0", product_id="8703")
    if not test_df.empty:
        print("\n--- Test API Fetch Successful ---")
        print(test_df.head())
        print("---------------------------------")
