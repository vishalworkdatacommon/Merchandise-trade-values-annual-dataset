import requests
import pandas as pd
import os

def get_comtrade_data(reporter_id, partner_id, product_id):
    """
    Fetches annual trade data from the UN Comtrade public API.
    """
    # This is the correct, public, key-less API endpoint.
    base_url = "https://comtradeapi.un.org/public/v1/get"
    
    params = {
        "r": reporter_id,
        "p": partner_id,
        "ps": "recent",
        "px": "HS",
        "cc": product_id,
        "rg": "1", # Imports
        "freq": "A",
        "type": "C",
        "head": "H",
        "fmt": "json"
    }
    
    print(f"Fetching data from UN Comtrade Public API for: r={reporter_id}, p={partner_id}, cc={product_id}")
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or not data.get('data'):
            print("No data returned from the API for this selection.")
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])
        
        df = df[['period', 'reporterDesc', 'partnerDesc', 'cmdDesc', 'primaryValue']]
        
        df.rename(columns={
            'period': 'Year',
            'reporterDesc': 'Reporter',
            'partnerDesc': 'Partner',
            'cmdDesc': 'Product',
            'primaryValue': 'Value'
        }, inplace=True)
        
        df['Value'] = df['Value'] / 1e6
        
        print(f"Successfully fetched and processed {len(df)} rows of data.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the Comtrade API: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    test_df = get_comtrade_data(reporter_id="842", partner_id="0", product_id="8703")
    if not test_df.empty:
        print("\n--- Test API Fetch Successful ---")
        print(test_df.head())
        print("---------------------------------")