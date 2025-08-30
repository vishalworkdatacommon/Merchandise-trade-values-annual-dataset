import pandas as pd
import wbgapi as wb
import os

def integrate_external_data(cleaned_data_path, enriched_output_path):
    """
    Fetches external economic data (GDP) and merges it with the cleaned trade data.
    """
    print("Starting data integration process...")

    # --- 1. Load Cleaned Trade Data ---
    try:
        trade_df = pd.read_csv(cleaned_data_path, index_col='Year', parse_dates=True)
        trade_df = trade_df.asfreq('YS')
        trade_df['Value'] = trade_df['Value'].fillna(method='ffill')
        print("Loaded cleaned trade data.")
    except FileNotFoundError:
        print(f"Error: Cleaned data file not found at {cleaned_data_path}. Please run the cleaning script first.")
        return

    # --- 2. Fetch World Bank GDP Data for China ---
    print("Fetching GDP data for China from the World Bank API...")
    try:
        gdp_raw = wb.data.DataFrame(
            'NY.GDP.MKTP.CD',
            'CHN',
            time=range(1960, 2024)
        )
        
        print("\n--- Raw Data from World Bank API ---")
        print(gdp_raw.head())
        print("------------------------------------")

        gdp_df = gdp_raw.transpose()
        
        gdp_df.index = pd.to_datetime(gdp_df.index.str.replace('YR', ''), format='%Y')
        gdp_df.index.name = 'Year'
        gdp_df.rename(columns={'CHN': 'GDP_USD'}, inplace=True)
        
        gdp_df['GDP_USD'] = gdp_df['GDP_USD'] / 1e6
        
        print("Successfully processed GDP data.")
        
    except Exception as e:
        print(f"An error occurred while processing the World Bank API data: {e}")
        return

    # --- 3. Merge Datasets ---
    print("Merging trade data with GDP data...")
    enriched_df = trade_df.join(gdp_df, how='left')
    
    enriched_df['GDP_USD'].fillna(method='ffill', inplace=True)
    enriched_df.dropna(inplace=True)

    # --- 4. Save Enriched Data ---
    enriched_df.to_csv(enriched_output_path)
    print(f"Enriched data successfully saved to {enriched_output_path}")
    print("\n--- Enriched Data Sample ---")
    print(enriched_df.head())
    print("--------------------------")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cleaned_csv_path = os.path.join(script_dir, '..', 'data', 'china_exports_cleaned.csv')
    enriched_output_path = os.path.join(script_dir, '..', 'data', 'china_exports_enriched.csv')
    
    integrate_external_data(cleaned_csv_path, enriched_output_path)