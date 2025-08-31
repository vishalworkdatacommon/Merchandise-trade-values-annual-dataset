
import pandas as pd
import wbgapi as wb
import os

def integrate_external_data(input_df, country_code="CHN"):
    """
    Fetches external economic data (GDP) and merges it with trade data.
    """
    print(f"Fetching GDP data for {country_code}...")
    
    trade_df = input_df.set_index('Year')
    trade_df.index = pd.to_datetime(trade_df.index)

    try:
        gdp_df = wb.data.DataFrame(
            'NY.GDP.MKTP.CD',
            country_code,
            time=range(1960, 2024)
        ).transpose()
        
        gdp_df.index = pd.to_datetime(gdp_df.index.str.replace('YR', ''), format='%Y')
        gdp_df.index.name = 'Year'
        gdp_df.rename(columns={country_code: 'GDP_USD'}, inplace=True)
        gdp_df['GDP_USD'] = gdp_df['GDP_USD'] / 1e6
        
    except Exception as e:
        print(f"Could not fetch GDP data: {e}. Proceeding without it.")
        trade_df['GDP_USD'] = 0 # Return a column of zeros if API fails
        return trade_df.reset_index()

    enriched_df = trade_df.join(gdp_df, how='left')
    enriched_df['GDP_USD'].fillna(method='ffill', inplace=True)
    enriched_df.dropna(inplace=True)

    return enriched_df.reset_index()

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    cleaned_csv_path = os.path.join(data_dir, 'china_exports_cleaned.csv')
    enriched_output_path = os.path.join(data_dir, 'china_exports_enriched.csv')
    
    df = pd.read_csv(cleaned_csv_path)
    enriched_df = integrate_external_data(df)
    enriched_df.to_csv(enriched_output_path, index=False)
    print(f"Enriched data saved to {enriched_output_path}")
