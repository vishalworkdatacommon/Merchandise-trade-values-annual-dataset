
import pandas as pd
import os

def process_trade_data(input_path, output_path, reporter="China", partner="World", product="Total merchandise"):
    """
    Reads the large trade data CSV in chunks, filters for a specific data series,
    and saves the result to a new CSV.
    """
    print(f"Processing {input_path} for Reporter='{reporter}', Partner='{partner}', Product='{product}'...")

    cols_to_use = ['Reporter', 'Partner', 'Product', 'Year', 'Value']
    chunk_iter = pd.read_csv(
        input_path,
        chunksize=1000000,
        usecols=cols_to_use,
        low_memory=False,
        encoding='latin1'
    )

    processed_chunks = []
    for chunk in chunk_iter:
        filtered_chunk = chunk[
            (chunk['Reporter'] == reporter) &
            (chunk['Partner'] == partner) &
            (chunk['Product'] == product)
        ]
        if not filtered_chunk.empty:
            processed_chunks.append(filtered_chunk)

    if not processed_chunks:
        print("No data found for the specified filters.")
        return pd.DataFrame() # Return empty dataframe

    final_df = pd.concat(processed_chunks, ignore_index=True)
    final_df['Year'] = pd.to_numeric(final_df['Year'], errors='coerce')
    final_df['Value'] = pd.to_numeric(final_df['Value'], errors='coerce')
    final_df.dropna(subset=['Year', 'Value'], inplace=True)
    final_df['Year'] = final_df['Year'].astype(int)
    final_df.sort_values('Year', inplace=True)

    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved filtered data to {output_path}")
    return final_df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    raw_csv_path = os.path.join(data_dir, 'merchandise_values_annual_input.csv')
    processed_csv_path = os.path.join(data_dir, 'processed_china_exports.csv')
    process_trade_data(raw_csv_path, processed_csv_path)
