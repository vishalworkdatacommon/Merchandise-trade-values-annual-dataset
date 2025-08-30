import pandas as pd
import os

def process_trade_data(input_path, output_path):
    """
    Reads the large trade data CSV in chunks, filters for a specific data series,
    and saves the result to a new CSV.

    Args:
        input_path (str): Path to the large input CSV file.
        output_path (str): Path to save the processed CSV file.
    """
    print(f"Starting to process {input_path}...")

    # Define the columns we want to keep for our analysis
    cols_to_use = ['Reporter', 'Partner', 'Product', 'Year', 'Value']
    
    # Create an iterator to read the CSV in chunks of 1 million rows
    chunk_iter = pd.read_csv(
        input_path,
        chunksize=1000000,
        usecols=cols_to_use,
        low_memory=False,
        encoding='latin1',  # Try a different encoding
        on_bad_lines='skip' # Skip rows that cause errors
    )

    processed_chunks = []
    chunk_num = 0
    for chunk in chunk_iter:
        chunk_num += 1
        print(f"Processing chunk {chunk_num}...")

        # Filter the chunk for the specific data we want:
        # - Reporter: China
        # - Partner: World
        # - Product: Total merchandise
        filtered_chunk = chunk[
            (chunk['Reporter'] == 'China') &
            (chunk['Partner'] == 'World') &
            (chunk['Product'] == 'Total merchandise')
        ]
        
        if not filtered_chunk.empty:
            print(f"  Found {len(filtered_chunk)} relevant rows in chunk {chunk_num}.")
            processed_chunks.append(filtered_chunk)

    if not processed_chunks:
        print("No data found for the specified filters.")
        return

    # Concatenate the filtered chunks into a single DataFrame
    final_df = pd.concat(processed_chunks, ignore_index=True)

    # Convert 'Year' to integer and 'Value' to numeric, coercing errors
    final_df['Year'] = pd.to_numeric(final_df['Year'], errors='coerce')
    final_df['Value'] = pd.to_numeric(final_df['Value'], errors='coerce')

    # Drop rows with invalid data and sort by year
    final_df.dropna(subset=['Year', 'Value'], inplace=True)
    final_df['Year'] = final_df['Year'].astype(int)
    final_df.sort_values('Year', inplace=True)

    # Save the processed data
    final_df.to_csv(output_path, index=False)
    print(f"Successfully processed and saved the filtered data to {output_path}")
    print("\n--- Processed Data Sample ---")
    print(final_df.head())
    print("---------------------------")


if __name__ == "__main__":
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the input and output file paths
    input_csv_path = os.path.join(script_dir, 'data', 'merchandise_values_annual_input.csv')
    processed_csv_path = os.path.join(script_dir, 'data', 'processed_china_exports.csv')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    
    process_trade_data(input_csv_path, processed_csv_path)
