import pandas as pd
import os

def clean_and_treat_outliers(input_path, output_path):
    """
    Loads, cleans, and treats outliers in the time-series data.
    """
    print(f"Loading data from {input_path} for cleaning and outlier treatment...")
    df = pd.read_csv(input_path)

    # --- Data Aggregation ---
    df_agg = df.groupby('Year')['Value'].sum().reset_index()
    df_agg = df_agg.set_index('Year')
    df_agg.index = pd.to_datetime(df_agg.index, format='%Y')
    df_agg = df_agg.asfreq('YS')

    print("\n--- Initial Data Profile ---")
    print(df_agg.head())
    print("----------------------------")

    # --- Outlier Detection and Treatment ---
    window_size = 5
    rolling_median = df_agg['Value'].rolling(window=window_size, center=True).median()
    rolling_std = df_agg['Value'].rolling(window=window_size, center=True).std()

    outlier_threshold = 2
    is_outlier = (df_agg['Value'] - rolling_median).abs() > (outlier_threshold * rolling_std)

    outliers = df_agg[is_outlier]

    if not outliers.empty:
        print(f"\nFound {len(outliers)} potential outlier(s):")
        print(outliers)
        
        print("\nTreating outliers by replacing them with the rolling median...")
        df_agg.loc[is_outlier, 'Value'] = rolling_median[is_outlier]
        print("Outlier treatment complete.")
    else:
        print("\nNo significant outliers were detected.")

    if (df_agg['Value'] <= 0).any():
        print("\nWarning: Found zero or negative values after cleaning. Inspecting...")
        print(df_agg[df_agg['Value'] <= 0])
        df_agg = df_agg[df_agg['Value'] > 0]

    df_agg.to_csv(output_path)
    print(f"\nCleaned data successfully saved to {output_path}")
    print("\n--- Cleaned Data Sample ---")
    print(df_agg.head())
    print("---------------------------")

if __name__ == "__main__":
    data_dir = 'data'
    processed_csv_path = os.path.join(data_dir, 'processed_china_exports.csv')
    cleaned_output_path = os.path.join(data_dir, 'china_exports_cleaned.csv')
    
    clean_and_treat_outliers(processed_csv_path, cleaned_output_path)
