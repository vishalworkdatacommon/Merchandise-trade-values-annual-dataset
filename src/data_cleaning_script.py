
import pandas as pd
import os

def clean_and_treat_outliers(input_path, output_path):
    """
    Loads, cleans, and treats outliers in the time-series data.

    Args:
        input_path (str): Path to the processed CSV file.
        output_path (str): Path to save the cleaned CSV file.
    """
    print(f"Loading data from {input_path} for cleaning and outlier treatment...")
    df = pd.read_csv(input_path)

    # --- Data Aggregation ---
    # Same aggregation as before to create a single time-series
    df_agg = df.groupby('Year')['Value'].sum().reset_index()
    df_agg = df_agg.set_index('Year')
    df_agg.index = pd.to_datetime(df_agg.index, format='%Y')
    df_agg = df_agg.asfreq('YS')

    print("\n--- Initial Data Profile ---")
    print(df_agg.head())
    print("----------------------------")

    # --- Outlier Detection and Treatment ---
    # We use a rolling window to calculate the median and standard deviation.
    # This helps identify points that are anomalous compared to their neighbors.
    window_size = 5  # 5-year rolling window
    rolling_median = df_agg['Value'].rolling(window=window_size, center=True).median()
    rolling_std = df_agg['Value'].rolling(window=window_size, center=True).std()

    # Identify outliers as points more than 2 standard deviations from the rolling median
    outlier_threshold = 2
    is_outlier = (df_agg['Value'] - rolling_median).abs() > (outlier_threshold * rolling_std)

    outliers = df_agg[is_outlier]

    if not outliers.empty:
        print(f"\nFound {len(outliers)} potential outlier(s):")
        print(outliers)
        
        # --- Outlier Treatment ---
        # Replace outliers with the rolling median value for that period
        print("\nTreating outliers by replacing them with the rolling median...")
        df_agg['Value'][is_outlier] = rolling_median[is_outlier]
        print("Outlier treatment complete.")
    else:
        print("\nNo significant outliers were detected.")

    # --- Final Checks ---
    # Check for negative or zero values which are not logical for trade data
    if (df_agg['Value'] <= 0).any():
        print("\nWarning: Found zero or negative values after cleaning. Inspecting...")
        print(df_agg[df_agg['Value'] <= 0])
        # For this case, we'll remove them, but in a real project, this would
        # require deeper investigation.
        df_agg = df_agg[df_agg['Value'] > 0]

    # Save the cleaned data
    df_agg.to_csv(output_path)
    print(f"\nCleaned data successfully saved to {output_path}")
    print("\n--- Cleaned Data Sample ---")
    print(df_agg.head())
    print("---------------------------")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    processed_csv_path = os.path.join(script_dir, 'data', 'processed_china_exports.csv')
    cleaned_output_path = os.path.join(script_dir, 'data', 'china_exports_cleaned.csv')
    
    clean_and_treat_outliers(processed_csv_path, cleaned_output_path)
