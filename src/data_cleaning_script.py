import pandas as pd
import os
import logging

def clean_and_treat_outliers(df):
    """Cleans, aggregates, and treats outliers in the trade data.

    This function performs several preprocessing steps:
    1. Aggregates the data by year, summing the 'Value'.
    2. Sets a proper yearly frequency ('YS') for the time series.
    3. Identifies outliers using a rolling median and standard deviation approach.
    4. Replaces identified outliers with the rolling median.
    5. Removes any rows with zero or negative values.

    Args:
        df (pd.DataFrame): The raw trade data DataFrame from the Comtrade API.
                           Expected to have 'Year' and 'Value' columns.

    Returns:
        pd.DataFrame: A cleaned and aggregated DataFrame with a DatetimeIndex
                      and treated outliers, ready for feature engineering.
    """
    logging.info("Cleaning and treating outliers in the DataFrame...")

    # --- Data Aggregation ---
    df_agg = df.groupby('Year')['Value'].sum().reset_index()
    df_agg = df_agg.set_index('Year')
    df_agg.index = pd.to_datetime(df_agg.index, format='%Y')
    df_agg = df_agg.asfreq('YS')

    logging.info(f"Initial Data Profile:\n{df_agg.head()}")

    # --- Outlier Detection and Treatment ---
    window_size = 5
    rolling_median = df_agg['Value'].rolling(window=window_size, center=True).median()
    rolling_std = df_agg['Value'].rolling(window=window_size, center=True).std()

    outlier_threshold = 2
    is_outlier = (df_agg['Value'] - rolling_median).abs() > (outlier_threshold * rolling_std)

    outliers = df_agg[is_outlier]

    if not outliers.empty:
        logging.warning(f"Found {len(outliers)} potential outlier(s):\n{outliers}")
        logging.info("Treating outliers by replacing them with the rolling median...")
        df_agg.loc[is_outlier, 'Value'] = rolling_median[is_outlier]
        logging.info("Outlier treatment complete.")
    else:
        logging.info("No significant outliers were detected.")

    if (df_agg['Value'] <= 0).any():
        logging.warning(f"Found zero or negative values after cleaning. Inspecting...\n{df_agg[df_agg['Value'] <= 0]}")
        df_agg = df_agg[df_agg['Value'] > 0]

    # Drop rows where rolling median could not be calculated
    df_agg.dropna(inplace=True)

    logging.info("Cleaned data successfully.")
    logging.info(f"Cleaned Data Sample:\n{df_agg.head()}")
    
    return df_agg.reset_index()


if __name__ == "__main__":
    # This block is for testing the function with a sample CSV file.
    data_dir = 'data'
    input_csv_path = os.path.join(data_dir, 'processed_china_exports.csv')
    
    if os.path.exists(input_csv_path):
        print(f"Loading test data from {input_csv_path}")
        test_df = pd.read_csv(input_csv_path)
        cleaned_df = clean_and_treat_outliers(test_df)
        
        print("\n--- Standalone Test Successful ---")
        print("Cleaned DataFrame head:")
        print(cleaned_df.head())
        print("------------------------------------")
    else:
        print(f"Test file not found at {input_csv_path}. Skipping standalone test.")