
import pandas as pd
import os

def clean_and_treat_outliers(input_df):
    """
    Cleans and treats outliers in a time-series DataFrame.
    """
    print("Cleaning data and treating outliers...")
    
    df_agg = input_df.groupby('Year')['Value'].sum().reset_index()
    df_agg = df_agg.set_index('Year')
    df_agg.index = pd.to_datetime(df_agg.index, format='%Y')
    df_agg = df_agg.asfreq('YS')

    window_size = 5
    rolling_median = df_agg['Value'].rolling(window=window_size, center=True).median()
    rolling_std = df_agg['Value'].rolling(window=window_size, center=True).std()

    outlier_threshold = 2
    is_outlier = (df_agg['Value'] - rolling_median).abs() > (outlier_threshold * rolling_std)

    outliers = df_agg[is_outlier]

    if not outliers.empty:
        print(f"Found {len(outliers)} potential outlier(s). Treating them...")
        df_agg.loc[is_outlier, 'Value'] = rolling_median[is_outlier]
    else:
        print("No significant outliers were detected.")

    if (df_agg['Value'] <= 0).any():
        df_agg = df_agg[df_agg['Value'] > 0]

    return df_agg.reset_index()

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    processed_csv_path = os.path.join(data_dir, 'processed_china_exports.csv')
    cleaned_output_path = os.path.join(data_dir, 'china_exports_cleaned.csv')
    
    df = pd.read_csv(processed_csv_path)
    cleaned_df = clean_and_treat_outliers(df)
    cleaned_df.to_csv(cleaned_output_path, index=False)
    print(f"Cleaned data saved to {cleaned_output_path}")
