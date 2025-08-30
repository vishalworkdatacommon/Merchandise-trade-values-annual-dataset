import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def perform_eda(input_path, output_plot_path):
    """
    Loads processed trade data, performs exploratory data analysis,
    and saves a time-series plot.

    Args:
        input_path (str): Path to the processed CSV file.
        output_plot_path (str): Path to save the output plot image.
    """
    print(f"Loading data from {input_path} for EDA...")
    df = pd.read_csv(input_path)

    # --- Data Aggregation and Preparation ---
    # The data may have multiple entries for the same year. We will sum these
    # to get a single, total value for each year.
    print("Aggregating data to create a clean time-series...")
    df_agg = df.groupby('Year')['Value'].sum().reset_index()
    df_agg = df_agg.set_index('Year')
    df_agg.index = pd.to_datetime(df_agg.index, format='%Y')
    df_agg = df_agg.asfreq('YS') # Ensure yearly frequency

    print("\n--- Summary Statistics ---")
    print(df_agg['Value'].describe())
    print("--------------------------")

    # --- Visualization ---
    print(f"\nGenerating and saving time-series plot to {output_plot_path}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_agg.index, df_agg['Value'], marker='o', linestyle='-')
    
    # Formatting the plot
    ax.set_title("China's Total Merchandise Exports to the World (1948-Present)", fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Export Value (in Millions of USD)", fontsize=12)
    ax.grid(True)
    
    # Improve y-axis readability
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print("Plot saved successfully.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    processed_csv_path = os.path.join(script_dir, 'data', 'processed_china_exports.csv')
    plot_output_path = os.path.join(script_dir, 'data', 'china_exports_timeseries.png')
    
    perform_eda(processed_csv_path, plot_output_path)
