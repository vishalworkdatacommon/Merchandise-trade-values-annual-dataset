
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- LSTM Helper Functions (adapted for backtesting) ---
def create_lstm_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i, 0]) # Corrected to align with input
    return np.array(dataX), np.array(dataY)

def run_backtest(enriched_data_path, results_output_path):
    """
    Performs a rigorous backtest of the SARIMAX and LSTM models.
    """
    print("Starting backtesting process...")
    df = pd.read_csv(enriched_data_path, index_col='Year', parse_dates=True)
    df = df.asfreq('YS').dropna()

    # --- Backtesting Configuration ---
    backtest_start_year = 2010
    backtest_end_year = df.index.max().year
    
    results = []

    print(f"Backtesting from {backtest_start_year} to {backtest_end_year}...")

    for year in range(backtest_start_year, backtest_end_year):
        print(f"\n--- Testing Year: {year + 1} ---")
        
        # --- 1. Split Data ---
        train_df = df[df.index.year <= year]
        actual_value = df[df.index.year == year + 1]['Value'].iloc[0]

        # --- 2. SARIMAX Backtest ---
        print("Training SARIMAX model...")
        endog = train_df['Value']
        exog = train_df[['GDP_USD']]
        
        sarimax_model = sm.tsa.statespace.SARIMAX(
            endog=endog, exog=exog, order=(1, 2, 1)
        ).fit(disp=False)
        
        # Forecast the next year's value
        exog_forecast = df[df.index.year == year + 1][['GDP_USD']]
        sarimax_pred = sarimax_model.get_forecast(steps=1, exog=exog_forecast).predicted_mean.iloc[0]
        print(f"SARIMAX Prediction for {year + 1}: {sarimax_pred:,.0f}")

        # --- 3. LSTM Backtest ---
        print("Training LSTM model...")
        scaler_value = MinMaxScaler(feature_range=(0, 1))
        scaler_gdp = MinMaxScaler(feature_range=(0, 1))
        
        train_scaled = train_df.copy()
        train_scaled['Value_Scaled'] = scaler_value.fit_transform(train_scaled['Value'].values.reshape(-1, 1))
        train_scaled['GDP_Scaled'] = scaler_gdp.fit_transform(train_scaled['GDP_USD'].values.reshape(-1, 1))
        
        dataset = train_scaled[['Value_Scaled', 'GDP_Scaled']].values
        look_back = 3
        trainX, trainY = create_lstm_dataset(dataset, look_back)
        
        lstm_model = Sequential([
            LSTM(4, input_shape=(look_back, 2)),
            Dense(1)
        ])
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=0)
        
        # Forecast the next year
        last_data_scaled = dataset[-look_back:]
        pred_input = np.reshape(last_data_scaled, (1, look_back, 2))
        lstm_pred_scaled = lstm_model.predict(pred_input)
        lstm_pred = scaler_value.inverse_transform(lstm_pred_scaled)[0][0]
        print(f"LSTM Prediction for {year + 1}: {lstm_pred:,.0f}")

        # --- 4. Store Results ---
        results.append({
            'Year': year + 1,
            'Actual_Value': actual_value,
            'SARIMAX_Forecast': sarimax_pred,
            'LSTM_Forecast': lstm_pred
        })

    # --- 5. Calculate and Report Metrics ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_output_path, index=False)
    print(f"\nBacktesting complete. Results saved to {results_output_path}")

    sarimax_mae = mean_absolute_error(results_df['Actual_Value'], results_df['SARIMAX_Forecast'])
    lstm_mae = mean_absolute_error(results_df['Actual_Value'], results_df['LSTM_Forecast'])
    
    sarimax_rmse = np.sqrt(mean_squared_error(results_df['Actual_Value'], results_df['SARIMAX_Forecast']))
    lstm_rmse = np.sqrt(mean_squared_error(results_df['Actual_Value'], results_df['LSTM_Forecast']))

    print("\n--- Backtesting Performance Metrics ---")
    print(f"SARIMAX Model:")
    print(f"  - Mean Absolute Error (MAE): ${sarimax_mae:,.0f}")
    print(f"  - Root Mean Squared Error (RMSE): ${sarimax_rmse:,.0f}")
    print(f"\nLSTM Model:")
    print(f"  - Mean Absolute Error (MAE): ${lstm_mae:,.0f}")
    print(f"  - Root Mean Squared Error (RMSE): ${lstm_rmse:,.0f}")
    print("---------------------------------------")
    
    winner = "LSTM" if lstm_mae < sarimax_mae else "SARIMAX"
    print(f"\nConclusion: The {winner} model performed better in this backtest.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    enriched_csv_path = os.path.join(script_dir, '..', 'data', 'china_exports_enriched.csv')
    backtest_results_path = os.path.join(script_dir, '..', 'data', 'backtest_results.csv')
    
    run_backtest(enriched_csv_path, backtest_results_path)
