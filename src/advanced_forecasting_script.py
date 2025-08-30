
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def create_dataset(dataset, look_back=1):
    """Create dataset for LSTM model with multiple features."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :] # Get all features
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) # Predict only the 'Value'
    return np.array(dataX), np.array(dataY)

def forecast_with_lstm(input_path, model_path, forecast_output_path):
    """
    Loads cleaned data, builds and trains an LSTM model, and saves the forecast.
    """
    print(f"Loading data from {input_path} for LSTM modeling...")
    df = pd.read_csv(input_path, index_col='Year', parse_dates=True)
    df = df.asfreq('YS')
    df['Value'] = df['Value'].fillna(method='ffill')

    # --- Data Scaling ---
    scaler_value = MinMaxScaler(feature_range=(0, 1))
    scaler_gdp = MinMaxScaler(feature_range=(0, 1))
    
    df['Value_Scaled'] = scaler_value.fit_transform(df['Value'].values.reshape(-1, 1))
    df['GDP_Scaled'] = scaler_gdp.fit_transform(df['GDP_USD'].values.reshape(-1, 1))
    
    dataset = df[['Value_Scaled', 'GDP_Scaled']].values

    # --- Train/Test Split ---
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # --- Reshape into LSTM format ---
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, 2))
    testX = np.reshape(testX, (testX.shape[0], look_back, 2))

    # --- Build and Train LSTM Model ---
    if os.path.exists(model_path):
        print("Loading pre-trained LSTM model...")
        model = load_model(model_path)
    else:
        print("Training a new LSTM model with external regressor...")
        model = Sequential()
        model.add(LSTM(4, input_shape=(look_back, 2))) # Input shape now has 2 features
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # --- Forecasting ---
    print("\nGenerating forecast with LSTM model...")
    last_data = dataset[-look_back:]
    
    # We need to forecast future GDP to use as input for the model
    future_gdp_scaled = scaler_gdp.transform(
        np.array([df['GDP_USD'].iloc[-1] * (1.04)**i for i in range(1, 6)]).reshape(-1, 1)
    )
    
    forecast = []
    current_input = last_data
    
    for gdp_val in future_gdp_scaled:
        # Reshape input for prediction
        pred_input = np.reshape(current_input, (1, look_back, 2))
        pred = model.predict(pred_input)
        forecast.append(pred[0][0])
        
        # Create new input for the next step
        new_row = np.array([pred[0][0], gdp_val[0]])
        current_input = np.append(current_input[1:], [new_row], axis=0)

    # --- Inverse Transform and Save ---
    forecast = scaler_value.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    last_year = df.index.max().year
    forecast_years = pd.date_range(start=f'{last_year + 1}-01-01', periods=5, freq='YS')
    
    forecast_df = pd.DataFrame(forecast, index=forecast_years, columns=['mean'])
    forecast_df.index.name = 'Year'
    
    forecast_df.to_csv(forecast_output_path)
    print(f"LSTM forecast successfully saved to {forecast_output_path}")
    print("\n--- LSTM Forecasted Values ---")
    print(forecast_df)
    print("------------------------------")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cleaned_csv_path = os.path.join(script_dir, '..', 'data', 'china_exports_enriched.csv')
    lstm_model_path = os.path.join(script_dir, '..', 'data', 'lstm_model.h5')
    lstm_forecast_path = os.path.join(script_dir, '..', 'data', 'china_exports_forecast_lstm.csv')
    
    forecast_with_lstm(cleaned_csv_path, lstm_model_path, lstm_forecast_path)
