
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

def create_lstm_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def forecast_lstm(input_df):
    """
    Builds and trains an LSTM model and returns the forecast.
    """
    print("Training LSTM model...")
    df = input_df.set_index('Year')
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('YS').dropna()

    scaler_value = MinMaxScaler(feature_range=(0, 1))
    scaler_gdp = MinMaxScaler(feature_range=(0, 1))
    
    df['Value_Scaled'] = scaler_value.fit_transform(df['Value'].values.reshape(-1, 1))
    df['GDP_Scaled'] = scaler_gdp.fit_transform(df['GDP_USD'].values.reshape(-1, 1))
    
    dataset = df[['Value_Scaled', 'GDP_Scaled']].values
    look_back = 3
    trainX, trainY = create_lstm_dataset(dataset, look_back)

    model = Sequential([
        LSTM(4, input_shape=(look_back, 2)),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

    print("Generating LSTM forecast...")
    last_data = dataset[-look_back:]
    future_gdp_scaled = scaler_gdp.transform(
        np.array([df['GDP_USD'].iloc[-1] * (1.04)**i for i in range(1, 6)]).reshape(-1, 1)
    )
    
    forecast = []
    current_input = last_data
    
    for gdp_val in future_gdp_scaled:
        pred_input = np.reshape(current_input, (1, look_back, 2))
        pred = model.predict(pred_input, verbose=0)
        forecast.append(pred[0][0])
        new_row = np.array([pred[0][0], gdp_val[0]])
        current_input = np.append(current_input[1:], [new_row], axis=0)

    forecast = scaler_value.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    forecast_years = pd.date_range(start=df.index.max() + pd.DateOffset(years=1), periods=5, freq='YS')
    
    forecast_df = pd.DataFrame(forecast, index=forecast_years, columns=['mean'])
    forecast_df.index.name = 'Year'
    
    return forecast_df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    enriched_csv_path = os.path.join(data_dir, 'china_exports_enriched.csv')
    forecast_output_path = os.path.join(data_dir, 'china_exports_forecast_lstm.csv')
    
    df = pd.read_csv(enriched_csv_path)
    forecast_df = forecast_lstm(df)
    forecast_df.to_csv(forecast_output_path)
    print(f"LSTM forecast saved to {forecast_output_path}")
