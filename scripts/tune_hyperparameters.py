
import pandas as pd
import numpy as np
import logging
import itertools
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Add src to path to import from custom modules
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging_config import setup_logging
from src.data_cleaning_script import clean_and_treat_outliers
from src.data_integration_script import integrate_external_data
from src.comtrade_api import get_comtrade_data
from src.advanced_forecasting_script import create_lstm_dataset

warnings.filterwarnings("ignore")

def tune_sarimax(train_df):
    """Performs a grid search to find the best SARIMAX order."""
    logging.info("--- Starting SARIMAX Hyperparameter Tuning ---")
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    
    best_aic = np.inf
    best_order = None
    
    endog = train_df['Value']
    exog = train_df[['GDP_USD']]

    for order in pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(
                endog=endog,
                exog=exog,
                order=order,
            ).fit(disp=False)
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
                logging.info(f"New best SARIMAX order: {best_order} with AIC: {best_aic}")

        except Exception as e:
            logging.error(f"Error with order {order}: {e}")
            continue
            
    logging.info(f"--- Finished SARIMAX Tuning. Best Order: {best_order} ---")
    return best_order

def tune_lstm(train_df):
    """Performs a grid search for LSTM hyperparameters."""
    logging.info("--- Starting LSTM Hyperparameter Tuning ---")

    # Hyperparameter grid
    neurons = [4, 8, 16]
    epochs = [50, 100]
    batch_sizes = [1, 2]
    look_backs = [2, 3, 4]

    best_rmse = np.inf
    best_params = {}

    # Further split training data for validation
    train, val = train_test_split(train_df, test_size=0.2, shuffle=False)

    for lb in look_backs:
        for n in neurons:
            for e in epochs:
                for bs in batch_sizes:
                    try:
                        # Create and train model
                        model = Sequential([
                            LSTM(n, input_shape=(lb, 2)),
                            Dense(1)
                        ])
                        model.compile(loss='mean_squared_error', optimizer='adam')
                        
                        # Note: For a real scenario, scaling should be fit on train only
                        scaled_train = train[['Value', 'GDP_USD']].values
                        trainX, trainY = create_lstm_dataset(scaled_train, lb)
                        
                        model.fit(trainX, trainY, epochs=e, batch_size=bs, verbose=0)

                        # Evaluate on validation set
                        scaled_val = val[['Value', 'GDP_USD']].values
                        valX, valY = create_lstm_dataset(scaled_val, lb)
                        
                        predictions = model.predict(valX, verbose=0)
                        rmse = np.sqrt(mean_squared_error(valY, predictions))

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'neurons': n, 'epochs': e, 'batch_size': bs, 'look_back': lb}
                            logging.info(f"New best LSTM params: {best_params} with RMSE: {rmse}")

                    except Exception as err:
                        logging.error(f"Error with params {lb, n, e, bs}: {err}")
                        continue

    logging.info(f"--- Finished LSTM Tuning. Best Params: {best_params} ---")
    return best_params


if __name__ == "__main__":
    setup_logging()
    logging.info("Fetching sample data for hyperparameter tuning...")
    
    # Use a standard dataset for tuning
    sample_df = get_comtrade_data(reporter_id="842", partner_id="0", product_id="TOTAL")
    
    if sample_df.empty:
        logging.warning("Could not fetch live data. Falling back to local cached data.")
        try:
            sample_df = pd.read_csv('data/merchandise_values_annual_input.csv')
            # Basic preprocessing to match comtrade_api output
            sample_df = sample_df[sample_df['Reporter ISO'] == 'USA']
            sample_df = sample_df[['Year', 'Trade Value (US$)', 'Reporter', 'Partner']]
            sample_df.rename(columns={'Trade Value (US$)': 'Value'}, inplace=True)
        except FileNotFoundError:
            logging.error("Local data file not found. Aborting tuning.")
            sample_df = pd.DataFrame()

    if not sample_df.empty:
        cleaned_df = clean_and_treat_outliers(sample_df)
        enriched_df = integrate_external_data(cleaned_df, country_code="USA")
        enriched_df['Year'] = pd.to_datetime(enriched_df['Year'])
        enriched_df = enriched_df.set_index('Year')

        # Tune SARIMAX
        best_sarimax_order = tune_sarimax(enriched_df)
        print(f"\nRECOMMENDATION: Update SARIMAX_ORDER in config.py to: {best_sarimax_order}")

        # Tune LSTM
        best_lstm_params = tune_lstm(enriched_df)
        print(f"\nRECOMMENDATION: Update LSTM parameters in config.py based on these results: {best_lstm_params}")
    else:
        logging.error("Could not fetch or load any sample data. Aborting tuning.")
