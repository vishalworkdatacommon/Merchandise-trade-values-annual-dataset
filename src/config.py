
# Configuration file for the AI Trade Forecaster

# --- Data Paths ---
DATA_DIR = 'data'
REPORTERS_JSON_PATH = f'{DATA_DIR}/reporters.json'
COMMODITIES_JSON_PATH = f'{DATA_DIR}/commodities.json'

# --- Comtrade API ---
COMTRADE_API_BASE_URL = "https://comtradeapi.un.org/public/v1/get/C/A/HS"

# --- World Bank API ---
WB_INDICATOR = 'NY.GDP.MKTP.CD'
WB_START_YEAR = 1960
WB_END_YEAR = 2024

# --- Forecasting ---
MIN_YEARS_FOR_FORECAST = 15
FORECAST_STEPS = 5
GDP_GROWTH_ASSUMPTION = 1.04

# --- SARIMAX Model ---
SARIMAX_ORDER = (0, 1, 1)

# --- LSTM Model ---
LSTM_LOOK_BACK = 2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 1
LSTM_NEURONS = 16

# --- Gradio App ---
COUNTRY_CODE_MAP = {"842": "USA", "156": "CHN", "276": "DEU", "392": "JPN", "356": "IND"}
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860

# --- LLM ---
LLM_MODEL = 'google/gemma-2b-it'
LLM_MAX_NEW_TOKENS = 512
