# AI-Powered Global Trade Forecasting and Analysis Platform

This project provides a comprehensive platform for forecasting and analyzing global trade data. It uses historical merchandise trade data to train a time-series forecasting model and offers a simple chatbot interface to retrieve future trade predictions.

## Project Structure

```
/
├── data/                # Holds raw, processed, and output data (ignored by Git)
├── notebooks/           # Jupyter notebooks for experimentation (if any)
├── src/                 # Source code for data processing, forecasting, and the chatbot
├── tests/               # Unit tests for the application
├── .gitignore           # Specifies files to be ignored by Git
├── Dockerfile           # Defines the Docker container for the application
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Features

- **Data Processing:** Scripts to handle large trade datasets efficiently.
- **Exploratory Data Analysis (EDA):** Generates visualizations and summary statistics.
- **Time-Series Forecasting:** Uses a SARIMAX model to predict future trade values.
- **Outlier Detection:** Identifies and treats anomalous data points before forecasting.
- **Chatbot Interface:** A simple, command-line-based AI assistant to retrieve forecasts.
- **Dockerized:** Fully containerized for easy and reproducible deployment.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Docker (for containerized deployment)

### 1. Clone the Repository

```bash
git clone git@github.com:vishalworkdatacommon/Merchandise-trade-values-annual-dataset.git
cd Merchandise-trade-values-annual-dataset
```

### 2. Set Up the Environment

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Running the Application

The project workflow is broken down into several scripts. You should run them in the following order:

**Step 1: Download the Data**
*(Note: The data is already included in the `data` directory for this project, but you can re-run this script if needed)*
```bash
python3 src/download_script.py
```

**Step 2: Process the Raw Data**
This script filters the main dataset to create a focused time-series for analysis.
```bash
python3 src/data_processing_script.py
```

**Step 3: Perform EDA (Optional)**
This will generate a plot of the time-series in the `data` directory.
```bash
python3 src/eda_script.py
```

**Step 4: Clean the Data and Handle Outliers**
This creates a cleaned version of the data for forecasting.
```bash
python3 src/data_cleaning_script.py
```

**Step 5: Generate the Forecast**
This trains the model and saves the 5-year forecast.
```bash
python3 src/forecasting_script.py
```

**Step 6: Interact with the Chatbot**
Run the chatbot to ask for forecasts for specific years.
```bash
python3 src/chatbot.py
```

## Deployment with Docker

This application is fully containerized, allowing you to build and run it in a consistent environment.

### 1. Build the Docker Image

From the project's root directory, run the following command:

```bash
docker build -t trade-forecasting-app .
```

### 2. Run the Docker Container

After the image is built, you can run the chatbot in a container:

```bash
docker run -it --rm trade-forecasting-app
```

This will start the chatbot, and you can begin asking it questions.
