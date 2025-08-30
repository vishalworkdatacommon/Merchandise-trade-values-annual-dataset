---
title: AI Trade Forecaster
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.29.0"
app_file: src/app.py
pinned: false
---

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

**Step 6: Run the Web Application**
This will start a local web server. You can open the provided URL in your browser to interact with the application.
```bash
python3 src/app.py
```

## Deployment with Docker

This application is fully containerized, allowing you to build and run it in a consistent environment.

### 1. Build the Docker Image

From the project's root directory, run the following command:

```bash
docker build -t trade-forecasting-app .
```

### 2. Run the Docker Container

After the image is built, you can run the web application in a container:

```bash
docker run -it --rm -p 7860:7860 trade-forecasting-app
```
Open your browser and navigate to `http://localhost:7860` to use the app.

## Deployment to Hugging Face Spaces

This project is optimized for deployment on Hugging Face Spaces.

### 1. Create a New Space

- Go to [huggingface.co/new-space](https://huggingface.co/new-space).
- Give your Space a name.
- Select **Gradio** as the SDK.
- Choose the **"Use the free CPU"** hardware.
- Select **"Public"** for the visibility.
- Click **"Create Space"**.

### 2. Push Your Code

You will be given instructions on how to push your repository to the new Space. It will look something like this:

```bash
# Add the Hugging Face remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push your code to the 'main' branch of the Space
git push --force space main
```

Once you push your code, Hugging Face will automatically build the Docker container and launch your Gradio application. Your AI-powered forecasting tool will then be live for anyone to use!