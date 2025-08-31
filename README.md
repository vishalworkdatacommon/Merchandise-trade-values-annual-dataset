---
title: AI Trade Forecaster
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.29.0"
app_file: app.py
---

# AI-Powered Global Trade Forecasting and Analysis Platform

This project provides a comprehensive, industry-level platform for forecasting and analyzing global trade data. It demonstrates a full MLOps lifecycle, including data processing, advanced modeling, automated testing, and deployment as an interactive web application.

## Features

- **Interactive Analysis:** Allows users to select any country, partner, and product combination for on-the-fly forecasting.
- **Dynamic AI Insights:** Uses a Large Language Model (Google's Gemma) to provide a custom analysis for each user query.
- **Advanced Forecasting:** Implements and compares a classical statistical model (SARIMAX) and a deep learning model (LSTM) for any selected data series.
- **Automated Data Pipeline:** A complete pipeline that processes, cleans, and enriches data with external GDP information in real-time.
- **Professional Project Structure:** Organized, documented, and version-controlled with Git.
- **Automated Testing & CI/CD:** Includes unit tests and a GitHub Actions workflow for continuous integration.
- **Deployable & Shareable:** Ready for deployment on Hugging Face Spaces.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vishalworkdatacommon/Merchandise-trade-values-annual-dataset.git
cd Merchandise-trade-values-annual-dataset
```

### 2. Set Up the Environment

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Running the Application Locally

The entire pipeline is now managed by a single main script. This will process the data, run the forecasts, and launch the web application.

```bash
python3 src/main.py
```
Open your browser and navigate to `http://localhost:7860` to use the app.

## Deployment to Hugging Face Spaces

This project is now fully configured for deployment on Hugging Face Spaces.

### 1. Create a New Space on Hugging Face

- Go to [huggingface.co/new-space](https://huggingface.co/new-space).
- Give your Space a name.
- Select **Gradio** as the Space SDK.
- Choose the **"Use the free CPU"** hardware.
- Select **"Public"**.
- Click **"Create Space"**.

### 2. Add Your API Keys as Secrets

This is a **critical step**. The application needs two API keys to function:

- In your Hugging Face Space, go to the **"Settings"** tab.
- Find the **"Repository secrets"** section.
- Add the following two secrets:
    1.  **Name:** `HF_TOKEN`
        - **Value:** Your Hugging Face access token.
    2.  **Name:** `UN_COMTRADE_API_KEY`
        - **Value:** Your UN Comtrade API subscription key. You can get one for free from the [UN Comtrade Portal](https://comtradeplus.un.org/portal).

### 3. Push Your Code

Push your local repository to the Hugging Face Space to trigger the deployment.

```bash
# Add the Hugging Face remote (if you haven't already)
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push your code to the 'main' branch of the Space
git push --force space main
```

The Space will now build the Docker container, run the full data pipeline, and launch your Gradio application. The initial startup will take a few minutes to download the models and process the data.
