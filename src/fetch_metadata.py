
import requests
import json
import os

def fetch_and_save_metadata():
    """
    Fetches the latest country and commodity codes from the UN Comtrade API
    and saves them to local JSON files.
    """
    print("Fetching UN Comtrade metadata...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # --- Fetch Reporter (Country) Codes ---
    try:
        # This is the correct, documented endpoint for the list of reporters
        reporters_url = "https://comtradeapi.un.org/files/v1/app/getReporterDataset"
        response = requests.get(reporters_url)
        response.raise_for_status()
        reporters_data = response.json()
        
        reporter_path = os.path.join(data_dir, 'reporters.json')
        with open(reporter_path, 'w') as f:
            json.dump(reporters_data, f)
        print(f"Successfully saved {len(reporters_data)} reporter codes to {reporter_path}")

    except Exception as e:
        print(f"Could not fetch reporter codes: {e}")

    # --- Fetch Commodity Codes (HS, simplified) ---
    try:
        # This is the correct, documented endpoint for the HS commodity guide
        commodities_url = "https://comtradeapi.un.org/files/v1/app/getHSGuide"
        response = requests.get(commodities_url)
        response.raise_for_status()
        commodities_data = response.json()
        
        commodity_path = os.path.join(data_dir, 'commodities.json')
        with open(commodity_path, 'w') as f:
            json.dump(commodities_data, f)
        print(f"Successfully saved {len(commodities_data)} commodity codes to {commodity_path}")

    except Exception as e:
        print(f"Could not fetch commodity codes: {e}")

if __name__ == "__main__":
    fetch_and_save_metadata()
