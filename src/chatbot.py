
import pandas as pd
import os

def load_forecast_data(file_path):
    """Loads the forecast data from the CSV file."""
    try:
        df = pd.read_csv(file_path, index_col='Year')
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
        print(f"Error: Forecast file not found at {file_path}")
        return None

def get_forecast_for_year(year, forecast_df):
    """Retrieves the forecast for a specific year."""
    try:
        # Create a datetime object for the requested year
        target_date = pd.to_datetime(f"{year}-01-01")
        
        if target_date in forecast_df.index:
            return forecast_df.loc[target_date]
        else:
            return None
    except ValueError:
        return None # Invalid year format

def generate_response(year, data):
    """Generates a natural language response for the forecast."""
    if data is None:
        return f"I'm sorry, but I don't have a forecast for the year {year}. I can only provide forecasts for the next 5 years from our last data point."

    mean_value = data['mean'] / 1e6  # Convert to trillions for readability
    ci_lower = data['mean_ci_lower'] / 1e6
    ci_upper = data['mean_ci_upper'] / 1e6

    response = (
        f"\n🤖 **Trade Forecast Analysis for {year}:**\n\n"
        f"Our model forecasts that the total merchandise export value for China in **{year}** will be approximately **${mean_value:.2f} trillion USD**.\n\n"
        f"We are 95% confident that the actual value will fall between ${ci_lower:.2f} trillion and ${ci_upper:.2f} trillion USD.\n\n"
        f"*Disclaimer: This is a simplified forecast based on historical data and does not account for unforeseen economic events.*"
    )
    return response

def start_chatbot(forecast_file):
    """Main function to run the chatbot interface."""
    forecast_df = load_forecast_data(forecast_file)
    if forecast_df is None:
        return

    print("--- Global Trade AI Chatbot ---")
    print("I can provide forecasts for China's total merchandise exports.")
    print("You can ask questions like 'What is the forecast for 2026?' or type 'exit' to quit.")

    while True:
        user_input = input("> ").strip().lower()

        if user_input == 'exit':
            print("Goodbye!")
            break

        # Simple keyword extraction for the year
        words = user_input.split()
        year = None
        for word in words:
            if word.isdigit() and len(word) == 4:
                year = int(word)
                break
        
        if year:
            forecast_data = get_forecast_for_year(year, forecast_df)
            response = generate_response(year, forecast_data)
            print(response)
        else:
            print("I'm sorry, I didn't understand that. Please ask for a forecast for a specific year (e.g., 'forecast for 2027').")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forecast_csv_path = os.path.join(script_dir, 'data', 'china_exports_forecast.csv')
    start_chatbot(forecast_csv_path)
