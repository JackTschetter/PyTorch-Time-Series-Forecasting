import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("POLYGON_API_KEY")  # Retrieves the API key

# Check if the API key is found
if not API_KEY:
    print("API Key is missing. Please check your .env file.")
    exit(1)

# Stock symbol (e.g., AAPL for Apple)
STOCK_SYMBOL = "TSLA"

# Date range (past 30 days)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

# Polygon.io API endpoint (1-minute interval)
BASE_URL = f"https://api.polygon.io/v2/aggs/ticker/{STOCK_SYMBOL}/range/1/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"

def fetch_stock_data():
    """Fetch historical stock data from Polygon.io API."""
    response = requests.get(BASE_URL)
    
    if response.status_code == 200:
        data = response.json()
        
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp
            
            # Rename columns
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            return df
        else:
            print("No data found.")
            return None
    else:
        print("Error:", response.status_code, response.text)
        return None

# Fetch data
df = fetch_stock_data()

if df is not None:
    # Save to CSV
    filename = f"{STOCK_SYMBOL}_minute_data.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    
