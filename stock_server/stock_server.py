from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Path to CSV file (replace with your actual file path)
CSV_FILE_PATH = 'TSLA_minute_data.csv'  # Update this path to where your CSV file is located

# Function to load CSV data
def load_csv_data():
    """Load the CSV data into a pandas DataFrame."""
    return pd.read_csv(CSV_FILE_PATH, parse_dates=['timestamp'])

# Helper function to filter data for one minute ago at the same time as current time today
def get_data_for_one_minute_ago():
    """Filter stock data for the same minute, one month ago."""
    # Load the CSV data
    df = load_csv_data()
    
    # Get the current time and calculate one month ago (same time)
    today = datetime.today()
    one_month_ago = today - timedelta(days=29)
    
    # Adjust the one_month_ago to match the same time of day
    one_month_ago = one_month_ago.replace(hour=today.hour, minute=today.minute, second=0, microsecond=0)
    
    # Debug: print the current time and the calculated one month ago time
    print(f"Today's Time: {today}")
    print(f"One Month Ago: {one_month_ago}")
    
    # Filter the data for the same minute, one month ago
    df_filtered = df[df['timestamp'] == one_month_ago]
    
    # Debug: check if we have any data in the filtered range
    if df_filtered.empty:
        print("No data for the specified time (one month ago).")
    else:
        print("Filtered Data:")
        print(df_filtered.head())
    
    # Return the filtered data in JSON format
    return df_filtered.to_dict(orient='records')

@app.route('/api/v1/stock_data', methods=['GET'])
def get_stock_data():
    """API endpoint that returns the stock data for the same minute, one month ago."""
    data = get_data_for_one_minute_ago()
    
    if data:
        return jsonify(data), 200
    else:
        return jsonify({"error": "No data available for the same minute, one month ago"}), 404

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
