from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize memory to store recent stock data
stock_data_memory = []

# The URL of your local stock data API
STOCK_DATA_API_URL = "http://localhost:5001/api/v1/stock_data"

# Prepare LSTM Model for prediction
def create_lstm_model(X_train):
    """Create and return an LSTM model for prediction."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_for_lstm(data):
    """Prepare the data for the LSTM model."""
    # Check if there are at least 5 data points
    if len(data) < 5:
        print("Not enough data to make a prediction. Waiting for more data...")
        return None, None, None  # Not enough data
    
    # If there are fewer than 60 data points, use the entire dataset
    if len(data) <= 60:
        data = data[-len(data):]  # Use all the available data

    # If there are more than 60 data points, limit the data to the most recent 60
    else:
        data = data[-60:]  # Use only the most recent 60 data points

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare X and Y datasets
    X = []
    Y = []
    for i in range(60, len(scaled_data)):  # Use the last 60 minutes for prediction
        X.append(scaled_data[i-60:i, 0])  # last 60 minutes
        Y.append(scaled_data[i, 0])  # next minute's close price

    # Check if X and Y are populated
    if len(X) == 0 or len(Y) == 0:
        print("Error: X and Y arrays are empty.")
        return None, None, None

    X_train = np.array(X)
    Y_train = np.array(Y)

    # Reshape X_train to be 3D for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, Y_train, scaler


# Function to fetch the most recent stock data from the local API
def fetch_latest_stock_data():
    """Fetch the latest stock data from the local server."""
    response = requests.get(STOCK_DATA_API_URL)
    if response.status_code == 200:
        data = response.json()
        return data[0]  # Assuming the API returns a list of stock data
    else:
        print("Error fetching stock data.")
        return None

def predict_next_5_minutes():
    """Predict the next 5 minutes based on the most recent stock data."""
    # Fetch the most recent stock data from the local API
    recent_data = fetch_latest_stock_data()
    
    if not recent_data:
        return []

    # Add the new data to memory
    stock_data_memory.append(recent_data)

    # Prepare the data for the LSTM model (using 'close' price)
    data_for_lstm = [row['close'] for row in stock_data_memory]
    
    # Ensure there are at least 5 data points for training
    X_train, Y_train, scaler = prepare_data_for_lstm(np.array(data_for_lstm).reshape(-1, 1))
    
    if X_train is None or Y_train is None:
        return []  # Return empty list if not enough data is available

    # Once there are at least 5 data points, create and train the LSTM model
    model = create_lstm_model(X_train)
    model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=0)

    # Predict the next 5 minutes
    predictions = []
    for _ in range(5):
        last_data = X_train[-1].reshape(1, X_train.shape[1], 1)
        predicted_price = model.predict(last_data)
        predictions.append(predicted_price[0][0])

        # Append the predicted price to the memory for the next prediction
        stock_data_memory.append({'close': predicted_price[0][0]})

    return predictions



# Scheduler function to run prediction every minute
def scheduled_task():
    """Task that runs every minute to fetch data and make predictions."""
    predictions = predict_next_5_minutes()
    print(f"Predictions for next 5 minutes: {predictions}")

# Setup APScheduler to run the task every minute
scheduler = BackgroundScheduler()
scheduler.add_job(func=scheduled_task, trigger="interval", minutes=1)
scheduler.start()

@app.route('/api/v1/predict_stock', methods=['GET'])
def get_stock_prediction():
    """API endpoint that predicts the stock price for the next 5 minutes."""
    predictions = predict_next_5_minutes()
    return jsonify({'predictions': predictions})

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
