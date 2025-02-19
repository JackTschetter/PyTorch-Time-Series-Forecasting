# Time Series Forecasting Using Python/PyTorch

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)

## Table of Contents

- [About the Project](#about-the-project)
  - [Background on Time Series Forecasting](#background-on-time-series-forecasting)
- [Algorithms Studied](#algorithms-studied)
- [Tools Used](#tools-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Results and Evaluation](#results-and-evaluation)
  - [Evaluating the Models](#evaluating-the-models)
  - [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About the Project

This repository contains code to implement various machine learning algorithms for time series forecasting at scale. Various algorithms were implemented in our Python code frame, and compared. 

### Background on Time Series Forecasting

---

## Models Studied

- Long Short Term Memory (LSTM)
- Prophet/Multi-Prophet (Meta)
- Temporal Convolutional Neural Network (TCN)
- Long Short Term Temporal Patterns (LSTNet)
- Informer Based Model
- Univariate Divide and Conquer Long Short Term Memory (UnivarDIVIDE)

---

## Tools Used

---

## Getting Started

### Prerequisites

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/JackTschetter/Predictions
   cd Predictions

## Usage

## Results and Model Evaluation

### Evaluating the Models

### Results

## Contact

Contact me anytime! Day or night! My email is jackrtschetter@gmail.com and my phone number is 612-380-1832. Sending me a brief text introducing yourself is the best way to reach me. I will respond ASAP.









### **How to Run the Stock Prediction Program**

#### **Overview**
This stock prediction program consists of three main components:
1. **`fetch_data.py`**: Fetches stock data for a specified stock symbol from the Polygon.io API and saves it to a CSV file.
2. **`stock_server.py`**: A Flask server that provides stock data for a specific time from the past (one month ago).
3. **`predict_stock.py`**: Another Flask server that uses a local API to fetch stock data, stores it in memory, and predicts the next 5 minutes of stock data using an LSTM model.

The setup involves fetching stock data, storing it, and using it for predictions. Here’s how to get everything running.

#### **Prerequisites**
- **Python 3.x** installed.
- Install necessary Python packages via pip:
    ```bash
    pip install requests pandas tensorflow apscheduler flask flask-cors scikit-learn
    ```
- **Create a `.env` file** in your project directory to store your Polygon.io API key:
    ```text
    POLYGON_API_KEY=your_polygon_api_key_here
    ```

#### **Steps to Run the Program**

1. **Fetch Stock Data (`fetch_data.py`)**:
   - **Purpose**: Fetch historical stock data for a specific stock symbol (e.g., TSLA) from the Polygon.io API for the past 30 days.
   - **How to Run**:
     - Modify the `STOCK_SYMBOL` in `fetch_data.py` to the stock symbol you want to track.
     - Run the script:
       ```bash
       python fetch_data.py
       ```
     - This will save the stock data in a CSV file (`TSLA_minute_data.csv`).

2. **Start the Stock Server (`stock_server.py`)**:
   - **Purpose**: A Flask server that provides stock data for a specific time (one month ago). It will use the saved CSV file (`TSLA_minute_data.csv`) for data.
   - **How to Run**:
     - Ensure `CSV_FILE_PATH` points to your saved `TSLA_minute_data.csv`.
     - Run the server:
       ```bash
       python stock_server.py
       ```
     - The server will start and listen on `http://localhost:5001`. You can access the stock data for the same minute, one month ago by navigating to:
       ```bash
       http://localhost:5001/api/v1/stock_data
       ```

3. **Start the Prediction Server (`predict_stock.py`)**:
   - **Purpose**: A Flask server that predicts the next 5 minutes of stock data based on the most recent stock data fetched from the `stock_server.py` API. The predictions are made using an LSTM model.
   - **How to Run**:
     - Run the server:
       ```bash
       python predict_stock.py
       ```
     - The server will start and listen on `http://localhost:5002`. Predictions will be made every minute.
     - To manually trigger a prediction, access the following endpoint:
       ```bash
       http://localhost:5002/api/v1/predict_stock
       ```

     The model will use the most recent data, train on it, and predict the next 5 minutes.

---

#### **Expected Outputs**

- After running `fetch_data.py`, a CSV file (`TSLA_minute_data.csv`) will be generated containing the historical stock data.
- **Stock Data** (from `stock_server.py`):
   - When accessing `http://localhost:5001/api/v1/stock_data`, you should see stock data for the same time one month ago:
     ```json
     [
       {
         "timestamp": "Tue, 21 Jan 2025 14:05:00 GMT",
         "open": 434.6,
         "high": 435.1299,
         "low": 434.48,
         "close": 435.08,
         "volume": 12837.0
       }
     ]
     ```
- **Predictions** (from `predict_stock.py`):
   - When accessing `http://localhost:5002/api/v1/predict_stock`, the response will include predictions for the next 5 minutes:
     ```json
     {
       "predictions": [
         435.12,
         435.14,
         435.13,
         435.15,
         435.18
       ]
     }
     ```

---

#### **Important Notes**
- Ensure the server is running continuously to make predictions. The background scheduler in `predict_stock.py` will run the predictions every minute.
- If the data in the CSV file is insufficient (less than 5 data points), the model will wait until more data is accumulated.
- The prediction model is updated every minute with the latest data.

