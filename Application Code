# Minor-Project
Prediction Of Volatility In Currency Market using Machine Learning
from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sqlite3
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def train_lstm_model(stock_symbol):
    data = yf.download(stock_symbol, period='5y', interval='1d')
    data['Date'] = data.index
    data['Date'] = data['Date'].map(lambda x: x.toordinal())
    X = np.array(data['Date']).reshape(-1, 1)
    y = np.array(data['Close'])
    
    X = X.reshape((X.shape[0], 1, 1))  # Reshape for LSTM
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)
    
    return model, data

@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('symbol', 'AAPL')
    model, data = train_lstm_model(stock_symbol)
    
    future_dates = [pd.Timestamp.today() + pd.DateOffset(days=i) for i in range(1, 31)]
    future_ordinals = np.array([[date.toordinal()] for date in future_dates])
    future_ordinals = future_ordinals.reshape((future_ordinals.shape[0], 1, 1))
    predictions = model.predict(future_ordinals)
    
    response = {"symbol": stock_symbol, "predictions": {}}
    for i, date in enumerate(future_dates):
        response["predictions"][date.strftime('%Y-%m-%d')] = float(predictions[i])
    
    save_to_db(stock_symbol, response["predictions"])
    return jsonify(response)

def save_to_db(stock_symbol, predictions):
    conn = sqlite3.connect('stocks.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS predictions (symbol TEXT, date TEXT, price REAL)")
    for date, price in predictions.items():
        cursor.execute("INSERT INTO predictions (symbol, date, price) VALUES (?, ?, ?)", (stock_symbol, date, price))
    conn.commit()
    conn.close()

@app.route('/history', methods=['GET'])
def history():
    stock_symbol = request.args.get('symbol', 'AAPL')
    data = yf.download(stock_symbol, period='5y', interval='1d')
    historical_prices = data['Close'].to_dict()
    return jsonify({"symbol": stock_symbol, "historical_prices": historical_prices})

if __name__ == '__main__':
    app.run(debug=True)
