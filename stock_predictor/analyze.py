import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator 
from qiskit.circuit.library import QuantumVolume
from datetime import datetime, timedelta
from data_model import *

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    # Use closing price for prediction
    close_prices = data['Close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create training and test datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        # for i in range(time_step, len(dataset)):
        #     X.append(dataset[i-time_step:i, 0])
        #     y.append(dataset[i, 0])
        for i in range(len(dataset) - time_step - 1):
            X.append(data.iloc[i:(i + time_step), 0])
            y.append(data.iloc[i + time_step, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, scaler

def predict_stock_price(model, X_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    predictions = predictions.numpy()
    predictions = scaler.inverse_transform(predictions)
    return predictions

def plot_predictions(real_data, predicted_data):
    plt.figure(figsize=(10,6))
    plt.plot(real_data, color='blue', label='Real Stock Price')
    plt.plot(predicted_data, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def quantum_inspired_stock_predictor(data):
    backend = StatevectorSimulator()
    qv = QuantumVolume(4)
    job = transpile(qv, backend)
    job.draw(output='mpl')
    # print("Quantum Feature Extraction Result:", result.get_statevector())

def main():
    ticker = "AAPL"
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*4)
    
    stock_data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    X_train, y_train, X_test, y_test, scaler = preprocess_data(stock_data)
    
    model = build_and_train_lstm_model(X_train, y_train, X_test, y_test)
    
    predictions = predict_stock_price(model, X_test, scaler)
    
    real_stock_price = scaler.inverse_transform(stock_data['Close'][len(stock_data) - len(predictions):].values.reshape(-1, 1))
    plot_predictions(real_stock_price, predictions)
    
    quantum_inspired_stock_predictor(stock_data)

if __name__ == "__main__":
    main()
