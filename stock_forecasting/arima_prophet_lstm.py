import pandas as pd
from preprocess import load_and_clean_data
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load data
df = load_and_clean_data("stock_data.csv")

# ARIMA
def run_arima(df):
    model = ARIMA(df, order=(5, 1, 0))
    results = model.fit()
    forecast = results.forecast(steps=30)
    forecast.plot(title="ARIMA Forecast")
    plt.show()

# Prophet
def run_prophet(df):
    prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    model.plot(forecast)
    plt.show()

# LSTM
def run_lstm(df):
    data = df.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)
    preds = model.predict(X[-30:])
    plt.plot(preds, label="LSTM forecast")
    plt.legend()
    plt.show()

# Run all
run_arima(df)
run_prophet(df)
run_lstm(df)
