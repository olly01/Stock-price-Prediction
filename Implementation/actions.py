import io
import os

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler

# Set configs
dark_mode = False

try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("Request Accepted: OpenAI API key accepted.")
except:
    print("Error: You currently do not have an OpenAI API key set. Please check "
          "the README.md file in the project directory for more information.")


# Set colors used for the webpage
def colors():
    if not dark_mode:
        # Colour palette 463f3a-8a817c-bcb8b1-f4f3ee-e0afa0
        return "#bcb8b1", "#8a817c", "#463f3a", "#f4f3ee", "#e0afa0"
    else:
        # NOT DONE
        return "#bcb8b1", "#8a817c", "#463f3a", "#f4f3ee", "#e0afa0"


# Fetch stock data
def fetch_data(symbol, start_date="2015-01-01", end_date="2023-01-01"):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data.ffill()
    stock_data["EMA"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
    return stock_data


# Fetch company data
def fetch_company_data(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return info


# Fetch stock information
def get_stock_info(symbol):
    try:
        stock_info = yf.Ticker(symbol).info
        return stock_info
    except:
        return None


# Create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)


def build_model(sequence_length):
    model = Sequential()

    # First Layer
    model.add(LSTM(200, activation="tanh", input_shape=(sequence_length, 1), return_sequences=True))
    model.add(Dropout(0.2))

    # Second Layer
    model.add(LSTM(200, activation="tanh", input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))

    # Dense Layer
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=['mean_absolute_error'])
    return model


# Preprocess data by scaling, splitting and sequencing
def preprocess_data(stock_data, sequence_length=25):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data["EMA"].values.reshape(-1, 1))
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train = create_sequences(train_data, sequence_length)
    X_test = create_sequences(test_data, sequence_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = train_data[sequence_length:]
    y_test = test_data[sequence_length:]

    return X_train, X_test, y_train, y_test, scaler


# Predict data
def predict_data(model, X_test):
    return model.predict(X_test)


# Inverse transform scaled data
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)


# Call to the OpenAI API to generate a response
def generate_response(request):
    context = ("You are a stock analyst assistant. You provide detailed stock information including current prices, "
               "historical data, future predictions based on market trends, and comparisons between different stocks. "
               "You also offer insights into market news and events that could impact stock performance.")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.4,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": request}
        ],
    )
    return completion.choices[0].message.content


# Calculate historical volatility
def calculate_historical_volatility(stock_data, window=252):
    # Calculate daily returns
    daily_returns = stock_data['Close'].pct_change()

    # Calculate historical volatility
    historical_volatility = daily_returns.rolling(window=window).std() * np.sqrt(window)

    return historical_volatility


# Function to save and download the prediction graph
def save_and_download_graph(symbol, stock_data, y_test_actual, y_pred_actual, date_range):
    img_data = io.BytesIO()
    plt.figure(figsize=(16, 9))
    plt.plot(stock_data.index, stock_data["EMA"], label="Dataset")
    plt.plot(date_range, y_test_actual.flatten(), label="Actual Data")
    plt.plot(date_range, y_pred_actual.flatten(), label="Predicted EMA")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{symbol} Stock Price Prediction")
    plt.legend()
    file_name = f"{symbol}_prediction_graph.png"
    plt.savefig(file_name)
    plt.close()

    # Return the download path
    return f"/download/{file_name}"
