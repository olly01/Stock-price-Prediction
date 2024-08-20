import warnings
import os
import dash
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set the Matplotlib to non-interactive
plt.switch_backend('Agg')



from actions import create_sequences, generate_response, save_and_download_graph
from main import update_company, update_plot_and_info, user_chat

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sre_constants")

# update_company()
output_update_company = update_company(1, "NVDA")

# List - of elements used in "company-info"
def test_update_company_company_info():
    print(output_update_company[0])
    assert type(output_update_company[0]) is list

# String - text promt used in "chat-input-text"
def test_update_company_chat_input_text():
    print(output_update_company[1])
    assert type(output_update_company[1]) is str

# Div - element used in "rt-stock-info"
def test_update_company_rt_stock_info():
    print(output_update_company[2])
    assert type(output_update_company[2]) is dash.html.Div

# List - of elements used in "stock_info"
def test_update_company_stock_info():
    print(output_update_company[3])
    assert type(output_update_company[3]) is list


# update_plot_and_info()
output_update_plot_and_info = update_plot_and_info(1, "NVDA", "2015-01-01", "2024-01-01", 2, 25, 32)

# Dictionary - for plot used in "prediction-plot"
def test_update_plot_and_info_prediction_plot():
    print(output_update_plot_and_info[0])
    assert type(output_update_plot_and_info[0]) is dict

# String - for link used in "download-link"
def test_update_plot_and_info_download_link():
    print(output_update_plot_and_info[1])
    assert type(output_update_plot_and_info[1]) is str

# Dictionary - for plot used in "loss-plot"
def test_update_plot_and_info_loss_plot():
    print(output_update_plot_and_info[2])
    assert type(output_update_plot_and_info[2]) is dict

# Dictionary - for plot used in "historical-volatility-plot"
def test_update_plot_and_info_historical_volatility_plot():
    print(output_update_plot_and_info[3])
    assert type(output_update_plot_and_info[3]) is dict


# user_chat()
output_user_chat = user_chat(1,
                             "Evaluate important events that caused the stock price of Nvidia (NVDA) to change.")

# List - of elements for chat log used in "chat-output-text"
def test_user_chat_output():
    print(output_user_chat)
    assert type(output_user_chat) is list

# Check model saving and loading
def test_update_plot_and_info_model_saving_and_loading():
    model_path = "LSTM_model.h5"
    # Trigger model training and saving
    output = update_plot_and_info(1, "NVDA", "2015-01-01", "2024-01-01", 2, 25, 32)

    # Check if the model file exists
    assert os.path.exists(model_path), "Model file does not exist after saving."

    # Load the saved model
    loaded_model = None
    try:
        loaded_model = load_model(model_path)
    except Exception as e:
        assert False, f"Error loading model: {str(e)}"

    # Assert that the loaded model is not None, meaning successful loading
    assert loaded_model is not None, "Loaded model is None, failed to load the model."

# Test create_sequences model
def test_create_sequences_length():
    # Generate mock data for testing
    mock_data = np.arange(100)
    sequence_length = 10

    # Call create_sequences function
    sequences = create_sequences(mock_data, sequence_length)

    # Check if sequences have the correct length
    assert all(len(seq) == sequence_length for seq in sequences)

# Test the save_and_download_graph to see functionality
def test_download_graph():
    # Mock data for testing
    symbol = "AAPL"
    stock_data = pd.DataFrame({"EMA": np.random.rand(100)})
    y_test_actual = np.random.rand(100)
    y_pred_actual = np.random.rand(100)
    date_range = pd.date_range(start="2022-01-01", periods=100)

    # Call to save_and_download_graph function
    download_path = save_and_download_graph(symbol, stock_data, y_test_actual, y_pred_actual, date_range)

    # Check if the file exists at the download path
    if os.path.exists(download_path):
        print(f"File '{download_path}' exists. Download successful.")
    else:
        print(f"File '{download_path}' does not exist. Download failed.")

# Test functionality of the generate_response function
def test_generate_response():
    # Call to generate_response function with a sample request
    request = "What is the current stock price of AAPL?"
    response = generate_response(request)

    # Check if the response is a string
    assert isinstance(response, str)

    # Check if there is a response
    assert response.strip() != ""
