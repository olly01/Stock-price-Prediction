import os

import dash
import flask
from dash import html
from dash.dependencies import (Input, Output, State)
from keras.models import load_model

import application
from actions import (colors, fetch_data, fetch_company_data, save_and_download_graph, preprocess_data, build_model,
                     predict_data, inverse_transform, get_stock_info, generate_response,
                     calculate_historical_volatility)

# Set colors used for the webpage
primary1, primary2, primary3, secondary1, secondary2 = colors()

# Define the path to save and load the model
model_path = "LSTM_model.h5"

# External CSS styles
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Initialise Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define app layout with improved styling
app.layout = application.web_app


# Update company information
@app.callback(
    [
        Output("company-info", "children"),
        Output("chat-input-text", "value"),
        Output("rt-stock-info", "children"),
        Output("stock-info", "children")
    ],
    [Input("search-button", "n_clicks")],
    [
        State("symbol-input", "value")
    ]
)
def update_company(n_clicks, symbol_input):
    if n_clicks > 0:
        print("Action: Updating company information.")
        # Fetch information about the company
        company_data = fetch_company_data(symbol_input)

        # Summarise long summary
        print("Request: Generating a summary from OpenAI API - Summary")
        try:
            summary = generate_response("Summarise in a short paragraph: " + company_data["longBusinessSummary"])
            company_info = [html.P([html.H2(company_data["longName"]),
                                    html.P(summary)])]
            print("Request Accepted: Returning OpenAPI call results - Summary.")
        except:
            company_info = [html.P([html.H2(company_data["longName"]),
                                    html.P(company_data["longBusinessSummary"])])]
            print("Request Denied: OpenAI API request cannot be completed. Please check "
                  "the README.md file in the project directory for more information.")

        print("Action: Updating response in ChatBot input.")
        # Prompt for stock symbol
        promt = ("Evaluate important events that caused the stock price of " + company_data["longName"]
                 + " (" + symbol_input + ") to change.")

        print("Action: Updating real-time stock information.")
        # Get real-time stock information
        stock_info = get_stock_info(symbol_input)
        current_stock = html.Div(f"Real-time Stock Price of {stock_info['longName']}",
                                 style={"paddingTop": "5px",
                                        "paddingBottom": "5px",
                                        "background": primary2,
                                        "color": secondary1,
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "width": "80%",
                                        "margin": "auto",
                                        "font-weight": "bold"})

        stock_info_display = []
        if stock_info:
            styles = {"padding": "10px", "background": primary1, "borderRadius": "5px"}
            stock_info_display = [
                html.Div(f"Current Price: {stock_info['ask']:.2f}", style=styles),
                html.Div(f"Open: {stock_info['open']:.2f}", style=styles),
                html.Div(f"Low: {stock_info['dayLow']:.2f}", style={**styles, "color": "#ff0000"}),
                html.Div(f"High: {stock_info['dayHigh']:.2f}", style={**styles, "color": "#008000"})
            ]

        # Return updates
        if company_info:
            return company_info, promt, current_stock, stock_info_display
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Update the company info, plot, and real-time stock info based on user input
@app.callback(
    [
        Output("prediction-plot", "figure"),
        Output("download-link", "href"),
        Output("loss-plot", "figure"),
        Output("historical-volatility-plot", "figure")
    ],
    [Input("search-button", "n_clicks")],
    [
        State("symbol-input", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
        State("epoch-input", "value"),
        State("sequence-input", "value"),
        State("batch-input", "value")
    ]
)
def update_plot_and_info(n_clicks, symbol_input, start_date, end_date, epochs, sequence, batch):
    if n_clicks > 0:
        # Fetching and preprocessing data
        symbol = symbol_input.upper()
        stock_data = fetch_data(symbol, start_date, end_date)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
        # Check if the saved model exists, if not, build and train the model
        if not os.path.exists(model_path):
            model = build_model(sequence_length=sequence)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch)
            model.save(model_path)
            print("Action: Model saved successfully.")
        else:
            # Load the saved model
            model = load_model(model_path)
            print("Action: Model loaded successfully.")

        # Use the loaded model for predictions
        y_pred = predict_data(model, X_test)

        # Train the model and get training history
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch)
        # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

        # Get the predicted data and actual data
        y_pred = predict_data(model, X_test)
        y_test_actual = inverse_transform(scaler, y_test)
        y_pred_actual = inverse_transform(scaler, y_pred)
        date_range = stock_data.index[-len(y_test):]

        # Plotly graph for Dash
        figure = {
            "data": [
                {"x": stock_data.index, "y": stock_data["EMA"], "type": "line", "name": "Dataset"},
                {"x": date_range, "y": y_test_actual.flatten(), "type": "line", "name": "Actual Data"},
                {"x": date_range, "y": y_pred_actual.flatten(), "type": "line", "name": "Predicted EMA"}
            ],
            "layout": {
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Stock Price"},
                "title": f"{symbol} Stock Price Prediction",
                "legend": {"x": 0, "y": 1}
            }
        }

        # Plotly graph for loss
        loss_figure = {
            "data": [
                {
                    "x": list(range(1, epochs + 1)),
                    "y": history.history["val_loss"],
                    "type": "line",
                    "name": "Validation Loss"
                }
            ],
            "layout": {
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Loss", "range": [0, 0.1]},
                "title": "Validation Loss Over Time",
                "legend": {"x": 0, "y": 1}
            }
        }

        # Calculate historical volatility
        historical_volatility = calculate_historical_volatility(stock_data)

        # Plotly graph for historical volatility
        volatility_figure = {
            "data": [
                {"x": historical_volatility.index, "y": historical_volatility, "type": "line",
                 "name": "Historical Volatility"}
            ],
            "layout": {
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Volatility"},
                "title": f"{symbol} Historical Volatility",
                "legend": {"x": 0, "y": 1}
            }
        }

        # Save and download the prediction graph
        download_href = save_and_download_graph(symbol, stock_data, y_test_actual, y_pred_actual, date_range)

        # Return updates
        return figure, download_href, loss_figure, volatility_figure
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.server.route("/download/<path:path>")
def download(path):
    return flask.send_from_directory(".", path, as_attachment=True)


chat_log = []


# Used for ChatGPT ChatBox
@app.callback(
    Output("chat-output-text", "children"),
    [Input("chat-submit-button", "n_clicks")],
    [State("chat-input-text", "value")]
)
def user_chat(n_clicks, input_text):
    if n_clicks is not None:
        # Send user input to function to call API
        print("Request: Generating a response from OpenAI API - ChatBox")
        try:
            user_message = {"user": True, "text": input_text}
            chat_log.append(user_message)
            output = generate_response(input_text)
            bot_response = {"user": False, "text": output}
            chat_log.append(bot_response)
            print("Request Accepted: Returning OpenAI API call result - ChatBox.")
        except:
            output = ("Request Denied: OpenAI API request cannot be completed. Please check "
                      "the README.md file in the project directory for more information.")
            print(output)
            return output

    chat_log_elements = []
    for message in chat_log:
        className = "user-bubble" if message["user"] else "bot-bubble"
        chat_log_elements.append(html.Div(message["text"], className=className, style={
            'backgroundColor': '#dcf8c6',
            'padding': '10px',
            'borderRadius': '10px',
            'margin': '5px 5px 5px 5px',
            "paddingBottom": "20px",
            'maxWidth': '100%'
        } if message["user"] else {
            'backgroundColor': '#147efb',
            'color': 'white',
            'padding': '10px',
            'borderRadius': '10px',
            'margin': '5px 5px 5px 5px',
            "paddingBottom": "20px",
            'maxWidth': '100%'
        }))

    return chat_log_elements


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
