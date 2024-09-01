# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

# ------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before.
# If so, load the saved data
# If not, save the data into a directory
# ------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"


def load_and_process_data(company, start_date, end_date, price_value, split_method, split_date, train_size, scale_features, save_data,
                          load_data):
    """
    Docstring:
    Function loads and processes stock data from Yahoo Finance.
    Takes various arguments to define the data and processing steps.
    Returns a tuple containing the processed data.

    Args:
        company (str): Ticker symbol of the company (e.g., "AAPL").
        start_date (str): Start date for data retrieval (YYYY-MM-DD format).
        end_date (str): End date for data retrieval (YYYY-MM-DD format).
        price_value (str): Specific price value to retrieve ("Open", "High", "Low", "Close", "Adj Close", "Volume").
        split_method (str): Method for splitting data ("ratio", "date", or "random").
        split_date (str): Date for splitting data using the "date" method (YYYY-MM-DD format).
        train_size (float): Proportion of data for training, used while splitting using "ratio" (between 0 and 1).
        scale_features (bool): Whether to scale features before training.
        save_data (bool): Whether to save the processed data locally.
        load_data (bool): Whether to load the data from a local file (if it exists).

    Returns:
        tuple: A tuple containing the scaled training data, raw split testing data, scaler object
    """

    # Defining a filename with high specificity, to avoid confusion with data with a different time frame
    data_file = f"storeddata/{company}_{start_date}_{end_date}_data.csv"

    # Checking if loading data locally
    if (load_data):
        if os.path.exists(data_file):
            # Load data from local file, using Date as the index column
            data = pd.read_csv(data_file, index_col="Date")
        else:
            raise ValueError("Invalid data file.")
    else:
        # Download data from Yahoo Finance
        data = yf.download(company, start_date, end_date)
        # Saving data to defined filename if requested
        if save_data:
            data.to_csv(data_file)

    # Handle NaN values using pandas to drop any rows with missing values.
    # Could also use fillna to fill with dummy data if this introduces issues
    data = data.dropna()

    # Split data based on defined method
    if split_method == "random":
        # Randomly split data using sklearn. Shuffles data before splitting to ensure randomness
        train_data, test_data = train_test_split(data, train_size=train_size, shuffle=True)
    elif split_method == "date":
        # Split data based on a specific date - split_date
        # Search data for given date, split before and after this date
        split_index = data.index.get_loc(split_date)
        train_data = data[:split_index]
        test_data = data[split_index:]
    elif split_method == "ratio":
        # Similar to date splitting, but using an arbitary ratio applied to the whole dataset - train_data
        split_index = int(train_size * len(data))
        train_data = data.iloc[:split_index, :]
        test_data = data.iloc[split_index:, :]
    else:
        raise ValueError("Invalid split method.")

    # Defining the scaler outside
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Note that, by default, feature_range=(0, 1). Thus, if you want a different
    # feature_range (min,max) then you'll need to specify it here

    # Scaling the data (or not scaling the data!)
    if scale_features:
        train_data_scaled = scaler.fit_transform(train_data[price_value].values.reshape(-1, 1))
    else:
        train_data_scaled = train_data

    # Returning scaled training data, split test data, scaler object
    return train_data_scaled, test_data, scaler, data

def candlestick_display(data, n_days=1, title="Stock Prices"):
    """
      Function to display financial data using candlestick charts.

      Args:
          data (pandas.DataFrame): DataFrame containing OHLC (Open, High, Low, Close) data.
          n_days (int, optional): Number of days to include in each candlestick. Defaults to 1.
          title (str, optional): Title for the chart. Defaults to "Stock Prices".
      """

    # Calculate OHLC values based on n_days aggregation
    # Using resample instead of rolling for better compatibility
    ohlc = data[['Open', 'High', 'Low', 'Close']].resample(f"{n_days}D").agg({'Open': 'first',
                                                                              'High': 'max',
                                                                              'Low': 'min',
                                                                              'Close': 'last'})

    # Convert data to format expected by candlestick_ohlc
    # Candlestick needs numerical timestamps generated by matplotlib (mpldates)
    dates = ohlc.index.to_pydatetime()
    timestamps =  mpldates.date2num(dates)
    # Generating list - each tuple represents a single candlestick and its elements
    candles = list(zip(timestamps, ohlc['Open'], ohlc['High'], ohlc['Low'], ohlc['Close']))

    # Create candlestick chart using mpl_finance
    fig, ax = plt.subplots(figsize=(12, 6))
    candlestick_ohlc(ax, candles, width=n_days, colorup='green', colordown='red')
    fig.autofmt_xdate()
    plt.title(title)
    plt.ylabel('Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def boxplot_display(data, n_days=1, title="Stock Prices"):
    """
      Function to display financial data using boxplot charts.

      Args:
          data (pandas.DataFrame): DataFrame containing OHLC (Open, High, Low, Close) data.
          n_days (int, optional): Number of days to represent in each boxplot. Defaults to 1.
          title (str, optional): Title for the plot. Defaults to "Stock Prices".
      """
    # Calculate OHLC values for each n-day window
    # First and last are not valid agg functions for rolling windows - using lambda grabbed by iloc as workaround
    ohlc = data[['Open', 'High', 'Low', 'Close']].rolling(window=n_days).agg({'Open': lambda x: x.iloc[0],
                                                                           'High': 'max',
                                                                           'Low': 'min',
                                                                           'Close': lambda x: x.iloc[-1]})

    # Create boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(ohlc[['Open', 'High', 'Low', 'Close']].dropna().values.T)
    plt.xticks(range(1, 5), ['Open', 'High', 'Low', 'Close'])
    plt.title(title)
    plt.ylabel('Price')
    plt.show()

# Assigning variables externally that are used later
COMPANY = 'CBA.AX'
PRICE_VALUE = "Close"

# Call function with args
train_data_scaled, test_data, scaler, data = load_and_process_data(COMPANY,'2020-01-01', '2024-07-02', PRICE_VALUE, "date", '2023-08-02', 0.8, True, True, False)

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60  # Original

# To store the training data
x_train = []
y_train = []

scaled_data = train_data_scaled[:, 0]  # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x - PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train
# is an array of p inputs with each input being a 2D array

# ------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before.
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
# ------------------------------------------------------------------------------
model = Sequential()  # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer.
# That's why we specify the input shape for this layer;
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True
# when stacking LSTM layers so that the next LSTM layer has a
# three-dimensional sequence input.

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of
# rate (= 0.2 above) at each step during training time, which helps
# prevent overfitting (one of the major problems of ML).

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.

# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to
# Lecture Week 6 (COS30018): If you update your model for each and every
# input sample, then there are potentially 2 issues: 1. If you training
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

# ------------------------------------------------------------------------------
# Test the model accuracy on existing data
# ------------------------------------------------------------------------------

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we
# can use part of it for training and the rest for testing. You need to
# implement such a way

# ------------------------------------------------------------------------------
# Make predictions on test data
# ------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation
# ------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
# ------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

candlestick_display(data, 10)
boxplot_display(data, 10)

# ------------------------------------------------------------------------------
# Predict next day
# ------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??