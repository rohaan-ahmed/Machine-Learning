# Recurrent Neural Networks

# 1. Data Pre-processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set= dataset_train.iloc[:,1:2].values      # Unsing only the Open Stock price

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating data structure with TS timesteps as input and 1 output
X_train = []
y_train = []
TS = 60
for i in range (TS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-TS:i, 0])  #Take last TS values until (i-1)th
    y_train.append(training_set_scaled[i, 0])       #Take i-th value
X_train = np.array(X_train)
y_train = np.array(y_train)
# The RNN will take the last TS values and predicts the next value

# Reshaping (for correct input shape into RNN)
num_indicators = 1              # Only using Open price as indicator. Could use more
batch_size = X_train.shape[0]
X_train = np.reshape(X_train, (batch_size, TS, num_indicators))

# 2. Building the RNN

## Load pre-saved model
#from keras.models import load_model 
#regressor = load_model('RNN_1')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# RNN Architecture
num_neurons_LSTM = 25
epochs = 50
batch_size = 32
# Initializing the  RNN
regressor = Sequential()
# Adding 4 LSTM and Dropout Layers
regressor.add(LSTM(units = num_neurons_LSTM, return_sequences = True, input_shape = (TS, num_indicators)))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = num_neurons_LSTM, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = num_neurons_LSTM, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = num_neurons_LSTM, return_sequences = False))
regressor.add(Dropout(rate = 0.2))
# Addint Output Layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting to the Training Set
history = regressor.fit(x = X_train, y = y_train, epochs = epochs, batch_size = batch_size)
regressor.save('RNN_1')

# 3. Prediction
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Compiling data for prediction
# The RNN uses the last TS values and predicts the next value
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
first_test_index = len(dataset_total) - len(dataset_test) - TS
inputs = dataset_total[first_test_index:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range (TS, TS + len(dataset_test)):
    X_test.append(inputs[i-TS:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_indicators))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualization
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('USD')
plt.legend()
plt.show()
