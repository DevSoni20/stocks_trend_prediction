"""
Importing Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

"""Extracting data"""
start = '2010-01-01'
end = '2022-12-31'

df = yf.download('TATAMOTORS.NS', start, end)

"""Data Description"""

df.head()

df.tail()

df = df.reset_index()

df.head()

df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

plt.plot(df.Close)

ma100 = df.Close.rolling(100).mean()
ma100

ma200 = df.Close.rolling(200).mean()
ma200

plt.figure(figsize = (10,5))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

df.shape

training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

print(training_data)

print(testing_data.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(training_data)

standardized_data = scaler.transform(training_data)

print(standardized_data)

x_train = []
y_train = []

for i in range(100,standardized_data.shape[0]):
  x_train.append(standardized_data[i-100: i])
  y_train.append(standardized_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train.shape

"""ML Model"""

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))


model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))


model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))


model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))


model.add(Dense(units = 1))

model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

model.save('keras_model.h5')

past_100_days = training_data.tail(100)
final_testing_data = pd.concat([past_100_days, testing_data], ignore_index = True)

final_testing_data.head()

scaler.fit(final_testing_data)

input_data = scaler.transform(final_testing_data)

input_data

input_data.shape

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_test.shape)
print(y_test.shape)

"""Final Prdeiction"""

y_predicted = model.predict(x_test)

y_predicted.shape

y_predicted

y_test

scaler.scale_

scale_factor = 1/0.00215123
original_y_predicted = y_predicted * scale_factor
original_y_test = y_test * scale_factor

plt.figure(figsize = (12,6))
plt.plot(original_y_test, 'b', label = 'Original Price')
plt.plot(original_y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

df = pd.DataFrame(original_y_predicted, columns = ['predicted_close'])

df1 = pd.DataFrame(original_y_test, columns = ['tested_close'])

df['tomorrow'] = df['predicted_close'].shift(-1)
df1['tomorrow'] = df1['tested_close'].shift(-1)

df['target'] = (df['tomorrow'] > df['predicted_close']).astype(int)
df1['target'] = (df1['tomorrow'] > df1['tested_close']).astype(int)

print(df['target'])
print(df1['target'])

from sklearn.metrics import mean_squared_error
accuracy = mean_squared_error(y_test,y_predicted)
print(accuracy)