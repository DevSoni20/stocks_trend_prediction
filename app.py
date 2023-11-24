import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2022-12-31'

st.title('Stock Trend Predictions')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (10,5))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (10, 5))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (10, 5))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#Splitting Data into Training and Testing
training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(training_data)
standardized_data = scaler.transform(training_data)

#Splitting Data into x_train and y_train
x_train = []
y_train = []

for i in range(100,standardized_data.shape[0]):
  x_train.append(standardized_data[i-100: i])
  y_train.append(standardized_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

#Load my model
model = load_model('keras_model.h5')

past_100_days = training_data.tail(100)
final_testing_data = pd.concat([past_100_days, testing_data], ignore_index = True)
scaler.fit(final_testing_data)
input_data = scaler.transform(final_testing_data)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
original_y_predicted = y_predicted * scale_factor
original_y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (10,5))
plt.plot(original_y_test, 'b', label = 'Original Price')
plt.plot(original_y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)