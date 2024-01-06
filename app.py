import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "TSLA", "FB", "NFLX", "DIS", "NVDA", "TATA", "PFE", "SNAP", "TATAMOTORS.NS", "TCS.NS", "TC", "GEO", "RELIANCE.NS")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)
st.subheader('Raw data')
st.write(data.info())
st.write(data.describe())
# st.write(data.head())

print(f'Dataframe contains stock prices between {data.Date.min()} {data.Date.max()}') 
print(f'Total days = {(data.Date.max() - data.Date.min()).days} days')


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

candlestick = go.Candlestick(x=data['Date'],
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'])

layout = {
    'title': f'{selected_stock} Candlestick Chart',
    'xaxis': {'title': 'Date'},
    'yaxis': {'title': 'Price'},
}

fig = go.Figure(data=[candlestick], layout=layout)
st.plotly_chart(fig)

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

from sklearn.metrics import mean_absolute_error, mean_squared_error

true_values = data[['Date', 'Close']][-period:]
forecast = forecast[-period:]

mae = mean_absolute_error(true_values['Close'], forecast['yhat'][-period:])
mse = mean_squared_error(true_values['Close'], forecast['yhat'][-period:])
rmse = np.sqrt(mse)

st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
st.write(f'Mean Squared Error (MSE): {mse:.2f}')
st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
