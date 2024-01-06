import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import plotly as py
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import 

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "TSLA", "FB", "NFLX", "DIS", "NVDA", "TATA", "PFE", "SNAP", "TATAMOTORS.NS", "TCS.NS", "TC", "GEO", "RELIANCE.NS", "TATASTEEL.NS", "RVNL.NS", "TATACONSUM.NS", "TATACHEM.NS", "TATAMTRDVR.NS", "MANKIND.NS", "SUNPHARAMA.NS")
selected_stocks = st.multiselect("Select stocks for prediction", stocks, ["TSLA"])

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data_dict = {stock: load_data(stock) for stock in selected_stocks}
data_load_state.text("Loading data... done!")

for stock, data in data_dict.items():
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    

def plot_raw_data():
    for stock, data in data_dict.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name=f'{stock}_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name=f'{stock}_close'))
        fig.layout.update(title_text=f"Time Series Data for {stock}", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

plot_raw_data()

def evaluate_model(stock, data):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, mse_scores, r2_scores = [], [], []
    
    for train_index, test_index in tscv.split(df_train):
        train_data, test_data = df_train.iloc[train_index], df_train.iloc[test_index]
        
        m = Prophet()
        m.fit(train_data)
        
        future = m.make_future_dataframe(periods=len(test_data))
        forecast = m.predict(future)
        
        y_true = test_data['y'].values
        y_pred = forecast['yhat'][-len(test_data):].values
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 =  r2_score(y_true, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)
    
    return np.mean(mae_scores), np.mean(mse_scores), np.mean(r2_scores)


selected_stock = st.selectbox("Select stock for visualization", selected_stocks)


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
st.write(forecast.head())

# st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], mode='lines', name='Volume', line=dict(color='teal')))
fig3.update_xaxes(title_text='Date')
fig3.update_yaxes(title_text='Trading Volume')
fig3.update_layout(title_text='Trading Volume Over Time')

st.plotly_chart(fig3)

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=data['Volume'], y=data['Close'], mode='markers', marker=dict(size=5, opacity=0.5, color='purple')))

fig4.update_layout(scene=dict(xaxis_title='Volume', yaxis_title='Closing Price', zaxis_title='Date'), title='Volume vs. Closing Price')
fig4.update_xaxes(title_text='Volume')
fig4.update_yaxes(title_text='Closing Price')
fig4.update_layout(title_text='Volume vs. Closing Price')
st.plotly_chart(fig4)

evaluation_results = {}
for stock, data in data_dict.items():
    mae, mse, r2 = evaluate_model(stock, data)
    evaluation_results[stock] = {
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'R-squared Error (R2)': r2
    }
