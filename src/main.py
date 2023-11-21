import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from get_prediction import get_prediction_train, get_prediction_test_future
from pathlib import Path
import os
# from your_model import predict_inflation 

# funciton to predict 6 months of inflation rates
def predict_inflation():
    # params to get prediction
    n_steps_in = 24
    n_steps_out = 6
    root_path = Path(os.getcwd())
    data_path = root_path / 'data' / 'merged_data.csv'
    ckpt_path = root_path / 'src' / 'model_cache' / 'LSTM.h5'
    
    #  6(n_steps_out) length list 
    train_pred = get_prediction_train(n_steps_in, n_steps_out, data_path, ckpt_path)
    test_pred, future_pred = get_prediction_test_future(n_steps_in, n_steps_out, data_path, ckpt_path)
    
    train_pred = [float(i) for i in train_pred]
    print(len(train_pred))
    # create df with index as month time series starting in 2023-10 and future predicted rates
    future_data = pd.DataFrame()
    future_data['Date'] = pd.date_range(start='2023-10-01', periods=6, freq='MS')
    future_data['predicted_rates'] = future_pred

    # create df with index as month time series starting in 2023-4 and test predicted rates
    pred_data = pd.DataFrame()
    pred_data['Date'] = pd.date_range(start='2023-04-01', periods=6, freq='MS')
    pred_data['predicted_rates'] = test_pred
    # create df for predicted on train data
    train_len = len(train_pred)
    train_pred = pd.DataFrame()
    train_pred['Date'] = pd.date_range(start='1993-01-01', periods=train_len, freq='MS')
    train_pred['predicted_rates'] = train_len
    return pred_data, future_data, train_pred

# Set the page config to wide mode with a title
st.set_page_config(layout="wide")

# Set the title
st.title('Food Inflation Prediction')

# Load data from food_inflation_analysis.db database
# connect to the sqlite database
conn = sqlite3.connect('food_inflation_analysis.db')
# get the data from the database
# historical data
query = '''SELECT * FROM OCED_USA_FOOD_INFLATION'''
cpi_food_data = pd.read_sql(query, conn)
cpi_food_data['Date'] = pd.to_datetime(cpi_food_data['Date'])
# sort by Date column
cpi_food_data.sort_values(by='Date', inplace=True)
# only keep data after Date 1990
historical_data = cpi_food_data[(cpi_food_data['Date'] >= '1990-01-01') & (cpi_food_data['Date'] < '2023-04-01')]
# test data
test_data = cpi_food_data[(cpi_food_data['Date'] >= '2023-04-01')]
# close the connection
conn.close()

# Define a placeholder for the graph
graph_placeholder = st.empty()
# Create a Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['CPI'],
                         mode='lines+markers', name='Historical Data'))

# Set graph layout
fig.update_layout(title='Food Inflation Rates', 
                  xaxis_title='Date', 
                  yaxis_title='Inflation Rate (CPI)')

# Create options for users to select prediciton horizon
# prediction_range = st.selectbox('Select Prediction Range (in months)', [1, 2, 3, 6, 12])

# show the graph
graph_placeholder.plotly_chart(fig, use_container_width=True)

# Create a button to predict the inflation rates
if st.button('Predict Food Inflation Rates'):
    pred_data, future_data, train_pred = predict_inflation()

    # Create a new Plotly graph
    updated_fig = go.Figure()

    # Add Historical Data Trace
    updated_fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['CPI'],
                                     mode='lines+markers', name='Historical Data'))

    # Add Test Data Trace
    updated_fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['CPI'],
                             mode='lines+markers', name='Test Data', line=dict(color='green')))

    # Add Predicted Test Data Trace
    updated_fig.add_trace(go.Scatter(x=pred_data['Date'], y=pred_data['predicted_rates'],
                             mode='lines+markers', name='Predicted Data - Test', line=dict(color='red')))

    # Add predicted future data trace
    updated_fig.add_trace(go.Scatter(x=future_data['Date'], y=future_data['predicted_rates'],
                             mode='lines+markers', name='Predicted Data - Future', line=dict(color='orange')))
    
    # Add predicted test data trace
    updated_fig.add_trace(go.Scatter(x=train_pred['Date'], y=train_pred['predicted_rates'],
                             mode='lines+markers', name='Predicted Data - Train', line=dict(color='purple')))
    

    # calculate the mse
    mse_test = mean_squared_error(test_data['CPI'], pred_data['predicted_rates'])
    train_dat = historical_data[(historical_data['Date'] >= train_pred['Date'].min()) & (historical_data['Date'] <= train_pred['Date'].max())]
    mse_train = mean_squared_error(train_dat['CPI'], train_pred['predicted_rates'])

    # calculate the MAPE
    mape_test = mean_absolute_percentage_error(test_data['CPI'], pred_data['predicted_rates']) * 100
    mape_train = mean_absolute_percentage_error(train_dat['CPI'], train_pred['predicted_rates']) * 100

    # Set graph layout
    updated_fig.update_layout(title='Food Inflation Rates', 
                    xaxis_title='Date', 
                    yaxis_title='Inflation Rate (CPI)')

    # Display MSE
    st.text(f'Mean Squared Error: Test = {mse_test:.2f} | Train = {mse_train:.2f}')
    # Display MAPE
    st.text(f'Mean Absolute Percentage Error: Test = {mape_test:.2f}% | Train = {mape_train:.2f}%')

    # show the graph
    graph_placeholder.plotly_chart(updated_fig, use_container_width=True)