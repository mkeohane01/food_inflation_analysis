import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import plotly.express as px
import plotly.graph_objs as go
# from your_model import predict_inflation 

# funciton to predict 6 months of inflation rates
def predict_inflation():
    # create dummy function to return df with index as month time series starting in 2023 and predicted rates
    future_data = pd.DataFrame()
    future_data['Date'] = pd.date_range(start='2023-01-01', periods=6, freq='MS')
    future_data['predicted_rates'] = np.random.rand(6)
    # return the df with index as Date and predicted rates as values
    return future_data

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
historical_data = cpi_food_data[(cpi_food_data['Date'] >= '1990-01-01') & (cpi_food_data['Date'] < '2023-01-01')]
# test data
test_data = cpi_food_data[(cpi_food_data['Date'] >= '2023-01-01')]
# close the connection
conn.close()

# Define a placeholder for the graph
graph_placeholder = st.empty()
# Create a Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Inflation'],
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
    future_data = predict_inflation()

    # Create a new Plotly graph
    updated_fig = go.Figure()

    # Add Historical Data Trace
    updated_fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Inflation'],
                                     mode='lines+markers', name='Historical Data'))

    # Add Test Data Trace
    updated_fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Inflation'],
                             mode='lines+markers', name='Test Data', line=dict(color='green')))

    # Add Predicted Data Trace
    updated_fig.add_trace(go.Scatter(x=future_data['Date'], y=future_data['predicted_rates'],
                             mode='lines+markers', name='Predicted Data', line=dict(color='red')))
    
    # Set graph layout
    updated_fig.update_layout(title='Food Inflation Rates', 
                    xaxis_title='Date', 
                    yaxis_title='Inflation Rate (CPI)')

    # show the graph
    graph_placeholder.plotly_chart(updated_fig, use_container_width=True)