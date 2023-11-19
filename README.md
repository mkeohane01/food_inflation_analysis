# Food Inflation Predictor
Project using Deep Learning to predict changing Food Inflation prices. We use CPI as our target.

**To run:**
```bash
pip install -r requirements.txt
streamlit run src/streamlit_main.py
```

## Data Procesing
Processed various data sets to use as features for our prediction model in the form of a time series dataframe.

Processing steps:

1. 

2.

3.

4.

 

## Model Structure


## Repo Management
- **data/**
    - Contains the data files used for feature creation
- **notebooks/**
    - Contains the various notebooks we used for data understanding and processing. We saved the processed data in to tables in the database from these notebooks.
- **src/**
    - Contains the executable files to load and combine the data from database, run the model, and start streamlit server.
- **food_inflation_analysis.db**
    - The sqlite database used to store the processed tables. The tables included: OCED_USA_FOOD_INFLATION; DOW_JONES_REAL; USA_MEAT_EXPORT_IMPORT; food_production; interest_rate; gas_prices.