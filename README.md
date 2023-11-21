# Food Inflation Predictor
Project using Deep Learning to predict changing Food Inflation prices. We use United States food CPI sourced from OECD as our target.

**To run:**
```bash
pip install -r requirements.txt
streamlit run src/main.py
```

## Data Procesing
Processed various data sets to use as features for our prediction model in the form of a time series dataframe.

Processing steps:

1. Load each data set in timeseries format per month or year. Some data was webscraped, pulled from API, or downloaded from csv.

2. Fill in any missing value such as the gas_prices data by extrapolating surrounding data.

3. Compare correlations between lagged data and the target, food CPI.

4. Save the features that had the most impactful correlations to the database.


## Model Structure

Adopted LSTM to coduct Multivariate Multistep time series prediction.

Pipeline:

1. Split the time-series data to train/test set.

2. Define optimization objective as average MSE of 5-fold time-series cross-validation.

3. Run 200 Optuna trials to find the optimal hyperparameters of LSTM by minimizing the objective.

4. Use optimal hyperparameters and full training set to retrain the model and save the final model checkpoint.


## Repo Management
- **data/**
    - Contains the data files used for feature creation
- **notebooks/**
    - Contains the various notebooks we used for data understanding and processing. We saved the processed data in to tables in the database from these notebooks.
- **src/**
    - Contains the executable files to load and combine the data from database, run the model, and start streamlit server.
- **food_inflation_analysis.db**
    - The sqlite database used to store the processed tables. The tables included: OCED_USA_FOOD_INFLATION; DOW_JONES_REAL; USA_MEAT_EXPORT_IMPORT; food_production; interest_rate; gas_prices.