
# multivariate multi-step stacked lstm example
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
import logging
#from tensorflow.keras.layers import LSTM
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import datetime
import optuna
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')
import keras

tf.random.set_seed(42)
np.random.seed(42)

from utils import *

def objective_tscv(trial, train):
    # Define the hyperparameters to be tuned
    hidden1 = trial.suggest_int('hidden1', 100, 200)
    hidden2 = trial.suggest_int('hidden2', 50, 100)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    
    # input 24 months and predict 6 months
    tscv = TimeSeriesSplit(n_splits = 4, test_size = n_steps_in + n_steps_out)
    mse = []
    
    n_features = train[:, 1:].shape[1]
    for fold, (train_index, val_index) in enumerate(tscv.split(train)):
        
        # split train dataset into train/validation set
        train_split, val_split = train[train_index], train[val_index]

        # split the train validation set into sequences
        X_train, y_train = split_sequences(train_split, n_steps_in, n_steps_out)
 
        X_val, y_val = split_sequences(val_split, n_steps_in, n_steps_out)
        
        # reshape X_val into one sample format
        X_val = X_val.reshape((1, n_steps_in, n_features))
        
        # initiate LSTM model
        model = LSTM_config(n_features, hidden1, hidden2, activation, learning_rate, dropout_rate)
        
        # compute  mse for each fold
        
        # print(X_train.shape, y_train.shape)
        model.fit(X_train, y_train, epochs = EPOCHS, verbose=0)
        preds = model.predict(X_val, verbose=1)
        mse.append(mean_squared_error(y_val[0], preds[0]))
        print(f"Fold - {fold}, MSE - {mse}")
        
    print(f"Final MSE - {np.mean(mse)}")
    return np.mean(mse)


    

def LSTM_config(n_features, hidden1, hidden2, activation, learning_rate, dropout_rate):
    model = Sequential()
    model.add(LSTM(hidden1, activation = activation, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(hidden2, activation = activation))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_steps_out))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model




def train_model(n_steps_in, n_steps_out, data_path, output_path, n_trails, logger):
    
    logger.info(f'********************** Loading Data **********************')

    train, _ = load_data(data_path, n_steps_in, n_steps_out)
    n_features = train[:, 1:].shape[1]

    logger.info(f'********************** Training LSTM **********************')

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_tscv(trial, train), n_trials=n_trails)  # Adjust the number of trials
    
    logger.info(f'********************** Saving Experiments Results **********************')
    #save optuna optimizing results table
    train_optuna_result = study.trials_dataframe()
    train_optuna_result.sort_values(by=['value'], inplace = True)
    train_optuna_result.to_csv(output_path / "trials.csv", index = False)
    
    logger.info(f'********************** Saving Optimal Model**********************')
    # load best params and retrain the model
    best_params = study.best_params  
    best_params.update({"n_features" : n_features})
    model = LSTM_config(**best_params)
    
    
    X_train, y_train = split_sequences(train, n_steps_in, n_steps_out)
    model.fit(X_train, y_train, epochs= EPOCHS, verbose=0)
    model.save(output_path / 'LSTM.h5')
    
    logger.info(f'********************** Done **********************')

    return study

            
# DO NOT CHANGE THIS
n_steps_in, n_steps_out = 24, 6


#SET EPCOHS AND TRAILS
EPOCHS = 150
n_trails = 50

# get data path
root_path = Path(os.getcwd())

trial_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

data_path = root_path / "data" / "merged_data.csv"
output_path = root_path / "results" / trial_time

print(f"Output Path {output_path}")
# initialize the output path for different trails
output_path.mkdir(parents=True, exist_ok=True)

# create logger
log_file = output_path / ('train_%s.log' % trial_time)
logger = create_logger(log_file)


study = train_model(n_steps_in, n_steps_out, data_path, output_path, n_trails, logger)


