
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler


def load_data(data_path, n_steps_in, n_steps_out):
    
    dataset = pd.read_csv(data_path).drop(columns = ["Date"])
    
    dataset.fillna(method='bfill', inplace = True)
    dataset = move_col(dataset, "CPI", 0)
    train = dataset.iloc[:-(n_steps_in + n_steps_out), :].values
    test = dataset.iloc[-(n_steps_in + n_steps_out):, :].values
    
    scaler = StandardScaler()
    train[:, 1:] = scaler.fit_transform(train[:, 1:])
    test[:, 1:] = scaler.transform(test[:, 1:])

    return train, test

def move_col(df, column_to_move, new_position):
    cols = list(df.columns)
    cols.remove(column_to_move)
    cols.insert(new_position, column_to_move)
    df = df[cols]
    return df

def create_logger(log_file=None, log_level=logging.INFO):
    """
    Creates and configures a logger object.

    Args:
        log_file (str, optional): The path to the log file. If not provided, logs will be printed to the console only.
        log_level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger



def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, 1:], sequences[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y) 