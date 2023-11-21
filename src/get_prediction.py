from utils import *
import keras



def get_prediction_train(n_steps_in, n_steps_out, data_path, ckpt_path):
    #load data
    
    # train,_ = load_data(data_path, n_steps_in, n_steps_out)
    train, test = load_data(data_path, n_steps_in, n_steps_out)
    train = np.vstack((train, test[:-6, :]))
    X_train, _ = split_sequences(train, n_steps_in, n_steps_out)
    
    train_pred = []
    model = keras.models.load_model(ckpt_path)
    preds = model.predict(X_train)
    
    for i, pred in enumerate(preds):
        if i < len(preds) - 1:
        
            train_pred.append(pred[0])
        else:
            train_pred.extend(pred)
            
    return train_pred
    
def get_prediction_test_future(n_steps_in, n_steps_out, data_path, ckpt_path):
    
    _, test = load_data(data_path, n_steps_in, n_steps_out)
    n_features = test[:, 1:].shape[1]
    
    X_test, _ = split_sequences(test, n_steps_in, n_steps_out)
    X_future = test[-n_steps_in:, 1:].reshape((1, n_steps_in, n_features))

    model = keras.models.load_model(ckpt_path)
    test_pred = model.predict(X_test)[0]
    future_pred = model.predict(X_future)[0]
    
    return test_pred, future_pred