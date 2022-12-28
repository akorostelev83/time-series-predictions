import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def train_predict_daily_vix_with_ltsm():
    vix = pd.read_csv(
        'C:/Users/SUPREME/Documents/GitHub/time-series-predictions/VIX-daily.csv',)\
            .dropna()

    print(len(vix.Close))

    n_samples = 3175 * 2
    sequence_length = 15 * 2

    data = vix.Close[len(vix.Close)-n_samples:len(vix.Close)]
    assert data[-1:].values[0] == vix.Close.values[-1:][0]
    data = data.values.astype(np.float32)    

    mmscaler = MinMaxScaler((-1.0,1.0))
    data = mmscaler.fit_transform(data.reshape(-1,1))
    X_ts = np.zeros(shape=(n_samples-sequence_length,sequence_length,1),dtype=np.float32)
    Y_ts = np.zeros(shape=(n_samples-sequence_length,1),dtype=np.float32)

    for i in range(0,data.shape[0]-sequence_length):
        X_ts[i] = data[i:i+sequence_length]
        Y_ts[i] = data[i+sequence_length]

    X_ts_train = X_ts[0:2600*2,:]
    Y_ts_train = Y_ts[0:2600*2]

    X_ts_test = X_ts[2600*2:n_samples,:]
    Y_ts_test = Y_ts[2600*2:n_samples]   

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            4,
            stateful=True,
            batch_input_shape=(1,sequence_length,1)),
        tf.keras.layers.Dense(1,activation='tanh')])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            decay=0.0001),
        loss = 'mse',
        metrics=['mse'])

    model.fit(
        X_ts_train,
        Y_ts_train,
        batch_size=1,
        epochs=1,
        shuffle=False,
        validation_data=(X_ts_test,Y_ts_test))

    y_pred = model.predict(X_ts_test, batch_size=1)
    y_pred_inversed = mmscaler.inverse_transform(y_pred)
    y_true_inversed = mmscaler.inverse_transform(Y_ts_test)

    plt.figure(figsize=(12,5))
    plt.plot(vix.Date.values[-y_true_inversed.shape[0]:],y_true_inversed, 'b', label="truth")
    plt.plot(vix.Date.values[-y_pred_inversed.shape[0]:],y_pred_inversed, 'r', label="predictions")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    y_pred_future = mmscaler.inverse_transform(
        model.predict(
            data[-30:].reshape(1,30,1),
            batch_size=1))

    print(f'next day prediction: {y_pred_future}')

    x=0

def train_predict_weekly_ltsm():
    vix = pd.read_csv(
        'C:/Users/SUPREME/Documents/GitHub/time-series-predictions/VIX-weekly.csv',)\
            .dropna()
    print(len(vix.Close))

    n_samples = 1720
    sequence_length = 5

    data = vix.Close[len(vix.Close)-n_samples:len(vix.Close)]
    assert data[-1:].values[0] == vix.Close.values[-1:][0]
    data = data.values.astype(np.float32)    

    # data = vix.Close.values.astype(np.float32) 
    # assert len(data) == len(vix.Close)

    mmscaler = MinMaxScaler((-1.0,1.0))
    data = mmscaler.fit_transform(data.reshape(-1,1))

    X_ts = np.zeros(shape=(n_samples-sequence_length,sequence_length,1),dtype=np.float32)
    Y_ts = np.zeros(shape=(n_samples-sequence_length,1),dtype=np.float32)

    for i in range(0,data.shape[0]-sequence_length):
        X_ts[i] = data[i:i+sequence_length]
        Y_ts[i] = data[i+sequence_length]

    train_test_split = 1500

    X_ts_train = X_ts[0:train_test_split,:]
    Y_ts_train = Y_ts[0:train_test_split]

    X_ts_test = X_ts[train_test_split:n_samples,:]
    Y_ts_test = Y_ts[train_test_split:n_samples]

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            12,
            stateful=True,
            batch_input_shape=(1,sequence_length,1)),
        tf.keras.layers.Dense(1,activation='tanh')])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            decay=0.0001),
        loss = 'mse',
        metrics=['mse'])

    model.fit(
        X_ts_train,
        Y_ts_train,
        batch_size=1,
        epochs=100,
        shuffle=False,
        validation_data=(X_ts_test,Y_ts_test))   

    y_pred = model.predict(X_ts_test, batch_size=1)
    y_pred_inversed = mmscaler.inverse_transform(y_pred)
    y_true_inversed = mmscaler.inverse_transform(Y_ts_test)

    plt.figure(figsize=(12,5))
    plt.plot(vix.Date.values[-y_true_inversed.shape[0]:],y_true_inversed, 'b', label="truth")
    plt.plot(vix.Date.values[-y_pred_inversed.shape[0]:],y_pred_inversed, 'r', label="predictions")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


    sequence_for_debug = mmscaler.inverse_transform(data[-sequence_length-1:-1])
    print(f'predicting next week close with this sequence:\n {sequence_for_debug}')

    # verify that current week's data is not being used for predictions
    # the current week's closing price is pending a close
    # wait until the end of the week to use the current week's closing price to predict next week's closing price
    assert mmscaler.inverse_transform(data[-1:]) not in sequence_for_debug

    y_pred_future = mmscaler.inverse_transform(
        model.predict(
            data[-sequence_length-1:-1].reshape(1,sequence_length,1),
            batch_size=1))

    print(f'next week close prediction: {y_pred_future}')

    x=0

#train_predict_daily_vix_with_ltsm()
train_predict_weekly_ltsm()