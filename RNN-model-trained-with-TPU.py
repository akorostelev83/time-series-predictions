import tensorflow as tf
import os

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

with strategy.scope():
  step = 1
  model = Sequential()
  model.add(SimpleRNN(units=32, input_shape=(1,step), activation="relu"))
  model.add(Dense(16, activation="relu"))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='rmsprop')

import pandas as pd
import numpy as np
from statistics import mean 

data = pd.read_csv('soybeans_monthly.csv')
ohlc = ['Open','High','Low','Close']
test_train_split = 0.9
times_to_run_model = 15000

def convert_to_matrix(data, step):
    X, Y =[], []
    for i in range(len(data)-step):
        d=i+step  
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

def convert_X_to_matrix(data,step):   
    X=[]
    for i in range(len(data)-step):
        d=i+step  
        X.append(data[i:d,])        
    return np.array(X)

for c in ohlc:
    model_predictions=[]
    N = len(data)    
    Tp = int(N*test_train_split)    
    #t=np.arange(0,N)    
    x=data[c]
    df = pd.DataFrame(x)
    values=df.values
    train,test = values[0:Tp,:], values[Tp:N,:]

    
    test = np.append(test,np.repeat(test[-1,],step))
    train = np.append(train,np.repeat(train[-1,],step))

    trainX,trainY = convert_to_matrix(train,step)
    testX,testY = convert_to_matrix(test,step)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    last_value = x[-1:]._values[0]
    p = pd.DataFrame([last_value])
    v = np.append(p,np.repeat(p,step))
    v_x = convert_X_to_matrix(v,step)
    v_x = np.reshape(v_x, (v_x.shape[0], 1, v_x.shape[1]))  

    for t in range(times_to_run_model):      
        model.fit(trainX,trainY, epochs=50, batch_size=16, verbose=0)
        trainPredict = model.predict(trainX)
        testPredict= model.predict(testX)
        predicted=np.concatenate((trainPredict,testPredict),axis=0)

        trainScore = model.evaluate(trainX, trainY, verbose=0)
        print(trainScore)

        prediction = model.predict(v_x)[0][0]
        model_predictions.append(prediction)
        print('{}: {} prediction for {} is {}'.format(t,c,last_value,prediction))
        print('"{}" average prediction so far is {}'.format(c,mean(model_predictions)))

        pd.plotting.lag_plot(df, lag=step)  
        index = df.index.values
        plt.plot(index,df,'C2',label=c)
        plt.plot(index,predicted,'C1',label='predicted')
        plt.legend()
        plt.show() 

    print('"{}" average prediction is {}'.format(c,mean(model_predictions)))

print('input row for predictions: {}'.format(data[-1:]._values))