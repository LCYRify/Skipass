from Skipass.data import DataSkipass
import Skipass.params as params
from Skipass.utils.split import df_2_nparray
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras


def model_run(shape1, shape2):

    model = Sequential()
    model.add(layers.GRU(10,activation='tanh',return_sequences=True,input_shape=(shape1, shape2)))
    model.add(layers.GRU(10, activation='tanh', return_sequences=True))
    model.add(layers.GRU(10, activation='tanh'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(8, activation='linear'))

    model.compile(loss='msle',
                  optimizer=RMSprop(learning_rate=0.01),
                  metrics=MAE)

    return model


data = DataSkipass()

X_train, y_train, X_valid, y_valid, X_test, y_test = data.split_X_y()
col = y_train[0].columns

X_train, y_train = df_2_nparray(X_train, y_train)
X_valid, y_valid = df_2_nparray(X_valid, y_valid)
X_test, y_test = df_2_nparray(X_test, y_test)

shape1 = X_train.shape[1]
shape2 = X_train.shape[2]

model = model_run(shape1, shape2)

es = EarlyStopping(patience=25, restore_best_weights=True)

history = model.fit(X_train,
                    y_train,
                    epochs=1000,
                    validation_data=(X_valid, y_valid),
                    callbacks=[es])

loss, mae = model.evaluate(X_test, y_test, verbose=2)

model.save('my_model')
