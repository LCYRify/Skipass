from Skipass.data import DataSkipass
import Skipass.params as params
from Skipass.utils.utils import save_model
from Skipass.utils.split import df_2_nparray
from Skipass.utils.utils import save_model
from Skipass.gcp import storage_upload
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def model_run(norm):

    model = Sequential()
    model.add(norm)
    model.add(layers.GRU(960, activation='tanh', return_sequences=True))
    model.add(layers.GRU(672, activation='tanh', return_sequences=True))
    model.add(layers.GRU(96, activation='tanh'))
    model.add(layers.Dense(424, activation='relu'))
    model.add(layers.Dense(304, activation='relu'))
    model.add(layers.Dense(8, activation='linear'))

    model.compile(loss='msle', optimizer=RMSprop(learning_rate=0.01), metrics=MAE)

    return model

data = DataSkipass()

X_train, y_train, X_valid, y_valid, X_test, y_test = data.split_X_y()
col = y_train[0].columns

X_train, y_train = df_2_nparray(X_train, y_train)
X_valid, y_valid = df_2_nparray(X_valid, y_valid)
X_test, y_test = df_2_nparray(X_test, y_test)

norm = Normalization()
norm.adapt(X_train)

model = model_run(norm)

es = EarlyStopping(patience=1, restore_best_weights=True)

history = model.fit(X_train,
                    y_train,
                    epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[es])

loss, mae = model.evaluate(X_test, y_test, verbose=2)
storage_upload(history)