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


class model_test(kt.HyperModel):
    def __init__(self, norm):
        self.norm = norm

    def build(self, hp):
        model = Sequential()
        model.add(self.norm)
        hp_GRU_units1 = hp.Int('GRU unit1',
                               min_value=32,
                               max_value=512,
                               step=32)
        hp_GRU_units2 = hp.Int('GRU unit2',
                               min_value=32,
                               max_value=512,
                               step=32)
        hp_GRU_units3 = hp.Int('GRU unit3',
                               min_value=32,
                               max_value=512,
                               step=32)
        hp_Dense_unit1 = hp.Int('hp_Dense_unit1',
                                min_value=16,
                                max_value=128,
                                step=8)
        hp_learning_rate = hp.Choice('learning_rate',
                                     values=[1e-2, 1e-3, 1e-4])
        hp_loss_metrics = hp.Choice('loss_metrics', values=['MSLE', 'MAE'])

        model.add(
            layers.GRU(units=hp_GRU_units1,
                       activation='tanh',
                       return_sequences=True))
        model.add(
            layers.GRU(units=hp_GRU_units2,
                       activation='tanh',
                       return_sequences=True))
        model.add(layers.GRU(units=hp_GRU_units3, activation='tanh'))
        model.add(layers.Dense(units=hp_Dense_unit1, activation='relu'))
        model.add(layers.Dense(8, activation='linear'))

        model.compile(loss=hp_loss_metrics,
                      optimizer=RMSprop(learning_rate=hp_learning_rate),
                      metrics=['msle', 'mae'])

        return model


def grid_search():

    data = DataSkipass()

    X_train, y_train, X_valid, y_valid, X_test, y_test = data.split_X_y()

    col = y_train[0].columns

    X_train, y_train = df_2_nparray(X_train, y_train)

    X_valid, y_valid = df_2_nparray(X_valid, y_valid)
    X_test, y_test = df_2_nparray(X_test, y_test)

    norm = Normalization()
    norm.adapt(X_train)

    hypermodel = model_test(norm=norm)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5)
    tuner = kt.Hyperband(
        hypermodel,
        objective='msle',
        max_epochs=10,
        factor=3,
        project_name='meteo1'
    )

    tuner.search(X_train,
                 y_train,
                 epochs=20,
                 validation_data=(X_valid, y_valid),
                 callbacks=[stop_early])

    return tuner

grid_search()
