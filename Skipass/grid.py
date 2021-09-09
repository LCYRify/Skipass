from Skipass.data import DataSkipass
from Skipass.utils.split import df_2_nparray
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan, split_X_y
from Skipass.utils.evaluation import baseline_mae, baseline_mse
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pickle
import numpy as np
import keras_tuner as kt



class model_test(kt.HyperModel):
    def __init__(self):
        self.name = 'grid'

    def build(self, hp):
        model = Sequential()

        hp_GRU_units1 = hp.Int('GRU unit1',
                               min_value=128,
                               max_value=512,
                               step=32)
        hp_GRU_units2 = hp.Int('GRU unit2',
                               min_value=64,
                               max_value=512,
                               step=64)
        hp_GRU_units3 = hp.Int('GRU unit3',
                               min_value=32,
                               max_value=512,
                               step=32)
        hp_Dense_unit1 = hp.Int('hp_Dense_unit1',
                                min_value=64,
                                max_value=256,
                                step=16)
        hp_Dense_unit2 = hp.Int('hp_Dense_unit2',
                                min_value=64,
                                max_value=256,
                                step=16)
        hp_Dense_unit3 = hp.Int('hp_Dense_unit3',
                                min_value=16,
                                max_value=128,
                                step=8)
        hp_learning_rate = hp.Choice('learning_rate',
                                     values=[1e-2, 1e-3, 1e-4, 1e-5])

        model.add(layers.GRU(units=hp_GRU_units1,activation='tanh',return_sequences=True))
        model.add(layers.GRU(units=hp_GRU_units2,activation='tanh',return_sequences=True))
        model.add(layers.GRU(units=hp_GRU_units3, activation='tanh'))
        model.add(layers.Dense(units=hp_Dense_unit1, activation='relu'))
        model.add(layers.Dense(units=hp_Dense_unit2, activation='relu'))
        model.add(layers.Dense(units=hp_Dense_unit3, activation='relu'))
        model.add(layers.Dense(8, activation='linear'))

        model.compile(loss=MAE,
                      optimizer=RMSprop(learning_rate=hp_learning_rate),
                      metrics=['mse', 'mae'])

        return model


df = DataSkipass().create_df()

df = filter_data(df)

df = fill_missing(df)

df_scaled = replace_nan(df, True, True)
X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test = split_X_y(
    df_scaled)

df = replace_nan(df, False, False)
X_train, y_train, X_valid, y_valid, X_test, y_test = split_X_y(df)

print('La baseline mse est de : ' + str(baseline_mse(X_train, y_train)))
print('La baseline mae est de : ' + str(baseline_mae(X_train, y_train)))

test_predict_X = X_train[0]
test_predict_y = y_train[0]

del X_train, X_valid, X_test, df_scaled, df

col = y_train[0].columns

X_train, y_train = df_2_nparray(X_train_scaled, y_train)
X_valid, y_valid = df_2_nparray(X_valid_scaled, y_valid)
X_test, y_test = df_2_nparray(X_test_scaled, y_test)

hypermodel = model_test()

stop_early = EarlyStopping(monitor='val_loss',patience=5)
tuner = kt.Hyperband(
    hypermodel,
    objective='msle',
    max_epochs=10,
    factor=3,
    project_name='meteo'
)

tuner.search(X_train,
                y_train,
                epochs=50,
                validation_data=(X_valid, y_valid),
                callbacks=[stop_early])

tuner.save('../log/tuner_trained')
