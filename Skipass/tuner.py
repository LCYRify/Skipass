from tensorflow.python.keras.activations import swish
from Skipass.data import DataSkipass
from Skipass.utils.split import df_2_nparray
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan, split_X_y
from Skipass.utils.evaluation import baseline_mae, baseline_mse
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import Hyperband, BayesianOptimization
import Skipass.params as params
import pandas as pd
import pickle
import numpy as np
import keras_tuner as kt



class model_test(kt.HyperModel):
    def __init__(self):
        self.name = 'grid'

    def build(self, hp):
        model = Sequential()

        hp_GRU_units1 = hp.Int('GRU unit1', min_value=128, max_value=512, step=32)
        hp_GRU_units2 = hp.Int('GRU unit2', min_value=64, max_value=512, step=64)
        hp_GRU_units3 = hp.Int('GRU unit3', min_value=64, max_value=512, step=32)
        hp_GRU_units4 = hp.Int('GRU unit4', min_value=64, max_value=512, step=32)
        # hp_GRU_units5 = hp.Int('GRU unit5', min_value=64, max_value=512, step=32)
        hp_Dense_unit1 = hp.Int('hp_Dense_unit1', min_value=16, max_value=256, step=16)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        hp_dense_activation = hp.Choice('dense activation',values=['relu','swish','selu'])
        hp_GRU_activation = hp.Choice('gru_activation', values=['tanh'])

        model.add(layers.GRU(units=hp_GRU_units1, activation=hp_GRU_activation, return_sequences=True))
        model.add(layers.GRU(units=hp_GRU_units2, activation=hp_GRU_activation, return_sequences=True))
        model.add(layers.GRU(units=hp_GRU_units3, activation=hp_GRU_activation, return_sequences=True))
        model.add(layers.GRU(units=hp_GRU_units4, activation=hp_GRU_activation))
        #model.add(layers.GRU(units=hp_GRU_units5, activation=hp_GRU_activation))
        model.add(layers.Dense(units=hp_Dense_unit1, activation=hp_dense_activation))
        model.add(layers.Dense(8, activation='linear'))

        model.compile(loss=MAE,
                      optimizer=RMSprop(learning_rate=hp_learning_rate),
                      metrics=['mse', 'mae'])

        return model

def hyperband_try():
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

    del X_train, X_valid, X_test, df_scaled, df

    X_train, y_train = df_2_nparray(X_train_scaled, y_train)
    X_valid, y_valid = df_2_nparray(X_valid_scaled, y_valid)
    X_test, y_test = df_2_nparray(X_test_scaled, y_test)

    hypermodel = model_test()

    stop_early = EarlyStopping(monitor='val_loss',patience=5)

    tuner = Hyperband(
        hypermodel,
        objective='mae',
        max_epochs=15,
        factor=3,
        project_name='hyperband_test'
    )

    tuner.search(X_train,
                    y_train,
                    epochs=50,
                    validation_data=(X_valid, y_valid),
                    callbacks=[stop_early])


def Bayesian_try():

    df = DataSkipass().create_df()

    print('Data : loaded in RAM')

    df = filter_data(df)

    print('Data : filtered')

    df = fill_missing(df)

    print('Data : missing values filled')

    df_scaled = replace_nan(df, True, True)

    print('Data : NaN treated')

    X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test = split_X_y(
        df_scaled)

    del df_scaled, y_train, y_valid, y_test

    print('Sequencing X Train / Valid / Test split done')

    df = replace_nan(df, False, False)

    print('Data : NaN treated')

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_X_y(df)

    print('Sequencing y Train / Valid / Test split done')

    print(
        f'Tuner pour une distance de {params.target} observation, un nombre de station de {len(params.Stations)}, et {params.sequence_train} sequence pour le training.'
    )

    print('La baseline mse est de : ' + str(baseline_mse(X_train, y_train)))
    print('La baseline mae est de : ' + str(baseline_mae(X_train, y_train)))

    del X_train, X_valid, X_test, df

    X_train, y_train = df_2_nparray(X_train_scaled, y_train)
    X_valid, y_valid = df_2_nparray(X_valid_scaled, y_valid)
    X_test, y_test = df_2_nparray(X_test_scaled, y_test)

    hypermodel = model_test()

    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner = BayesianOptimization(hypermodel,
                         objective='val_loss',
                         max_trials=5,
                         project_name='Bayes_test_mae_1obs_18sta_2k5seq')

    tuner.search(X_train,
                 y_train,
                 epochs=50,
                 validation_data=(X_valid, y_valid),
                 callbacks=[stop_early])

    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

    pickle.dump(best_hyperparameters, open("Best_Bayesian_mae.p", "wb"))

Bayesian_try()
