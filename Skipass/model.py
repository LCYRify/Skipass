from Skipass.utils.evaluation import baseline_mse
from Skipass.data import DataSkipass
from Skipass.utils.evaluation import baseline_mse, baseline_mae
import Skipass.params as params
from Skipass.utils.split import df_2_nparray
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan, split_X_y
from Skipass.utils.split import df_2_nparray
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

def model_run(shape1, shape2):

    model = Sequential()
    model.add(layers.GRU(320,activation='tanh',return_sequences=True,input_shape=(shape1, shape2)))
    model.add(layers.GRU(128, activation='tanh', return_sequences=True))
    model.add(layers.GRU(256, activation='tanh'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(8, activation='linear'))

    model.compile(loss='mae',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=MSE)

    return model

df = DataSkipass().create_df()

df = filter_data(df)

df = fill_missing(df)

df_scaled = replace_nan(df, True, True)
X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test = split_X_y(df_scaled)

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

del X_test_scaled, X_train_scaled, X_valid_scaled

shape1 = X_train.shape[1]
shape2 = X_train.shape[2]

model = model_run(shape1, shape2)

es = EarlyStopping(patience=1, restore_best_weights=True)

history = model.fit(X_train,
                    y_train,
                    epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[es])

loss, mae = model.evaluate(X_test, y_test, verbose=2)
storage_upload(history)
