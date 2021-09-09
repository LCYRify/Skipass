import pandas as pd
import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError


def baseline_mae(X_train, y):

    col = y[0].columns

    X = pd.DataFrame()
    Y = pd.DataFrame()
    for i in X_train:
        X = pd.concat([X,i[-1:][col]])
    mse = MeanAbsoluteError()

    for i in y:
        Y = pd.concat([Y,i])

    mae = MeanAbsoluteError()

    return mae(X, Y)

def baseline_mse(X_train, y):

    col = y[0].columns

    X = pd.DataFrame()
    Y = pd.DataFrame()
    for i in X_train:
        X = pd.concat([X,i[-1:][col]])
    mse = MeanAbsoluteError()

    for i in y:
        Y = pd.concat([Y,i])

    mse = MeanSquaredError()
    return mse(X, Y)
