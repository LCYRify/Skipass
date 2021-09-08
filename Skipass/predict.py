from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import Skipass.params as params
import pandas as pd
import numpy as np
import pickle

def predict(df):

    scaler = MinMaxScaler()
    scaler = pickle.load(open(params.model_path + 'scaler.pkl', 'rb'))

    df[['x', 'y', 'z', 'Altitude', 'pmer', 'ff', 't', 'u', 'ssfrai', 'rr3', 'dd_sin', 'dd_cos']] = \
    scaler.transform(df[['x', 'y', 'z', 'Altitude', 'pmer', 'ff', 't', 'u', 'ssfrai', 'rr3', 'dd_sin', 'dd_cos']])

    model = load_model(params.model_path + 'my_model')

    X = df.to_numpy()
    X = X.reshape(1, X.shape[0], X.shape[1])

    result = model.predict(df)

    print(pd.DataFrame(result, columns=df.columns[1:9]))

    return result
