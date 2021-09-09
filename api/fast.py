"""
IMPORTS
"""

from fastapi import FastAPI
from google.cloud import storage
from tempfile import TemporaryFile
import joblib
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

from Skipass.predict import predict
from Skipass.utils.evaluation import baseline_mse
from Skipass.data import DataSkipass
from Skipass.utils.evaluation import baseline_mse, baseline_mae
from Skipass.utils.split import df_2_nparray
from Skipass.gcp import storage_upload
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan, split_X_y
from Skipass.utils.split import df_2_nparray
from Skipass.utils.merger import create_last15_csv
from Skipass.model import model_run

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping

from fastapi.middleware.cors import CORSMiddleware

"""
API CREATION
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# define a root `/` endpoint
@app.get("/")
def index():
    return {"greeting":'halo'}

@app.get("/predict")
def predict(name):
    name = name
    
    #scaler = pickle.load(open('model_test/scaler.pkl', 'rb'))
    #print(scaler)
    
    # model = tf.keras.models.load_model('model_test/meteo1/')
    # print(model)
    # df = pd.read_csv('last_15.csv')
    # df_scaled = scaler.transform(df)
    
    # print(df_scaled.head())
    # df = DataSkipass().create_df()

    # df = filter_data(df)

    # df = fill_missing(df)

    # df_scaled = replace_nan(df, True, True)
    # X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test = split_X_y(df_scaled)

    # df = replace_nan(df, False, False)
    # X_train, y_train, X_valid, y_valid, X_test, y_test = split_X_y(df)
    
    """
    Import df
    """
    df = create_last15_csv()
    print(df.head())
    
    """
    Clean Data
    """
    
    df_scaled = replace_nan(df, True, True)
    
    """
    Split Data
    """
    
    X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test = split_X_y(df_scaled)

    df = replace_nan(df, False, False)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_X_y(df)

    """
    Base line
    """
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

    es = EarlyStopping(patience=25, restore_best_weights=True)

    history = model.fit(X_train,
                        y_train,
                        epochs=1000,
                        validation_data=(X_valid, y_valid),
                        callbacks=[es])

    loss, mae = model.evaluate(X_test, y_test, verbose=2)
    """
    Predict based on df
    """
    
    
    return {'loss':loss,
            'mae':mae}
    
    # predict()

@app.get("/model")
def call_model():
    storage_client = storage.Client()
    bucket_name='skipass_325207_model'
    model_bucket='model.joblib'
    bucket = storage_client.get_bucket(bucket_name)
    print(bucket)
    #select bucket file
    blob = bucket.blob(model_bucket)
    print(blob)
    
        #download blob into temp file
    r = blob.download_to_filename('../model.joblib')
    print(r)

    #load into joblib
    model=joblib.load('../model.joblib')
    print(model)
    #use the model
    return {'model':'test'}


