from fastapi import FastAPI
from google.cloud import storage
from tempfile import TemporaryFile
import joblib
import tensorflow as tf
from Skipass.predict import predict
import pickle
from Skipass.utils.evaluation import baseline_mse
from Skipass.data import DataSkipass
from Skipass.utils.evaluation import baseline_mse, baseline_mae
from Skipass.utils.split import df_2_nparray
from Skipass.gcp import storage_upload
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan, split_X_y
from Skipass.utils.split import df_2_nparray
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE, MSE, MSLE, MAE
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

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
    # triangulation des points
    # on recupere le bon model
    scaler = pickle.load(open('model_test/scaler.pkl', 'rb'))
    print(scaler)
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
    
    return 'hey ho'
    
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


