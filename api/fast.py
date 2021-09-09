from fastapi import FastAPI
from google.cloud import storage
from tempfile import TemporaryFile
import joblib
import tensorflow as tf

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"greeting":'halo'}

@app.get("/predict")
def predict(query):
    pass

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


