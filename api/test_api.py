from fastapi import FastAPI
from google.cloud import storage
from tempfile import TemporaryFile
import joblib
import tensorflow as tf
import pickle


# option 1: 

storage_client = storage.Client()
bucket_name='skipass_325207_model'
model_bucket='model.joblib'
bucket = storage_client.get_bucket(bucket_name)
print('In bucket')
#select bucket file
blob = bucket.blob(model_bucket)
print('Blob Loaded')
with TemporaryFile() as temp_file:
    #download blob into temp file
    blob.download_to_file(temp_file)
    print('Model downloaded')
    temp_file.seek(0)
    #load into joblib
    print('Prep to loading')
    model=joblib.load(temp_file)
    #data = pickle.load(temp_file)
    print(model)
    print('Model loaded')

# option 2: 
gcs_path = 'gs://skipass_325207_model/model.joblib'

#loaded_model = joblib.load(tf.io.gfile.GFile(gcs_path, 'rb'))

with open(gcs_path, "rb") as f:
     data = pickle.load(f)
print(data)