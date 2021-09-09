import os
from termcolor import colored
from google.cloud import storage
import joblib
from Skipass.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, STORAGE_LOCATION

"""
def storage_upload(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)
    print("Client created")
    storage_location = STORAGE_LOCATION
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print('blob uploaded')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')
def save_model(reg):
    #method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')
"""

def save_model(model):
        """Save the model into a .joblib format"""
        joblib.dump(model,'model.joblib')
        print(colored("model.joblib saved", "green"))

def upload_model_to_gcp(model):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    #blob.upload_from_file(model,'model.joblib')
    with open("model", "rb") as model:
        blob.upload_from_file(model,)
