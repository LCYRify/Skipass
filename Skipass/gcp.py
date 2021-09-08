import os
from termcolor import colored
from google.cloud import storage
from Skipass.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION


def storage_upload(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)
    print("Client created")
    storage_location = 'model.joblib'
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print('blob uploaded')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')