from fastapi import FastAPI
import os
import joblib
import tensorflow as tf
import pickle
model_path = '/Users/devasou/code/LCYRify/Skipass/model.joblib'
scaler_path = '/Users/devasou/code/LCYRify/Skipass/save_model/scaller.joblib'

X_test = ""
Y_test = ""

# loaded_model = pickle.load(open(model_path, 'rb'))
loaded_model = joblib.load(model_path)
#result = loaded_model.score(X_test, Y_test)
print(loaded_model)
loaded_model = pickle.load(open(model_path, 'rb'))
#result = loaded_model.score(X_test, Y_test)
print(loaded_model)
