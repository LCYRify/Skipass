from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import MinMaxScaler


def Predict(minmaxscaler, model, sequence):

    sequence = minmaxscaler.transform(sequence)
    result = model.predict(sequence)
    result = result.inverse_tranform(result)

return result
