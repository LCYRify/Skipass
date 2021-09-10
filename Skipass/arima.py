import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from datetime import date, timedelta
from sklearn.impute import SimpleImputer
import scipy.optimize as op
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from Skipass.data import DataSkipass
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib


def my_fit(t, data, guess_freq):

    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / (2**0.5) / (2**0.5)
    guess_phase = 0
    #guess_freq = 11
    guess_amp = 1

    data_first_guess = guess_std * np.sin(guess_freq * 2 * np.pi * t / len(t) +
                                          guess_phase) + guess_mean

    optimize_func = lambda x: x[0] * np.sin(x[1] * 2 * np.pi * t / len(t) + x[
        2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = op.leastsq(
        optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    return (est_amp * np.sin(est_freq * 2 * np.pi * t / len(t) + est_phase) +
            est_mean), data_first_guess


def arima(df, col_name=['t', 'u', 'pmer']):
    '''Import Raw Dataset (station 7577) and clearing (remove NaN, mq, missing values) '''
    # df = DataSkipass().create_df()
    # df = filter_data(df)
    # df = fill_missing(df)
    # df = replace_nan(df, False, False)
    print('Preprocesing done : dataset clear')

    df_f = df
    df_f = df_f.reset_index(drop=True)

    t = df_f.index
    col_name = ['t','u','pmer']
    answer = []

    for i in col_name:

        data = df_f[col_name[i]]
        data_fit, data_first=my_fit(t, data, 11)
        df_f["unyearly"] = data - data_fit
        df_f.set_index('date', inplace=True)
        df_f = df_f.asfreq(freq='3H')

        total_slot = 8 * 25
        train_slot = 8 * 20
        serie = df_f["unyearly"]
        min_slot = len(serie) - total_slot
        df_slot = df_f[min_slot:]

        result_add = seasonal_decompose(df_slot["unyearly"], model='additive')

        df_deseasonal = df_slot["unyearly"] - result_add.seasonal

        # Build Model
        model = ARIMA(df_deseasonal, order=(0, 1, 3))
        arima = model.fit()

        # Forecast
        forecast, std_err, confidence_int = arima.forecast(2, alpha=0.05)  # 95% confidence

        data_fit_slot = data_fit[min_slot:]

        #test_saisonal = result_add.seasonal[train_slot:] + data_fit_slot[train_slot:]

        test_saisonal= 0

        #train_saisonal = result_add.seasonal[0:train_slot] + data_fit_slot[0:train_slot]

        forecast_recons = forecast + test_saisonal
        #train_recons = train + train_saisonal
        #test_recons = test + test_saisonal
        lower_recons = confidence_int[:, 0] + test_saisonal
        upper_recons = confidence_int[:, 1] + test_saisonal

        answer.append(forecast_recons)

    return answer

def rain_model():

    root_path  = '/home/romain/code/LCYRify/Skipass/'

    csv_path = root_path + 'raw_data/stations_arima.csv'
    df = pd.read_csv(csv_path)

    model = Pipeline([('scaler', MinMaxScaler()), ('logistic', LogisticRegression())])

    directory = root_path + 'save_model/rain_predict'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for station in stations:
        # Select the data that corresponds to the station
        df_station = df[df['numer_sta'] == station]
        # Calculate the number of rainy and no-rainy segments (called "days")
        days_of_rain    = (df_station['rr3']==1).sum()
        days_of_no_rain = (df_station['rr3']==0).sum()
        print('=====================================================================')
        print(f'Working on station: {station}')
        print(f'days_of_no_rain: {days_of_no_rain}, days_of_rain: {days_of_rain}')
        # Eliminating no-rainy days at random to balance the number of instances of each class
        df_station = df_station[df_station['rr3'] == 1].append(df_station[df_station['rr3'] == 0].sample(days_of_rain))
        #  print(df_station.shape)
        #  print(df_of_rain.shape)
        #  print(df_of_no_rain.shape)
        X = df_station[['t', 'u', 'pmer']]
        y = df_station['rr3']
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=51)
        #if station != 7643.0:
        model.fit(X_train, y_train)
        model.score(X_train,y_train)
        y_test_pred = model.predict(X_test)
        print(model.score(X_train,y_train))
        print(model.score(X_test,y_test))
        print(precision_score(y_test, y_test_pred, average='macro'))
        print(recall_score(y_test, y_test_pred, average='macro'))
        model_path  = directory + f'/model{int(station)}.pkl'
        joblib.dump(model, model_path)
        print(f'Saved in {model_path}')

def get_rain(numer_sta, t, u, pmer): # numer_sta is the station number integer
    model_path = directory + f'/model{int(station)}.pkl'
    model = joblib.load(model_path)
    #  print(lcl[variable_name])
    X_test = np.array([[t, u, pmer]])
    return model.predict(X_test)
