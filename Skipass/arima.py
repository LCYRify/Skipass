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
import Skipass.utils
from pmdarima import auto_arima


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


def arima(df, col_name):
    '''Import Raw Dataset (station 7    577) and clearing (remove NaN, mq, missing values) '''
    # df = DataSkipass().create_df()
    # df = filter_data(df)
    # df = fill_missing(df)
    # df = replace_nan(df, False, False)
    print('Preprocesing done : dataset clear')

    df_f = df
    df_f.reset_index(drop=True)

    t = df_f.index

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
