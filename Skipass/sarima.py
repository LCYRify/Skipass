from pmdarima import auto_arima
from Skipass.data import DataSkipass
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan
import numpy as np
import pickle


def sarimax_fourier():

    df = DataSkipass().create_df()

    df = filter_data(df)

    df = fill_missing(df)

    df = replace_nan(df, False, False)

    print('data preprocesing is done')

    '''
    Take df, put date in index then drop
    y_train = all temp data - last year
    y_test = last year
    8 obs/day * 365 = 2920
    to take into account bisextile years : 8 * 365,25 = 2922
    '''

    df10 = df.loc[df['numer_sta'] == 7577]
    df10 = df10[['date', 't']]
    df10 = df10.set_index(df10['date'])
    y = df10[['t']]
    y_to_train = y.iloc[:(len(y) - 2920)]
    y_to_test = y.iloc[(len(y) - 2920):]  # last year for testing

    print('data splitting is done')

    '''prepare Fourier terms'''
    exog = df10
    exog['sin365'] = np.sin(2 * np.pi * exog.index.dayofyear / 2922)
    exog['cos365'] = np.cos(2 * np.pi * exog.index.dayofyear / 2922)
    exog['sin365_2'] = np.sin(4 * np.pi * exog.index.dayofyear / 2922)
    exog['cos365_2'] = np.cos(4 * np.pi * exog.index.dayofyear / 2922)
    exog = exog.drop(columns=['date'])
    exog_to_train = exog.iloc[:(len(y) - 2920)]
    exog_to_test = exog.iloc[(len(y) - 2920):]

    print('data fourier is done')

    # Fit model
    arima_exog_model = auto_arima(y=y_to_train,
                                  exogenous=exog_to_train,
                                  seasonal=True,
                                  trace=True,
                                  m=2922)

    print('data fitting is done')

    # Forecast
    y_arima_exog_forecast = arima_exog_model.predict(n_periods=2918,exogenous=exog_to_test)

    print('data F is done')

    pickle.dump(arima_exog_model, open("sarima_model.p", "wb"))
    pickle.dump(y_arima_exog_forecast, open("sarima_forecast.p", "wb"))

sarimax_fourier()
