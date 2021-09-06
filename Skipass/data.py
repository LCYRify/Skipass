"""
BASIC IMPORTS
"""
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE
from tensorflow.keras.callbacks import EarlyStopping

"""
IMPORTS FROM SKIPASS PACKAGE
"""
from Skipass.utils.DataCleaner import replace_values,delete_bad_measures,select_stations
from Skipass.utils.df_typing import mf_date_conv_filtered, mf_date_totime
from Skipass.station_filter.station_filter import station_filter_nivo,station_filter_synop, station_mapping
#from Skipass.utils.utils import
from Skipass.utils.cleaner import replace_nan_0, replace_nan_mean_2points, replace_nan_most_frequent, pmer_compute, categorize_rain
from Skipass.utils.split import create_subsample, sequence, splitdata, df_2_nparray
import Skipass.params as params

"""
PATHS
"""
path_to_data = '../raw_data/weather_synop_data.csv'
path_to_station_list = '../documentation/liste_stations_rawdata_synop.txt'

class DataSkipass:

    def __init__(self):
        self.df = self.create_df()

    """
    DATA CREATION
    """

    def import_data(self):
        """
        Output (Pandas Dataframe containing Synop data):
        """
        return pd.read_csv(path_to_data)

    def import_list_stations(self):
        """
        Output (Pandas DataFrame):
            ID;Nom;Latitude;Longitude;Altitude
            07510;BORDEAUX-MERIGNAC;44.830667;-0.691333;47
        """
        return pd.read_csv(path_to_station_list, sep=';')

    def create_df(self):
        """
        Output: Pandas DataFrame from 'df Synop Data' and 'df Station list'
        """
        df_data = self.import_data()
        df_stations = self.import_list_stations()

        # Drop columns
        df_data = df_data.drop(columns = ['Unnamed: 0','Unnamed: 59'])
        df_stations = df_stations.rename(columns={'ID': 'numer_sta'})
        # Merge on station numbers
        df = df_stations.merge(df_data, on='numer_sta')

        return df

    """
    DATA TRANSFORMATIONS
    """

    def filter_data(self, replace_value = np.nan):
        """
        Output:
            Get a DF filtered without 'mq' and '/' values and a datetime type
        """
        # get df
        df = self.df[self.df.numer_sta.isin(params.Stations)][params.Col_select]
        # replace mq as nan
        df = df.replace("mq",value=replace_value)
        df = df.replace("/",value=replace_value)
        # convert to datetime
        df['date'] = pd.to_datetime(df['date'],format='%Y%m%d%H%M%S',errors='coerce')
        # sort via datetime
        df = df.sort_values('date')
        # convert str as float
        for i in params.col_synop_float:
            df[i] = df[i].astype(float,errors='ignore')
        return df

    def replace_nan(self):
        df_ = self.filter_data()
        list_df = create_subsample(df_)
        lm2p, lmf, l0 = params.extract_list_target()

        list_new_df1,list_new_df2,list_new_df3 = [],[],[]
        for df in list_df:
            list_new_df1.append(replace_nan_mean_2points(df,lm2p))
        for df in list_new_df1:
            list_new_df2.append(replace_nan_most_frequent(df,lmf))
        for df in list_new_df2:
            list_new_df3.append(replace_nan_0(df,l0))
        df = list_new_df3[0]

        for df_new in list_new_df3[1:]:
            df = pd.concat([df,df_new])

        # create sin and cos from wind direction
        df['dd_sin'] = np.sin(2 * np.pi * df.dd / 360)
        df['dd_cos'] = np.cos(2 * np.pi * df.dd / 360)
        # convert t to °C
        df['t'] = df['t'] - 273.15
        # categorize rain
        df = categorize_rain(df,'rr3')
        # compute pmer from temp, station pressure and Alt
        df['pmer'] = df.apply(
            lambda row: pmer_compute(row['t'], row['pres'], row['Altitude'])
            if pd.isnull(row['pmer']) else row['pmer'],
            axis=1)
        return df

    def split_set(self):
        """
        Output: A splitdata of DF
        """
        return splitdata(self.replace_nan())

    def split_X_y(self):
        """
        Output: A train, valid and test subsample of the DF
        """
        df_train, df_valid, df_test = self.split_set()

        X_train, y_train = sequence(df_train,params.obs_per_seq,params.target,params.sequence_train)
        X_valid, y_valid = sequence(df_valid,params.obs_per_seq,params.target,params.sequence_valid)
        X_test, y_test = sequence(df_test,params.obs_per_seq,params.target,params.sequence_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def create_model(self):
        """
        Input: subsample of df (train, valid, test)
        Output: a fitted DL model and its evaluation values as a tuple
        """
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_X_y()
        X_train,y_train = df_2_nparray(X_train,y_train)
        X_valid, y_valid = df_2_nparray(X_valid, y_valid)
        X_test, y_test = df_2_nparray(X_test, y_test)
        norm = Normalization()
        norm.adapt(X_train)

        model = Sequential()
        model.add(norm)
        model.add(layers.LSTM(50,activation = 'tanh', return_sequences=True))
        model.add(layers.GRU(50,activation= 'tanh'))
        model.add(layers.Dense(100,activation = 'relu'))
        model.add(layers.Dense(7,activation = 'linear'))

        model.compile(loss = 'mse', optimizer = RMSprop(), metrics = MAPE)

        es = EarlyStopping(patience = 10, restore_best_weights = True)

        history = model.fit(X_train,y_train, epochs = 2, validation_data = (X_valid,y_valid), callbacks = [es])

        eval = model.evaluate(X_test, y_test)

        return history,eval




if __name__ == '__main__':
    """
    create df:
    """
    data = DataSkipass()
    """
    Split df:
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test = data.split_X_y()
    col = y_train[0].columns
    X_train1, y_train1, X_valid1, y_valid1, X_test1, y_test1 = data.split_X_y()
    """
    Transform them to np array:
    """
    X_train,y_train = df_2_nparray(X_train,y_train)
    X_valid, y_valid = df_2_nparray(X_valid, y_valid)
    X_test, y_test = df_2_nparray(X_test, y_test)
    """
    Create model:
    """
    # normalization
    norm = Normalization()
    norm.adapt(X_train)
    # dumping
    with open("X_train.pkl","wb") as file:
        pickle.dump(X_train, file)
    with open("y_train.pkl","wb") as file:
        pickle.dump(y_train, file)
    # model creation
    model = Sequential()
    model.add(norm)
    model.add(layers.GRU(384,activation = 'tanh', return_sequences=True))
    model.add(layers.GRU(96,activation = 'tanh', return_sequences=True))
    model.add(layers.GRU(96,activation= 'tanh'))
    model.add(layers.Dense(100,activation = 'relu'))
    model.add(layers.Dense(8,activation = 'linear'))
    # model compilation
    model.compile(loss = 'mse', optimizer = RMSprop(learning_rate=0.01), metrics = MAPE)
    # Early Stopping creation
    es = EarlyStopping(patience = 25, restore_best_weights = True)
    # Fitting
    history = model.fit(X_train,y_train, epochs = 1000, validation_data = (X_valid,y_valid), callbacks = [es])
    # evaluation
    eval = model.evaluate(X_test, y_test)
    # plots
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.plot(history.history['mean_absolute_percentage_error'])
    # predictions
    result = model.predict(X_test[0])
    pd.DataFrame(y_test[0].reshape(1,8), columns=col)
    pd.DataFrame(result[0].reshape(1,8),columns=col)
    result[0].T.shape
