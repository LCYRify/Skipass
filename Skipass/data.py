"""
BASIC IMPORTS
"""
from os import sep
import pandas as pd
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from google.cloud import storage

from sklearn.preprocessing import MinMaxScaler
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
from Skipass.utils.cleaner import replace_nan_0, replace_nan_mean_2points, replace_nan_most_frequent, pmer_compute, categorize_rain, my_custom_ts_multi_data_prep
from Skipass.utils.split import create_subsample, sequence, splitdata, df_2_nparray
#from Skipass.grid import model_test
from Skipass.utils.utils import save_model
import Skipass.params as params

"""
PATHS
"""

path_to_data = 'gs://skipass_325207_model/skipass_325207_data/weather_synop_data.csv'
path_to_station_list = 'gs://skipass_325207_model/skipass_325207_data/liste_stations_rawdata_synop.txt'
#path_to_data = '/Users/devasou/code/LCYRify/Skipass/raw_data/weather_synop_data.csv'
#path_to_station_list = '/Users/devasou/code/LCYRify/Skipass/documentation/liste_stations_rawdata_synop.txt'


class DataSkipass:

    def __init__(self):
        pass

    """
    DATA CREATION
    """

    def import_data(self):
        """
        Output (Pandas Dataframe containing Synop data):
        """
        print('Importing data')
        return pd.read_csv(path_to_data)

    def import_list_stations(self):
        """
        Output (Pandas DataFrame):
            ID;Nom;Latitude;Longitude;Altitude
            07510;BORDEAUX-MERIGNAC;44.830667;-0.691333;47
        """
        print('Importing station list')
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
        print('DF created')
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
        df = self.create_df()
        df = df[df.numer_sta.isin(params.Stations)][params.Col_select]
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

        # replace nan according to strategies
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

        # compute pmer from temp, station pressure and Alt
        df['pmer'] = df.apply(
            lambda row: pmer_compute(row['t'], row['pres'], row['Altitude'])
            if pd.isnull(row['pmer']) else row['pmer'],
            axis=1)

        list_df2 = create_subsample(df)

        # replace nan for pmer
        list_new_df4 = []
        for df in list_df2:
            list_new_df4.append(replace_nan_mean_2points(df, ['pmer']))
        df = list_new_df4[0]
        for df_new in list_new_df4[1:]:
            df = pd.concat([df, df_new])

        # convert K to °C
        df['t'] = df['t'] - 273.15

        # categorize rain
        df['rr3'] = df.apply(lambda row: 0 if row['rr3'] < 4 else 1, axis=1)

        return df
        #df = categorize_rain(df,'rr3')


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

    def run_test(self):
        df = self.replace_nan()
        df.set_index('date', inplace=True)
        df.Latitude = np.radians(df.Latitude)
        df.Longitude = np.radians(df.Longitude)
        df['x'] = np.cos(df.Latitude) * np.cos(df.Longitude)
        df['y'] = np.cos(df.Latitude) * np.sin(df.Longitude)
        df['z'] = np.sin(df.Latitude)
        df.drop(columns=['Latitude', 'Longitude'], inplace=True)
        df = df.astype({"numer_sta": int, "Altitude": int, "dd": int})
        df.numer_sta.unique()
        df[df.numer_sta == 7630]
        
        scaler = MinMaxScaler()
        df[['x', 'y', 'z', 'Altitude', 'pmer', 'dd', 'ff', 't', 'u', 'ssfrai', 'rr3', 'pres', 'dd_sin', 'dd_cos']] = \
        scaler.fit_transform(df[['x', 'y', 'z', 'Altitude', 'pmer', 'dd', 'ff', 't', 'u', 'ssfrai', 'rr3', 'pres', 'dd_sin', 'dd_cos']])
        
        dataX = df[['numer_sta', 'x', 'y', 'z', 'Altitude', 'pmer', 'dd', 'ff', 't', 'u', 'ssfrai', 'rr3', 'pres', 'dd_sin', 'dd_cos']]
        dataY = df[['numer_sta', 't']]
        
        what_to_predict = [1]
        hist_window = 120 # 15 days * 8 measures per day
        horizon = len(what_to_predict)
        split = 0.8
        x_train, y_train, x_val, y_val = my_custom_ts_multi_data_prep(dataX, dataY, split, hist_window, horizon)
        
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        ind = np.random.randint(0, x_train.shape[0], 2)

        fig, axs = plt.subplots(nrows=x_train.shape[2], ncols=2, sharex=True, figsize=(8, 20))
        fig.suptitle('Two random samples \n [{:d} & {:d}]'.format(*ind))

        the_range = [x+x_train.shape[1]-1 for x in what_to_predict]

        for j in range(2):
            for i in range(x_train.shape[2]):
                axs[i, j].set_title(dataX.columns[i+1], fontsize=9)
                axs[i, j].plot(x_train[ind[j], :, i])
                if dataX.columns[i+1] == 't':
                    axs[i, j].scatter(the_range, y_train[ind[j]])
        
        


if __name__ == '__main__':
    dsp = DataSkipass()
    dsp.run_test()
    
