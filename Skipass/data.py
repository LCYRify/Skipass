"""
BASIC IMPORTS
"""
from os import sep
import os
import pandas as pd
import numpy as np
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
from Skipass.station_filter.station_filter import station_filter_nivo,station_filter_synop, station_mapping
from Skipass.utils.cleaner import replace_nan_0, replace_nan_mean_2points, replace_nan_most_frequent, pmer_compute, categorize_rain, my_custom_ts_multi_data_prep
from Skipass.utils.split import create_subsample, sequence, splitdata, df_2_nparray
import Skipass.params as params

"""
PATHS
"""

chemin = os.path.dirname(os.path.realpath('__file__'))
path_CSV = chemin + '/raw_data/weather_synop_data.csv'
path_txt = chemin + '/documentation/liste_stations_rawdata_synop.txt'

#path_to_data = 'gs://skipass_325207_model/skipass_325207_data/weather_synop_data.csv'
#path_to_station_list = 'gs://skipass_325207_model/skipass_325207_data/liste_stations_rawdata_synop.txt'
path_to_data = path_CSV
path_to_station_list = path_txt


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
        ra = pd.read_csv(path_to_data)
        print('ok')
        return ra

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
        df = df[params.Col_select]
        #df = df[df.numer_sta.isin(params.Stations)][params.Col_select]
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

        # convert lat lon en cos sin
        df.Latitude = np.radians(df.Latitude)
        df.Longitude = np.radians(df.Longitude)
        df['x'] = (np.cos(df.Latitude) * np.cos(df.Longitude))
        df['y'] = (np.cos(df.Latitude) * np.sin(df.Longitude))
        df['z'] = (np.sin(df.Latitude))
        df.drop(columns=['Latitude', 'Longitude'], inplace=True)
        df = df.astype({"numer_sta": int, "Altitude": int, "dd": int})

        # scaling des datas en min max
        scaler = MinMaxScaler()
        scaler.fit(df[['x', 'y', 'z', 'Altitude', 'pmer', 'dd', 'ff', 't', 'u', 'ssfrai','rr3', 'pres', 'dd_sin', 'dd_cos']])
        df[['x', 'y', 'z', 'Altitude', 'pmer', 'dd', 'ff', 't', 'u', 'ssfrai', 'rr3', 'pres', 'dd_sin', 'dd_cos']] = \
        scaler.transform(df[['x', 'y', 'z', 'Altitude', 'pmer', 'dd', 'ff', 't', 'u', 'ssfrai', 'rr3', 'pres', 'dd_sin', 'dd_cos']])

        # save the scaler
        pickle.dump(scaler, open('scaler.pkl', 'wb'))

        df = df[df.numer_sta.isin(params.Stations)]

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
