"""
BASIC IMPORTS
"""
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from Skipass.utils.utils import sequence, splitdata, df_2_nparray
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

    def filter_data(self):
        """
        Output:
            Get a DF filtered without 'mq' and '/' values and a datetime type 
        """
        df = self.df[self.df.numer_sta.isin(params.Stations)][params.Col_select]
        df = df.replace("mq",value=0)
        df = df.replace("/",value=0)
        #df = df.replace("mq",value=np.nan)
        #df = df.replace("/",value=np.nan)
        df['date'] = pd.to_datetime(df['date'],format='%Y%m%d%H%M%S',errors='coerce')
     
        for i in params.col_synop_float:
            df[i] = df[i].astype(float,errors='ignore')

        return df
    
    def split_set(self):
        """
        Output: A splitdata of DF 
        """
        return splitdata(self.filter_data())
    
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