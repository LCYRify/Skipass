"""
BASIC IMPORTS
"""
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
IMPORTS FROM SKIPASS PACKAGE
"""
from Skipass.utils.DataCleaner import replace_values,delete_bad_measures,select_stations
from Skipass.utils.df_typing import mf_date_conv_filtered, mf_date_totime
from Skipass.station_filter.station_filter import station_filter_nivo,station_filter_synop, station_mapping
from Skipass.utils.utils import sequence, splitdata
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
        df = self.df[self.df.numer_sta.isin(params.Stations)][params.Col_select]
        df = df.replace("mq",value=np.nan)
        df = df.replace("/",value=np.nan)
        df['date'] = pd.to_datetime(df['date'],format='%Y%m%d%H%M%S',errors='coerce')
        df = df.sort_values('date')

        for i in params.col_synop_float:
            df[i] = df[i].astype(float,errors='ignore')

        df['dd_sin'] = np.sin(2 * np.pi * df.dd / 360)
        df['dd_cos'] = np.cos(2 * np.pi * df.dd / 360)
        #df.drop('dd', axis=1, inplace=True)

        return df

    def split_set(self):
        return splitdata(self.filter_data())

    def split_X_y(self):

        df_train, df_valid, df_test = self.split_set()

        X_train, y_train = sequence(df_train,params.obs_per_seq,params.target,params.sequence_train)
        X_valid, y_valid = sequence(df_valid,params.obs_per_seq,params.target,params.sequence_valid)
        X_test, y_test = sequence(df_test,params.obs_per_seq,params.target,params.sequence_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test
