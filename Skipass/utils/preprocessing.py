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
from Skipass.data import DataSkipass
from Skipass.station_filter.station_filter import station_filter_nivo,station_filter_synop, station_mapping
from Skipass.utils.cleaner import replace_nan_0, replace_nan_mean_2points, replace_nan_most_frequent, pmer_compute, categorize_rain, my_custom_ts_multi_data_prep
from Skipass.utils.split import create_subsample, sequence, splitdata, df_2_nparray
from Skipass.utils.utils import save_model
import Skipass.params as params
from Skipass.gcp import upload_model_to_gcp

"""
PATHS
"""

chemin = os.path.dirname(os.path.realpath('__file__'))
path_CSV = chemin + '/../' + 'raw_data/weather_synop_data.csv'
path_txt = chemin + '/../' + 'documentation/liste_stations_rawdata_synop.txt'

# path_to_data = 'gs://skipass_325207_model/skipass_325207_data/weather_synop_data.csv'
# path_to_station_list = 'gs://skipass_325207_model/skipass_325207_data/liste_stations_rawdata_synop.txt'
path_to_data = path_CSV
path_to_station_list = path_txt

def fill_missing(df):

    df = df.drop_duplicates()

    df_full = []

    for i in df.numer_sta.unique():
        df2 = df[df.numer_sta == i]
        alt = df2.Altitude.mean()
        lat = df2.Latitude.mean()
        lon = df2.Longitude.mean()
        start_date = df2.date.min()
        end_date = df2.date.max()
        all_date = pd.date_range(start_date, end_date, freq='3H')
        all_date = pd.DataFrame({'date': all_date})
        all_date['numer_sta'] = i
        all_date['Altitude'] = alt
        all_date['Latitude'] = lat
        all_date['Longitude'] = lon
        df_full.append(pd.merge(df2, all_date, how="outer", on="date"))

    df = pd.DataFrame()
    for i in df_full:
        df = pd.concat([df,i])

    df.numer_sta_x = df.numer_sta_y
    df = df.drop(columns='numer_sta_y')
    df = df.rename(columns={'numer_sta_x': 'numer_sta'})

    df.Altitude_x = df.Altitude_y
    df = df.drop(columns='Altitude_y')
    df = df.rename(columns={'Altitude_x': 'Altitude'})

    df.Latitude_x = df.Latitude_y
    df = df.drop(columns='Latitude_y')
    df = df.rename(columns={'Latitude_x': 'Latitude'})

    df.Longitude_x = df.Longitude_y
    df = df.drop(columns='Longitude_y')
    df = df.rename(columns={'Longitude_x': 'Longitude'})

    return df


def filter_data(df, replace_value = np.nan):
    """
        Output:
            Get a DF filtered without 'mq' and '/' values and a datetime type
        """
    # get df

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

def replace_nan(df, scaler, to_scaled):
    '''
    df = dataframe
    scaler = boolean, True if ou want to use a scaling
    to_scaled = boolean, if scaler is true, True to scale, False to load the scaled model
    '''

    list_df = create_subsample(df)
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
    if scaler == True:
        if to_scaled == True:
            scaler = MinMaxScaler()
            scaler.fit(df[['x', 'y', 'z', 'Altitude', 'pmer', 'ff', 't', 'u', 'ssfrai','rr3', 'dd_sin', 'dd_cos']])
            # save the scaler
            
            upload_model_to_gcp(scaler)
            #pickle.dump(scaler, open(params.model_path + 'scaler.pkl', 'wb'))
        else:
            # load the scaler
            pickle.load(scaler, open(params.model_path + 'scaler.pkl', 'rb'))
        df[['x', 'y', 'z', 'Altitude', 'pmer', 'ff', 't', 'u', 'ssfrai', 'rr3', 'dd_sin', 'dd_cos']] = \
        scaler.transform(df[['x', 'y', 'z', 'Altitude', 'pmer', 'ff', 't', 'u', 'ssfrai', 'rr3', 'dd_sin', 'dd_cos']])

    df = df[df.numer_sta.isin(params.Stations)]

    return df
    #df = categorize_rain(df,'rr3')

def split_X_y(df):
    """
    Output: A train, valid and test subsample of the DF
    """
    df_train, df_valid, df_test = splitdata(df)

    X_train, y_train = sequence(df_train,params.obs_per_seq,params.target,params.sequence_train)
    X_valid, y_valid = sequence(df_valid,params.obs_per_seq,params.target,params.sequence_valid)
    X_test, y_test = sequence(df_test,params.obs_per_seq,params.target,params.sequence_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
