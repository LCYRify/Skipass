from os import listdir, path
import pandas as pd
import gzip
from Skipass.utils.preprocessing import fill_missing, filter_data, replace_nan, split_X_y
from Skipass.data import DataSkipass


def synop_merger_tocsv():
    df = pd.DataFrame()
    # path_archive_data = path.abspath("../Skipass/data/archive_data/Synop")
    path_archive_data = '/home/romain/code/LCYRify/Skipass/raw_data/archive_data/Synop'
    # path_2_save = path.abspath("../Skipass/data/")
    path_2_save = '/home/romain/code/LCYRify/Skipass/raw_data'
    marker = False
    for i in listdir(path_archive_data):
        with open(path.join(path_archive_data, i), 'rb') as file:
            gzip_fd = gzip.GzipFile(fileobj=file)
            df_temp = pd.read_csv(gzip_fd, delimiter=';')
        if marker is False:
            df = df_temp
            marker = True
        else:
            df = pd.concat([df, df_temp])

    #today = date.today().strftime("%d-%m-%Y")
    dumpname = 'weather_synop_data.csv'
    path_2_save_date = path.join(path_2_save, dumpname)
    df.to_csv(path_2_save_date)

def nivo_merger_tocsv():
    path_archive_data = path.abspath("../Skipass/scrapper/archive_data/")
    path_2_save = path.abspath("../Skipass/data/")
    marker = False
    for i in listdir(path_archive_data):
        with open(path.join(path_archive_data,i), 'rb') as file:
            gzip_fd = gzip.GzipFile(fileobj=file)
            df_temp = pd.read_csv(gzip_fd,delimiter=';')
        if marker is False:
            df = df_temp
            marker = True
        else:
            df = pd.concat([df,df_temp])

    #today = date.today().strftime("%d-%m-%Y")
    dumpname = 'weather_nivodata.csv'
    path_2_save_date = path.join(path_2_save,dumpname)
    df.to_csv(path_2_save_date)

def create_last15_csv():
    #path_last_15 = path.abspath('/Users/devasou/code/LCYRify/Skipass/raw_data/last_15.csv')

    df = pd.read_csv('/Users/devasou/code/LCYRify/Skipass/raw_data/last_15.csv', delimiter=',')
    print('csv read')
    print(df.head())
    df_stat = DataSkipass().import_list_stations()
    print('stations imported')
    df = df.drop(columns=['Unnamed: 59'])
    df_stat = df_stat.rename(columns={'ID': 'numer_sta'})
    df = df_stat.merge(df, on='numer_sta')
    df = filter_data(df)
    df = fill_missing(df)

    return df

def save_arima_station():

    synop_merger_tocsv()

    df = DataSkipass().create_df()

    df = filter_data(df)

    df = fill_missing(df)

    df = replace_nan(df, False, False)

    df.to_csv('/home/romain/code/LCYRify/Skipass/raw_data/stations_arima.csv')

save_arima_station()
