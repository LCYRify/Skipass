from os import listdir, path
import pandas as pd
import gzip
from Skipass.utils.preprocessing import filter_data,fill_missing
from Skipass.data import DataSkipass


def synop_merger_tocsv():
    df = pd.DataFrame()
    path_archive_data = path.abspath("../Skipass/data/archive_data/Synop")
    path_2_save = path.abspath("../Skipass/data/")
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
    path_last_15 = path.abspath('../raw_data/last_15.csv.gz')

    df = pd.read_csv(path_last_15, delimiter=';')
    df_stat = DataSkipass().import_list_stations()
    df = df.drop(columns=['Unnamed: 59'])
    df_stat = df_stat.rename(columns={'ID': 'numer_sta'})
    df = df_stat.merge(df, on='numer_sta')
    df = filter_data(df)
    df = fill_missing(df)

    return df
