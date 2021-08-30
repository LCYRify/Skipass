from os import listdir, path
import pandas as pd
import gzip

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
