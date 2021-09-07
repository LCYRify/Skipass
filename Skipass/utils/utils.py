import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from google.cloud import storage


def mf_date_totime(df):
    df['date'] = pd.to_datetime(df['date'],
                                format='%Y%m%d%H%M%S',
                                errors='coerce')
    return df


def mf_date_filter(df, year):
    df_filtered = df[df['date'].dt.year == year]
    return df_filtered


def mf_date_conv_filtered(df, year):
    df = mf_date_totime(df)
    df = mf_date_filter(df, year)
    return df

def draw_sample(X, y, n):
    fig, axs = plt.subplots(3, 3, figsize=(25, 12))
    fig.suptitle(f'Sample n°{n+1}.', fontsize=30)
    fig.tight_layout()
    axs[0, 0].set_title('pression au niveau de la mer', fontsize=20)
    sns.lineplot(x=X.index, y=X['pmer'] / 100, ax=axs[0, 0])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['pmer'] / 100,
                    ax=axs[0, 0],
                    sizes=40,
                    color='red')
    axs[0, 1].set_title('direction du vent', fontsize=20)
    sns.lineplot(x=X.index, y=X['dd'], ax=axs[0, 1])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['dd'],
                    ax=axs[0, 1],
                    sizes=40,
                    color='red')
    axs[0, 2].set_title('vitesse du vent', fontsize=20)
    sns.lineplot(x=X.index, y=X['ff'], ax=axs[0, 2])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['ff'] / 100,
                    ax=axs[0, 2],
                    sizes=40,
                    color='red')
    axs[1, 0].set_title('température', fontsize=20)
    sns.lineplot(x=X.index, y=X['t'] - 262, ax=axs[1, 0])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['t'] - 273.15,
                    ax=axs[1, 0],
                    sizes=40,
                    color='red')
    axs[1, 1].set_title('humidité', fontsize=20)
    sns.lineplot(x=X.index, y=X['u'], ax=axs[1, 1])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['u'],
                    ax=axs[1, 1],
                    sizes=40,
                    color='red')
    axs[1, 2].set_title('hauteur neige fraiche', fontsize=20)
    sns.lineplot(x=X.index, y=X['ssfrai'], ax=axs[1, 2])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['ssfrai'],
                    ax=axs[1, 2],
                    sizes=40,
                    color='red')
    axs[2, 0].set_title('précipitation sur les 3 dernières heures',
                        fontsize=20)
    sns.lineplot(x=X.index, y=X['rr3'], ax=axs[2, 0])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['rr3'],
                    ax=axs[2, 0],
                    sizes=40,
                    color='red')
    axs[2, 1].set_title('Direction du vent (sin)',
                        fontsize=20)
    sns.lineplot(x=X.index, y=X['dd_sin'], ax=axs[2, 1])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['dd_sin'],
                    ax=axs[2, 1],
                    sizes=40,
                    color='red')
    axs[2, 2].set_title('Direction du vent (cos)', fontsize=20)
    sns.lineplot(x=X.index, y=X['dd_cos'], ax=axs[2, 2])
    sns.scatterplot(x=X.index.max() + 1,
                    y=y['dd_cos'],
                    ax=axs[2, 2],
                    sizes=40,
                    color='red')
    print(fig)


def draw_station(X):
    fig, axs = plt.subplots(3, 3, figsize=(25, 12))
    fig.suptitle(f'Station', fontsize=30)
    fig.tight_layout()
    axs[0, 0].set_title('pression au niveau de la mer', fontsize=20)
    sns.lineplot(x=X.index, y=X['pmer'] / 100, ax=axs[0, 0])

    axs[0, 1].set_title('direction du vent', fontsize=20)
    sns.lineplot(x=X.index, y=X['dd'], ax=axs[0, 1])

    axs[0, 2].set_title('vitesse du vent', fontsize=20)
    sns.lineplot(x=X.index, y=X['ff'], ax=axs[0, 2])

    axs[1, 0].set_title('température', fontsize=20)
    sns.lineplot(x=X.index, y=X['t'] - 273.15, ax=axs[1, 0])

    axs[1, 1].set_title('humidité', fontsize=20)
    sns.lineplot(x=X.index, y=X['u'], ax=axs[1, 1])

    axs[1, 2].set_title('hauteur neige fraiche', fontsize=20)
    sns.lineplot(x=X.index, y=X['ssfrai'], ax=axs[1, 2])

    axs[2, 0].set_title('précipitation sur les 3 dernières heures',
                        fontsize=20)
    sns.lineplot(x=X.index, y=X['rr3'], ax=axs[2, 0])
    axs[2, 1].set_title('direction du vent (sin)',
                        fontsize=20)
    sns.lineplot(x=X.index, y=X['dd_sin'], ax=axs[2, 1])
    axs[2, 2].set_title('direction du vent (cos)',
                        fontsize=20)
    sns.lineplot(x=X.index, y=X['dd_cos'], ax=axs[2, 2])

    print(fig)

def upload_model_to_gcp():
    STORAGE_LOCATION = 'skipass_325207_model/model.joblib'
    BUCKET_NAME='skipass_325207_model'
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')

def save_model(reg):
    STORAGE_LOCATION = 'skipass_325207_model/model.joblib'
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")
    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

def plot_guillaume():

    fig, axs = plt.subplots(nrows=x_train.shape[2], ncols=2, sharex=True, figsize=(8, 20))
    fig.suptitle('Two random samples \n [{:d} & {:d}]'.format(*ind))

    the_range = [x+x_train.shape[1]-1 for x in what_to_predict]

    for j in range(2):
        for i in range(x_train.shape[2]):
            axs[i, j].set_title(dataX.columns[i+1], fontsize=9)
            axs[i, j].plot(x_train[ind[j], :, i])
            if dataX.columns[i+1] == 't':
                axs[i, j].scatter(the_range, y_train[ind[j]])
