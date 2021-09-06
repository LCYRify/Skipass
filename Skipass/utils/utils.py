import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

<<<<<<< HEAD
def replace_nan_0(df, column_name):
    df[column_name] = df[column_name].replace(np.nan,value=0)
    return df

def replace_nan_mean_2points(df, column_name):
    df.reset_index(drop=True)
    df = df.sort_values(['numer_sta', 'date'])
    df[column_name] = pd.concat([df['t'].ffill(), df['t'].bfill()]).groupby(level=0).mean()
    df = df.sort_index()
    return df.sort_index().sort_values(['date'])

def replace_nan_most_frequent(df,column_name):
    df[column_name] = df[column_name].fillna(df[column_name].mode().iloc[0])
    return df

def categorize_rain(df, column_name):
    '''Transforme la colonne rrN en catégorielle (pluie):
    1 si précipitation > 4mm, 0 si inférieur'''
    df[column_name] = np.where(df[column_name] >= 4, 1, 0)
=======
>>>>>>> 2810b2fdb83bd40a4add6c468aa3e4e2241f72bc
