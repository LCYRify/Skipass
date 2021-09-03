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


def subsample_sequence(df, length, target):
    """
    Given the initial dataframe `df`, return a shorter dataframe sequence of length `length`.
    This shorter sequence should be selected at random
    """
    x = np.random.randint(1, df.shape[0] - length - target)
    df_sample = df[x:x + length]
    df_target = df[x + length + target - 1:x + length + target]
    return df_sample, df_target


def sequence(df, lenght, target, sequence):

    """
    Given the initial dataframe `df`, return a sequence X and y.
    df = dataframe
    lenght = number of observation in a sequence (int)
    target = number of observation between sequence and prediction (int)
    sequence = number of sequence per stations (int)
    """

    d = {}
    X_ = []
    y_ = []

    for i in df.numer_sta.unique():
        d[int(i)] = df.loc[df.numer_sta == i].sort_values('date').reset_index(drop=True)

    for i in d.keys():
        for j in range(sequence):
            df_sample, df_target = subsample_sequence(d[i], lenght, target)
            X_.append(df_sample)
            y_.append(df_target)

    for i in X_:
        i.drop(columns=['date', 'numer_sta'], inplace=True)

    for i in y_:
        i.drop(
            columns=['date', 'numer_sta', 'Latitude', 'Longitude', 'Altitude'],
            inplace=True)

    return X_, y_

def df_2_nparray(X_,y_):

    X, y = [], []

    for i in X_:
        X.append(i.to_numpy())
    X = np.array(X)
    for i in y_:
        y.append(i.to_numpy())
    y = np.array(y)
    y = y.reshape(y.shape[0], y.shape[-1])

    return X, y


def splitdata(df):

    """
    Given the initial dataframe `df`, return a df_train, df_valid, df_test.
    df = dataframe
    """

    training_l = int(0.8 * len(df))
    test_l = len(df) - training_l
    train_l = int(0.8 * training_l)
    valid_l = int(training_l - train_l)

    test = df.tail(test_l)
    training = df.head(training_l)
    valid = training.tail(valid_l)
    train = training.head(train_l)

    return train, valid, test


def random_trainsample(X, y, n):
    '''
    Given a list of sequence (X), a list of target(y), the number of sample(n)
    it return a dict of 10 samples and targets
    '''

    sequence_len = len(X)
    dictyX = {'X': [], 'y': []}

    for i in range(n):
        x = np.random.randint(0, sequence_len)
        dictyX['X'].append(X[x])
        dictyX['y'].append(y[x])
    return dictyX


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
