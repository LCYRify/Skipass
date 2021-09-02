import pandas as pd
import numpy as np


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
        d[int(i)] = df.loc[df.numer_sta == i]

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
