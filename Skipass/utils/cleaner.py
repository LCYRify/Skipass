import pandas as pd
import numpy as np



'''

'''












'''
Features Engineering:
'''


def pmer_compute(temperature, pression, altitude):
    ''' Serie_temperature, température en C ou K
        serie_pression, en pascal
        altitude, en metre
        retourne la pression en Pascal
    '''
    P = pression
    g = 9.81
    Cp = 1006
    T0 = T_mer_calc(temperature, altitude)

    P0 = P / (np.exp(((-7 * g) / (2 * Cp * T0)) * altitude))

    return P0


def T_mer_calc(serie, alt):
    ''' Calcul la température au niveau de la mer.
        INPUT :
            serie = température (C ou K)
            alt = altidude en M
    '''
    alt = alt / 1000
    DegM = 6.5
    Var = alt * DegM
    return (serie + Var)

def categorize_rain(df, column_name):
    df[column_name] = np.where(df[column_name] >= 4, 1, 0)

"""
Cleaner strategies:
"""

"""
Replace by 0:
"""
def replace_nan_0(df, columns_name):
    for column in columns_name:
        df[column] = df[column].replace(np.nan, value=0)
    return df

"""
Replace by mean of 2 proxi points:
"""
def replace_nan_first_last_elem(df,columns_name):
    for column in columns_name:
        if pd.isnull(df[column].iloc[0]) == True:
            df[column].iloc[0]  = df[column].mean()
        if pd.isnull(df[column].iloc[-1]) == True:
            df[column].iloc[-1]  = df[column].mean()
    return df

def replace_nan_mean_2points(df, columns_name):
    df = df.sort_values(['date'])
    replace_nan_first_last_elem(df,columns_name)
    for column in columns_name:
        df[column] = pd.concat([df[column].ffill(), df[column].bfill()]).groupby(level=0).mean()
    return df

"""
Replace by most frequent values
"""
def replace_nan_most_frequent(df,columns_name):
    for column in columns_name:
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    return df

def my_custom_ts_multi_data_prep(dataset, target, split, window, horizon):
    Xt, Xv, yt, yv, X, y = [], [], [], [], [], []
    start = window

    # Let's do a sequencing per station (same x, y, z & altitude)
    for i_station in dataset.numer_sta.unique():
        sub_dataset = dataset[dataset.numer_sta == i_station].copy()
        sub_dataset.drop(columns=['numer_sta'], inplace=True)

        sub_target = target[target.numer_sta == i_station].copy()
        sub_target.drop(columns=['numer_sta'], inplace=True)

        print("station {:d} has {:d} time steps".format(i_station, len(sub_dataset)))

        end = len(sub_dataset) - horizon + 1

        for i in range(start, end):
            indices = range(i-window, i)
            X.append(sub_dataset.iloc[indices].to_numpy())

            indicey = range(i, i+horizon)
            y.append(sub_target.iloc[indicey].to_numpy())

        sub_split = int(0.8*(end-start))
        print("  ... pushing {} elements in training set and {:d} in validation set".format(sub_split-1, end-start-sub_split+1))

        Xt += X[:sub_split]
        yt += y[:sub_split]
        Xv += X[sub_split:]
        yv += y[sub_split:]
        print("  ... training set has {:d} elements and validation set {:d}\n".format(len(Xt), len(Xv)))

        X.clear()
        y.clear()

    return np.array(Xt), np.array(yt), np.array(Xv), np.array(yv)
