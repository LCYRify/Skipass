import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

#Import DF from Deva's class
#df = pd.read_csv()

def replace_by_mean(col):
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df[[col]])
    df[col] = imputer.transform(df[[col]])
    df.f'{col}'.replace('mq', df.col.mean())

def replace_by_zero(col):
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    imputer.fit(df[[col]])
    df[col] = imputer.transform(df[[col]])
    df.f'{col}'.replace('mq', 0)
    df[df<0] = 0


def replace_by_last_preceding_value(col):
