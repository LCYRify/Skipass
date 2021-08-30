import pandas as pd

def mf_date_totime(df):
    df['date'] = pd.to_datetime(df['date'],format='%Y%m%d%H%M%S',errors='coerce')
    return df

def mf_date_filter(df, year):
    df_filtered = df[df['date'].dt.year == year]
    return df_filtered
