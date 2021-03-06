import folium
import pandas as pd

def station_filter_nivo():
    df = pd.read_csv('../documentation/liste_stations_rawdata_nivo.txt', delimiter = ",")
    alps = df.loc[(df['Longitude'] >= 6) & (df['Longitude'] < 7.84) & (df['Latitude'] >= 44.48) & (df['Latitude'] <= 46.37)]
    return alps

def station_filter_synop():
    df = pd.read_csv('../documentation/liste_stations_rawdata_synop.txt', delimiter= ";")
    synop_stations = df.loc[(df['Longitude'] >= 6) & (df['Longitude'] < 7.84) &
                  (df['Latitude'] >= 44.48) & (df['Latitude'] <= 46.37)]
    return synop_stations

def station_mapping(alps):
    ''' créer un carte a partir d'un dataframe'''
    m = folium.Map(location=[45, 6], zoom_start=7)
    for i in range(0,len(alps)):
        marker = folium.Marker(
            location=[alps.iloc[i]['Latitude'], alps.iloc[i]['Longitude']],
            popup=alps.iloc[i]['Nom'].capitalize(),)
        marker.add_to(m)
    return m
