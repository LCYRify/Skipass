# Contains all variables for the package


"""
DATA PARAMS
"""

"""
REPLACE NAN STRATEGY: 
"""

column_dict = {
    'date': ['ns', 'dat', 'date'],
    'numer_sta': ['ns', 'nt', 'numéro de station'],
    'Latitude': ['ns', 'nt', 'Latitude'],
    'Longitude': ['ns', 'nt', 'Longitude'],
    'Altitude': ['ns', 'int', 'Altitude'],
    'pmer': ['ss', 'int', 'Pression au niveau de la mer', 'replace_nan_mean_2points'],
    'dd': ['ss', 'int', 'Direction du vent', 'replace_nan_most_frequent'],
    'ff': ['ss', 'flt', 'Vitesse du vent', 'replace_nan_0'],
    't': ['ss', 'flt', 'Température', 'replace_nan_0'],
    'u': ['ss', 'int', 'Humidité', 'replace_nan_mean_2points'],
    'ssfrai': ['rs', 'flt', 'Hauteur de neige fraiche', 'replace_nan_mean_2points'],
    'rr3': ['ms', 'flt', 'Précipitation sur les 3 dernières heures', 'replace_nan_mean_2points'],
    'dd_sin': ['ms', 'flt', 'Direction du vent (sin)'],
    'dd_cos': ['ms', 'flt', 'Direction du vent (cos)']
}

Dtype_col = {'int':[],'flt':[],'dat':[]}
for i in column_dict:
    if column_dict[i][1] == 'int':
        Dtype_col['int'].append(i)
    elif column_dict[i][1] == 'flt':
        Dtype_col['flt'].append(i)
    elif column_dict[i][1] == 'dat':
        Dtype_col['dat'].append(i)

Not_encoded = ['date','numer_sta','Latitude','Longitude','Altitude']
Num_col_standard = ['pmer','dd','ff','t','u']
Num_col_robust = ['ssfrai']
Num_col_minmax = ['rr3']
Num_col_engineer = ['dd_sin','dd_cos']
Cat_col = []
Col_select = (Not_encoded + Num_col_standard + Num_col_robust + Num_col_minmax + Cat_col )
Col_improved = Col_select +  Num_col_engineer
Col_base = (Num_col_standard + Num_col_robust + Num_col_minmax + Cat_col + Num_col_engineer)
Stations = [7481,7650,7661,7690,7591,7577,7643]
day_per_seq = 15
obs_per_day = 24/3
obs_per_seq = int(day_per_seq * obs_per_day)
target = 1
sequence_train = 250
sequence_test = int(sequence_train * 0.5)
sequence_valid = int(sequence_train * 0.2)
col_synop_float = Col_select[1:]
