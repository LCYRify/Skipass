# Contains all variables for the package


"""
DATA PARAMS
"""

"""
REPLACE NAN STRATEGY:
"""
column_dict = {
    'date': ['ns', 'dat', 'date', 'na'],
    'numer_sta': ['ns', 'nt', 'numéro de station', 'na'],
    'Latitude': ['ns', 'nt', 'Latitude', 'na'],
    'Longitude': ['ns', 'nt', 'Longitude', 'na'],
    'Altitude': ['ns', 'int', 'Altitude', 'na'],
    'pmer':
    ['ss', 'int', 'Pression au niveau de la mer', 'na'],
    'dd': ['ss', 'int', 'Direction du vent', 'replace_nan_most_frequent'],
    'ff': ['ss', 'flt', 'Vitesse du vent', 'replace_nan_0'],
    't': ['ss', 'flt', 'Température', 'replace_nan_mean_2points'],
    'u': ['ss', 'int', 'Humidité', 'replace_nan_mean_2points'],
    'ssfrai': ['rs', 'flt', 'Hauteur de neige fraiche', 'replace_nan_0'],
    'rr3':
    ['ms', 'flt', 'Précipitation sur les 3 dernières heures', 'replace_nan_0'],
    'dd_sin': ['ms', 'flt', 'Direction du vent (sin)', 'na'],
    'dd_cos': ['ms', 'flt', 'Direction du vent (cos)', 'na'],
    'pres': ['ss', 'int', 'Pression à la station', 'na']
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
Num_col_todrop = ['pres']
Cat_col = []
Col_select = (Not_encoded + Num_col_standard + Num_col_robust +
              Num_col_minmax + Cat_col + Num_col_todrop)
Col_improved = Col_select +  Num_col_engineer
Col_base = (Num_col_standard + Num_col_robust + Num_col_minmax + Cat_col + Num_col_engineer)
Stations = [7510,7434,7643,7690,7481,7630,7255,7240,7460,7222,7577,7280,7299,7650]
#Stations = [7481, 7650, 7630, 7690, 7591, 7577, 7643]
# Stations = [7577]
day_per_seq = 7
obs_per_day = 24/3
obs_per_seq = int(day_per_seq * obs_per_day)
target = 2
sequence_train = 2500
sequence_test = int(sequence_train * 0.5)
sequence_valid = int(sequence_train * 0.2)
col_synop_float = Col_select[1:]

model_path = '../../saved_model/'


def extract_list_target():
    lm2p, lmf, l0 = [],[],[]
    for key in column_dict.keys():
        if column_dict[key][3] == 'replace_nan_mean_2points':
            lm2p.append(key)
        elif column_dict[key][3] == 'replace_nan_most_frequent':
            lmf.append(key)
        elif column_dict[key][3] == 'replace_nan_0':
            l0.append(key)
    return lm2p, lmf, l0

"""
GCP CONFIGURATION
"""
# - - - GCP Project - - -
PROJECT_ID='skipass-325207'
# - - - GCP Storage - - -
BUCKET_NAME='skipass_325207_model'
REGION='europe-west1'
# - - - Data - - -
BUCKET_TRAIN_DATA_PATH = 'skipass_325207_data/weather_synop_data.csv'
# - - - Model - - -
MODEL_NAME = 'skipass'
MODEL_VERSION = 'v1'
