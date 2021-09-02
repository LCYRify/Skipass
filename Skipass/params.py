# Contains all variables for the package


"""
DATA PARAMS
"""

column_dict = {
    'date':['ns','dat','date'],
    'numer_sta':['ns','nt','numéro de station'],
    'Latitude':['ns','nt','Latitude'],
    'Longitude':['ns','nt','Longitude'],
    'Altitude':['ns','int','Altitude'],
    'pmer':['ss','int','Pression au niveau de la mer'],
    'dd':['ss','int','Direction du vent'],
    'ff':['ss','flt','Vitesse du vent'],
    't':['ss','flt','Température'],
    'u':['ss','int','Humidité'],
    'ssfrai':['rs','flt','Hauteur de neige fraiche'],
    'rr3':['ms','flt','Précipitation sur les 3 dernières heures']
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
Cat_col = []
Col_select = (Not_encoded + Num_col_standard + Num_col_robust + Num_col_minmax + Cat_col)
Col_base = (Num_col_standard + Num_col_robust + Num_col_minmax + Cat_col)
Stations = [7481,7650,7661,7690,7591,7577,7643]
day_per_seq = 15
obs_per_day = 24/3
obs_per_seq = int(day_per_seq * obs_per_day)
target = 1
sequence_train = 250
sequence_test = int(sequence_train * 0.5)
sequence_valid = int(sequence_train * 0.2)
col_synop_float = Col_select[1:]
