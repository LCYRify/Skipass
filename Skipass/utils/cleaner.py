import pandas as pd
import numpy as np

def pmer_compute(temperature, pression, altitude):
    ''' Serie_temperature, température en C ou K
        serie_pression, en pascal
        altitude, en metre
        retourne la pression en Pascal
    '''
    if np.nan in [temperature, pression, altitude]:
        return 101400
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
    return (serie - Var)
