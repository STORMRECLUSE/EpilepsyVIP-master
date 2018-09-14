'''
Uses the MATLAB engine to get scattering coefficients from a window of data
'''
import numpy as np
import matlab.engine

def flat_scat_coeffs(data,T,Q,mat_eng = None):
    '''
    Outputs all the scattering coefficients as a vector
    :param data:
    :param T: Period for computing coefficients over a window
    :param Q: Number of filters/octave
    :param mat_eng: matlab engine, used to compute the data
    :return:
    '''

    coeffs = scat_coeffs(data,T,Q,mat_eng)

    flat_coeffs = []
    for layer in coeffs:
        flat_coeffs.extend(layer.flatten())

    return np.array(flat_coeffs)


def scat_coeffs(data, T, Q, mat_eng = None):
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()

    if len(data.shape)==1:
        data = data[:,None]

    data2 = matlab.double(data.tolist())
    coeffs = mat_eng.data_to_scatter_f(data2,float(T),float(Q))
    for index,_ in enumerate(coeffs):
        coeffs[index] = np.array(coeffs[index]['signal'])


    return coeffs