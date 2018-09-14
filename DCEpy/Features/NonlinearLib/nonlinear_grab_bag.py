'''
A file collecting nonlinear measures to use in conjunction with the autoregressive
coefficients to use as features.
***NOTE FOR CODERS***
If the feature requires some smaller subfunctions, start their names with underscores
***
'''
from __future__ import print_function
import numpy as np
import scipy
import scipy.stats

def teager_energy(x,axis = -1):
    """
    :param x: a numpy array containing data from a window
    :param axis:
    :return:

    """
    n= x.shape[0]
    tmp = x[1:n - 1, :] ** 2 - x[0:n - 2, :] * x[2:n,:]  # Teager energy
    TE = np.log10(np.mean(np.abs(tmp), axis=axis))  # ABSOLUTE Teager energy
    return TE

def mcl(x,axis = -1):
    """
    :param x:
    :param axis:
    :return:
    """
    n = x.shape[0]
    mcl=np.log10(np.mean(np.abs(np.diff(x, axis=axis)), axis=axis))
    return mcl



def lyapunov_exponent(x,axis = -1,eps = 1e-11):
    '''
    Compute the Lyapunov exponent along an arbitrary axis.
    Detailed in Haddad, et al 2014: "Temporal Epilepsy Seizure Prediction Using Graph and Chaos Theory"

    :param x: a numpy array containing data from a window
    :param axis: the axis along which it's done
    :param eps: the added noise term to prevent negative infinite logs
    :return :
    '''
    x_abs_deriv = np.abs(np.diff(x,axis = axis))
    lyap = np.mean(np.log(x_abs_deriv + eps),axis=axis)
    return lyap

nonlinear_func_dict = {'lyapunov_exponent':lyapunov_exponent}

def signal_entropy(x,num_bins=None, axis = -1):
    '''
    Compute
    :param x: the signal
    :param num_bins: the number of (evenly-spaced) bins used to approximate the pdf (with a histogram)
    :param axis: axis
    :return:
    '''
    def entropy_on_row(row):
        approx_pdf,_ = np.histogram(row,bins = num_bins)
        return scipy.stats.entropy(approx_pdf)
    if num_bins is None:
        num_bins = np.ceil(np.sqrt(np.size(x,axis)))

    return np.apply_along_axis(entropy_on_row,axis,x)

