"""
This file stores all the fuction for computing features over a number of samples.
Each function returns a 1*p array where p is the number of features.

"""

import json
import os

import numpy as np

from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import lyapunov_exponent
from DCEpy.Features.Spectral_Energy import spectral_energy


# Hjorth complexity and complexity

def first_order_diff(X):
    """
    :param X: time series with size (number of samples,)
    :return: a list containing first order difference of a time series, with size equal to number of samples-1
    """
    diffs = []

    for i in xrange(1, len(X)):
        diffs.append(float(X[i])- X[i - 1])

    return diffs



def hjorth(X):
    """
    :param X: signal array
    :return: vector containing the mobility and complexity with shape (2,)

    """
    n = len(X)

    diffs=first_order_diff(X)
    diffs.insert(0, X[0])  # pad the first difference
    diffs = np.array(diffs)

    M2 = float(sum(diffs ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0;
    for i in xrange(1, len(diffs)):
        M4 += (diffs[i] - diffs[i - 1]) ** 2
    M4 = M4 / n

    # return np.sqrt(M2 / TP)
    return np.sqrt(float(M4) * TP / M2 / M2)

    # return np.hstack((np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)))


# lyapunov exponent

def lyapunov_exp(X,axis=-1):

    """
    :param X: signal array of a window
    :return: numpy array containing the lyapunov exponent of the window
    """
    return lyapunov_exponent(X,axis=axis)

# def lyapunov_exponents(patient_id,seizure_file_name, X_test_raw, X_test_one_channel, win_len,win_overlap):
#
#     """
#     :param X: signal array with shape (number of samples, number of channel=1)
#     :return: 1D vector with shape (number of windows, 1)
#     """
#
#     # extract windows and compute features by window
#     n, p = X_test_one_channel.shape
#     n_windows = n / (win_len - win_overlap) - 1  # evaluates to floor( n / (L - O ) - 1 since ints
#     X_feat = []
#
#     for j in range(win_len, X_test_one_channel.shape[0], win_len - win_overlap):
#         window = X_test_one_channel[(j - win_len):j, :]  # select window
#         f = [lyapunov_exponent(window)]  # extract lyapunov exponent from the window
#         X_feat.append(f)
#
#     return np.array(X_feat)




## gardner's: mean curve length CL, mean energy E, mean Teager energy TE

def energy_features_one_window(X_test_one_channel):
    """
    :param X:  signal array with shape (number of samples, number of channels=1). Number of samples usually equals the number of samples within a window.
    :param order: some useless stuff for SOM, ignore
    :return Y: 1D vector with shape (number of window=1, number of features)
    """

    # get data dimensions
    n,p = X_test_one_channel.shape   # n is the number of samples within X, p is the number of channels

    # compute energy based statistics
    CL = np.log10(np.mean(np.abs(np.diff(X_test_one_channel, axis=0)), axis=0)) # mean curve length
    E = np.log10(np.mean(X_test_one_channel ** 2, axis=0)) # mean energy
    tmp = X_test_one_channel[1:n-1,:] ** 2 -  X_test_one_channel[0:n-2,:] * X_test_one_channel[2:n,:] # Teager energy
    TE = np.log10(np.mean(np.abs(tmp), axis=0)) # ABSOLUTE Teager energy


    Y = np.hstack((CL, E, TE))

    return Y


def energy_features(patient_id,seizure_file_name, X_test_raw, X_test_one_channel, win_len,
                                           win_overlap):
    """

    :param X: data array with shape (number of windows, )
    :param feat_lst: a list of features names whose corresponding functions compute only features from a single window.
    :param feat_func: a dictionary where all the features names are mapped to feature functions.
    :return:  : feature matrix with shape (number of windows, number of windows)

    """

    # extract windows and compute features by window
    n, p = X_test_one_channel.shape
    n_windows = n/(win_len - win_overlap) - 1  # evaluates to floor( n / (L - O ) - 1 since ints
    X_feat=[]

    for j in range(win_len, X_test_one_channel.shape[0], win_len - win_overlap):
        window = X_test_one_channel[(j - win_len):j, :]  # select window
        f = energy_features_one_window(window)  # extract energy statistics
        X_feat.append(f)

    return np.array(X_feat)





## Burns, NEEDS THE FILE FOR PATIENT 23!!!!
def burns_features(patient_id, seizure_file_path,*args):

    """
    Reads in pre-computed burns features for patient 41(NEEEEDDD PATIENT 23!!!!).
    :parameters:  seizure_file: the name of the seizure file, string

    :return: 2D matrix with shape (number of windows, number of channels(burns features))

    """
    burns_data_path = os.path.dirname(os.path.dirname(os.path.dirname(seizure_file_path)))
    burns_evc_path = os.path.join(burns_data_path, 'DCEpy','Features','BurnsStudy','StoredFeatures',patient_id)
    file_of_interest = os.path.basename(seizure_file_path)
    file_of_interest = file_of_interest[:-4]
    file_of_interest = file_of_interest + 'eigenvec_centrality.json'
    file_path = os.path.join(burns_evc_path,file_of_interest)

    # seizure_id = seizure_file_name[:-4]
    # file_name=burns_evc_path+'*.json'
    # to_data= os.getcwd()
    # print to_data
    # burns_data_path = os.path.join(to_data,'StoredFeatures',file_name)

    # print data_path

    with open(file_path) as data_file:
        raw_data = json.load(data_file)

    num_windows = len(raw_data)
    num_channels = len(raw_data["0"])

    data = np.empty([num_windows, num_channels])
    for k in xrange(num_windows):
        for j in xrange(num_channels):
            data[k, j] = raw_data["%d" % k]["%d" % j]
    win_array = np.array(data)  # just to be sure...

    # print win_array.shape

    return win_array


# print burns_features("CA00100E_1-1_03oct2010_01_03_05_Awake")


## Zhang's
def spectral_energy_features(patient_id, seizure_file_name, raw_data, *args):

    """
    Computes spectral energy on given edf file.

    :parameter: patient_id:
                seizure_file:

    :returns:  spectral features: 2D matrix with shape (number of windows, number of features/channels)

    """

    selected_features = spectral_energy.feature_selection(raw_data)

    return selected_features


# for SOM, energy and lya...
def comb_1_one_window(X):
    x1=energy_features_one_window(X)
    x2=lyapunov_exp(X,axis=0)

    # print "shape 1", x1.shape
    # print "shape 2", x2.shape
    return np.hstack((x1,x2))

