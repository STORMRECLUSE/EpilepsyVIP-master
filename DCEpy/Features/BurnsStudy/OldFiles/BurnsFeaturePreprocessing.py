import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import signal
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from DCEpy.Features.Spectral_Energy import spectral_energy
import itertools

# from edfread import edfread
import array
import random
#import feature functions
# from feature_functions import burns_features
# from feature_functions import energy_features
# from feature_functions import spectral_energy_features
# from AnalyzePatient import analyze_patient_raw
# from AnalyzePatient import analyze_patient_denoise
# from AnalyzePatient import analyze_patient_noise
from scipy.signal import iirfilter
from scipy.signal import lfilter
from scipy.signal import welch


'''
notch_filter applies a notch filter on the raw segmented iEEG data.

inputs:
X = raw segmented iEEG
fs = sampling frequency (1000)
band = bandwidth (Burns = 4)
filter_freq = notch frequency (Burns = 60)
ripple = maximum ripple (dB)
order = order of filter (usually 2 or 3)
filter_type = 'butter', 'ellipses', etc.

outputs:
notch-filtered segmented iEEG data as specified by Burns
'''
def notch_filter(X, fs, band, filter_freq, ripple, order, filter_type):

    # calculate useful variables
    nyq = fs/2.0
    low = filter_freq - band/2.0
    high = filter_freq + band/2.0
    low = low/nyq
    high = high/nyq

    # apply built-in filters
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    notchfiltered_data = lfilter(b, a, X)

    return notchfiltered_data


'''
preprocessing preprocesses the segmented raw iEEG data by:
1) filtering with notch filter
2) normalizing by subtracting mean and dividing by STD
3) obtaining spectrogram of each window by Welch's method
4) extracting power spectrum in specified frequency bands
5) select the frequency band with the highest r-spectrum ratio
'''
def preprocessing(X, num_channels): # X is the segmented raw iEEG data: 2.5 s windows with 1.5 s overlap (sampling freq = 1000)

    notchfiltered_data = notch_filter(X, 1000, 4, 60, 0.01, 2, 'butter') # notch filter

    notchmean = np.mean(notchfiltered_data) # calculate mean
    notchSTD = np.std(notchfiltered_data) # calculate standard deviation
    normalized_data = np.subtract(notchfiltered_data, notchmean) # subtract average
    normalized_data = np.divide(normalized_data, notchSTD) # divide by standard deviation

    power_spectrum = welch(normalized_data, fs=1000, window='hanning', nperseg=1000, noverlap=750) # spectrogram using Welch method

    power_spectrum = list(power_spectrum) # convert to list

    spectral_power_densities = [] # initialize storage list
    frequencies = [(1, 4), (5, 8), (9, 13), (14, 25), (25, 90), (100, 200)]
    for i in range(0, 6):  # run forloop to obtain spectral power densities in the 6 specified frequency bands
        freq_indices = np.where((power_spectrum[0] > frequencies[i][0]) & (power_spectrum[0] < frequencies[i][1]))[0]  # determine indices of frequency bands
        selected_spectro = power_spectrum[1][freq_indices, :]  # select the spectrogram values that correspond to those indices
        spectral_power_densities[i, :] = np.sum(selected_spectro, 0)  # store

    return spectral_power_densities

'''
freq_band_select computes the ratio of the average interictal spectra to ictal spectra over their respective windows
and determines the ideal frequency band
'''
def freq_band_select(X, num_channels, num_windows):

    spectral_power_densities = []
    for num in range(num_windows): # preprocess and obtain power spectrum of all windows
        spectral_power_densities_per_window = preprocessing(X, num_channels)
        spectral_power_densities[num, :] = spectral_power_densities_per_window

    # identify which windows are interictal, which are ictal
    interictal_window_indices = 1
    ictal_window_indices = 2


    average_interictal_spectral_power_densities = []
    average_ictal_spectral_power_densities = []
    for i in range(0, 6):
        sum_interictal = np.sum(spectral_power_densities[interictal_window_indices, i], 0)
        average_interictal = np.divide(sum_interictal, num_windows)
        average_interictal_spectral_power_densities[i, :] = average_interictal

        sum_ictal = np.sum(spectral_power_densities[ictal_window_indices, i], 0)
        average_ictal = np.divide(sum_ictal, num_windows)
        average_ictal_spectral_power_densities[i, :] = average_ictal

    ratio = np.divide(average_interictal_spectral_power_densities, average_ictal_spectral_power_densities)
    frequencies = [(1, 4), (5, 8), (9, 13), (14, 25), (25, 90), (100, 200)]
    best_freq_band_index = np.where(np.max(ratio))[1]
    best_freq_band = frequencies[best_freq_band_index]

    return best_freq_band



if __name__ == '__main__':

    # parameters -- sampling data
    win_len = 2.5  # in seconds
    win_overlap = 1.5  # in seconds
    f_s = float(1e3)  # sampling frequency
    win_len = int(win_len * f_s)
    win_overlap = int(win_overlap * f_s)

    # cut raw data into windows
    # num_channels = X.shape[1]
    # num_windows =

    # obtain training ictal or interictal windows here

    # purpose of preprocessing is selecting ideal frequency band
    # best_freq_band = freq_band_select(X, num_channels, num_windows)
