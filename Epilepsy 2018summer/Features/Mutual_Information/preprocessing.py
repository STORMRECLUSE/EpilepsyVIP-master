from DataInterfacing.edfread import edfread
import matplotlib.pyplot as plt
from Preprocessing.preprocess import notch, high_pass, butter_lowpass_filter
import numpy as np
import scipy
from scipy import signal
import mne
from sklearn.preprocessing import minmax_scale


# notch filter data
def notch_filter_data(data, highest_frequency):
    """Notch filter data.
    Data should already be low passed with Cf = 300Hz using a butterworth filter."""

    power_harmonics = [60, 120, 180, 240, 300, 360, 420]
    fs = 1000

    # remove around the power harmonics
    remove_harmonics = list(map(lambda x: x if x < highest_frequency else None, power_harmonics))
    remove_harmonics = [x for x in remove_harmonics if x is not None]
    if len(remove_harmonics) != 0:
        print "harmonics to remove:",remove_harmonics
        for h in remove_harmonics:
                data = notch(data, h-1, h+1, fs)

    cf = 0.5  # 0.5Hz cutoff filter to ensure dc is removed
    data = high_pass(data, cf, fs, axis = 0)
    return data


"""
Get all frequency bands.
"""
def get_freq_bands():
    return ["delta","theta", "alpha", "beta", "gamma", "high", "very high"], \
           [[1,4],[4, 8], [8, 13], [13, 32], [30, 80], [60, 100], [100, 200]]
