from __future__ import print_function

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

from DataInterfacing.edfread import edfread



# high pass filter
def high_pass(data,cf,fs,axis = 0,order=5,mode=False):

    """
    high pass filtering on given data.

    :param data: single channel time array
    :param fs: sampling frequency
    :param cf: cutoff frequency
    :param order:
    :param mode:
    :return:
    """

    def butter_highpass_filter(data, cf, fs, order=5, mode=False):

        def butter_high_pass(cf, fs, order=5):
            nyq = fs / 2.
            normalized_cf = cf / nyq
            b, a = butter(order, normalized_cf, btype='highpass')
            return b,a

        b, a = butter_high_pass(cf, fs, order=order)
        if mode:
            y_lpf = filtfilt(b, a, data, axis=axis)
        else:
            y_lpf = lfilter(b, a, data, axis=axis)
        return y_lpf


    return butter_highpass_filter(data,cf,fs,order=order,mode=mode)



# creates lowpass Butterworth filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


# applies filter to data using filtfilt so that it is zero-phase
def butter_lowpass_filter(data, cutoff, fs, order=5, mode=False):
    b, a = butter_lowpass(cutoff, fs, order=order)
    if mode: #mode is nonzero: zero-phase
        y_lpf = filtfilt(b, a, data, axis=0)
    else: #mode is false: causal filter
        y_lpf= lfilter(b, a, data, axis=0)
    return y_lpf



def notch(data,flow,fhigh,fs,mode=0):
    """
    Applies a notch filter to the data
    Inputs:
    -------
    data = data to be filtered: nxN numpy array,
        N = number of channels, n = number of samples
    flow = lower-bound cutoff frequency in Hz: float
    fhigh = upper-bound cutoff frequency in Hz: float
    fs = sampling rate of data in Hz: float
    mode = whether zero-phase filtering is used: 0 = no, 1 = yes
    Outputs:
    --------
    y_notch = filtered data: Nxn numpy arrat,
        N = number of channels, n = number of samples
    """


    # Creates band stop filter
    bp_stop_hz = np.array([flow,fhigh])
    b, a = butter(2,bp_stop_hz/(fs/2.0),'bandstop')


    #filter all channels with mode selected
    if mode == 1: # zero-phase
        y_notch = filtfilt(b, a, data, axis=0)
    if mode == 0: # phase-altering (causal)
        y_notch = lfilter(b, a, data, axis=0)


    return y_notch





