from __future__ import print_function

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

from DCEpy.General.DataInterfacing.edfread import edfread


def box_filter(data,filter_len,axis=0):
    '''
    Performs a box filter on the input data
    :param filter_len: length of the filter (adapt_rate)
    :param axis: along which axis?
    :return:
    '''
    return lfilter(np.ones(filter_len)/filter_len,[1],data,axis=axis)

def easy_seizure_preprocessing(x,fs, axis = 0,order = 5):

    """
    The preprocessing function:
    (1) applies 49-51Hz band-reject filter to remove the power line noise
    (2) applies .5Hz cutoff high pass filter to remove the dc component

    Your one-stop-shop for preprocessing!

    :param x: single channel time series with shape (number of samples,)
    :param fs: sampling frequency
    :return: preprocessed signal

    """
   # empirical maximum
    reject=(59,61)          # band-reject frequency range (59Hz,61Hz) to remove power line noise
    x = bandreject(x, reject, fs,axis = axis, order=order)
    cf = .5                 # 0.5Hz cutoff filter to remove dc
    x= high_pass(x,cf,fs,axis = axis,order=order)

    return x


# band reject filter
def bandreject(data,band,fs,axis = 0,order=5,mode=False):

    """
    Band reject filtering on given data.

    :param data: single channel time series
    :param fs: sampling frequency
    :param band: reject range, tuple/sequence with 2 elements
    :param order:
    :param mode:
    :return:
    """

    def butter_bandreject_filter(data, band, fs, order=5, mode=False):

        def butter_bandreject(band, fs, order=5):
            nyq = fs / 2.
            normalized_band = np.asarray(band) / nyq  # normalized band for digital filters
            b, a = butter(order, normalized_band, btype='bandstop')
            return b, a

        b, a = butter_bandreject(band, fs, order=order)

        if mode:
            y_lpf = filtfilt(b, a, data, axis=axis)
        else:
            y_lpf = lfilter(b, a, data, axis=axis)
        return y_lpf

    return butter_bandreject_filter(data,band,fs,order=order,mode=mode)


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

def lpf(data,fs,fc,order,mode=False):

    """
    Applies a low pass filter to the input data

    Inputs:
    -------
    data = the data to be filtered: nxN numpy matrix,
        N = number of channels, n = number of samples
    order = order of the low pass filter: int
    fs =  sample rate of the data in Hz: int
    fc = cutoff frequency of the filter in Hz: int
    mode = whether zero-phase filtering is used: False = no, True = yes

    Outputs:
    --------
    y_lpf = filtered data: nxN numpy matrix,
        N = number of channels, n = number of samples
    """

    # Filter the data
    y_lpf = butter_lowpass_filter(data, fc, fs, order, mode)
    return y_lpf

def butter_bandpass(band,fs,order=5):
    nyq = fs/2.
    normalized_band = np.asarray(band)/nyq
    b,a = butter(order,normalized_band,btype='bandpass')
    return b,a

def butter_bandpass_filter(data,band,fs,order=5,mode=False):
    b, a = butter_bandpass(band, fs, order=order)
    if mode:
        y_lpf = filtfilt(b, a, data, axis=0)
    else:
        y_lpf= lfilter(b, a, data, axis=0)
    return y_lpf

def bandpass(data,fs,band,order=5,mode=False):

    return butter_bandpass_filter(data,band,fs,order=order,mode=mode)



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


def normalize(data):
    """
    Normalizes the data to mean and scales the data by standard deviation
    Inputs:
    -------
    data = data to be normalized: Nxn numpy array,
        N = number of channels, n = number of samples
    Outputs:
    --------
    y_normal = normalized data Nxn numpy array,
        N = number of channels, n = number of samples
    """

    # subtract mean and divide by standard deviation to normalize and scale
    mean = np.mean(data,axis=0)
    std =  np.std(data, axis=0)
    y_normal = (data - mean)/std
    return y_normal


def downsample(data,fs,dsrate):
    """
    Downsamples data
    Inputs:
    -------
    data = the data to be downsampled: Nxn numpy array,
        N = number of channels
        n = number of samples
    fs = the sampling rate of data in Hz: int
    dsrate = the desired downsampled sampling rate in Hz: int
    Outputs:
    --------
    y_ds = the downsampled data
    """
    data = data.T

    interval = fs/dsrate
    N = data.shape[0]
    n = data.shape[1]
    y_ds = np.zeros((N,n/interval))

    # downsample the data by taking sample every interval
    for ch in range(0,N):
        j = 0
        for i in range(0,n):
            if i % interval == 0:
                y_ds[ch,j] = data[ch,i]
                j += 1

    y_ds = y_ds.T
    return y_ds

# from DCEpy.Features.BurnsStudy.rstat_42 import calc_rstat
# class RstatPreprocessor(object):
#     '''
#     A class that computes the r-statistic on a file
#     '''
#     def __init__(self,inter_file,ictal_file, seizure_times,fs=1000):
#         self.inter_file,self.ictal_file = inter_file,ictal_file
#         self.seizure_times = seizure_times
#         self.fs = fs
#
#
#     def _get_windows(self,good_channels = None,bad_channels=None,
#                         window_len = 20000):
#         all_ictal_data,_,_ = edfread(self.ictal_file,bad_channels=bad_channels,
#                                      good_channels=good_channels)
#
#         ictal_window_end = np.random.randint(self.fs*self.seizure_times[0] + window_len,
#                                               self.fs*self.seizure_times[1])
#
#         self.ictal_window = all_ictal_data[ictal_window_end-window_len:ictal_window_end,:]
#
#         all_inter_data,_,_ = edfread(self.ictal_file,bad_channels=bad_channels,
#                                      good_channels=good_channels)
#
#         inter_window_end = np.random.randint(window_len,np.size(all_inter_data,axis=0))
#
#         self.inter_window = all_inter_data[inter_window_end-window_len:inter_window_end,:]
#
#     def _compute_rstat(self,bands,window_len=2500,window_interval=1500,mode=0):
#         self.good_band = calc_rstat(self.ictal_window,self.inter_window,self.fs,bands,
#                    window_len=window_len,
#                    window_interval=window_interval,mode=mode)
#
#     def prepare_rstat(self,bands,good_channels=None,bad_channels=None,window_len=20000,
#                       rstat_win_len = 2500,rstat_win_interval=1500,mode=0):
#         self._get_windows(good_channels=good_channels,bad_channels=bad_channels,window_len=window_len)
#         self._compute_rstat(bands,window_len=rstat_win_len,window_interval=rstat_win_interval,mode=mode)
#
#     def optimal_bandpass(self,data,order=5,mode=0):
#         return bandpass(data,self.fs,self.good_band,order,mode)




