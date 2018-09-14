'''
Calculates the r-statistic, mentioned in Burns' paper,to see which frequency bands are changing the most.
'''
from __future__ import print_function
import scipy.signal
from scipy.integrate import trapz
from scipy import stats
import numpy as np
import math
from DCEpy.Features.Preprocessing.preprocess import lpf, notch, normalize


def calc_rstat(x_ict,x_inter,fs,bands,window_len,window_interval,mode=1):
    """
    Choose the frequency band of interest by calculating the band with the maximum R statistic
    (the ratio of power spectra in ictal state to interictal state).

    Inputs:
    -------
    x_ict = ictal time-series data: nxN array, n = number of samples, N = number of channels
    x_inter = interictal time-series data: nxN array, n = number of samples, N = number of channels
    fs = sampling rate, in Hz: int,float
    mode = whether zero-phase filtering is used: 0 = no, 1 = yes
    bands = the frequency bands of interest: Mx2 array, M = number of bands

    Output:
    -------
    band = selected frequency band for channel, Hz: 1x2 array
    """
    band_ch = np.zeros([x_ict.shape[1],2])
    for ch in range(x_ict.shape[1]):
        band_ch[ch],_,_,_,_ = calc_band(x_ict[:,ch],x_inter[:,ch],fs,bands,window_len,window_interval,mode)
    band,count = stats.mode(band_ch)
    return band[0]


def preprocess_sig(window,fs,mode):
    window = lpf(window,fs,120.0,6,mode)
    window = notch(window,56.0,64.0,fs,mode)
    window = normalize(window)
    return window

def sliding_window_freq_stuff(data,window_len,window_interval,fs,
                              seg=1000,noverlap=750,scaling='spectrum',mode=0):
    '''
    Compute power spectrum vectors, frequencies, on a sliding window
    :param data:            data to evaluate power spectrum on. Nx1 array.
    :param window_len:      the window length, an int, in nsamples
    :param window_interval: the step between, an int, in nsamples
    :param fs:              sampling frequency of the data
    :param seg:             parameter used by welch function
    :param noverlap:        parameter used by welch fxn
    :param scaling:         parameter used by welch fxn
    :return:
    '''


    if np.shape(data)[0]==1:
        data = data.T

    data_len = np.size(data,axis=0)

    # initialize vectors in sliding window. Can return t if you so choose.
    t = np.arange(window_len,step=window_interval,stop=data_len)
    Pxx_vect =[0]*len(t)
    f = None
    for index,end_time in enumerate(t):
        #get the signal window and preprocess it.
        window = data[end_time-window_len:end_time,:]
        #process the signal window
        window = preprocess_sig(window,fs,mode)


        f, Pxx = scipy.signal.welch(np.squeeze(window), fs,
                                    nperseg=seg, noverlap=noverlap, scaling=scaling,axis=0)
        Pxx_vect[index]= Pxx


    f,Pxx_vect = map(np.asarray,(f,Pxx_vect))
    return f,Pxx_vect

def calc_band(x_ict,x_inter,fs,bands,window_len,window_interval,mode):
    """
    Choose the frequency band of interest for a certain channel by calculating the band with the maximum R statistic
    (the ratio of power spectra in ictal state to interictal state).

    Inputs:
    -------
    x_ict = 1 channel of preprocessed ictal time-series data: nx1 array
    x_inter = 1 channel of preprocessed interictal time-series data: nx1 array
    fs = sampling rate, in Hz: int,float
    mode = whether zero-phase filtering is used: 0 = no, 1 = yes
    bands = the frequency bands of interest: Mx2 numpy array, M = number of bands

    Output:
    -------
    band = selected frequency band for channel, Hz: 1x2 array
    """

    # Preprocessing and compute power spectrum for ictal and interictal data
    x_ict,x_inter = np.atleast_2d(x_ict,x_inter)
    #x_inter = np.atleast_2d(x_inter)



    seg = 1000
    f_inter,Pxx_inter = sliding_window_freq_stuff(x_inter,window_len,window_interval,
                                                  fs,seg,mode)

    f_ict,Pxx_ict = sliding_window_freq_stuff(x_ict,window_len,window_interval,
                                                  fs,seg,mode)
    # TODO: fix logic to work with any frequencies, integral function.
    # Frequency bands of interest

    Pxx_ict = np.mean(Pxx_ict,axis=0)
    Pxx_inter = np.mean(Pxx_inter,axis=0)
    # Average the power spectra over the frequency bands of interest
    # and calculate r statistic
    rstat = np.zeros(bands.shape[0])
    for b,band in enumerate(bands):
        b_ict = np.mean(Pxx_ict[band[0]:band[1]])
        b_inter = np.mean(Pxx_inter[band[0]:band[1]])
        rstat[b] = b_ict/b_inter
    #print(rstat)

    # Choose band with maximum r statistic
    I = np.argmax(rstat)
    band = bands[I]


    return band, f_ict, Pxx_ict, f_inter, Pxx_inter


def mod_trapz(y,x=None,a=None,b=None):
    '''
    Approximate the integral the signal y(x) from a to b using the trapezoidal method.
    Assumes x is monotonically increasing..
    :param y: a vector of y vals
    :param x: a vector of x vals
    :param a:
    :param b:
    :return:
    '''
    if x:
        min_x = np.min(x)
        max_x = np.max(x)

    else:
        min_x = 0
        max_x = len(y)


    if ((a is None) or (b is None)):
        a,b = min_x,max_x

    if a<min_x or b>max_x:
        raise ValueError('Cannot integrate outside of signal bounds!')

    min_check, max_check =(a<=x).nonzero(),(b>=x).nonzero()

    min_index = min_check[-1]
    max_index = max_check[1]

    if x[min_index] !=a:
        min_index -=1
    if x[max_index] !=b:
        max_index +=1

    y= y[min_index:max_index+1]
    if x:
        dx = np.diff(x[min_index:max_index+1])
    else:
        dx = np.ones_like(y[min_index:max_index+1])
    pass



