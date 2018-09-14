import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy
from scipy.ndimage.filters import gaussian_filter
import pylab as pl
from pylab import norm
from numpy import linalg
from sklearn import svm
from sklearn.decomposition import pca
from pykalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import math
from sklearn.cluster import KMeans
# how to import scipy.ndimage.filter

from DCEpy.Features.BurnsStudy.eig_centrality import eig_centrality
from DCEpy.General.DataInterfacing.edfread import edfread

'''
divide input signal into segments of window size 4 s with 2 s overlap.
Input signal is a sxn array where s is # of samples and n is # of channels
fs: sampling frequency=1000
nperseg: number of samples per window segment length=4s
noverlap: number of samples per window segment overlap=2s
'''

'''
PSD: power spectral density estimation calculated by:
1) Fast Fourier Transform (win_len*f_s=1000*4=4000-point real FFT but used power of 2 to gain efficiency)
2) compute magnitude square of FFT coefficients
Input: raw iEEG signal x
Output: PSD coefficients
FFT_coeff: FFT coefficients
PSD_coeff: Power spectral density estimate
'''

def preprocessing(x, n):
    #x = np.nan_to_num(x)
    #x = x+np.random.randn(x.size)
    #x = x/pl.norm(x)
    # pass segmented iEEG signal into spectrogram to obtain FFT coefficients
    print 'Preprocessing channel', n
    print 'Data for channel',n,'=',x
    spectro = scipy.signal.spectrogram(x,fs=1000,nperseg=4096,noverlap=2048)
    # spectro is a list of 3 nparrays: 0-frequencies 1-times 2-spectrogram (PSD Coefficients)
    spectro=list(spectro)
    num_windows=spectro[1].shape[0] # find number of window segments by counting how many times there are
    frequencies=[(4,8),(8,13),(13,30),(30,50),(50,70),(70,90),(90,110),(110,128)] # initialize frequencies tuple
    PSD_coeff=np.empty([8,num_windows])
    for i in range(0,8): # run forloop to calculate FFT coefficients in the 8 specified frequency bands
        freq_indices=np.where((spectro[0]>frequencies[i][0]) & (spectro[0]<frequencies[i][1]))[0] # determine indices of frequency bands
        selected_spectro=spectro[2][freq_indices,:] # select the spectrogram values that correspond to those indices
        FFT_coeff=np.sum(selected_spectro**2,0) # sum up all coefficients in column
        PSD_coeff[i,:]=FFT_coeff # store
    return PSD_coeff, num_windows

'''
Feature set contains 3 different feature types. For each electrode, there will be 44 features:
1) 8 absolute spectral powers,
2) 8 relative spectral powers,
3) 28 spectral power ratios
'''

'''
compute absolute spectral power in each frequency band i for each window
abs_spec: absolute spectral power
        computed by taking the log of the sum of the PSD coefficients within a window segment within a frequency band.
PSD_coeff: PSD coefficient
PSD is the distribution of the signal's total average power over frequency
i: frequency band (8 total)
Input: ixnxp matrix that displays PSD coefficients p of each time window n in each frequency band i.
Output: time series ixn where ixn-th element is spectral power of input signal in the n-th window segment in frequency band i
'''

def absolute_spectral_density(PSD_coeff):
# extract absolute spectral density features
    # abs_spec_density is a ixn array of the log of the sums of the p
    abs_spec_density=np.log(PSD_coeff)
    return abs_spec_density

'''
compute relative spectral power in each frequency band i for each window:
rel_spec: relative spectral power
        computed by dividing the sum of the PSD coefficients (absolute spectral power) within a frequency band by
        the total sum of the PSD coefficients in all frequency bands (total power of the signal).
Input: ixnxp matrix that displays PSD coefficients p of each time window n in each frequency band i.
Output: time series ixl where l-th element is relative spectral power of input signal in the l-th window segment with frequency band i.
'''

def relative_spectral_power(PSD_coeff):
# extract relative spectral power features
    # total_power is a ix1 array with sums of all PSD coefficients in each row.
    total_power=np.sum(PSD_coeff,0)
    # rel_spec found by taking the log of each column of PSD_coeff_sums by row array total_power.
    rel_spec=np.log(PSD_coeff/total_power)
    return rel_spec

'''
compute spectral power ratio in each frequency band i for each window:
spec_ratio: spectral power ratio
        indicate change in power distribution between two frequency bands when EEG transfers from interictal to ictal and vv.
        computed by subtracting the absolute spectral ratios of two frequency bands in an l-th window segment.
        *There are 8 frequency bands for one channel, and since the ratio is a difference between two frequency band
        spectral power values, there can be 28 different ratios.*
i: one frequency band
j: the other frequency band
Input: abs_spec_density
Output: 28xnumber of window segments array that contains the power ratios
'''

def power_spectral_ratio(abs_spec_density,num_windows):
# extract power spectral ratios
    num_rows=abs_spec_density.shape[0] # 7 frequency bands because band 8 is not in ratio with any other band.
    # combinations: B0/B1, B0/B2, B0/B3, ... B0/B7, B1/B2, B1/B3, ... B1/B7, B2/B3, B2/B4, ... B7/B8 {8C2=28}
    power_spec_ratio=np.zeros([28,num_windows]) # initialize output array
    k=0 # forloop indexing variable
    for i in range(num_rows): # loop through the first element of ratio
        for j in range(i+1,num_rows): # loop through the second element of ratio
            # calculation uses subtract sign because log(a)-log(b) is log(a/b), which is a logged ratio
            spec_ratio=abs_spec_density[i,:]-abs_spec_density[j,:] # first element / second element to get spectral ratios
            power_spec_ratio[k,:]=spec_ratio # store spectral ratios
            k+=1 # update indexing variable
    return power_spec_ratio

'''
Postprocessing: Kalman filter ideally
Using Gaussian filter for now for simplicity: Can try replacing with Kalman filter later and see difference.
If not much difference then keep at Gaussian for speed and simplicity.
scipy.ndimage.filters.gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
Input: 44 features
Output: array of 44xnum_windows of filtered features.
Cross Validate STD
'''

def postprocessing(abs_spec_density,rel_spec,power_spec_ratio,n):
# create filter
# Kalman_filter=KalmanFilter(dim_x=,dim_z=num_windows)
    print 'Postprocessing features for channel',n,'...'
    filtered_abs = scipy.ndimage.filters.gaussian_filter(abs_spec_density,0.1) # check suitable variables for filter (STD)
    filtered_rel = scipy.ndimage.filters.gaussian_filter(rel_spec,0.1) # check STD
    filtered_ratio = scipy.ndimage.filters.gaussian_filter(power_spec_ratio,0.1) # check STD
# if all 3 feature types have same STD can combine before passing through filter.
    # combine 3 types of features into one big feature vector with 8+8+28=44 rows and num_windows columns
    filtered_features = np.concatenate((filtered_abs,filtered_rel,filtered_ratio),0)
    return filtered_features

'''
Feature selection:
1) Choose feature basis using linear separability criteria
2) Choose best channel using scatter matrix method
3) Choose best features using Branch and Bound method
'''

'''
Feature Basis Selection: selection of R linearly independent features according to the linear separability criteria.
R is determined by eigenvalue analysis of covariance matrix.
1) Determine maximum number of features that are linearly independent of each other using
    eigenvalue analysis of covariance matrix of features.
    * count how many significantly large eigenvalues there are, and that count is R *
2) Select R features using class separability method in a "greedy manner". Start with an empty feature set, and
    keep adding more features to maximize J until R features are selected. Repeat for each electrode.
S is the feature set of dimension 1xR
R is the maximum number of features that are linearly independent of each other.
'''
def myPCA(filtered_features):
    filtered_features /= np.var(filtered_features,1).reshape((-1,1))
    cov_matrix = np.cov(filtered_features)
    eigvals,eigvecs = np.linalg.eigh(cov_matrix)   # return eigenvalues in ascending order and their eigenvectors
    eigvals /= sum(eigvals)  # normalize eigenvalues by dividing each value by the total sum
    eigvals = np.flipud(eigvals)  # flip from ascending to descending order
    eigvals = np.cumsum(eigvals) # cumulative sum
    first_index = np.where(eigvals>0.925)[0][0]   # find first index above 92.5%
    selected_eigvecs = eigvecs[:,-first_index:] # select eigenvectors corresponding to first index
    selected_features = np.dot(np.transpose(selected_eigvecs), filtered_features)

    return selected_features

def feature_selection(x):
# select suboptimal features for each channel using linear separability criteria.
    num_channels = x.shape[1] # count number of channels
    features_all_channels=[None]*num_channels
    for n in range(num_channels): # for all channels
        PSD_coeff,num_windows=preprocessing(x[:,n],n+1) # preprocess
        print 'Extracting features for channel', n+1, '...'
        abs_spec_density=absolute_spectral_density(PSD_coeff) # extract absolute spectral density
        rel_spec=relative_spectral_power(PSD_coeff) # extract relative spectral ratio
        power_spec_ratio=power_spectral_ratio(abs_spec_density,num_windows) # extract power spectral ratio
        filtered_features=postprocessing(abs_spec_density,rel_spec,power_spec_ratio,n+1) # postprocess
        features_all_channels[n]=filtered_features
    features_all_channels=np.vstack(features_all_channels)
    print 'Selecting optimal features from all channels...'
    selected_features = myPCA(features_all_channels)

    return selected_features

def clustering(selected_features):
    print 'Clustering optimal features...'
    clusters = KMeans(4) # make cluster class
    clusters.fit(np.transpose(selected_features)) # train model
    labels = clusters.predict(np.transpose(selected_features)) # use trained model to label clusters
    plt.plot(labels)
    plt.show()
    return labels



#Classifier: Linear SVM


#sklearn.svm.LinearSVC

if __name__ == '__main__':
    include_awake = False # include only ictal files
    include_asleep = False # include only ictal files
    patients=['TS039'] # patient files
    long_interictal = [False]
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) # direct to EpilepsyVIP
    data_path = os.path.join(to_data, 'data') # direct to data folder

    for i, patient_id in enumerate(patients):
        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)
        print "---------------------------Analyzing patient ", patient_id, "----------------------------\n"
        # if data path does not work out
        if not os.path.isdir(data_path):
            sys.exit('Error: Specified data path does not exist')
        # get pickle file
        p_file = os.path.join(p_data_path, 'patient_pickle.txt')
        # open pickle file and load
        with open(p_file, 'r') as pickle_file:
            print("Open Pickle: {}".format(p_file) + "...\n")
            patient_info = pickle.load(pickle_file)
        data_filenames = patient_info['seizure_data_filenames']
        seizure_times = patient_info['seizure_times']
        file_type = ['ictal'] * len(data_filenames)
        seizure_print = [True] * len(data_filenames)  # mark whether is seizure

        print 'Getting Data...'
        # read seizure file
        for i, seizure_file in enumerate(data_filenames):
            path_to_seizure = os.path.join(p_data_path,seizure_file)
            print path_to_seizure
            x, _, labels = edfread(path_to_seizure)
            # output x, raw iEEG signal
            print 'There are',x.shape[1],'channels'
            selected_features=feature_selection(x)
            labels=clustering(selected_features)