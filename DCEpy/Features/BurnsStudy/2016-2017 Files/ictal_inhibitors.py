import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef,hinge_loss,hamming_loss,jaccard_similarity_score,fbeta_score,average_precision_score
import math
import matplotlib.pyplot as plt
from scipy.signal import csd
from scipy.signal import welch
import scipy.io as sio
from scipy.stats import mode
from sklearn import svm
import networkx as nx
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
from scipy import fftpack
from scipy.signal import signaltools
from DCEpy.Features.Preprocessing.preprocess import box_filter
from copy import copy
from math import sqrt
from sklearn import metrics
from memory_profiler import profile
from sys import getsizeof

"""
find_mean()

Purpose: find the mean of a vector

Inputs:
    vector: the vector input of which the mean is desired
    length: the length of the above vector

Outputs:
    mean: the mean of the vector
"""
def find_mean(vector, length):
    mean_sum = 0
    for mean_index in range(length):
        mean_sum += vector[mean_index]
    mean = mean_sum / float(length)
    return mean

"""
create_hann_window()

Purpose: Create a Hann window for Welch spectral density.

Inputs:
   M : int number of points in the output window. If zero or less, an empty
       array is returned.
   sym : bool, optional When True (default), generates a symmetric window, for use in filter design.
       When False, generates a periodic window, for use in spectral analysis.

Output:
   w : array the window, with the maximum value normalized to 1 (though the value 1
       does not appear if `M` is even and `sym` is True).
"""
def create_hann_window(M, sym = True):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w

"""
create_decision()

Purpose: to create the binary seizure prediction decision

Inputs:
    outlier_fraction: one dimensional ndarray containing the outlier fraction in units of windows
    threshold: int indicating the threshold over which a seizure is predicted
    persistence_time: the amount of time after a seizure during which no other flag is raised
    win_len: float indicating window length
    win_overlap: float indicating window overlap

Outputs:
    decision: one dimensional ndarray containing a decision {-1: no seizure; 1: seizure} for every window

"""
def create_decision(outlier_fraction, threshold, persistence_time, win_len, win_overlap):

    # determining where the outlier fraction meets or exceeds the threshold
    raw_decision = np.sign(outlier_fraction - threshold)
    raw_decision[raw_decision==0] = 1

    # initializing the final decision
    decision = np.ones(raw_decision.size)*-1

    # finding the prediction time (persistence_time) in units of windows
    persistence_windows = float(persistence_time) / float(win_len - win_overlap)

    # determining the final decision
    for i in np.arange(0,len(raw_decision)):
        if (raw_decision[i]==1):
            decision[i] = 1
            raw_decision[i+1:i+int(persistence_windows)] = -1

    return decision

"""
window_to_samples()

Purpose: to transform an array of windows to an array of samples

Inputs:
    window_array: ndarray in units of windows
    win_len: float indicating length of window
    win_overlap: float indicating window overlap
    f_s: int indicating sampling frequency

Outputs:
    sample_array: ndarray in units of samples
"""
def window_to_samples(window_array, win_len, win_overlap, f_s):

    #finding the total number of windows
    num_windows = np.size(window_array)

    #finding the time indices corresponding to each window
    time_indices = np.arange(num_windows)*(win_len-win_overlap) + win_len

    #finding the sample inices corresponding to each time/window indice
    sample_indices = time_indices*f_s

    #filling an array in units of samples with the values taken from the appropriate window
    sample_array = np.zeros(int(max(sample_indices)))
    for i in np.arange(0,num_windows-1):
        sample_array[int(sample_indices[i]):int(sample_indices[i+1])] = window_array[i]

    return sample_array

"""
create_outlier_frac()

Purpose: to create the outlier fraction given binary outputs

Inputs:
    predicted_labels: ndarray of size (number of predictions) containing binary predictions from svm
    adapt_rate: int indicating the length of time over which to average the predicted labels and compute the outlier fraction

Outputs:
    out_frac: ndarray of size (number of predictions) containing the outlier fraction
"""
def create_outlier_frac(predicted_labels, adapt_rate):

    # finding the total number of predicted labels
    tot_obs = predicted_labels.size
    out_frac = np.zeros(tot_obs)

    # finding the mean of the adapt_rate using a window size of adapt_rate
    for i in np.arange(adapt_rate,tot_obs):
        out_frac[i] = np.mean(predicted_labels[i-adapt_rate:i])

    return out_frac

"""
find_evc()

Purpose: to find the eigenvector centrality of given graphs

Inputs:
    list_of_matrices: list of ndarrays; each entry represents a file of coherency data and is of size (number of channels x number of channels x number of windows)

Outputs:
    centrality_all_files: list of ndarrays; each entry is all eigenvector centralies for the given files and is of size (number of windows x number of channels)
"""
def find_evc(list_of_matrices):

    centrality_all_files = []

    # for each file in the input...
    for matrix in list_of_matrices:

        # initalize the centrality as num windows x num channels
        centrality = np.zeros((matrix.shape[2],matrix.shape[1]))

        # and compute the eigenvector centrality for each window
        for i in xrange(matrix.shape[2]):
            sub_matrix = matrix[:,:,i].copy()
            G = nx.Graph(sub_matrix)
            try:
                evc = nx.eigenvector_centrality(G, max_iter=500)
                #TODO: change this if you want, such that order stays the same. Not urgent.
                centrality[i,:] = np.asarray(evc.values())
            except:
                centrality[i,:] = np.ones(matrix.shape[1])/float(matrix.shape[1])
                print"Convergence failure in EVC; bad vector returned"

        centrality_all_files += [centrality]

    return centrality_all_files

"""
itemized_find_evc()

Purpose: to analyze Networkx's eigenvector_centrality() call, which uses the power iteration method to find
            the EVC for a given coherency matrix

Input:
    coherency_matrix: 2d weighted adjacency matrix (num channels x num_channels)
    max_iter: max iterations to run power-iteration method before failing (scalar int)
    tol: tolerance input for power-iteration method (scalar float)
    nstart: optional starting vector
    weight: graph edge weight attribute

Output:
    eigenvector centrality matrix (num windows x num channels)

Computations:
    1) Initialize and verify nstart:
        - n additions (OPTIONAL error check)
    2) Normalize nstart to create initial vector:
        - n divides
        - **float cast**
    3) Create convergence check:
        - 1 multiply
    4) For each iteration:
        a) Generate copy of x:
            - n assignments
        b) X = A * X_last:
            - n^2 multiplies
            - n^2 additions
        c) Calculate l2 norm of x
            - n squares (multiplies)
            - n additions
            - 1 square root
        d) Normalize x with its l2 norm
            - n divides
            - **float cast**
        e) Find l1 norm of normalized x
            - n subtractions
            - n absolute values
            - n additions
        f) Check for convergence
            - 1 less than comparison

Overall complexity: O(m*n^2)
    - m = max_iter
    - n = num_channels
"""
def itemized_find_evc(coherency_matrix, max_iter=500, tol=1.0e-6, nstart=None, weight='weight'):
    # Create graph representation of coherency_matrix (num_rows = num_cols)

    # Avoiding graphic representation for now, leaving as coherency matrix
    # Assumption: num_rows = num_cols
    num_rows = coherency_matrix.shape[0]
    num_cols = coherency_matrix.shape[1]

    num_channels = num_rows

    # Compute eigenvector centrality of that graphic repr
    if num_rows == 0 or num_cols == 0:
        print 'Cannot compute centrality for NULL graph'
        return (-1, [-1])

    # If no inital vector specified, initialize to vector of 1s
    if nstart == None:
        nstart = [1 for v in range(num_channels)]

    # Assert initial vector is not a zero-vector
    # Assumption: every element in nstart is non-negative
    nstart_sum = sum(nstart) # Would have to iterate num_channels times
    if (nstart_sum == 0):
        print 'Initial vector cannot have all zero values'
        return (-1, nstart)

    # Initial vector is normalized
    x = [val / float(nstart_sum) for val in nstart]

    nnodes = num_channels
    convergence_check = nnodes * tol

    # Iterate max_iter times or until convergence
    for it in range(max_iter):
        xlast = x
        x = np.copy(xlast)
        # X = A*X_last
        # update the vector foreach node
        for n in range(len(x)): # iterating num_channels times
            for nbr in range(len(x)): # iterating num_channels times
                # foreach neighbor of node n, update the vector based on the last iteration
                # and the weight of its edge with node n
                edge_weight = coherency_matrix[n, nbr]
                tmp = xlast[n] * edge_weight
                x[nbr] += tmp

        # Calculate the l2 norm of x
        sum_squares = 0
        for z in x: # Looping num_channels times
            z_t = z ** 2
            sum_squares += z_t
        norm = sqrt(sum_squares) or 1

        # Normalize each value in x
        x = [val / float(norm) for val in x]

        # Find the l1 norm of the normalized vector to check for convergence
        l1_norm = 0
        for n in range(len(x)): # Iterating num_channels times
            d_t = x[n] - xlast[n]
            d_t = abs(d_t)
            l1_norm += d_t

        # Check for convergence
        if l1_norm < convergence_check:
            return (0, x)
    print 'Power iteration method for finding EVC did not converge'
    return (-2, x)

"""
test_find_evc()

Purpose: wrapper around the above 2 find_evc functions

Input:
    coherency_matrix -- adjacency matrix for network of channels [np.ndarray (2d -- num_channels x num_channels)]

Output:
    centrality -- eigenvector centrality array, containing EVC for each channel [np.ndarray (1d -- 1 x num_channels)]
"""
def test_find_evc(coherency_matrix):
    run_itemized = 1
    if not run_itemized:
        # Original implementation
        G = nx.Graph(coherency_matrix)
        evc = nx.eigenvector_centrality(G, max_iter=500)
        centrality = np.asarray(evc.values())
    else:
        # Itemized implementation
        evc = itemized_find_evc(coherency_matrix=coherency_matrix, max_iter=500)

        if evc[0] == -1:
            # Input error -- should never happen
            print 'Invalid input error on test_find_evc()'
        elif evc[0] == -2:
            # Failed to converge
            print 'Convergence failure for find_evc... bad EVC vector returned'

        centrality = evc[1] # type = list
        centrality = np.asarray(centrality)

    return centrality

"""
transform_coherency()

Purpose: to normalize the input coherencies and aid in classification accuracy

Inputs:
    coherencies_list: list of ndarrays containing each file's coherency matrices; each ndarray is (number of channels x number of channels x number of windows)
    mean: ndarray (number of channels x number of channels) containing the mean for each entry
    std: ndarray (number of channels x number of channels) containing the standard deviation for each entry

Outputs:
    transformed_coherencies: list of ndarrays containing each file's normalized coherency arrays
"""
def transform_coherency(coherencies_list, mean, std):

    std[std==0] = 1
    transformed_coherencies = []

    # for each file's coherency matrices...
    for coherency_matrices_one_file in coherencies_list:

        # for each window's coherency matrix...
        num_windows = coherency_matrices_one_file.shape[2]
        for i in xrange(num_windows):
            matrix = coherency_matrices_one_file[:,:,i].copy()
            # normalize the matrix. This is done according to Burns et. al.
            matrix -= mean
            matrix = np.divide(matrix, std)
            matrix = np.divide(np.exp(matrix), 1 + np.exp(matrix))
            coherency_matrices_one_file[:,:,i] = matrix

        transformed_coherencies += [coherency_matrices_one_file]

    return transformed_coherencies

def test_transform_coherency(coherency_matrix, mean, std, num_channels):
    for row in np.arange(num_channels):
        for col in np.arange(num_channels):
            coherency_matrix[row,col] -= mean[row,col]
            coherency_matrix[row,col] = float(coherency_matrix[row,col]) / float(std[row,col])
            coherency_matrix[row,col] = (2.7182818284590452353602874713527 ** coherency_matrix[row,col])
            denominator = coherency_matrix[row,col] + 1
            coherency_matrix[row,col] = coherency_matrix[row,col] / float(denominator)
    return coherency_matrix

"""
find_normalizing_coherency()

Purpose: to find the mean and standard deviation for a given matrix to normalize the data

Inputs:
    matrix: ndarray (number of channels x number of channels x number of windows) for which to find the mean and standard deviation

Outputs:
    mean_mat: matrix of means for each entry in the first two dimensions of the input matrix
    std_mat: matrix of standard deviations for each entry in the first two dimensions of the input matrix
"""
def find_normalizing_coherency(matrix):
    # compute the mean of each entry along the third dimension
    mean_mat = np.mean(matrix, axis=2)
    # compute the standard deviation of each entry along the third dimension
    std_mat = np.std(matrix, axis=2)
    return mean_mat, std_mat

def create_hann_window(M, sym = False):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w

def welch_new(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):

    freqs, Pxx = csd_new(x, x, fs, window, nperseg, noverlap, nfft, detrend,
                     return_onesided, scaling, axis)

    return freqs, Pxx.real

def csd_new(x, y, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):

    freqs, _, Pxy = _spectral_helper_new(x, y, fs, window, nperseg, noverlap, nfft,
                                     detrend, return_onesided, scaling, axis,
                                     mode='psd')

    # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            Pxy = Pxy.mean(axis=-1)  # can be broken down, we have a mean function right?
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    return freqs, Pxy

def _spectral_helper_new(x, y, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum', axis=-1, mode='psd'):

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    axis = int(axis)

    nperseg = int(nperseg)

    if nfft is None:
        nfft = nperseg

    # Handle detrending and window functions
    # I'm not entirely sure how this works - not broken down
    if not detrend:
        def detrend_func(d):
            return d

    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return signaltools.detrend(d, type=detrend, axis=-1)

    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.rollaxis(d, -1, axis)
            d = detrend(d)
            return np.rollaxis(d, axis, len(d.shape))

    else:
        detrend_func = detrend

    win = create_hann_window(nperseg, False)


    scale = 0
    for entry_in_win in win:
        entry_in_win_multi = entry_in_win * entry_in_win
        scale = scale + entry_in_win_multi


    scale = fs * scale
    scale = 1 / scale

    sides = 'onesided'

    if nfft % 2:
        num_freqs = (nfft + 1)
        num_freqs = num_freqs//2
    else:
        num_freqs = nfft//2
        num_freqs = num_freqs + 1


    ################################
    # Following this is FFT mumbo jumbo which is mostly not broken down

    # Perform the windowed FFTs
    result = _fft_helper_new(x, win, detrend_func, nperseg, noverlap, nfft)
    result = result[..., :num_freqs]
    freqs = fftpack.fftfreq(nfft, 1/fs)[:num_freqs]

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper_new(y, win, detrend_func, nperseg, noverlap, nfft)
        result_y = result_y[..., :num_freqs]
        result = np.conjugate(result) * result_y

    elif mode == 'psd':
        result = np.conjugate(result) * result


    result *= scale
    if sides == 'onesided':
        if nfft % 2:
            result[...,1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[...,1:-1] *= 2
    if sides != 'twosided' and not nfft % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1
    # we unwrap the phase here to handle the onesided vs. twosided case
    # All imaginary parts are zero anyways

    if same_data and mode != 'complex':
        result = result.real

    # Output is going to have new last axis for window index

    result = np.rollaxis(result, -1, -2)


    t = 0
    return freqs, t, result

def _fft_helper_new(x, win, detrend_func, nperseg, noverlap, nfft):

    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Detrend each data segment individually

    result = detrend_func(result)


    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    result = fftpack.fft(result, n=nfft)

    return result

"""
construct_coherency_matrix

Purpse: to create the coherency graph for a given window of data

Inputs:
    X: ndarray (number of samples x number of channels) containing all of the iEEG data for a given window of time
    f_s: int indicating sampling frequency
    freq_band: list containing the frequency band over which to compute the coherence graph

Outputs:
    A: the coherency graph as an ndarray (number of channels x number of channels)
"""
def construct_coherency_matrix(X, f_s, freq_band):
    n,p = X.shape

    # initialize the adjacency matrix
    A = np.zeros((p,p))

    # construct adjacency matrix filled with coherency values
    for i in range(p):
        for j in range(i+1,p):

            # compute cross power spectral density for each frequency and each pair of signals
            fxy, Pxy = csd_new(X[:,i], X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)

            # approximate power spectral density at each frequency for each signal with Welch's method
            fxx, Pxx = welch_new(X[:,i], fs = f_s, nperseg = 1000, noverlap = 500)
            fyy, Pyy = welch_new(X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)

            # find the power in each chosen band for each signal
            Pxy_band = np.mean([Pxy[n] for n in xrange(len(Pxy)) if fxy[n] <= freq_band[1] and fxy[n] >= freq_band[0]])
            Pxx_band = np.mean([Pxx[n] for n in xrange(len(Pxx)) if fxx[n] <= freq_band[1] and fxx[n] >= freq_band[0]])
            Pyy_band = np.mean([Pyy[n] for n in xrange(len(Pyy)) if fyy[n] <= freq_band[1] and fyy[n] >= freq_band[0]])

            # compute the coherence
            c = abs(Pxy_band)**2/(Pxx_band*Pyy_band)
            A[i,j] = c # upper triangular part
            A[j,i] = c # lower triangular part

    # return adjacency matrix filled with coherence
    return A

"""
build_coherency_array()

Purpose: to create the three dimensional array containing all of the coherency matrices given a full file of data

Inputs:
    data: ndarray containing the iEEG data (number of samples x number of channels)
    win_len: float indicating window length
    win_ovlap: float indicating window overlap
    f_s: integer indicating sampling frequency
    freq_band: list indicating the frequency band within which to compute coherency matrix

Outputs:
    coherency_array: ndarray of coherency matrices (number of channels x number of channels x number of windows in raw_data
"""
def build_coherency_array(data, win_len, win_ovlap, f_s, freq_band):

    n, p = data.shape

    # getting window information in units of samples
    win_len = win_len * f_s
    win_ovlap = win_ovlap * f_s

    # computing the number of windows from the given data in units of samples
    num_windows = int( math.floor( float(n) / float(win_len - win_ovlap)) )
    coherency_array = np.zeros((p,p,num_windows))

    # compute the coherency matrix for each window of data in the file
    for index in np.arange(num_windows):
        start = index*(win_len - win_ovlap)
        end = min(start+win_len, n)
        window_of_data = data[int(start):int(end),:] # windowed data
        coherency_array[:,:,index] = construct_coherency_matrix(window_of_data, f_s, freq_band)

    return coherency_array

"""
test_predict()

Purpose: the code for using a one class support vector machine to classify a new test point
Inputs and outputs are all components of support vector machine equation
"""
def test_predict(num_supp_vec, dim_supp_vec, supp_vecs, test_point, gamma, dual_coeff, intercept):

    dual_sum = 0

    for support in range(num_supp_vec):

        l2_norm = 0
        for dim in range(dim_supp_vec):
            difference = supp_vecs[support,dim]-test_point[dim]
            diff_sqrd = difference * difference
            l2_norm = l2_norm + diff_sqrd

        exp_expression = -1 * gamma * l2_norm
        exp_result = np.exp(exp_expression)
        weighted_exp = dual_coeff[support] * exp_result
        dual_sum = dual_sum + weighted_exp

    dual_sum = dual_sum + intercept

    if dual_sum <= 0:
        label = 1
    else:
        label = 0

    return label

"""
online_testing()

Purpose: pipeline to direct the online testing portion of this algorithm on all of the unseen test data; computes per window

Inputs:
    test_data: ndarray (number of samples x number of channels) of one window of raw iEEG data
    mean_mat: ndarray (number of channels x number of channels) containing the normalizing means
    std_mat: ndarray (number of channels x number of channels) containing the normalizing standard deviations
    threshold: float indicating the level above which a seizure is flagged
    adapt_rate: int indicating the number of windows over which to average the predicted labels
    win_len: float indicating the window length
    win_overlap: float indicating the window overlap
    f_s: int indicating the sampling frequency
    freq_band: list indicating the frequencies over which to compute coherency
    test_index: the index of the test file (used only for labeling visualizations_
    patient_id: string containing patient name
    persistence_time: the amount of time after a seizure during which no other flag is raised
    num_channels: number of electrodes used for this patient
    vector_testing_labels: keeps track of past adapt_rate values to produce outlier fraction
    label_index: used to track filling in the vector of testing labels
    alarm_timer: used to track persistance time
    num_supp_vec: number of support vectors in SVM
    dim_supp_vec: dimension of each support vector
    supp_vecs: the support vectors in the SVM
    gamma: tuned parameter for the rbf kernel
    dual_coeff: dual coefficients of the SVM solution
    intercept: the float indicating the intercept in the SVM

Outputs:
    decision: a single number indicating seizure time or not; e [0,1]
    test_outlier_fraction: a single number indicating outlier fraction for this unit of time
    vector_testing_labels: a 1xadapt_rate dimenionsal ndarray tracking the outlier fraction
    label_index: counter indicating location in label array
    alarm_timer: counter to track persistance time
    test_evc: the eigenvector centrality returned for this window of data
"""
def online_testing(window_of_data, mean_mat, std_mat, threshold, adapt_rate, f_s, freq_band, test_index, patient_id, persistence_time, num_channels, vector_testing_labels, label_index, alarm_timer, num_supp_vec, dim_supp_vec, supp_vecs, gamma, dual_coeff, intercept):

    coherency_matrix = construct_coherency_matrix(window_of_data, f_s, freq_band)

    transformed_coherency_matrix = test_transform_coherency(coherency_matrix, mean_mat, std_mat, num_channels)

    test_evc = test_find_evc(transformed_coherency_matrix)

    testing_label = test_predict(num_supp_vec, dim_supp_vec, supp_vecs, test_evc, gamma, dual_coeff, intercept) # predict with test set

    if label_index < adapt_rate:
        vector_testing_labels[label_index] = testing_label
        label_index += 1
    else:
        label_index = 0
        vector_testing_labels[label_index] = testing_label

    test_outlier_fraction = find_mean(vector_testing_labels, adapt_rate)

    if alarm_timer <= 0:
        if test_outlier_fraction >= threshold:
            decision = 1
            alarm_timer = persistence_time
        else:
            decision = -1

    else:
        decision = -1
        alarm_timer -= 1

    return decision, test_outlier_fraction, vector_testing_labels, label_index, alarm_timer, test_evc

"""
score_decision()

Purpose: to assign a quantitative measure of goodness given performance measures of different folds/files

Inputs:
    sensitivity: ndarray containing the prediction sensitivities
    latency: ndarray containing latencies
    fp: ndarray containing the number of false positives
    time: ndarray containing the total amount of interictal data
    persistence_time: the amount of time after a seizure during which no other flag is raised

Outputs:
    score: float indicating how well the model performed
"""
def score_decision(sensitivity, fp, time, persistence_time):

    # get aggregate mean on sensitivity and false positive rate
    S = np.nanmean(sensitivity)
    FPR = float(np.nansum(fp)) / float(np.nansum(time))

    # calculate quantitative descriptor
    alpha1 = S
    alpha2 = -FPR*persistence_time/3600
    score = min(alpha1+alpha2, 0)

    return score

"""
tune_decision()

Purpose: to tune the threshold and adaptation rate given a patient

Inputs:
    mynu: the best one class svm nu for this data set
    mygamma: the best one class svm gamma for this data set
    cv_data_list: list containing all training data (features)
    cv_label_dict: dictionary containing all labels for the data in above list (sorry for the data type inconsistancy)
    seizure_indices: list indicating which entries in the dictionary are seizure files
    interictal_indices: list indicating which entires in the dictionary are interictal files
    win_len: float indicating the window length
    win_overlap: float indicating the window overlap
    f_s: int indicating the sampling frequency
    seizure_times: list of tuples containing the start and end time of seizures or None for interictal files
    persistence_time: the amount of time after a seizure during which no other flag is raised
    preictal_time: the amount of time before a seizure where the decision is counted as a prediction and not a false positive

Outputs:
    threshold: the level above which a seizure will be flagged
    adapt_rate: the number of windows over which the predicted labels will be smoothed
"""
def tune_decision(mynu, mygamma, cv_data_list,cv_label_dict,seizure_indices,interictal_indices, win_len, win_overlap,f_s,seizure_times,persistence_time, preictal_time):

    # divisions for cross validation
    div = 4

    # initialize a very bad best score
    best_score = -float("inf")
    best_score2 = -float("inf")

    # define the search range
    threshold_range = np.arange(0,1,.05)
    adapt_range = np.arange(30,61, 10)

    # search over all combinations in search range
    for mythresh in threshold_range:
        for myadapt in adapt_range:

            all_scores = np.zeros(div)

            for mult in xrange(div):

                interictal_fit_data = []
                interictal_validate_data = []
                interictal_validate_labels = []

                # obtain the validate and fit sets
                for index in interictal_indices:

                    length = cv_data_list[index].shape[0]
                    start = int(np.floor(length*mult/div))
                    end = int(np.floor(length*(mult+1)/div))

                    interictal_fit_data += [np.vstack((cv_data_list[index][:start],cv_data_list[index][end:]))]
                    interictal_validate_data += [cv_data_list[index][start:end]]
                    interictal_validate_labels += [cv_label_dict[index][start:end]]

                # fit the model with stacked matrix of interictal data
                fit_data = np.vstack(interictal_fit_data[i] for i in xrange(len(interictal_fit_data)))
                clf = svm.OneClassSVM(nu=mynu, kernel='rbf', gamma=mygamma)
                clf.fit(fit_data)

                # instantiate lists to keep track of all predicted labels and seizure times for the validate data
                pred_labels = []
                val_times = []

                # use the fit model to predict ictal files
                for seiz_index in seizure_indices:
                    one_seiz_pred_labels = clf.predict(cv_data_list[seiz_index])
                    one_seiz_pred_labels[one_seiz_pred_labels==1] = 0
                    one_seiz_pred_labels[one_seiz_pred_labels==-1] = 1
                    pred_labels += [one_seiz_pred_labels]
                    val_times += [seizure_times[seiz_index]]

                # use the fit model to predict all interictal fit data
                inter_validate_data = np.vstack(interictal_validate_data[i] for i in xrange(len(interictal_validate_data)))
                inter_pred_labels = clf.predict(inter_validate_data)
                inter_pred_labels[inter_pred_labels==1] = 0
                inter_pred_labels[inter_pred_labels==-1] = 1
                pred_labels += [inter_pred_labels]
                val_times += [None]

                # initializaing the performance statistic trackers
                pred_s = np.zeros(len(val_times))
                fp = np.zeros(len(val_times))
                time = np.zeros(len(val_times))

                # compute the outlier fractions, decisions, and performance metrics for this set of parameters
                for ind, one_file_pred_labels in enumerate(pred_labels):
                    outlier_fraction = create_outlier_frac(one_file_pred_labels, myadapt)
                    decision = create_decision(outlier_fraction, mythresh, persistence_time, win_len, win_overlap)
                    decision_sample = window_to_samples(decision, win_len, win_overlap, f_s)
                    pred_s[ind], _, _, fp[ind], time[ind] = performance_stats(decision_sample, val_times[ind], f_s, preictal_time, win_len, win_overlap)

                # assign a score to this model using performance metrics
                all_scores[mult] = score_decision(pred_s, fp, time, persistence_time)

            # take the average score of all validate sets to represent the score of this model
            avg_score = np.mean(all_scores)

            # track the best performing set of parameters
            if avg_score >= best_score:
                best_score = avg_score
                threshold = mythresh
                adapt_rate = myadapt

            if avg_score > best_score2:
                best_score2 = avg_score
                threshold2 = mythresh
                adapt_rate2 = myadapt

    print'\t\t\tBest parameters are: Threshold = ', (threshold+threshold2)/2, 'Adaptation rate (sec) = ', (adapt_rate+adapt_rate2)*(win_len-win_overlap)/2
    return (threshold+threshold2)/2, (adapt_rate+adapt_rate2)/2

"""
label_classes()

Purpose: create the array of labels to classify the training data

Inputs:
    num_windows: the number of windows that need to be labeled
    preictal_time: float, the amount of time in seconds before seizure that should be labeled as ictal data
    postictal_time: float, the amount of time in seconds after seizure that should be labeled as ictal data
    win_len: float indicating window length
    win_overlap: float indicating window overlap
    seizure_time: tuple containing the start and end times in seconds of seizure
    file_type: string indicating the type of file {'ictal','awake','sleep'}

Outputs:
    labels: one dimensional ndarray containing labels for the training data.
            {1: interictal data; -1: preictal/ictal/postictal data}


"""
def label_classes(num_windows, preictal_time, postictal_time, win_len, win_overlap, seizure_time, file_type):

    # labeling the seizure files
    if file_type is 'ictal':

        labels = np.empty(num_windows)

        # determine seizure start/end times in seconds
        seizure_start_time = seizure_time[0]
        seizure_end_time = seizure_time[1]

        # determining which window the seizure starts in
        if seizure_start_time < win_len:
            seizure_start_window = 0
        else:
            seizure_start_window = int((seizure_start_time - win_len) / (win_len - win_overlap) + 1)

        # determining which window the seizure ends in
        if seizure_end_time < win_len:
            seizure_end_window = 0
        else:
            seizure_end_window = int((seizure_end_time - win_len) / (win_len - win_overlap) + 1)

        # in case the seizure end window is larger than the max window index
        if seizure_end_window > num_windows - 1:
            seizure_end_window = num_windows - 1

        # label the ictal period
        labels[seizure_start_window:seizure_end_window+1] = -np.ones(seizure_end_window + 1 - seizure_start_window)

        # label the preictal (and interictal period if that exists) period
        if seizure_start_time > preictal_time:  # if there is a long period before seizure onset

            # determine the time in seconds where preictal period begins
            preictal_start_time = seizure_start_time - preictal_time

            # determine the time in windows where preictal period begins
            preictal_start_win = int((preictal_start_time - win_len) / (win_len - win_overlap) + 1)

            # label the preictal time
            labels[preictal_start_win:seizure_start_window] = -np.ones(seizure_start_window - preictal_start_win)

            # label the interical time
            labels[:preictal_start_win] = np.ones(preictal_start_win)

        else: # if there is not a long time in file before seizure begins
            # label preictal time
            labels[:seizure_start_window] = -np.ones(seizure_start_window)

        # determining how long the postical period lasts in seconds
        postictal_period = (num_windows - seizure_end_window) * (win_len - win_overlap)

        # if there is a long period of time after seizure in the file
        if postictal_period > postictal_time:

            # determine where in seconds the postical period ends
            postictal_end_time = seizure_end_time + postictal_time

            # determine where in windows the postical period ends
            postictal_end_win = int((postictal_end_time - win_len) / (win_len - win_overlap)+1)

            # in the case that the postictal end window exceeds the maximum number of windows...
            if postictal_end_win > num_windows - 1:
                postictal_end_win = num_windows - 1

            # label the postical period
            labels[seizure_end_window+1:postictal_end_win+1] = -np.ones(postictal_end_win - seizure_end_window)

            # label the interictal period
            labels[postictal_end_win+1:] = np.ones(num_windows - 1 - postictal_end_win)

        else: # if there is a short amount of time in the file after the seizure ends
            # label the postictal period
            labels[seizure_end_window+1:] = -np.ones(num_windows - 1 - seizure_end_window)

    # label awake interictal files
    elif file_type is 'awake':
        labels = np.ones(num_windows)

    # label asleep interictal files
    elif file_type is 'sleep':
        labels = np.ones(num_windows)

    # return the data labels
    return list(labels)

"""
score_model()
Purpose:
    The function uses cross validation to evaluate the one class svm defined by a given pair of nu and gamma
    and returns the average f_1 score for all folds of cross validation.

Input:
    mynu: float. Nu parameter for one class svm model
    mygamma: float. Gamma parameter for one class svm model
    cv_data_dict: dictionary that maps the index of the file in the list of  cv files to the corresponding ndarray feature matrix of this file
    cv_label_dict: dictionary that maps the index of the file in the list of cv files to the corresponding array of labels of this file
    seizure_indices: list of indices of the seizure files in the list of cv files
    interictal_indices: list of indices of the interictal files in the list of cv files

Output:
    avg_score: average f_1 score for all folds of leave-one-out cross validation.


"""
def score_model(mynu,mygamma,cv_data_dict,cv_label_dict,seizure_indices,interictal_indices):

    # compiling all data and labels for seizure files
    ictal_validate_data = np.vstack(cv_data_dict[index] for index in seizure_indices)
    ictal_validate_labels = np.hstack(cv_label_dict[index] for index in seizure_indices)

    # defining the training/validate set division
    div = 4

    # using cross validation to evaluate the classifier
    score = np.zeros(div)
    for mult in xrange(div):

        interictal_fit_data = []
        interictal_validate_data = []
        interictal_validate_labels = []

        # obtain 1/div  of all interictal data for validating, (div-1)/div of all interictal data for training
        for index in interictal_indices:

            length = cv_data_dict[index].shape[0]
            start = int(np.floor(length*mult/div))
            end = int(np.floor(length*(mult+1)/div))

            interictal_fit_data += [np.vstack((cv_data_dict[index][:start],cv_data_dict[index][end:]))]
            interictal_validate_data += [cv_data_dict[index][start:end]]
            interictal_validate_labels += [cv_label_dict[index][start:end]]

        # stacking all fit data
        fit_data = np.vstack(interictal_fit_data[i] for i in xrange(len(interictal_fit_data)))

        # stacking all validate data (interictal and ictal)
        inter_validate_data = np.vstack(interictal_validate_data[i] for i in xrange(len(interictal_validate_data)))
        inter_validate_labels = np.ones(inter_validate_data.shape[0])
        validate_data = np.vstack((inter_validate_data, ictal_validate_data))
        true_labels = np.hstack((inter_validate_labels, ictal_validate_labels))

        # training a model to test on our validate set
        clf = svm.OneClassSVM(nu=mynu, kernel='rbf', gamma=mygamma)
        clf.fit(fit_data)
        pred_labels = clf.predict(validate_data)

        # scoring our model using the fbeta score
        score[mult] = metrics.fbeta_score(true_labels, pred_labels, beta=1)

    # taking the average score from all validation sets
    avg_score = np.mean(score)

    return avg_score

"""
classifier_gridsearch()
Purpose:
    Tunes nu and gamma parameters for nu and gamma using loocv_scoring.
Inputs:
    data_dict: dictionary that maps the index of the file in the list of  cv files to the corresponding ndarray feature matrix of this file
    label_dict: dictionary that maps the index of the file in the list of cv files to the corresponding array of labels of this file
    seizure_indices: list of indices of the seizure files in the list of cv files
    interictal_indices: list of indices of the interictal files in the list of cv files
Output:
    (best_nu,best_gamma): the (nu,gamma) pair with the highest score returned by loocv_scoring.

"""
def classifier_gridsearch(data_dict,label_dict,seizure_indices,interictal_indices):

    # defining the search range
    gamma_range = [5e-6,1e-5,5e-5]
    nu_range = [.005,.0075,.01,.02,.03]

    best_score = -float("inf")
    best_nu = None
    best_gamma = None

    # searching through all combinations of nu and gamma to find the best classifier
    for nu in nu_range:
        for gamma in gamma_range:
            score = score_model(nu,gamma,data_dict,label_dict,seizure_indices,interictal_indices)
            if score>best_score:
                best_nu = nu
                best_gamma = gamma
                best_score = score

    print '\t\t\tNu =',best_nu,'Gamma =',best_gamma

    return best_nu, best_gamma

"""
offline_training()

Purpose: the pipeline directing the offline training portion of this algorithm on the known training data

Inputs:
    cv_file_type: list of strings containing the type of each training file {'ictal','awake','sleep'}
    cv_seizure_times: list of tuples indicating the start and end time of each seizure or None if interictal file
    cv_test_files: list of ndarrays containing the raw iEEG data for each training file
    win_len: float indicating the window length
    win_overlap: float indicating the window overlap
    f_s: int indicatin the sampling frequency
    freq_band: list indicatinezg the frequencies over which to compute coherency
    test_index: the index of the test file (used only for naming visualization files)
    patient_id: string indicating name of patient
    persistence_time: the amount of time after a seizure during which no other flag is raised
    preictal_time: float, the amount of time in seconds before a seizure that is labeled as preictal data
    postictal_time: float, the amount of time in seconds after a seizure that is labeled as postictal data
    calc_local: binary number indicating if features are calculated on local computer or loaded from storage
    feature_path: file path pointing to stored features

Outputs:
    mean_coherency_matrix: ndarray of means (number of channels x number of channels) used to normalize the coherency matrices
    sd_coherency_matrix: ndarray of standard deviations (number of channels x number of channels) used to normalize the coherency matrices
    best_clf: classification model
    threshold: float indicating level above which to flag seizre
    adapt_rate: int indicating length of time over which to average the predicted labels
    training_evc_cv_files: list of ndarrays containing all of the training features (each entry is number of windows x number of channels)
"""
def offline_training(cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, freq_band, test_index, patient_id, persistence_time, preictal_time, postictal_time, calc_local, features_path):

    if calc_local: #if calculating features on the local computer

        print'\t\tBuilding small coherency matrices for all training files'
        training_coherency_cv_files = []
        first_file = 0
        for n,small_test_file in enumerate(cv_test_files):
            print '\t\tSmall test file coherency...'
            test_file_coherency = build_coherency_array(small_test_file, win_len, win_overlap, f_s, freq_band)
            training_coherency_cv_files += [test_file_coherency]
            if cv_file_type[n] is not "ictal":
                if first_file == 0:
                    interictal_coherency_files = test_file_coherency
                    first_file = 1
                else:
                    interictal_coherency_files = np.dstack((interictal_coherency_files, test_file_coherency))

        print '\t\tFinding mean and standard deviation of interictal features'
        mean_coherency_matrix, sd_coherency_matrix = find_normalizing_coherency(interictal_coherency_files)

        print'\t\tTransforming small coherency matrices'
        transformed_coherency_cv_files = transform_coherency(training_coherency_cv_files, mean_coherency_matrix, sd_coherency_matrix)

        print'\t\tFinding eigenvector centrality'
        training_evc_cv_files = find_evc(transformed_coherency_cv_files)

        # saving the features as a .mat file
        sio.savemat('mf_split_training_{}_fold_{}.mat'.format(patient_id[-2:],test_index), {'evc':training_evc_cv_files, 'mean':mean_coherency_matrix, 'std':sd_coherency_matrix, })

    else: # loading the features from a .mat file
        to_data = os.path.join(features_path, 'mf_split_training_{}_fold_{}.mat'.format(patient_id[-2:],test_index))
        load_data = sio.loadmat(to_data)
        training_evc_cv_files = load_data.get('evc')[0]
        mean_coherency_matrix = load_data.get('mean')
        sd_coherency_matrix = load_data.get('std')
        training_evc_cv_files = np.ndarray.tolist(training_evc_cv_files)

    print '\t\tTraining the classifier'
    training_data = training_evc_cv_files

    interictal_indices = []
    seizure_indices = []
    interictal_data = np.vstack((training_data[ind] for ind in xrange(len(training_data)) if cv_file_type[ind] is not 'ictal'))

    # organizing the labels and data
    cv_label_dict = {}  # store labels of CV files to the label dictionary
    for index in xrange(len(cv_test_files)):
       # if file is seizure, store cv index to seizure indices
        if cv_file_type[index] == "ictal":
            seizure_indices.append(index)  # otherwise, store cv index to interictal indices
        else:
            interictal_indices.append(index)
        cv_label_dict[index] = label_classes(len(training_data[index]), preictal_time, postictal_time, win_len, win_overlap, cv_seizure_times[index], cv_file_type[index])

    # tuning the SVM parameterss
    bestnu, bestgamma = classifier_gridsearch(training_data,cv_label_dict,seizure_indices,interictal_indices)
    best_clf = svm.OneClassSVM(nu=bestnu, kernel='rbf', gamma=bestgamma)
    best_clf.fit(interictal_data)
    dual_coeff = best_clf.dual_coef_[0]
    supp_vecs = best_clf.support_vectors_
    intercept = best_clf.intercept_
    dim_supp_vec = supp_vecs.shape[1]
    num_supp_vec = supp_vecs.shape[0]
    gamma = bestgamma

    print '\t\tTraining the decision rule'
    threshold, adapt_rate = tune_decision(bestnu, bestgamma, training_data, cv_label_dict,seizure_indices,interictal_indices, win_len, win_overlap,f_s,cv_seizure_times,persistence_time,preictal_time)

    return best_clf, mean_coherency_matrix, sd_coherency_matrix, threshold, adapt_rate, training_evc_cv_files, num_supp_vec, dim_supp_vec, supp_vecs, gamma, dual_coeff, intercept

"""
viz_single_outcome

Purpose: to visualize iEEG data, the outlier fraction, and the seizure prediction decision on one plot

Inputs:
    decision: one dimensional ndarray in units of samples indicating whether or not a seizure is predicted
    out_frac: one dimensional ndarray in units of samples containing the outlier fraction
    raw_ieeg: one dimensional ndarray in units of samples containing the iEEG data from the .edf file
    test_times: tuple containing the start and end time of the seizure, or None if interictal file
    test_index: int indicating which patient file we are visualizing
    patient_id: list indicating patient name
    f_s: int indicating sampling frequency
    show_fig: if show_fig is 1, the plot will show; if show_fig is 0, the plot will save to the current working directory

Outpus:
    No output. The graph will either be visualized or be saved; graph contains three subplots: the raw iEEG data,
    the outlier fraction, and the decision rule. All share an x axis
"""
def viz_single_outcome(decision, out_frac, raw_ieeg, test_times, thresh, test_index, patient_id, f_s, show_fig=1):

    # initialize the subplots
    fig, axes = plt.subplots(3, sharex=True)

    # plot the raw iEEG signal
    axes[0].plot(raw_ieeg)
    axes[0].set_title('Raw iEEG signal', size=14)
    axes[0].set_ylabel('Voltage', size=10)

    # plot the outlier fraction and mark the threshold and seizure times
    axes[1].plot(out_frac)
    axes[1].set_title('Outlier fraction', size=14)
    axes[1].set_ylabel('Liklihood of upcoming seizure', size=10)
    axes[1].set_ylim(ymin=0, ymax=1)
    if test_times is not None:
        axes[1].axvline(x=test_times[0]*f_s,lw=2,c='r')
        axes[1].axvline(x=test_times[1]*f_s,lw=2,c='r')
    axes[1].axhline(y=thresh,lw=2,c='k')

    #plot the seizure prediction decision
    axes[2].plot(decision)
    axes[2].set_title('Prediction Decision', size=14)
    axes[2].set_xlabel('Samples', size=10)
    axes[2].set_yticks([-1,1])
    axes[2].set_yticklabels(['No seizure','Seizure'], size=10)
    axes[2].set_ylim(ymin=-1.5, ymax=1.5)

    fig.suptitle('Patient {}, file {}'.format(patient_id, test_index), size=16)
    fig.subplots_adjust(hspace=.8)

    if show_fig:
        plt.show()
    else:
        plt.savefig('single_Patient_{}_file_{}'.format(patient_id,test_index))
        plt.clf()
        plt.close()

    return

"""
viz_many_outcomes()

Purpose: to visualize all of the outlier fractions of the training data

Inputs:
    all_outlier_fractions: list of one dimensional ndarrays containing many files' outlier fractions
    seizure_times: list of tuples indicating start and end time of seizure or None for interictal files
    patient_id: string containing the patient name
    threshold: list of ints indicating the threshold above which to flag a seizure for each file
    test_index: int indicating which file currently being tested
    f_s: sampling frequency
    show_fig: if show_fig is 1, the plot will show; if show_fig is 0, the plot will save to the current working directory

Outputs:
    No output. The plot will either be visualized or saved. The single plot will contain subplots of all given
    outlier fractions.
"""
def viz_many_outcomes(all_outlier_fractions, seizure_times, patient_id, threshold, test_index, f_s, show_fig=1):

    # initialize subplots
    fig, axes = plt.subplots(len(all_outlier_fractions))

    # visualize each outlier fraction
    for i in xrange(len(all_outlier_fractions)):
        this_out_frac = np.asarray(all_outlier_fractions[i])
        axes[i].plot(this_out_frac)
        axes[i].set_ylim(ymin=0, ymax=1)

        # mark the seizure times and determined threshold
        if seizure_times[i] is not None:
            axes[i].axvline(x=seizure_times[i][0]*f_s, lw=2, c='r')
            axes[i].axvline(x=seizure_times[i][1]*f_s, lw=2, c='r')
        axes[i].axhline(y=threshold[i], lw=2, c='k')

    if show_fig:
        plt.show()
    else:
        plt.savefig('all_Patient_{}_file_{}'.format(patient_id,test_index))
        plt.clf()
        plt.close()

    return

"""
update_log_stats()

Purpose: to record the performance statistics of this run

Inputs:
    log_file: path to the log file
    patient_id: string indicating name of patient
    prediction_sensitivity: list containing each fold's prediction sensitivity
    detection_sensitivity: list containing each fold's detection sensitivity
    latency: list containing each fold's latency in seconds
    fp: list containing each fold's total false positives
    time: list containing each fold's total time of interictal data

Outputs:
    None. The log file will be saved as a .txt in the save_path specificed in parent_function()
"""
def update_log_stats(log_file, patient_id, prediction_sensitivity, detection_sensitivity, latency, fp, time):

    # print to results file
    f = open(log_file, 'a')
    f.write('\nPatient ' + patient_id + '\n=========================\n')

    # print the results -- aggregates and total
    f.write('Mean Prediction Sensitivity: \t%.2f\n' %(np.nanmean(prediction_sensitivity)))
    f.write('Mean Detection Sensitivity: \t%.2f\n' %(np.nanmean(detection_sensitivity)))
    f.write('Mean Latency: \t%.4f\n' %(np.nanmean(latency)))
    f.write('False Positive Rate: \t%.5f (fp/Hr) \n\n' % (np.nansum(fp) / np.nansum(time)))

    f.write('All files prediction Sensitivity: ' + str(prediction_sensitivity) + '\n')
    f.write('All files detection Sensitivity: ' + str(detection_sensitivity) + '\n')
    f.write('All files latency: ' + str(latency) + '\n')
    f.write('All files false ositives: ' + str(fp) + '\n')
    f.write('All files FP Time: ' + str(time) + '\n' + '\n')
    f.close()
    return

"""
update_log_params()

Inputs:
    log_file: path to the log file
    win_len: float indicating window length
    win_overlap: float indicating window overlap
    include_awake: boolean specifying whether or not to include awake data in the training
    include_asleep: boolean specifying whether or not to include asleep data in the training
    f_s: int indicating sampling frequnecy
    patients: list holding strings of patient names

Outputs:
    None. The log file will be saved as a .txt in the save_path specificed in parent_function()
"""
def update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients, freq_band, preictal_time, postictal_time, persistence_time):

    f = open(log_file, 'w')

    # write the first few lines
    f.write("Results file for Burns Pipeline\n")

    # write the parameters
    f.write('Parameters used for this test\n================================\n')
    f.write('Feature used is Burns Features\n')
    f.write('Window Length \t%.3f\nWindow Overlap \t%.3f\n' %(win_len, win_overlap))
    f.write('Sampling Frequency \t%.3f\n' % f_s)
    f.write('Awake Times are ' + (not include_awake)*'NOT ' + ' included in training\n')
    f.write('Asleep Times are ' + (not include_asleep)*'NOT ' + ' included in training\n\n')
    f.write('Frequency band is {} to {}\n'.format(freq_band[0],freq_band[1]))
    f.write('Preictal time is {}\n'.format(preictal_time/f_s))
    f.write('Postictal time is {}\n'.format(postictal_time/f_s))
    f.write('Prediction/persistance time is {}\n\n'.format(persistence_time/f_s))

    # write the patients
    f.write('Patients are ' + " ".join(patients) + "\n\n")
    f.close()
    return

"""
performance_stats()

Purpose: to calculate the sensitivity, false positives, total interictal time elapsed and latency given a decision array

Inputs:
    decision: one dimensional ndarray containing {0: no seizure; 1: seizure} predictions in samples
    seizure_time: tuple of actual start and end time of seizure in seconds described by the input decision
    f_s: int indicating the sampling frequency
    preictal_time: the amount of time before a seizure that is defined as preictal
    win_len: length in seconds of each window
    win_overlap: window overlap in seconds
Outputs:
    detection_sensitivity: float indicating if seizure was detected
    prediction_sensitivity: float indicated if seizure was predicted
    latency: float indicating the length of time in seconds that the decision did or did not predict seizure
    FP: total number of false positives in decision
    time: total elapsed time of all interictal data

"""
def performance_stats(decision, seizure_time, f_s, preictal_time, win_len, win_ovlap):

    # if interictal file
    if seizure_time is None:

        # get the amount of time in the interictal file and the number of false positives
        time = float(decision.size) / float(f_s * 60. * 60.)
        FP = float(np.size(decision[decision>0])) / float(f_s) / float(win_len - win_ovlap)

        prediction_sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        detection_sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        latency = np.nan # latency is meaningless since there is no seizure

    else:

        # computing the seizure start time and end time in units of samples
        seizure_start = int(f_s * seizure_time[0])
        seizure_end = int(f_s* seizure_time[1])

        # determining the amount of time before a seizure occurs where a false positive is countred
        false_positive_range = int(max(0, seizure_start - preictal_time*f_s))
        false_positive_data = np.copy(decision[:false_positive_range])

        # calculating the time in which false positives could occur and the number of false positives
        time = float(false_positive_range) / float(f_s * 60. * 60.)
        FP = float(np.size(false_positive_data[false_positive_data > 0])) / float(f_s) / float(win_len - win_ovlap)

        # determining the prediction and detection sensitivity
        if not np.any(decision[false_positive_range:seizure_end] > 0):
            prediction_sensitivity = 0.0 # seizure not detected
            detection_sensitivity = 0.0
        elif not np.any(decision[false_positive_range:seizure_start] > 0):
            prediction_sensitivity = 0.0 # seizure detected late
            detection_sensitivity = 1.0
        elif np.any(decision[false_positive_range:seizure_start]):
            prediction_sensitivity = 1.0 # seizure detected early
            detection_sensitivity = 1.0

        # compute latency
        if np.any(decision[false_positive_range:seizure_end] > 0):
            detect_time = np.argmax(decision[false_positive_range:seizure_end] > 0) + false_positive_range
            latency = float(detect_time - seizure_start) / float(f_s) # (in seconds)
        else:
            latency = np.nan

    # put time and FP as nan if there was no time during which false posivities could have occured
    if time <= 0.0:
        time = np.nan
        FP = np.nan

    return prediction_sensitivity, detection_sensitivity, latency, FP, time

"""
choose_best_channels()

Purpose: to inform which channels should be used in feature extraction for each patient

Inputs:
    patient_id: string indicating patient name

Outputs:
    dimensions_to_keep: list of strings containing the names of the electrodes that should be used for a given patient
"""
def choose_best_channels(patient_id):
    #TODO: experiement with other ways of choosing channels.

    good_channel_dict = {'TS041':['LAH2','LAH3','LAH4','LPH1','LPH2','LPH3',],
                         'TS039':['RAH1','RAH2','RAH3','RPH2','RPH3','RPH4',],
                         'TA023':['MST1','MST2','HD1','TP1','AST2','HD2',],
                         'TA533': ['PD2','PD3','PD4','PD5','LF28','LP4',]}

    return good_channel_dict[patient_id]

"""
analyze_patient_raw()

Purpose: read in the raw iEEG data, file names, file types, and seizure times from the stored .edf and pickle files

Inputs:
    data_path: file path that leads to where .edf and pickle files are stored
    f_s: integer indicating sampling frequency
    include_awake: boolean indicating if awake interictal data should be included for patient
    include_asleep: boolean indicating if asleep interictal data should be included for patient
    patient_id: string containing the name of the patient
    win_len: float indicating window length
    win_overlap: float indicating window overlap
    calc_local: if 1, real iEEG data will be read in to compute features; if 0, toy matrix will be instantiated
Outputs:
    all_files: list of ndarrays containing raw iEEG data from edf files
    data_filenames: list of strings indicating names of .edf file s
    file_type: list of strings indicating file type; can be {'ictal','awake','sleep'}
    seizure_times: list of tuples indicating start and end time of seizure; None if interictal file

"""
def analyze_patient_raw(data_path, f_s, include_awake, include_asleep, patient_id, win_len, win_overlap, calc_local):

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    # open the patient pickle file containing relevant information
    p_file = os.path.join(data_path, 'patient_pickle.txt')
    with open(p_file, 'r') as pickle_file:
        print("\tOpen Pickle: {}".format(p_file) + "...")
        patient_info = pickle.load(pickle_file)

    # add data file names, seizure times, file types
    data_filenames = list(patient_info['seizure_data_filenames'])
    seizure_times = list(patient_info['seizure_times'])
    file_type = ['ictal'] * len(data_filenames)
    file_durations = patient_info['file_durations']

    # include awake data if indicated in parent_function()
    if include_awake:
        data_filenames += patient_info['awake_inter_filenames']
        seizure_times += [None] * len(patient_info['awake_inter_filenames'])
        file_type += ['awake'] * len(patient_info['awake_inter_filenames'])

    # include asleep data if indicated in parent_funtion()
    if include_asleep:
        data_filenames += patient_info['asleep_inter_filenames']
        seizure_times += [None] * len(patient_info['asleep_inter_filenames'])
        file_type += ['sleep'] * len(patient_info['asleep_inter_filenames'])

    # getting the data paths that include the data filenames
    data_filenames = [os.path.join(data_path, filename) for filename in data_filenames]

    # initialize lists of information
    all_files = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []

    print '\tGetting Data...'
    # for each relvant file for this  patient...
    for i, single_data_filename in enumerate(data_filenames):

        min_per_chunk = 30
        sec_per_min = 60
        just_path, just_name = os.path.split(single_data_filename)

        if file_durations[just_name] >= min_per_chunk*sec_per_min and not (file_type[i] is 'ictal'):

            print '\t\tFile %d is long...' % (i + 1)

            j = 0
            while True:
                # get chunk start and end times
                start = j * sec_per_min * min_per_chunk
                end = (j + 1) * sec_per_min * min_per_chunk

                try:

                    # extract the chunk
                    print '\t\t\tChunk ' + str(j + 1) + ' reading...\n',
                    if calc_local:
                        dimensions_to_keep = choose_best_channels(patient_id)
                        X_chunk, _, labels = edfread(single_data_filename, rec_times=[start, end], good_channels=dimensions_to_keep)
                    else:
                        X_chunk = np.ones((10,6))

                    # update file information
                    all_files.append(X_chunk)
                    tmp_data_filenames += [single_data_filename]
                    tmp_file_type += [file_type[i]]
                    tmp_seizure_times += [seizure_times[i]]

                    j += 1
                    # if less than an entire chunk was read, then this is the last one!
                    if j==2 and not calc_local:
                        break
                    if X_chunk.shape[0] < sec_per_min * min_per_chunk and calc_local:
                        break

                except ValueError:
                    print "\t\t\tFinished reading chunks of file"
                    break

        else:
            print '\t\tFile %d reading...\n' % (i + 1),
            # read data in
            if calc_local:
                dimensions_to_keep = choose_best_channels(patient_id)
                X, _, labels = edfread(single_data_filename, good_channels=dimensions_to_keep)
            else:
                X = np.ones((10,10))

            # update file information
            all_files.append(X)  # add raw data to files
            tmp_data_filenames += [single_data_filename]
            tmp_file_type += [file_type[i]]
            tmp_seizure_times += [seizure_times[i]]

    # finalize file information
    data_filenames = tmp_data_filenames
    file_type = tmp_file_type
    seizure_times = tmp_seizure_times

    # double check that no NaN values appear in the features
    for X, i in enumerate(all_files):
        if np.any(np.isnan(X)):
            print 'There are NaN in raw data of file', i
            sys.exit('Error: Uh-oh, NaN encountered while extracting features')

    return all_files, data_filenames, file_type, seizure_times

"""
parent_function()

Purpose: the function that directs the cross validation, training, testing, and evaluation of each run

Inputs:
    No inputs.

Outputs:
    No outputs.
"""
def parent_function():

    # setting model parameters
    win_len = 3.0  # seconds
    win_overlap = 2.0  # seconds
    f_s = float(1e3)  # Hz
    freq_band = [80,100] # Hz
    persistence_time = 3 * 60 # minutes times seconds, the amount of time after a seizure prediction for which no alarm is raised
    preictal_time = 5 * 60 # minutes times seconds, the amount of time prior to seizure onset defined as preictal
    postictal_time = 5 * 60 # minutes times seconds, the amount of time after seizure end defined as postictal
    include_awake = True
    include_asleep = True

    # setting data parameters
    patients = ['TS039']
    calc_local = 0 # if 1, calculate all features locally; if 0, load features from load_path
    show_fig = 0 # if 1, figures show; if 0, figures save to current working directory

    # create paths to the data folder
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(to_data, 'DCEpy', 'Features', 'BurnsStudy')
    features_path = os.path.join(save_path, 'StoredFeatures', 'features')

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(features_path):
        os.makedirs(features_path)

    # create results file
    log_file = os.path.join(save_path, 'log_file.txt')
    update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients,freq_band, preictal_time, postictal_time, persistence_time)

    # evaluate each patient
    for patient_index, patient_id in enumerate(patients):

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)
        p_features_path = os.path.join(features_path, patient_id)
        print p_data_path
        print 'Retreiving stored raw data'
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(p_data_path, f_s, include_awake, include_asleep, patient_id, win_len, win_overlap, calc_local)

        number_files = len(all_files)

        # intializing performance stats
        prediction_sensitivity = np.zeros(len(all_files))
        detection_sensitivity = np.zeros(len(all_files))
        latency = np.zeros(len(all_files))
        fp = np.zeros(len(all_files))
        times = np.zeros(len(all_files))

        # for visualization purposes
        all_test_out_fracs = []
        all_thresholds = []

        # beginning leave one out cross-validation
        for i in xrange(number_files):
            print '\nCross validations, k-fold %d of %d ...' % (i+1, number_files)
            testing_file = all_files[i]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_test_files = all_files[:i] + all_files[i+1:]
            cv_file_type = file_type[:i] + file_type[i+1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i+1:]

            print '\tEntering offline training'
            svm, mean_mat, std_mat, threshold, adapt_rate, traindat, num_supp_vec, dim_supp_vec, supp_vecs, gamma, dual_coeff, intercept = offline_training(cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, freq_band, i, patient_id, persistence_time, preictal_time, postictal_time, calc_local, p_features_path)
            all_thresholds += [threshold]

            print'\tEntering online testing'

            if calc_local:

                print '\tCalculating testing features locally'

                total_test_samples = testing_file.shape[0]
                win_len_samples = win_len * f_s
                win_ovlap_samples = win_overlap * f_s
                num_windows = int( math.floor( float(total_test_samples) / float(win_len_samples - win_ovlap_samples)) )
                num_channels = testing_file.shape[1]
                full_file_decision = np.zeros(num_windows)
                full_file_outlier_fraction = np.zeros(num_windows)

                vector_testing_labels = np.zeros(adapt_rate)
                label_index = 0
                alarm_timer = 0

                testing_tracker = 1
                for index in np.arange(num_windows):

                    # getting a single window of data
                    start = index*(win_len_samples - win_ovlap_samples)
                    end = min(start+win_len_samples, total_test_samples)
                    window_of_data = testing_file[start:end,:] # windowed data

                    # putting window of data through prediction algorithm
                    decision, outlier_fraction, vector_testing_labels, label_index, alarm_timer, test_evc = online_testing(window_of_data, mean_mat, std_mat, threshold, adapt_rate, f_s, freq_band, i, patient_id, persistence_time, num_channels, vector_testing_labels, label_index, alarm_timer, num_supp_vec, dim_supp_vec, supp_vecs, gamma, dual_coeff, intercept) #REAL TIME

                    # storing the outlier fraction and decision for calculating performance metrics and visualization
                    full_file_outlier_fraction[index] = outlier_fraction
                    full_file_decision[index] = decision

                    # compiling all outputs for saving the features
                    if testing_tracker:
                        test_matrix = test_evc
                        testing_tracker = 0
                    else:
                        test_matrix = np.vstack((test_matrix,test_evc))

                # saving computed features
                sio.savemat('mf_split_testing_{}_fold_{}.mat'.format(patient_id[-2:],i), {'evc':test_matrix})

            else: #features are already stored

                print '\tLoading saved testing features'
                to_saved_data = os.path.join(p_features_path, 'mf_split_testing_{}_fold_{}.mat'.format(patient_id[-2:],i))
                load_data = sio.loadmat(to_saved_data)
                testing_evc = load_data.get('evc')

                # using trained classifier to predict testing data classes
                pred_labels = svm.predict(testing_evc)
                pred_labels[pred_labels==1] = 0
                pred_labels[pred_labels==-1] = 1

                # storing the outlier fraction and decision for calculating performance metrics and visualization
                full_file_outlier_fraction = create_outlier_frac(pred_labels, adapt_rate)
                full_file_decision = create_decision(full_file_outlier_fraction, threshold, persistence_time, win_len, win_overlap)

            print'\tCalculating performance stats'
            test_outlier_frac_sample = window_to_samples(full_file_outlier_fraction, win_len, win_overlap, f_s)
            test_decision_sample = window_to_samples(full_file_decision, win_len, win_overlap, f_s)
            prediction_sensitivity[i], detection_sensitivity[i], latency[i], fp[i], times[i] = performance_stats(test_decision_sample, seizure_times[i], f_s, preictal_time, win_len, win_overlap)
            all_test_out_fracs += [test_outlier_frac_sample]

            print '\tPrediction sensitivity = ', prediction_sensitivity[i], 'Detection sensitivity = ', detection_sensitivity[i], 'Latency = ', latency[i], 'FP = ', fp[i], 'Time = ', times[i]
            # viz_single_outcome(test_decision_sample, test_outlier_frac_sample, testing_file[:,0], seizure_times[i], threshold, i, patient_id, f_s)

        fpr = float(np.nansum(fp)) / float(np.nansum(times))
        print '\nMean predition sensitivity = ', np.nanmean(prediction_sensitivity), 'Mean detection sensitivity = ', np.nanmean(detection_sensitivity), 'Mean latency = ', np.nanmean(latency), 'Mean FPR = ', fpr
        print 'Median prediction sensitivity = ', np.nanmedian(prediction_sensitivity), 'Median detection sensitivity = ', np.nanmedian(detection_sensitivity), 'Median latency = ', np.nanmedian(latency)
        update_log_stats(log_file, patient_id, prediction_sensitivity, detection_sensitivity, latency, fp, times)
        viz_many_outcomes(all_test_out_fracs, seizure_times, patient_id, all_thresholds, 'One class SVM 4 partitions', f_s, show_fig)

if __name__ == '__main__':
    parent_function()
