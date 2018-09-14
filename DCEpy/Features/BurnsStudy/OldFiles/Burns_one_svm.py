import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from DCEpy.Features.BurnsStudy.BurnsFinal import analyze_patient_raw, tune_pi
from scipy.signal import csd
from scipy.signal import welch
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

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
    sample_array = np.zeros(max(sample_indices))
    for i in np.arange(0,num_windows-1):
        sample_array[sample_indices[i]:sample_indices[i+1]] = window_array[i]

    return sample_array



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
            evc = nx.eigenvector_centrality(G, max_iter=500)
            #TODO: change this if you want, such that order stays the same. Not urgent.
            centrality[i,:] = np.asarray(evc.values())

        centrality_all_files += [centrality]

    return centrality_all_files



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
            fxy, Pxy = csd(X[:,i], X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)

            # approximate power spectral density at each frequency for each signal with Welch's method
            fxx, Pxx = welch(X[:,i], fs = f_s, nperseg = 1000, noverlap = 500)
            fyy, Pyy = welch(X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)

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
        window_of_data = data[start:end,:] # windowed data
        coherency_array[:,:,index] = construct_coherency_matrix(window_of_data, f_s, freq_band)

    return coherency_array



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
Notice:
    Labels for one class svm are different from those for csvm. One class svm outputs +1 for normal(interictal) data
    and -1 for preictal/ictal/postictal data


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
f_1()

Purpose: computes the f_1 score of the estimator on given data.
Input:
    estimator: one_class_svm, csvm or nusvm
    data: ndarray containing eigenvalues for each window with shape (number of samples, number of channels)
    actual_labels: 1d array containing expected labels.
Output:
    score: int type. F_1 score of the estimator on data.

"""
def f_1(estimator, data, actual_labels):
    predicted_labels = estimator.predict(data)
    score = fbeta_score(actual_labels, predicted_labels,beta=1)
    return score

"""
loocv_scoring()
Purpose:
    The function uses leave-one-out cross validation to evaluate the one class svm determined by a given pair of nu and gamma
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

def loocv_scoring(mynu,mygamma, cv_data_dict,cv_label_dict,seizure_indices,interictal_indices, win_len_secs=3.0, win_overlap_secs=2.0, f_s=float(1000)):

    scores = []
    if len(interictal_indices)<=1:
        raise ValueError("Eh-Oh, not enough interictal files to perform loocv!")
    # loocv
    for inter_idx in interictal_indices:
        fit_idx = interictal_indices[:inter_idx]+interictal_indices[inter_idx+1:]
        validate_idx = seizure_indices+[inter_idx]
        fit_data = np.vstack([cv_data_dict[i] for i in fit_idx])
        validate_data = np.vstack([cv_data_dict[i] for i in validate_idx])
        validate_labels = np.hstack([cv_label_dict[i] for i in validate_idx]) # TODO: inconsisitency

        # TODO: to try different kernels
        clf = svm.OneClassSVM(nu=mynu, kernel='rbf', gamma=mygamma)
        clf.fit(fit_data)
        score = f_1(clf, validate_data, validate_labels)
        scores.append(score)

    avg_score = float(sum(scores))/len(interictal_indices)
    return avg_score

"""
Gridsearch_loocv()
Purpose:
    Tunes nu and gamma parameters for nu and gamma using loocv_scoring.
Inputs:
    data_dict: dictionary that maps the index of the file in the list of  cv files to the corresponding ndarray feature matrix of this file
    label_dict: dictionary that maps the index of the file in the list of cv files to the corresponding array of labels of this file
    seizure_indices: list of indices of the seizure files in the list of cv files
    interictal_indices: list of indices of the interictal files in the list of cv files
    f_s: int indicatin the sampling frequency
    win_len_secs: float indicating the window length
    win_overlap_secs: float indicating the window overlap

Output:
    (best_nu,best_gamma): the (nu,gamma) pair with the highest score returned by loocv_scoring.

"""

def gridsearch_loocv(data_dict,label_dict,seizure_indices,interictal_indices, win_len_secs=3.0, win_overlap_secs=2.0, f_s=float(1000)):

    # TODO: find appropriate grid range
    nu_range = [  .05, .1, .15, .2,.3]
    gamma_range = [.001, .005,  .01, .05, .1, .5, 1,4, 8]
    best_score =-float("inf")
    best_nu = None
    best_gamma = None

    for nu in nu_range:
        for gamma in gamma_range:
            #leave-one-out
            score = loocv_scoring(nu,gamma, data_dict,label_dict,seizure_indices,interictal_indices, win_len_secs=3.0, win_overlap_secs=2.0, f_s=float(1000))
            if score>best_score:
                best_nu = nu
                best_gamma = gamma
                best_score = score

    return best_nu,best_gamma

"""
prep_offline_data()

Purpose: prepares offline training data
Inputs:
    cv_file_type: list of strings containing the type of each training file {'ictal','awake','sleep'}
    cv_seizure_times: list of tuples indicating the start and end time of each seizure or None if interictal file
    cv_test_files: list of ndarrays containing the raw iEEG data for each training file
    freq_band: list indicating the frequencies over which to compute coherency
    preictal_time: float, the amount of time in seconds before a seizure that is labeled as ictal data
    postictal_time: float, the amount of time in seconds after a seizure that is labeled as ictal data
    f_s: int indicatin the sampling frequency
    win_len_secs: float indicating the window length
    win_overlap_secs: float indicating the window overlap


Outputs:
    training_data: ndarray that contains interictal training data with a shape of (number of samples, number of windows)
    cv_data_dict: dictionary that maps the index of the file in the list of  cv files to the corresponding ndarray feature matrix of this file
    cv_label_dict: dictionary that maps the index of the file in the list of cv files to the corresponding array of labels of this file
    seizure_indices: list of indices of the seizure files in the list of cv files
    interictal_indices: list of indices of the interictal files in the list of cv files
    mean_mat: mean matrix of all interictal training data
    std_mat: std matrix of all interictal data
"""

def prep_offline_data(cv_file_type, cv_seizure_times,  cv_test_files, freq_band, preictal_time, postictal_time,f_s = float(1000),win_len_secs = 3, win_overlap_secs=2):

    print "\t\t Preparing interictal training data "
    # compile interictal data
    print'\t\t\tCompiling all interictal data from chosen channels'
    interictal_raw_data = np.vstack((cv_test_files[n] for n in xrange(len(cv_test_files)) if
                                     cv_file_type[n] is not "ictal"))  # number of window * number of channels
    # compute mean coherency matrix and standard coherency matrix
    print '\t\t\tFinding coherency matrix of all interictal data'
    interictal_coherency_matrix = build_coherency_array(interictal_raw_data, win_len_secs, win_overlap_secs, f_s,
                                                        freq_band)
    print'\t\t\tFinding mean and std of all interictal data'
    mean_mat, std_mat = find_normalizing_coherency(interictal_coherency_matrix)

    # prepare interictal training data
    print "\t\t\t Compiling training eigen values"
    training_data = transform_coherency([interictal_coherency_matrix], mean_mat, std_mat)

    training_evc = find_evc(training_data)
    training_evc = np.asarray(training_evc)
    training_evc = training_evc[0, :, :]
    training_data = np.vstack(training_evc)         # all interictal training data

    # preparing data dictionaries for leave-one-out
    print "\t\t Preparing data for leave-one-out cross validation"
    interictal_indices = []
    seizure_indices = []
    cv_data_dict = {}
    cv_label_dict = {}  # store labels of CV files to the label dictionary
    for index, file in enumerate(cv_test_files):
        # transform and computer burns features for each file
        file_coherency_matrices = [build_coherency_array(file, win_len_secs, win_overlap_secs, f_s, freq_band)]
        transformed_file_coherency_matrices = transform_coherency(file_coherency_matrices, mean_mat, std_mat)
        file_evc = find_evc(transformed_file_coherency_matrices)
        file_evc = np.asarray(file_evc)
        file_evc = file_evc[0, :, :]
        file_data = np.vstack(file_evc)
        cv_data_dict[index] = file_data  # store features of CV files to the data dictionary

        # if file is seizure, store cv index to seizure indices
        if cv_file_type[index] == "ictal":
            seizure_indices.append(index)  # otherwise, store cv index to interictal indices
        else:
            interictal_indices.append(index)
        cv_label_dict[index] = label_classes(len(file_data), preictal_time, postictal_time, win_len_secs, win_overlap_secs,
                                          cv_seizure_times[index], cv_file_type[index])
    print "\t\t Finished preparing data for this Fold"
    return training_data, cv_data_dict, cv_label_dict, seizure_indices,interictal_indices,mean_mat,std_mat

"""
offline_training_loocv()
Purpose:
    Performs Parameter Tuning for one_class_svm
Inputs:
    training_data: ndarray that contains all interictal training data with shape (number of samples, number of channels)
    cv_data_dict: dictionary that maps the index of the file in the list of  cv files to the corresponding ndarray feature matrix of this file
    cv_label_dict: dictionary that maps the index of the file in the list of cv files to the corresponding array of labels of this file
    seizure_indices: list of indices of the seizure files in the list of cv files
    interictal_indices: list of indices of the interictal files in the list of cv files

Output:
    best_clf: best one class svm with optimal parameters and fit with all interictal training data

"""
def offline_training_loocv(training_data, cv_data_dict, cv_label_dict, seizure_indices,interictal_indices):

    print "\t\tStart Parameter Tuning"
    # grid search on the best nu and best gamma using leave-one-out cross validation
    bestnu, bestgamma = gridsearch_loocv(cv_data_dict,cv_label_dict,seizure_indices,interictal_indices)
    best_clf = svm.OneClassSVM(nu=bestnu, kernel='rbf', gamma=bestgamma)
    best_clf.fit(training_data)
    return best_clf



"""
online_testing()
Purpose:
    Performs online testing on the trained model

Inputs:
    test_file: ndarray holding the test data with shape (number of samples, number of windows)
    mean_mat: mean matrix of all interictal training data
    std_mat: std matrix of all interictal data
    best_clf: trained one class svm object
    patient_id: identification of the patient
    test_file_type: string containin the type of the file. "ictal" for seizure file and "interictal" for interictal file.
    test_seizure_time: tuple containing the start and end time of the seizure, or None if interictal file

Outputs:
    testing_data: ndarray containing eigen value centrality of the test file with shape (number of samples, number of channels)
    actual_labels: 1d array containing expected labels
    predicted_labels: 1d array containing labels predicted by one class svm

"""

def online_testing(test_file, mean_mat, std_mat,best_clf,patient_id, test_file_type, test_seizure_time, freq_band, preictal_time, postictal_time, f_s = float(1000),win_len_secs=3.0, win_overlap_secs=2.0):
    print "\t\tPreparing testing data..."
    print'\t\t\tBuilding coherency matrices'
    test_coherency_matrices = [build_coherency_array(test_file, win_len_secs, win_overlap_secs, f_s, freq_band)]

    print'\t\t\tTransforming coherency matrices'
    transformed_test_coherency_matrices = transform_coherency(test_coherency_matrices, mean_mat, std_mat)

    print'\t\t\tFinding eigenvec centrality'
    test_evc = find_evc(transformed_test_coherency_matrices)
    test_evc = np.asarray(test_evc)
    test_evc = test_evc[0, :, :]
    testing_data = np.vstack(test_evc)
    actual_labels = label_classes(len(testing_data),preictal_time=preictal_time,postictal_time=postictal_time,win_len=win_len_secs,win_overlap=win_overlap_secs,seizure_time=test_seizure_time,file_type=test_file_type)

    print "\t\tPredicting Lables..."
    predicted_labels = best_clf.predict(testing_data)
    return testing_data,actual_labels,predicted_labels

"""
viz_labels
Purpose:
    visualize labels using plots
Inputs:
    labels: a list of labels to be plotted
    testing_file_name: string. Name of the test file

Output:
    None
"""

def viz_labels(labels,testing_file_name,test_data_seizure_time=None,f_s=1000,win_len_secs=2.0,win_overlap_secs=1.0):


    num_labels= len(labels)
    num_windows = len(labels[0])

    plt.figure()

    seizure_start_win = None
    seizure_end_win = None

    if test_data_seizure_time:
        seizure_start,seizure_end = test_data_seizure_time
        seizure_start_win = int((seizure_start - win_len_secs) / (win_len_secs - win_overlap_secs) + 1)
        seizure_end_win = int((seizure_end - win_len_secs) / (win_len_secs - win_overlap_secs) + 1)

    #in case the seizure end window is rounded to an index larger than the max window index
    if seizure_end_win>num_windows-1:
        seizure_end_win=num_windows-1


    for i in range(len(labels)):
        label = labels[i]
        # num_windows = len(label)

        plt.subplot(num_labels,1,i+1)
        plt.title(testing_file_name)
        plt.ylim([-1.5,1.5])
        windows = np.linspace(0, num_windows-1, num_windows)
        plt.plot(windows, label, 'b')

        plt.axvline(x=seizure_start_win,color="r")
        plt.axvline(x=seizure_end_win, color="r")

    # plt.savefig(testing_file_name+".png")
    #
    plt.show()


"""
parent_function()

Purpose: the function that directs the testing, offline training, online testing, and evaluation/vizualization of each run

Inputs:
    No inputs.

Outputs:
    No outputs.
"""

def parent_function():

    # setting parameters
    win_len = 3.0  # seconds
    win_overlap = 2.0  # seconds
    f_s = float(1e3)  # Hz

    patients = ['TS039']
    long_interictal = [False]
    include_awake = True
    include_asleep = False

    #TODO: tune the frequency band, the preictal and postictal time limits
    freq_band = [30,35] # Hz
    pred_time = 3.5 * 60 # minutes times seconds
    preictal_time = 3 * 60 # minutes times seconds
    postictal_time = 3 * 60 # minutes times seconds

    # create paths to the data folder
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(to_data, 'DCEpy', 'Features', 'BurnsStudy')

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # create results file
    log_file = os.path.join(save_path, 'log_file.txt')
    # update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients,freq_band, preictal_time, postictal_time, pred_time)

    # evaluate each patient
    for patient_index, patient_id in enumerate(patients):

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)

        print 'Retreiving stored EEG raw data'
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(p_data_path, f_s, include_awake, include_asleep, long_interictal[patient_index], patient_id, win_len, win_overlap)
        number_files = len(all_files)

        # intializing performance stats
        # prediction_sensitivity = np.zeros(len(all_files))
        # detection_sensitivity = np.zeros(len(all_files))
        # latency = np.zeros(len(all_files))
        # fp = np.zeros(len(all_files))
        # time = np.zeros(len(all_files))

        # beginning leave one out cross-validation
        for i in xrange(number_files):
            print '\nTest and Cross Validation, k-fold %d of %d ...' % (i + 1, number_files)

            # determine testing and training data for for this k-fold
            testing_file = all_files[i]
            test_seizure_time = seizure_times[i]
            test_file_type = file_type[i]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_test_files = all_files[:i] + all_files[i+1:]
            cv_file_type = file_type[:i] + file_type[i+1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i+1:]

            print '\tEntering offline training'
            training_data, cv_data_dict, cv_label_dict, seizure_indices, interictal_indices, mean_mat, std_mat = prep_offline_data(cv_file_type, cv_seizure_times,  cv_test_files, freq_band, preictal_time, postictal_time,f_s = float(1000),win_len_secs = 3, win_overlap_secs=2)
            best_clf = offline_training_loocv( training_data, cv_data_dict, cv_label_dict, seizure_indices,interictal_indices)

            print'\tEntering online testing'
            test_data, actual_labels, predicted_labels= online_testing(testing_file, mean_mat, std_mat, best_clf,patient_id,test_file_type, test_seizure_time, freq_band, preictal_time, postictal_time, f_s = float(1000),win_len_secs=3.0, win_overlap_secs=2.0)
            print'\tEnded online testing'

            # Visualize Classification Results
            print classification_report(actual_labels, predicted_labels)
            viz_labels([actual_labels,predicted_labels], "Test Fold "+str(i)+data_filenames[i], test_data_seizure_time=test_seizure_time, f_s=1000, win_len_secs=3.0,
                       win_overlap_secs=2.0)


            # Tuning Decision Rule, currently using the tune_pi from Burn's Final
            # TODO: Find good method for tuning decision rule
            cv_file_list = [cv_data_dict[j] for j in xrange(len(cv_data_dict))]
            threshold, adapt_rate = tune_pi(best_clf,cv_file_type,cv_seizure_times,cv_file_list, win_len,win_overlap,f_s,pred_time,verbose = 0)




parent_function()
