import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef,hinge_loss,hamming_loss,jaccard_similarity_score,fbeta_score
import math
import matplotlib.pyplot as plt
from scipy.signal import csd
from scipy.signal import welch
import scipy.io as sio
from sklearn import svm
import networkx as nx
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread

"""
create_decision()

Purpose: to create the binary seizure prediction decision

Inputs:
    outlier_fraction: one dimensional ndarray containing the outlier fraction in units of windows
    threshold: int indicating the threshold over which a seizure is predicted
    pred_time: the amount of time after a seizure during which no other flag is raised AND the amount of time before a
                seizure where the decision is counted as a prediction and not a false positive
    win_len: float indicating window length
    win_overlap: float indicating window overlap

Outputs:
    decision: one dimensional ndarray containing a decision {-1: no seizure; 1: seizure} for every window

"""

def create_decision(outlier_fraction, threshold, pred_time, win_len, win_overlap):

    # determining where the outlier fraction meets or exceeds the threshold
    raw_decision = np.sign(outlier_fraction - threshold)
    raw_decision[raw_decision==0] = 1

    # initializing the final decision
    decision = np.ones(raw_decision.size)*-1

    # finding the prediction time (pred_time) in units of windows
    pred_windows = float(pred_time) / float(win_len - win_overlap)

    # determining the final decision
    for i in np.arange(0,len(raw_decision)):
        if (raw_decision[i]==1):
            decision[i] = 1
            raw_decision[i+1:i+pred_windows] = -1

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
    sample_array = np.zeros(max(sample_indices))
    for i in np.arange(0,num_windows-1):
        sample_array[sample_indices[i]:sample_indices[i+1]] = window_array[i]

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


def read_energy_file(file_path):
    # get data dimensions
    n = sum(1 for row in csv.reader(open(file_path)))
    p = len(next(csv.reader(open(file_path)))) / 3
    X_feat = np.empty((n, 3, p))
    # open up reader
    f = open(file_path, 'rb')
    reader = csv.reader(open(file_path, 'rb'))

    for i, row in enumerate(reader):
        a = np.array([float(b) for b in row])
        a = np.reshape(a, (3, p), 'F')
        X_feat[i, :, :] = a

    return X_feat

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

    # cannot divide by 0
    std[std == 0] = 0.0001
    num_files = len(coherencies_list)

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
online_testing()

Purpose: pipeline to direct the online testing portion of this algorithm on all of the unseen test data

Inputs:
    test_data: ndarray (number of samples x number of channels) of raw iEEG data
    mean_mat: ndarray (number of channels x number of channels) containing the normalizing means
    std_mat: ndarray (number of channels x number of channels) containing the normalizing standard deviations
    classifier: the trained classification model
    threshold: float indicating the level above which a seizure is flagged
    adapt_rate: int indicating the number of windows over which to average the predicted labels
    win_len: float indicating the window length
    win_overlap: float indicating the window overlap
    f_s: int indicating the sampling frequency
    freq_band: list indicating the frequencies over which to compute coherency
    test_index: the index of the test file (used only for labeling visualizations_
    patient_id: string containing patient name
    pred_time: the amount of time after a seizure during which no other flag is raised AND the amount of time before a
                seizure where the decision is counted as a prediction and not a false positive

Outputs:
    decision: one dimensional ndarray containing decision of seizure prediction
    test_outlier_fraction: one dimensional ndarray containing the outlier fraction of the test data
"""
def online_testing(test_data, classifier, threshold, adapt_rate, win_len, win_overlap, f_s, freq_band, test_index, patient_id, pred_time):


    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(data_path, 'energy_stats', patient_id)

    file_begin = os.path.join(save_path, os.path.splitext(os.path.basename(test_data))[0])
    load_path_gardner = os.path.join(to_data, 'data', 'energy_stats', patient_id, file_begin + '.es')

    # read_energy_file takes in the data path as input
    X_feat = read_energy_file(load_path_gardner).T
    X_feat = list(X_feat)
    testing_es_files = [None] * len(X_feat)
    for j, X_file in enumerate(X_feat):
        X_file = X_file.T
        testing_es_files[j] = X_file

    testing_es_cv_files = np.hstack(testing_es_files)

    testing_data = testing_es_cv_files

    print '\t\tPredicting test labels'
    testing_labels = classifier.predict(testing_data) # predict with test set

    print '\t\tCreating outlier fraction'
    test_outlier_fraction = create_outlier_frac(testing_labels, adapt_rate)

    print '\t\tCreating prediciton decision'
    decision = create_decision(test_outlier_fraction, threshold, pred_time, win_len, win_overlap)

    return decision, test_outlier_fraction

"""
gardner_score()

Purpose: to assign a quantitative measure of goodness given performance measures of different folds/files

Inputs:
    sensitivity: ndarray containing the prediction sensitivities
    latency: ndarray containing latencies
    fp: ndarray containing the number of false positives
    time: ndarray containing the total amount of interictal data
    pred_time: the amount of time after a seizure during which no other flag is raised AND the amount of time before a
                seizure where the decision is counted as a prediction and not a false positive

Outputs:
    score: float indicating how well the model performed
"""
def gardner_score(sensitivity, latency, fp, time, pred_time):

    # get aggregate mean on sensitivity and false positive rate
    S = np.nanmean(sensitivity)
    FPR = float(np.nansum(fp)) / float(np.nansum(time))

    # if any seizure is detected, get aggregate mean for latency and early detection fraction
    if np.any(np.isfinite(latency)):
        mu = np.nanmean(latency)
        detected = latency[np.isfinite(latency)]
        EDF = float(np.where(detected < 0)[0].size) / float(detected.size)

    # if the model failed to detect any seizure, give terrible values to latency and early detection fraction
    else:
        mu = 500.0
        EDF = 0.0

    # calculate quantitative descriptor...
    # weighting sensitivity
    alpha1 = 100*S

    # weighting false positive rate
    beta = 10
    allowable_fpr = 6.0
    random_fpr = float(3600)/float(pred_time)
    if FPR >= random_fpr:
        alpha2 = -100
    else:
        alpha2 = min(float(-beta*FPR)/allowable_fpr, float((-100-beta)*(FPR-random_fpr))/float(random_fpr-allowable_fpr)-100)

    # weighting the latency
    desired_latency = -15
    alpha3 = min(30,30.0/float(desired_latency)*mu)

    score = alpha1 + alpha2 + alpha3

    return score

"""
tune_pi()

Purpose: to tune the threshold and adaptation rate given a patient

Inputs:
    prediction_model: model used to predict eigenvector centrality vectors
    file_types: list of strings containing the type of file {'ictal','awake','sleep}
    seizure_times: list of tuples containing the start and end time of seizures or None for interictal files
    data_files: list of ndarrays containing the eigenvector centralities for each file (ndarray: number of windows x number of channels)
    win_len: float indicating the window length
    win_overlap: float indicating the window overlap
    f_s: int indicating the sampling frequency
    pred_time: the amount of time after a seizure during which no other flag is raised AND the amount of time before a
                seizure where the decision is counted as a prediction and not a false positive
    verbose: detemines coarseness of search; 1: fine search; 0: coarse search

Outputs:
    threshold: float indicting the level above which a seizure will be flagged
    adapt_rate: int indicating the number of windows over which the predicted labels will be smoothed
"""
def tune_pi(prediction_model, file_types, seizure_times, data_files, win_len, win_overlap, f_s, pred_time, verbose=0):

    # find the predicted labels for all training data
    all_predicted_labels = []
    for file in (data_files):
        all_predicted_labels += [prediction_model.predict(file)]

    # initialize a very bad best score
    best_score = -5000

    # define the search range
    if verbose==1:
        threshold_range = np.arange(.2,1,.01)
        adapt_range = np.arange(30,71,10)
    else:
        threshold_range = np.arange(.3,1,.1)
        adapt_range = np.arange(50,51,10)

    # search over all combinations in search range
    for mythresh in threshold_range:
        for myadapt in adapt_range:

            # initializaing the performace statistic trackers
            pred_s = np.zeros(len(seizure_times))
            det_s = np.zeros(len(seizure_times))
            latency = np.zeros(len(seizure_times))
            fp = np.zeros(len(seizure_times))
            time = np.zeros(len(seizure_times))

            # compute the outlier fractions, decisions, and performance score for this set of parameters
            for i,predicted_labels in enumerate(all_predicted_labels):
                outlier_fraction = create_outlier_frac(predicted_labels, myadapt)
                decision = create_decision(outlier_fraction, mythresh, pred_time, win_len, win_overlap)
                decision_sample = window_to_samples(decision, win_len, win_overlap, f_s)
                pred_s[i], det_s[i], latency[i], fp[i], time[i] = performance_stats(decision_sample, seizure_times[i], f_s, pred_time)
            score = gardner_score(pred_s, latency, fp, time, pred_time)

            # track the best performing set of parameters
            if score > best_score:
                best_score = score
                threshold = mythresh
                adapt_rate = myadapt

    print'\t\t\tBest parameters are: Threshold = ', threshold, 'Adaptation rate = ', adapt_rate

    return threshold*0.85, adapt_rate

"""
grid_tune_svm()

Purpose: to tune the parameters for the svm classifier

Inputs:
    train_data: ndarray containing training data (number of windows x number of channels)
    train_labels: ndarray containing training labels (number of windows)

Outputs:
    clf.best_estimator: the best classifier model
"""
def grid_tune_svm(train_data, train_labels):

    # # define the grid over which we will search
    # c_range = [1,2,4,8]
    # gamma_range = [.0156,.0315,.0625,.125,.25,.5,1,2]
    # # c_range = [1,2,4]
    # # gamma_range = [.0156,.0315,.0625,.125]
    # grid = [{"C": c_range, "gamma": gamma_range}]


    # define the grid over which we will search
    # c_range = [1,2,4,8]
    nu_range = [.05,.1,.15,.2]
    gamma_range = [1e-5,1e-3,1e-1,1]
    # c_range = [1,2,4]
    # gamma_range = [.0156,.0315,.0625,.125]
    grid = [{"nu": nu_range, "gamma": gamma_range}]


    # define the scoring function for model evaluation
    # Options see: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = 'f1'

    #initiate svc object
    #TODO: with more interictal data, we will want to tune a one class SVM
    svr = svm.NuSVC(class_weight='balanced', kernel='rbf')

    # initiate grid search object
    clf = GridSearchCV(svr,grid,scoring=scoring)
    clf.fit(train_data,train_labels)

    print "\t\t\tThe best parameters are: ", clf.best_params_

    #return the best model
    return clf.best_estimator_

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
            {-1: asleep data; 0: awake data; 1: ictal data}

"""
def label_classes(num_windows, preictal_time, postictal_time, win_len, win_overlap, seizure_time, file_type):

    if file_type is 'ictal':

        labels = np.empty(num_windows)

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

        # label the seizure period
        labels[seizure_start_window:seizure_end_window+1] = np.ones(seizure_end_window + 1 - seizure_start_window)

        # label the preictal(and interictal period if that exists) period of ictal file
        if seizure_start_time > preictal_time:  # if there is a long period before seizure onset

            preictal_start_time = seizure_start_time - preictal_time
            preictal_start_win = int((preictal_start_time - win_len) / (win_len - win_overlap) + 1)

            labels[preictal_start_win:seizure_start_window] = np.ones(seizure_start_window - preictal_start_win)
            labels[:preictal_start_win] = np.zeros(preictal_start_win)

        else:
            labels[:seizure_start_window] = np.ones(seizure_start_window)


        # label postictal period
        postictal_period = (num_windows - seizure_end_window) * (win_len - win_overlap)

        if postictal_period > postictal_time:

            postictal_end_time = seizure_end_time + postictal_time
            postictal_end_win = int((postictal_end_time - win_len) / (win_len - win_overlap)+1)

            if postictal_end_win > num_windows - 1:
                postictal_end_win = num_windows - 1

            labels[seizure_end_window+1:postictal_end_win+1] = np.ones(postictal_end_win - seizure_end_window)
            labels[postictal_end_win+1:] = np.zeros(num_windows - 1 - postictal_end_win)

        else:
            labels[seizure_end_window+1:] = np.ones(num_windows - 1 - seizure_end_window)

    elif file_type is 'awake':
        labels = np.zeros(num_windows)

    elif file_type is 'sleep':
        labels = -np.ones(num_windows)

    return labels

"""
offline_training()

Purpose: the pipeline directing the offline training portion of this algorithm on the known training data

Inputs:
    cv_file_names: list of strings containing the names of the training files
    cv_file_type: list of strings containing the type of each training file {'ictal','awake','sleep'}
    cv_seizure_times: list of tuples indicating the start and end time of each seizure or None if interictal file
    cv_test_files: list of ndarrays containing the raw iEEG data for each training file
    win_len: float indicating the window length
    win_overlap: float indicating the window overlap
    f_s: int indicatin the sampling frequency
    freq_band: list indicating the frequencies over which to compute coherency
    test_index: the index of the test file (used only for naming visualization files)
    patient_id: string indicating name of patient
    pred_time: the amount of time after a seizure during which no other flag is raised AND the amount of time before a
                seizure where the decision is counted as a prediction and not a false positive
    preictal_time: float, the amount of time in seconds before a seizure that is labeled as ictal data
    postictal_time: float, the amount of time in seconds after a seizure that is labeled as ictal data

Outputs:
    mean_coherency_matrix: ndarray of means (number of channels x number of channels) used to normalize the coherency matrices
    sd_coherency_matrix: ndarray of standard deviations (number of channels x number of channels) used to normalize the coherency matrices
    best_clf: classification model
    threshold: float indicating level above which to flag seizre
    adapt_rate: int indicating length of time over which to average the predicted labels
    training_evc_cv_files: list of ndarrays containing all of the training features (each entry is number of windows x number of channels)
"""
def offline_training(cv_file_names, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, freq_band, test_index, patient_id, pred_time, preictal_time, postictal_time):

    # get gardner after es file stored with same win_len and win_overlap with Burns
    # for each file in cv_file_names
    training_es_cv_files = [None] * len(cv_file_names)
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(data_path, 'energy_stats', patient_id)

    for i, file in enumerate(cv_file_names):

        file_begin = os.path.join(save_path, os.path.splitext(os.path.basename(file))[0])
        load_path_gardner = os.path.join(to_data, 'data', 'energy_stats', patient_id, file_begin + '.es')

        # read_energy_file takes in the data path as input
        X_feat = read_energy_file(load_path_gardner).T
        X_feat = list(X_feat)
        training_es_files = [None] * len(X_feat)
        for j, X_file in enumerate(X_feat):
            X_file = X_file.T
            training_es_files[j] = X_file
        training_es_cv_files[i] = training_es_files
    training_es_cv_files = [np.hstack(es) for es in training_es_cv_files]


    print '\t\tTraining the classification model'
    training_data = training_es_cv_files
    training_labels = [label_classes(training_data[i].shape[0], preictal_time, postictal_time, win_len, win_overlap, cv_seizure_times[i], cv_file_type[i]) for i in xrange(len(cv_test_files))]
    best_clf = grid_tune_svm(np.vstack(training_data),np.hstack(training_labels))

    print '\t\tTraining the decision model'
    threshold, adapt_rate = tune_pi(best_clf, cv_file_type, cv_seizure_times, training_data, win_len, win_overlap, f_s, pred_time, verbose=1)

    return best_clf, threshold, adapt_rate, training_data

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

Outpus:
    No output. The graph will either be visualized or be saved; graph contains three subplots: the raw iEEG data,
    the outlier fraction, and the decision rule. All share an x axis
"""
def viz_single_outcome(decision, out_frac, raw_ieeg, test_times, thresh, test_index, patient_id, f_s):

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

    plt.savefig('single_Patient_{}_file_{}'.format(patient_id,test_index))
    plt.clf()
    # plt.show()
    return

"""
viz_many_outcomes()

Purpose: to visualize all of the outlier fractions of the training data

Inputs:
    all_outlier_fractions: list of one dimensional ndarrays containing many files' outlier fractions
    seizure_times: list of tuples indicating start and end time of seizure or None for interictal files
    patient_id: string containing the patient name
    threshold: int indicating the threshold above which to flag a seizure
    test_index: int indicating which file currently being tested

Outputs:
    No output. The plot will either be visualized or saved. The single plot will contain subplots of all given
    outlier fractions.
"""
def viz_many_outcomes(all_outlier_fractions, seizure_times, patient_id, threshold, test_index):

    # initialize subplots
    fig, axes = plt.subplots(len(all_outlier_fractions))

    # visualize each outlier fraction
    for i in xrange(len(all_outlier_fractions)):
        this_out_frac = np.asarray(all_outlier_fractions[i])
        axes[i].plot(this_out_frac)
        axes[i].set_ylim(ymin=0, ymax=1)

        # mark the seizure times and determined threshold
        if seizure_times[i] is not None:
            axes[i].axvline(x=seizure_times[i][0], lw=2, c='r')
        axes[i].axhline(y=threshold, lw=2, c='k')

    # plt.show()
    plt.savefig('all_Patient_{}_file_{}'.format(patient_id,test_index))
    plt.clf()
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
def update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients, freq_band, preictal_time, postictal_time, pred_time):

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
    f.write('Prediction/persistance time is {}\n\n'.format(pred_time/f_s))

    # write the patients
    f.write('Patients are ' + " ".join(patients) + "\n\n")
    f.close()
    return

"""
performance_stats()

Purpose: to calculate the sensitivity, false positives, total interictal time elapsed and latency given a decision array

Inputs:
    decision: one dimensional ndarray containing {-1: no seizure; 1: seizure} predictions
    seizure_time: tuple of actual start and end time of seizure described by the input decision
    f_s: int indicating the sampling frequency
    pred_time: the amount of time after a seizure during which no other flag is raised AND the amount of time before a
                seizure where the decision is counted as a prediction and not a false positive

Outputs:
    detection_sensitivity: float indicating if seizure was detected
    prediction_sensitivity: float indicated if seizure was predicted
    latency: float indicating the length of time in seconds that the decision did or did not predict seizure
    FP: total number of false positives in decision
    time: total elapsed time of all interictal data

"""
def performance_stats(decision, seizure_time, f_s, pred_time):

    # if inter-ictal file
    if seizure_time is None:

        # get the amount of time in the interictal file and the number of false positives
        time = float(decision.size) / float(f_s * 60. * 60.)
        FP = float(np.size(decision[decision>0])) / float(f_s)

        prediction_sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        detection_sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        latency = np.nan # latency is meaningless since there is no seizure

    else:

        # computing the seizure start time and end time in units of samples
        seizure_start = int(f_s * seizure_time[0])
        seizure_end = int(f_s* seizure_time[1])

        # determining the amount of time before a seizure occurs where a false positive is countred
        false_positive_range = int(max(0, seizure_start - pred_time*f_s))
        false_positive_data = np.copy(decision[:false_positive_range])

        # calculating the time in which false positives could occur and the number of false positives
        time = float(false_positive_range) / float(f_s * 60. * 60.)
        FP = float(np.size(false_positive_data[false_positive_data > 0])) / float(f_s)

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
def choose_best_channels(patient_id, labels):
    #TODO: experiement with other ways of choosing channels.

    good_channel_dict = {'TS041':['LAH2','LAH3','LAH4','LPH1','LPH2',],
                         'TS039':['RAH1','RAH2','RAH3','RPH2','RPH3','RPH4',],
                         'TA023':['MST1','MST2','HD1','TP1']}
    dimensions_to_keep = [labels.index(good_channel_dict[patient_id][i]) for i in
                          xrange(len(good_channel_dict[patient_id]))]

    return dimensions_to_keep

"""
analyze_patient_raw()

Purpose: read in the raw iEEG data, file names, file types, and seizure times from the stored .edf and pickle files

Inputs:
    data_path: file path that leads to where .edf and pickle files are stored
    f_s: integer indicating sampling frequency
    include_awake: boolean indicating if awake interictal data should be included for patient
    include_asleep: boolean indicating if asleep interictal data should be included for patient
    long_interictal: boolean indicating if patient has a long (>15 min) interictal file that needs to be read
    patient_id: string containing the name of the patient
    win_len: float indicating window length
    win_overlap: float indicating window overlap

Outputs:
    all_files: list of ndarrays containing raw iEEG data from edf files
    data_filenames: list of strings indicating names of .edf file s
    file_type: list of strings indicating file type; can be {'ictal','awake','sleep'}
    seizure_times: list of tuples indicating start and end time of seizure; None if interictal file

"""
def update_list(original, update):
    count = 0
    for a in update:
        place = a[0] + count
        original = original[:place] + a[1] + original[place+1:]
        count += len(a[1]) - 1
    return original

def analyze_patient_raw(data_path, f_s, include_awake, include_asleep, long_interictal, patient_id, win_len, win_overlap):

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

    for i, seizure_file in enumerate(data_filenames):
        # if there is a long interictal file (>15 minutes) present that needs to be read in in chunks
        if long_interictal and not (file_type[i] is 'ictal'):

            min_per_chunk = 15
            sec_per_min = 60

            print '\tFile %d is long...' % (i + 1)

            j = 0
            while True:
                # get chunk start and end times
                # start = j * sec_per_min * min_per_chunk
                # end = (j + 1) * sec_per_min * min_per_chunk
                #
                # try:
                #     # extract the chunk
                #     print '\t\tChunk ' + str(j + 1) + ' reading...\n',
                #     X_chunk, _, labels = edfread(seizure_file, rec_times=[start, end])
                #     dimensions_to_keep = choose_best_channels(patient_id, labels)
                #     X_chunk = X_chunk[:, dimensions_to_keep]
                    X_chunk = np.ones((300000,150))

                    # update file information
                    all_files.append(X_chunk)
                    tmp_data_filenames += [seizure_file]
                    tmp_file_type += [file_type[i]]
                    tmp_seizure_times += [seizure_times[i]]

                    j += 1
                    # if less than an entire chunk was read, then this is the last one!
                #     if X_chunk.shape[0] < sec_per_min * min_per_chunk:
                #         break
                # except ValueError:
                #     break  # the start was past the end!



        else:
            print '\t\tFile %d reading...\n' % (i + 1),
           # read data in
           #  X, _, labels = edfread(seizure_file)
           #  dimensions_to_keep = choose_best_channels(patient_id, labels)
           #  X = X[:, dimensions_to_keep]
            X = np.ones((300000,150))

            # update file information
            all_files.append(X)  # add raw data to files
            tmp_data_filenames += [seizure_file]
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

    # setting parameters
    win_len = 3.0  # seconds
    win_overlap = 2.0  # seconds
    f_s = float(1e3)  # Hz

    patients = ['TS041'] # can only do one patient at a time
    long_interictal = [False]
    include_awake = True
    #TODO: try with asleep data
    include_asleep = True

    #TODO: tune the frequency band
    freq_band = [30,35] # Hz
    pred_time = 5 * 60 # minutes times seconds
    preictal_time = 5 * 60 # minutes times seconds
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
    update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients,freq_band, preictal_time, postictal_time, pred_time)

    # evaluate each patient
    for patient_index, patient_id in enumerate(patients):

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)

        print 'Retreiving stored raw data'
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(p_data_path, f_s, include_awake, include_asleep, long_interictal[patient_index], patient_id, win_len, win_overlap)
        number_files = len(all_files)

        # intializing performance stats
        prediction_sensitivity = np.zeros(len(all_files))
        detection_sensitivity = np.zeros(len(all_files))
        latency = np.zeros(len(all_files))
        fp = np.zeros(len(all_files))
        time = np.zeros(len(all_files))

        # begining leave one out cross-validation
        # for i in xrange(number_files):
        for i in xrange(number_files):
            # determine training vs. testing data for for this k-fold
            print '\nCross validations, k-fold %d of %d ...' % (i+1, number_files)
            testing_file = data_filenames[i]
            test_file = all_files[i]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_test_files = all_files[:i] + all_files[i+1:]
            cv_file_type = file_type[:i] + file_type[i+1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i+1:]

            print '\tEntering offline training'
            cs_svm, threshold, adapt_rate, traindat = offline_training(cv_file_names, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, freq_band, i, patient_id, pred_time, preictal_time, postictal_time)

            print'\tEntering online testing'
            test_decision_windows, test_outlier_frac_windows = online_testing(testing_file, cs_svm, threshold, adapt_rate, win_len, win_overlap, f_s, freq_band, i, patient_id, pred_time)
            print'\tEnded online testing'

            print'\tCalculating performance stats'
            test_decision_sample = window_to_samples(test_decision_windows, win_len, win_overlap, f_s)
            test_outlier_frac_sample = window_to_samples(test_outlier_frac_windows, win_len, win_overlap, f_s)
            prediction_sensitivity[i], detection_sensitivity[i], latency[i], fp[i], time[i] = performance_stats(test_decision_sample, seizure_times[i], f_s, pred_time)

            print '\tBeginning visualization process'
            all_fracs = []
            for file in traindat:
                labels = cs_svm.predict(file)
                all_fracs += [create_outlier_frac(labels, adapt_rate)]
            viz_many_outcomes(all_fracs, cv_seizure_times, patient_id, threshold, i)
            viz_single_outcome(test_decision_sample, test_outlier_frac_sample, test_file[:,0], seizure_times[i], threshold, i, patient_id, f_s)
            print '\tPrediction sensitivity = ', prediction_sensitivity[i], 'Detection sensitivity = ', detection_sensitivity[i], 'Latency = ', latency[i], 'FP = ', fp[i], 'Time = ', time[i]

        #fpr = float(np.nansum(fp)) / float(np.nansum(time))
        #print 'Mean predition sensitivity = ', np.nanmean(prediction_sensitivity), 'Mean detection sensitivity = ', np.nanmean(detection_sensitivity), 'Mean latency = ', np.nanmean(latency), 'Mean FPR = ', fpr
        #print 'Median prediction sensitivity = ', np.nanmedian(prediction_sensitivity), 'Median detection sensitivity = ', np.nanmedian(detection_sensitivity), 'Median latency = ', np.nanmedian(latency)
        #update_log_stats(log_file, patient_id, prediction_sensitivity, detection_sensitivity, latency, fp, time)

if __name__ == '__main__':
    parent_function()

    #1: fix the bug that just came up with labels; finish testing long interictal/ictal files; try gardner with this pipeline; try burns+gardner