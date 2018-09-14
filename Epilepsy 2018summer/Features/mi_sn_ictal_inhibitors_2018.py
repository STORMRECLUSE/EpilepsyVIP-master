import sys
import os
import numpy as np
import pickle
import math
import scipy
from scipy import io
sys.path.insert(0,r"/Users/stormrecluse/Desktop/Epilepsy 2018summer")
from Features.Mutual_Information.centralities import compute_katz, compute_eigen_centrality, compute_stats, eigen, pagerank_centrality
from Features.Mutual_Information.preprocessing import get_freq_bands,notch_filter_data
from Features.Mutual_Information.mutual_information import window_data, cross_channel_MI
from DataInterfacing.edfread import edfread
from sklearn import svm
import os, pickle, sys
from ictal_inhibitors_2018 import classifier_gridsearch, label_classes, window_to_samples, choose_best_channels, performance_stats,find_katz, analyze_patient_raw
# from Experiments.extract_scattering import extract_scattering_coefficients
import matplotlib.pyplot as plt









########################################################## OFF LINE TRAINING BLOCK ##########################################################

"""
filename should correspond to the keys of the .mat file
returns a list of MI matrices for frequencies requested
"""

def get_MI_features(patient_id, filename, freq_bands = ["theta", "gamma"], chunk_len = 300):
    # h_path = "/Volumes/Brain_cleaner/Seizure Data/data"

    # TODO: change this to whatever directory you store the raw patient data
    h_path = "/Users/stormrecluse/Desktop/Epilepsy 2018summer/Data"
    d_path = os.path.join(h_path, patient_id)

    # TODO: modify the directories based on where you place the Epilepsy VIP folder
    if int(chunk_len) == 300:
        data_mat = scipy.io.loadmat(os.path.join("/Users/Stormrecluse/Desktop/Epilepsy 2018summer/Features/MI_features", patient_id + "CMI_5m_30shift.mat"))[filename]
    if int(chunk_len) == 180:
        data_mat = scipy.io.loadmat(os.path.join("/Users/Stormrecluse/Desktop/Epilepsy 2018summer/Features/MI_features", patient_id + "CMI_3m_30shift.mat"))[filename]
    all_band_names, bands = get_freq_bands()

    for i, band_name in enumerate(freq_bands):
        band_index = all_band_names.index(band_name)
        if i == 0:
            MI = data_mat[:, [band_index], :, :]
        else:
            MI = np.hstack((MI, data_mat[:, [band_index], :, :]))
    return np.array(MI)


def find_normalizing_MI(matrix):
    # compute the mean of each entry along the third dimension
    mean_mat = np.mean(matrix, axis= 0)

    # compute the standard deviation of each entry along the third dimension
    std_mat = np.std(matrix, axis= 0)

    return mean_mat, std_mat


def transform_coherency(coherencies_list, mean, std):
    # fail-safe to avoid dividing by zero
    std[std == 0] = 1
    transformed_coherencies = []

    # for each file's coherency matrices...
    for coherency_matrices_one_file in coherencies_list:

        # for each window's coherency matrix...
        num_windows = coherency_matrices_one_file.shape[0]

        for i in xrange(num_windows):
            matrix = coherency_matrices_one_file[i, :,  :, :].copy()

            # normalize the matrix. This is done according to Burns et. al.
            matrix -= mean
            matrix = np.divide(matrix, std)
            matrix = np.divide(np.exp(matrix), 1 + np.exp(matrix))

            # store all transformed coherence matrices for this file
            coherency_matrices_one_file[i, :, :, :] = matrix

        # store all transformed coherence matrices for all files
        transformed_coherencies += [coherency_matrices_one_file]

    return transformed_coherencies


def find_centrality_multibands(training_MI_files, centrality_type = 'katz'):
    # input: list of (n_samples for this file, num_freq, num_channels, num_channels)
    # output: list of (n_samples for this file, num_freq, num_channels)
    training_centrality_files = []

    for file in training_MI_files:
        n_samples, n_freq, n_channels, _ = file.shape
        interictal_centrality = np.zeros((n_samples, n_freq, n_channels))
        for i in range(n_samples):
            for j in range(n_freq):
                # TODO: experiment with different centrality values. functions are imported from centralities.py. find_centralities_multiband only supports the centrality functions not the stats one.

                # interictal_centrality[i, j, :] = compute_transitivity(file[i, j, :, :])
                interictal_centrality[i, j, :] = compute_katz(file[i, j, :, :])
                # interictal_centrality[i, j, :] = eigen(file[i, j, :, :]);

        r_interictal_centrality = np.reshape(interictal_centrality, (n_samples, n_freq * n_channels))
        training_centrality_files.append(r_interictal_centrality)

    return training_centrality_files


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
        labels[seizure_start_window:seizure_end_window + 1] = -np.ones(seizure_end_window + 1 - seizure_start_window)

        # label the preictal (and interictal period if that exists) period
        if seizure_start_time > preictal_time + win_len:  # if there is a long period before seizure onset

            # determine the time in seconds where preictal period begins
            preictal_start_time = seizure_start_time - preictal_time

            # determine the time in windows where preictal period begins
            preictal_start_win = int((preictal_start_time - win_len) / (win_len - win_overlap) + 1)

            # label the preictal time
            labels[preictal_start_win:seizure_start_window] = -np.ones(seizure_start_window - preictal_start_win)

            # label the interical time
            labels[:preictal_start_win] = np.ones(preictal_start_win)

        else:  # if there is not a long time in file before seizure begins
            # label preictal time
            labels[:seizure_start_window] = -np.ones(seizure_start_window)

        # determining how long the postical period lasts in seconds
        postictal_period = (num_windows - seizure_end_window) * (win_len - win_overlap)

        # if there is a long period of time after seizure in the file
        if postictal_period > postictal_time:

            # determine where in seconds the postical period ends
            postictal_end_time = seizure_end_time + postictal_time

            # determine where in windows the postical period ends
            postictal_end_win = int((postictal_end_time - win_len) / (win_len - win_overlap) + 1)

            # in the case that the postictal end window exceeds the maximum number of windows...
            if postictal_end_win > num_windows - 1:
                postictal_end_win = num_windows - 1

            # label the postical period
            labels[seizure_end_window + 1:postictal_end_win + 1] = -np.ones(postictal_end_win - seizure_end_window)

            # label the interictal period
            labels[postictal_end_win + 1:] = np.ones(num_windows - 1 - postictal_end_win)

        else:  # if there is a short amount of time in the file after the seizure ends
            # label the postictal period
            labels[seizure_end_window + 1:] = -np.ones(num_windows - 1 - seizure_end_window)

    # label awake interictal files
    elif file_type is 'awake':
        labels = np.ones(num_windows)

    # label asleep interictal files
    elif file_type is 'sleep':
        labels = np.ones(num_windows)

    # return the data labels
    return list(labels)


def offline_training(cv_file_type, cv_file_names, cv_file_idxs, cv_seizure_times, cv_train_files, chunk_len, chunk_overlap,
                     win_len, win_overlap, f_s, i,patient_id, persistence_time, preictal_time, postictal_time,
                     freq_bands = ["theta", "gamma"], svm_kernel = 'rbf'):

    calc_features_local = 0
    calc_params_local = 1
    first_file = 0

    # read pre-calculated features
    if not calc_features_local:
        print'\t\tBuilding MI matrices for all training files'
        training_MI_cv_files = []
        for n, small_test_file in enumerate(cv_train_files):
            # fetch data
            filename = cv_file_names[n]
            key = str(cv_file_idxs[n]) + "_" + filename.split("/")[-1]
            test_file_MI = get_MI_features(patient_id, key, freq_bands = freq_bands, chunk_len = chunk_len)    #(111, 2, 6, 6)

            training_MI_cv_files += [test_file_MI]

            # store specifically interictal MI matrices
            if cv_file_type[n] is not "ictal":
                if first_file == 0:
                    # interictal files are for finding normalizing parameters
                    interictal_MI_files = test_file_MI     # should be all number of samples, 2 * 6 * 6
                    first_file = 1
                else:
                    interictal_MI_files = np.vstack((interictal_MI_files,
                                                      test_file_MI))

    # After getting okay results: implement local feature calculation
    else:
        raise ValueError("Eh-oh, local feature calculation not implemented yet.")


    print '\t\tFinding mean and standard deviation of interictal features'
    mean_MI_matrix, sd_MI_matrix = find_normalizing_MI(interictal_MI_files)

    print'\t\tTransforming all coherency matrices'
    transformed_MI_cv_files = transform_coherency(training_MI_cv_files, mean_MI_matrix,
                                                         sd_MI_matrix)
    training_katz_cv_files = find_centrality_multibands(transformed_MI_cv_files)  # should be list of (n_samples, 2, 6, 6)

    # initializations
    training_data = training_katz_cv_files
    interictal_indices = []
    seizure_indices = []

    # stack all interictal data
    interictal_data = np.vstack(
        (training_data[ind] for ind in xrange(len(training_data)) if cv_file_type[ind] is not 'ictal'))

    # organizing the data and labeling it
    # create label dictionary to store labels of CV training files
    cv_label_dict = {}
    # label each training file
    for index in xrange(len(cv_train_files)):

        # if file is seizure, store this training file index to seizure indices
        if cv_file_type[index] == "ictal":
            seizure_indices.append(index)

        # otherwise, store this training file index to interictal indices
        else:
            interictal_indices.append(index)

        # label the windows in the file
        cv_label_dict[index] = label_classes(len(training_data[index]), preictal_time, postictal_time, chunk_len,
                                             chunk_overlap, cv_seizure_times[index], cv_file_type[index])

    print '\t\tTraining the classifier'
    # tuning the SVM parameterss
    bestnu, bestgamma = classifier_gridsearch(training_data, cv_label_dict, seizure_indices, interictal_indices, svm_kernel)

    # define your classifier
    best_clf = svm.OneClassSVM(nu=bestnu, kernel=svm_kernel, gamma=bestgamma)

    # train your classifier
    best_clf.fit(interictal_data)
    print "shape of interictal data: ", interictal_data.shape

    return best_clf, mean_MI_matrix, sd_MI_matrix


def online_testing(feature,f_s,i,pred_time,label_index,alarm_timer,best_clf, chunk_len = 5*60, chunk_overlap = 270):
    testing_label = best_clf.predict(feature)
    # switch the labels to our code's convention: 0 is normal, 1 is seizure
    testing_label[testing_label == 1] = 0
    testing_label[testing_label == -1] = 1


    # determining where the outlier fraction meets or exceeds the threshold
    if alarm_timer <= 0:
        if testing_label == 1:
            decision = 1  # predicted a seizure
            alarm_timer = pred_time  # set the refractory period
        else:
            decision = -1  # did not predict a seizure

    else:  # do not output a positive prediction within an alarm_timer period of time of the last positive prediction
        decision = -1
        alarm_timer -= 1

    return decision, label_index, alarm_timer

def parent_function():
    # TODO: set patients
    patients = ["TS068"]  # set which patients you want to test
    # show_fig = 0  # if 1, figures show; if 0, figures save to current working directory

    # create paths to the data folder
    to_data = os.path.dirname(os.getcwd())
    data_path = os.path.join(to_data, 'Data')
    save_path = os.path.join(to_data, 'Features')
    features_path = os.path.join(save_path, 'StoredFeatures')

    # setting model parameters
    # TODO: some patients also have 5 min windowing available. If you want to play with it, chase chunk_len to 300 and chunk_overlap to 270
    chunk_len = 180
    chunk_overlap = 150

    # MI parameters
    mi_win_len = 0.25  # seconds
    mi_win_overlap = 0  # seconds
    f_s = float(1e3)  # Hz

    # TODO: set frequency bands here. Mapping see function get_freq_bands(). Delta band is not available now!
    freq_bands = ["theta","beta","gamma"]

    persistence_time = 300/(chunk_len - chunk_overlap) + 1
    # persistence_time = 3.33 * 60  # minutes times seconds, the amount of time after a seizure prediction for which no alarm is raised
    preictal_time = 5 * 60  # minutes times seconds, the amount of time prior to seizure onset defined as preictal
    postictal_time = 5 * 60  # minutes times seconds, the amount of time after seizure end defined as postictal
    include_awake = True
    include_asleep = True

    # TODO: set rbf kernel here.
    svm_kernel = 'linear'

    # evaluate each patient
    for patient_index, patient_id in enumerate(patients):

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths to be specific to each patient
        p_data_path = os.path.join(data_path, patient_id)

        print 'Retreiving stored raw data'
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(p_data_path, f_s, include_awake,
                                                                                  include_asleep, patient_id, chunk_len,
                                                                                  chunk_overlap, calc_train_local=True)
        number_files = len(all_files)

        # intializing performance stats
        prediction_sensitivity = np.zeros(len(all_files))
        detection_sensitivity = np.zeros(len(all_files))
        latency = np.zeros(len(all_files))
        fp = np.zeros(len(all_files))
        times = np.zeros(len(all_files))

        # beginning leave one out cross-validation
        for i in xrange(number_files):

            print '\nCross validations, k-fold %d of %d ...' % (i + 1, number_files)
            # defining which files are training files vs testing files for this fold
            testing_file = all_files[i]
            testing_file_idx = i
            cv_file_names = data_filenames[:i] + data_filenames[i + 1:]
            cv_file_idxs = range(i) + range(i+1,number_files)

            cv_train_files = all_files[:i] + all_files[i + 1:]
            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            print '\tEntering offline training'
            my_svm, mean_MI_matrix, sd_MI_matrix = offline_training(cv_file_type, cv_file_names,  cv_file_idxs, cv_seizure_times, cv_train_files, chunk_len, chunk_overlap, mi_win_len, mi_win_overlap, f_s, i, patient_id,
                                      persistence_time, preictal_time, postictal_time, freq_bands, svm_kernel)


            print'\tEntering online testing'
            # computing prediction on testing file for this fold

            print '\tCalculating testing features locally'
            # determine how many samples, windows, and channels we have in this test file
            total_test_samples = testing_file.shape[0]
            chunk_len_samples = chunk_len * f_s
            chunk_ovlap_samples = chunk_overlap * f_s
            num_chunks = int(math.floor(float(total_test_samples) / float(chunk_len_samples - chunk_ovlap_samples)))
            num_channels = testing_file.shape[1]


            # load test file
            test_key = str(testing_file_idx) + "_" + data_filenames[testing_file_idx].split("/")[-1]
            test_MI = get_MI_features(patient_id, test_key, freq_bands = freq_bands, chunk_len = chunk_len)

            # transform (normalize) MI matrix
            transformed_MI_test = transform_coherency([test_MI], mean_MI_matrix,
                                                          sd_MI_matrix)

            test_features = find_centrality_multibands(transformed_MI_test)[0]  # should be list of (n_samples, 2, 6, 6)  # for loop to process each window in the test file
            # initializing for computing performance metrics
            # full_file_decision = np.zeros(num_chunks)
            t_samples = test_features.shape[0]
            full_file_decision = np.zeros(t_samples)

            alarm_timer = 0

            for index in np.arange(t_samples):
                # getting the single window of data for this iteration of the for loop
                feature = test_features[index].reshape(1, -1)
                decision, label_index, alarm_timer = online_testing(feature, f_s, testing_file_idx, persistence_time, index,
                                                                    alarm_timer, my_svm)

                # storing the outlier fraction and decision for calculating performance metrics and visualization
                full_file_decision[index] = decision

            # using outputs from test file to compute performance metrics
            print'\tCalculating performance stats'

            print "\tFile Type: ", file_type[i]

            print "\t Full File Decision: ", full_file_decision

            # convert from units of windows to units of samples
            test_decision_sample = window_to_samples(full_file_decision, chunk_len, chunk_overlap, f_s)

            # find performance metrics for this fold of cross validation
            prediction_sensitivity[i], detection_sensitivity[i], latency[i], fp[i], times[i] = performance_stats(
                test_decision_sample, seizure_times[i], f_s, preictal_time, chunk_len, chunk_overlap)


            # print the performance metrics and visualize the algorithm output on a graph
            print '\tPrediction sensitivity = ', prediction_sensitivity[i], 'Detection sensitivity = ', \
            detection_sensitivity[i], 'Latency = ', latency[i], 'FP = ', fp[i], 'Time = ', times[i]
            # viz_single_outcome(test_decision_sample, test_outlier_frac_sample, testing_file[:,0], seizure_times[i], threshold, i, patient_id, f_s)

        # compute false positive rate
        fpr = float(np.nansum(fp)) / float(np.nansum(times))

        # print mean and median performance metrics
        print '\nMean prediction sensitivity = ', np.nanmean(
            prediction_sensitivity), 'Mean detection sensitivity = ', np.nanmean(
            detection_sensitivity), 'Mean latency = ', np.nanmean(latency), 'Mean FPR = ', fpr
        print 'Median prediction sensitivity = ', np.nanmedian(
            prediction_sensitivity), 'Median detection sensitivity = ', np.nanmedian(
            detection_sensitivity), 'Median latency = ', np.nanmedian(latency)


        # save performance metrics to the log text file
        # update_log_stats(log_file, patient_id, prediction_sensitivity, detection_sensitivity, latency, fp, times)

        # visualize all of the testing file outputs on a single graph
        # viz_many_outcomes(all_test_out_fracs, seizure_times, patient_id, all_thresholds, f_s, show_fig)

if __name__ == '__main__':
    parent_function()
