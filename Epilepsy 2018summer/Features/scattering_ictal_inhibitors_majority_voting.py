import sys
sys.path.insert(0,r"/Users/stormrecluse/Desktop/Epilepsy 2018summer")
import os
import numpy as np
import pickle
import math
import scipy
from scipy import io
from Features.Mutual_Information.centralities import compute_katz, compute_eigen_centrality,compute_stats, eigen
from sklearn import svm
import os, pickle, sys
from ictal_inhibitors_2018 import classifier_gridsearch, choose_best_channels, performance_stats,create_outlier_frac, update_log_stats, create_decision, window_to_samples
from Features.Mutual_Information.mutual_information import window_data, cross_channel_MI
from Features.Mutual_Information.preprocessing import notch_filter_data
import matplotlib.pyplot as plt
from ictal_inhibitors_2018 import find_katz, analyze_patient_raw
from scipy import stats

# Note: when tuning SVM, make sure the scoring function is roc_auc and the grid range I am using is
# gamma_range = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 500, 1000]
# nu_range = [5e-6, 1e-5, 5e-5, 0.001 ,0.01 ,0.05, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9]

x=scipy.io.loadmat("/Users/stormrecluse/Desktop/Epilepsy 2018summer/Features/ScatteringPV/TA023_scatteringcoeffP_channel0.mat")
print x

def plot_single_channel(fig, channel_idx, testing_file, file_seizure_time, seizure_start_sample, seizure_end_sample, test_label_sample_indices, full_file_labels, channel_names, num_samples):
    ax = fig.add_subplot(9, 1, channel_idx + 1)
    ax.plot(testing_file[:num_samples, channel_idx])
    ax.set_ylabel(channel_names[channel_idx])
    if file_seizure_time != None:
        ax.axvline(x=seizure_start_sample, color='r')
        ax.axvline(x=seizure_end_sample, color='r')



def score_decision_my(sensitivity, fp, time, persistence_time):
    # get aggregate mean on sensitivity and false positive rate
    S = np.nanmean(sensitivity)
    FPR = float(np.nansum(fp)) / float(np.nansum(time))

    # calculate quantitative descriptor
    alpha1 = S
    alpha2 = -FPR * persistence_time / 3600
    score = alpha1 + alpha2

    return score

"""
preictal time and postictal time should be in seconds.
"""

def label_classes_samples(num_windows, preictal_time, postictal_time, win_len_samples, win_overlap_samples, seizure_time, file_type, f_s = 1000):

    # labeling the seizure files
    if file_type is 'ictal':

        labels = np.empty(num_windows)

        # determine seizure start/end times in seconds
        seizure_start_time = seizure_time[0]
        seizure_end_time = seizure_time[1]

        # convert into samples
        preictal_time_samples = preictal_time * f_s
        postictal_time_samples = postictal_time * f_s
        seizure_start_time_samples = seizure_start_time * f_s
        seizure_end_time_samples = seizure_end_time * f_s

        # determining which window the seizure starts in
        if seizure_start_time_samples < win_len_samples:
            seizure_start_window = 0
        else:
            seizure_start_window = int((seizure_start_time_samples - win_len_samples) / (win_len_samples - win_overlap_samples) + 1)

        # determining which window the seizure ends in
        if seizure_end_time_samples < win_len_samples:
            seizure_end_window = 0
        else:
            seizure_end_window = int((seizure_end_time_samples - win_len_samples) / (win_len_samples - win_overlap_samples) + 1)

        # in case the seizure end window is larger than the max window index
        if seizure_end_window > num_windows - 1:
            seizure_end_window = num_windows - 1

        # label the ictal period
        labels[seizure_start_window:seizure_end_window + 1] = -np.ones(seizure_end_window + 1 - seizure_start_window)

        # label the preictal (and interictal period if that exists) period
        if seizure_start_time_samples > preictal_time_samples + win_len_samples:  # if there is a long period before seizure onset

            # determine the time in seconds where preictal period begins
            preictal_start_time_samples = seizure_start_time_samples - preictal_time_samples

            # determine the time in windows where preictal period begins
            preictal_start_win = int((preictal_start_time_samples - win_len_samples)
                                     / (win_len_samples - win_overlap_samples) + 1)

            # label the preictal time
            labels[preictal_start_win:seizure_start_window] = -np.ones(seizure_start_window - preictal_start_win)

            # label the interical time
            labels[:preictal_start_win] = np.ones(preictal_start_win)

        else:  # if there is not a long time in file before seizure begins
            # label preictal time
            labels[:seizure_start_window] = -np.ones(seizure_start_window)

        # determining how long the postical period lasts in seconds
        postictal_period = (num_windows - seizure_end_window) * (win_len_samples - win_overlap_samples)

        # if there is a long period of time after seizure in the file
        if postictal_period > postictal_time_samples:

            # determine where in seconds the postical period ends
            postictal_end_time_samples = seizure_end_time_samples + postictal_time_samples

            # determine where in windows the postical period ends
            postictal_end_win = int((postictal_end_time_samples - win_len_samples)
                                    / (win_len_samples - win_overlap_samples) + 1)

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

def find_stats(training_MI_files):
    # input: list of (n_samples for this file, num_freq, num_channels, num_channels)
    # output: list of (n_samples for this file, num_freq, num_channels)
    training_centrality_files = []

    for file in training_MI_files:
        n_samples, n_freq, n_channels, _ = file.shape
        interictal_centrality = np.zeros((n_samples, n_freq, 10))
        for i in range(n_samples):
            for j in range(n_freq):
                interictal_centrality[i, j, :] = np.hstack(compute_stats(file[i, j, :, :]),
                                                           compute_eigen_centrality(file[i, j, :, :]))
            r_interictal_centrality = np.reshape(interictal_centrality, (n_samples, n_freq * 10))
        training_centrality_files.append(r_interictal_centrality)
    return training_centrality_files


"""
Find the mean and standard deviation across matrices.
"""

def find_normalizing_matrix(matrix):
    # compute the mean of each entry along the third dimension
    mean_mat = np.mean(matrix, axis= 0)

    # compute the standard deviation of each entry along the third dimension
    std_mat = np.std(matrix, axis= 0)

    return mean_mat, std_mat


"""
Loads and returns scattering features for a patient and one CV file.
"""

def get_sc_features_by_channel(patient_id, file_key, win_len_samples, win_overlap_samples, channel_idx):
    if (win_len_samples == 2**14) and (win_overlap_samples == 2**13):
        # load feature matrices from .mat files
        # TODO: change to the corresponding directory where you store these .mat files
        P_mat = scipy.io.loadmat(
            os.path.join("/Users/stormrecluse/Desktop/Epilepsy 2018summer/Features/ScatteringPV",
                         patient_id + "_scatteringcoeffP_channel{}.mat".format(channel_idx)))[
            file_key]  # shape is (num windows, 3, 32)

        V_mat = scipy.io.loadmat(
            os.path.join("/Users/stormrecluse/Desktop/Epilepsy 2018summer/Features/ScatteringPV/",
                         patient_id + "_scatteringcoeffV_channel{}.mat".format(channel_idx)))[file_key]  # shape is (num windows, 3, 32)
        # Examples of different operations on features
        # V features
        data_mat = P_mat    # P features
        # data_mat = V_mat    # V features
        # data_mat = np.concatenate([P_mat, V_mat], axis = 2)   # stack P, V

        new = np.log(np.reshape(data_mat, (data_mat.shape[0], data_mat.shape[1] * data_mat.shape[2])))
        # timediff = np.diff(new)
        # return np.vstack(np.array(new[0:, ]), timediff)
        return new   # shape should be (num_windows, 96)


"""
tune_decision()

Purpose: to tune the threshold and adaptation rate given a patient. Currently only tuning the threshold
and the adapt rate is fixed to 10 windows.

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


def tune_decision(mynus, mygammas, cv_data_list, cv_label_dict, seizure_indices, interictal_indices, win_len, win_overlap,
                  f_s, seizure_times, persistence_time, preictal_time):
    # folds for cross validation
    div = 4

    # initialize a very bad best score
    best_score = -float("inf")
    best_score2 = -float("inf")

    # define the search range
    threshold_range = np.arange(0, 1, .05)
    # adapt_range = np.arange(1, 20, 1)
    adapt_range = [10]

    # search over all combinations in search range
    for mythresh in threshold_range:
        for myadapt in adapt_range:

            all_scores = np.zeros(div)

            # cross validation folds for these parameters
            for mult in xrange(div):

                interictal_fit_data = []
                interictal_validate_data = []
                interictal_validate_labels = []

                # obtain the validate and fit sets of the interictal training files
                for index in interictal_indices:
                    # determine the length of this file
                    length = cv_data_list[index].shape[1]    # number of channels is in the 0th dimension
                    # determine the start/end indices of the validate set
                    start = int(np.floor(length * mult / div))
                    end = int(np.floor(length * (mult + 1) / div))

                    # obtain the fit data and the validate data for the interictal file
                    interictal_fit_data += [np.concatenate((cv_data_list[index][:, :start, :], cv_data_list[index][:, end:, :]), axis = 1)]
                    interictal_validate_data += [cv_data_list[index][:, start:end, :]]
                    interictal_validate_labels += [cv_label_dict[index][start:end]]

                # fit the model with stacked matrix of interictal data

                best_clfs = []
                for channel_idx in range(6):
                    # data of this channel
                    fit_data = np.concatenate([interictal_fit_data[i][channel_idx, :, :] for i in xrange(len(interictal_fit_data))], axis = 0)
                    # svm of this channel
                    best_clf = svm.OneClassSVM(nu=mynus[channel_idx], kernel='rbf', gamma=mygammas[channel_idx])
                    best_clf.fit(fit_data)
                    best_clfs.append(best_clf)


                # instantiate lists to keep track of all predicted labels and seizure times for the validate data
                pred_labels = []
                val_times = []

                # use the fit model to predict ictal files
                # loop through each seizure file
                for seiz_index in seizure_indices:
                    # # predict the labels of this seizure file
                    all_channel_data = cv_data_list[seiz_index]
                    one_seiz_pred_labels = predict_all_channels(all_channel_data, best_clfs)

                    # switch the labels to our code's convention: 1 is seizure, 0 is normal
                    one_seiz_pred_labels[one_seiz_pred_labels == 1] = 0
                    one_seiz_pred_labels[one_seiz_pred_labels == -1] = 1

                    # store the predicted labels and the correct seizure times
                    pred_labels += [one_seiz_pred_labels]
                    val_times += [seizure_times[seiz_index]]

                # use the fit model to predict all interictal fit data
                # get all interictal validate data
                inter_validate_data = np.concatenate(
                    [interictal_validate_data[i] for i in xrange(len(interictal_validate_data))], axis = 1)


                # predict the labels of all interictal validate data
                inter_pred_labels = predict_all_channels(inter_validate_data, best_clfs)

                # switch the labels to our code's convention: 0 is normal, 1 is seizure
                inter_pred_labels[inter_pred_labels == 1] = 0
                inter_pred_labels[inter_pred_labels == -1] = 1

                # store the predicted labels
                pred_labels += [inter_pred_labels]
                val_times += [None]

                # initializaing the performance statistic trackers
                pred_s = np.zeros(len(val_times))
                fp = np.zeros(len(val_times))
                time = np.zeros(len(val_times))

                # compute the outlier fractions, decisions, and performance metrics for this set of parameters
                # loop through each seizure file and the interictal validate set, each of which has an entry in pred_labels
                for ind, one_file_pred_labels in enumerate(pred_labels):
                    # find the outlier fraction for this set of predicted labels
                    outlier_fraction = create_outlier_frac(one_file_pred_labels, myadapt)

                    # determine the prediction decision
                    decision = create_decision(outlier_fraction, mythresh, persistence_time, win_len, win_overlap)

                    # transform that decision from units of windows to units of samples
                    decision_sample, decision_timing = window_to_samples_insamples(decision, win_len*f_s, win_overlap*f_s, f_s)
                    # find performance metrics
                    pred_s[ind], _, _, fp[ind], time[ind] = performance_stats(decision_sample, val_times[ind], f_s,
                                                                              preictal_time, win_len, win_overlap)

                # assign a score to this model using performance metrics
                all_scores[mult] = score_decision_my(pred_s, fp, time, persistence_time)

            # take the average score of all validate sets to represent the score of this model
            avg_score = np.mean(all_scores)

            print "Threshold value: ", mythresh
            print "Resulting score: ", avg_score

            # track the best performing set of parameters
            if avg_score > best_score:
                best_score = avg_score
                threshold = mythresh
                adapt_rate = myadapt
            if avg_score > best_score2:
                best_score2 = avg_score
                threshold2 = mythresh
                adapt_rate2 = myadapt

    print'\t\t\tBest parameters are: Threshold = ', (threshold + threshold2) / 2, 'Adaptation rate (sec) = ', (
                                                                                                              adapt_rate + adapt_rate2) * (
                                                                                                              win_len - win_overlap) / 2
    return (threshold + threshold2) / 2, (adapt_rate + adapt_rate2) / 2



"""
classifier_gridsearch_all_channels()
Purpose:
    Tunes nu and gamma parameters for nu and gamma using loocv_scoring.
Inputs:
    data_dicts: list of dictionaries that maps the index of the file in the list of  cv files to the corresponding ndarray feature matrix of this file.
                Each dictionary correspond to features for a channel.
    label_dicts: dictionary that maps the index of the file in the list of cv files to the corresponding array of labels of this file.

    seizure_indices: list of indices of the seizure files in the list of cv files
    interictal_indices: list of indices of the interictal files in the list of cv files
Output:
    (best_nu,best_gamma): the (nu,gamma) pair with the highest score returned by loocv_scoring.

"""


def classifier_gridsearch_all_channels(interictal_all_channels, training_data_all_channels, cv_label_dict, seizure_indices, interictal_indices, train_roc = None):

    num_channels = 6

    # store all the channel hyperparameters or models
    best_nus = []
    best_gammas = []
    best_clfs = []

    for channel_idx in range(num_channels):
        channel_data = [training_data[channel_idx, :, :] for training_data in training_data_all_channels]
        bestnu, bestgamma = classifier_gridsearch(channel_data, cv_label_dict, seizure_indices, interictal_indices, train_roc)
        # append
        best_nus.append(bestnu)
        best_gammas.append(bestgamma)

        # define your classifier
        best_clf = svm.OneClassSVM(nu=bestnu, kernel= 'rbf', gamma=bestgamma)

        # train your classifier on that corresponding channel; TODO: make sure dim is 2
        best_clf.fit(interictal_all_channels[channel_idx, :])
        best_clfs.append(best_clf)

    return best_nus, best_gammas, best_clfs


"""
Fetches scattering features for all channels.
Returns an array with shape (num_channels, num_windows, num_families * dim( 3*32 = 96))

"""
def get_sc_features_all_channels(patient_id, file_key,
                                           win_len_samples,
                                           win_overlap_samples, num_channels):
    for channel_idx in range(num_channels):
        channel_sc_features = get_sc_features_by_channel(patient_id, file_key, win_len_samples,
                                   win_overlap_samples, channel_idx)
        num_windows = channel_sc_features.shape[0]
        dim = channel_sc_features.shape[1]

        if channel_idx == 0:
            all_channel_sc_features = np.zeros((num_channels, num_windows, dim))
        all_channel_sc_features[channel_idx, :, :] = channel_sc_features
    return all_channel_sc_features             #shape is number of channels , number of samples, 3*32 currently



def offline_training(cv_file_type, cv_file_names, cv_file_idxs, cv_seizure_times, cv_train_files, win_len_samples, win_overlap_samples, f_s, i,patient_id, persistence_time, preictal_time, postictal_time,
                      train_roc = None):

    calc_features_local = 0

    # read pre-calculated features
    if not calc_features_local:
        print'\t\tBuilding MI matrices for all training files'
        training_sc_cv_files_all_channels = []
        for n, small_test_file in enumerate(cv_train_files):
            # fetch data
            filename = cv_file_names[n]
            file_key = str(cv_file_idxs[n]) + "_" + filename.split("/")[-1]

            test_file_sc_all_channels = get_sc_features_all_channels(patient_id, file_key,
                                           win_len_samples = win_len_samples,
                                           win_overlap_samples = win_overlap_samples, num_channels=6)
            training_sc_cv_files_all_channels += [test_file_sc_all_channels]


    # initializations
    training_data = training_sc_cv_files_all_channels   # list with 6/7 (num_channels, num_windows, dim)
    interictal_indices = []
    seizure_indices = []

    # stack all interictal data
    interictal_data = np.concatenate(
        [training_data[ind] for ind in xrange(len(training_data)) if cv_file_type[ind] is not 'ictal'], axis = 1)

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
        cv_label_dict[index] = label_classes_samples(training_data[index].shape[1], preictal_time, postictal_time, win_len_samples,
                                             win_overlap_samples, cv_seizure_times[index], cv_file_type[index], f_s = f_s)

    print '\t\tTraining the 6 classifiers'

    bestnus, bestgammas, best_clfs = classifier_gridsearch_all_channels(interictal_data, training_data, cv_label_dict, seizure_indices, interictal_indices,
                                              train_roc)


    print "\t\t\tbest nu values: ", bestnus
    print "\t\t\tbest gamma values: ", bestgammas

    # tune decisions
    threshold, adapt_rate = tune_decision(bestnus, bestgammas, training_data, cv_label_dict, seizure_indices,
                                          interictal_indices, win_len_samples * 1.0 / f_s,
                                          win_overlap_samples * 1.0 / f_s, f_s, cv_seizure_times,
                                          persistence_time, preictal_time)

    return best_clfs, threshold, adapt_rate


def find_mean(vector, length):
    mean_sum = 0
    for mean_index in range(length):
        mean_sum += vector[mean_index]
    mean = mean_sum / float(length)
    return mean


def update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients,
                      preictal_time, postictal_time, persistence_time, adapt_rate = None, threshold = None):
    f = open(log_file, 'a')

    # write the first few lines
    f.write("Results file for Burns Pipeline\n")

    # write the parameters
    f.write('Parameters used for this test\n================================\n')
    f.write('Feature used is Burns Features\n')
    f.write('Window Length \t%.3f\nWindow Overlap \t%.3f\n' % (win_len, win_overlap))
    f.write('Sampling Frequency \t%.3f\n' % f_s)
    f.write('Awake Times are ' + (not include_awake) * 'NOT ' + ' included in training\n')
    f.write('Asleep Times are ' + (not include_asleep) * 'NOT ' + ' included in training\n\n')


    # write the patients
    f.write('Patients are ' + " ".join(patients) + "\n\n")
    f.close()
    return

def update_log_decision(adapt_rate, threshold):
    f = open(log_file, 'a')
    f.write('Adapt rate is {}\n'.format(adapt_rate))
    f.write('Threshold is {}\n'.format(threshold))
    f.close()




"""
Helper function for predict_all_channels. Performs majority votes to generate a test label on
one window of scattering data from all 6 channels.

Inputs:
    - all_channel_feature: single-window test feature with shape (number of channels, feature dimension)
    - best_clfs: a list of trained SVMs for each SOZ channel. The length of this list should be 6.
Outputs:
    - testing_label: predicted label. 1 is seizure, -1 is normal.

"""
def predict_all_channels_single_feature(all_channel_feature, best_clfs):
    all_channel_testing_labels = []

    for channel_idx in range(all_channel_feature.shape[0]):
        this_channel_feature = all_channel_feature[channel_idx, :].reshape(-1, 1).transpose()
        # predict testing label
        this_channel_svm = best_clfs[channel_idx]
        this_channel_testing_label = this_channel_svm.predict(this_channel_feature)
        all_channel_testing_labels.append(this_channel_testing_label)

    testing_label = stats.mode(all_channel_testing_labels)[0]

    return testing_label



"""
Helper function for predict_all_channels. Performs majority votes to generate a test label on
one window of scattering data from all 6 channels.

Inputs:
    - all_channel_data: multi-window test features with shape (number of windows, number of channels, feature dimension)
    - best_clfs: a list of trained SVMs for each SOZ channel. The length of this list should be 6.
Outputs:
    - one_seiz_pred_labels: array of predicted labels.

"""

def predict_all_channels(all_channel_data, best_clfs):

    data_length = all_channel_data.shape[1]
    one_seiz_pred_labels = np.zeros(data_length)

    for data_idx in range(data_length):
        # feature for all seizure onset channels
        single_feature = all_channel_data[:, data_idx, :]
        single_label = predict_all_channels_single_feature(single_feature, best_clfs)[0][0]
        one_seiz_pred_labels[data_idx] = single_label
    return one_seiz_pred_labels




"""
online_testing()

Purpose: pipeline to direct the online testing portion of this algorithm on this window of the unseen test data

Inputs:
    all_channel_features: scattering features from 6 channels
    pred_time: the number of windows after a seizure during which no other flag is raised AND the amount of time before a
    seizure where the decision is counted as a prediction and not a false positive.

Outputs:
    decision: one dimensional ndarray containing decision of seizure prediction
    test_outlier_fraction: one dimensional ndarray containing the outlier fraction of the test data


others variables: please refer to ictal_inhibitors_final.py. The naming/meanings of the variables should be identical.

"""



def online_testing_all_channels(all_channel_features,f_s,i,pred_time,label_index,alarm_timer, best_clfs, threshold, adapt_rate,
                   vector_testing_labels):

    testing_label = predict_all_channels_single_feature(all_channel_features, best_clfs)

    # switch the labels to our code's convention: 0 is normal, 1 is seizure
    testing_label[testing_label == 1] = 0
    testing_label[testing_label == -1] = 1

    # store the label inside the vector of past testing labels
    if label_index < adapt_rate:
        vector_testing_labels[label_index] = testing_label[0]  # 0th to extract element in array
        label_index += 1
    else:
        label_index = 0
        vector_testing_labels[label_index] = testing_label[0]

    # find the outlier fraction as the mean of the vector of past testing labels
    test_outlier_fraction = find_mean(vector_testing_labels, adapt_rate)

    # determining where the outlier fraction meets or exceeds the threshold
    if alarm_timer <= 0:
        if test_outlier_fraction >= threshold:
            decision = 1  # predicted a seizure
            alarm_timer = pred_time  # set the refractory period
        else:
            decision = -1  # did not predict a seizure

    else:  # do not output a positive prediction within an alarm_timer period of time of the last positive prediction
        decision = -1
        alarm_timer -= 1

    return decision, test_outlier_fraction, vector_testing_labels, label_index, alarm_timer, testing_label


"""
Similar to the window_to_sample in ictal_inhibitors_final.py but adapts to our definition of
window size using number of samples instead of number of seconds.
"""

def window_to_samples_insamples(window_array, win_len_samples, win_overlap_samples, f_s):
    # finding the total number of windows
    num_windows = np.size(window_array)

    # finding the time indices corresponding to each window
    sample_indices = np.arange(num_windows) * (win_len_samples - win_overlap_samples) + win_len_samples

    # filling an array in units of samples with the values taken from the appropriate window
    sample_array = np.zeros(int(max(sample_indices)))
    for i in np.arange(0, num_windows - 1):
        sample_array[int(sample_indices[i]):int(sample_indices[i + 1])] = window_array[i]

    return sample_array, sample_indices


def parent_function(plot = 1):
    # setting run parameters
    patients = ["TA023", "TS057"]
    # show_fig = 0  # if 1, figures show; if 0, figures save to current working directory

    # create paths to the data folder
    to_data = os.path.dirname(os.getcwd())
    data_path = os.path.join(to_data, 'Data')
    save_path = os.path.join(to_data, 'Features')
    features_path = os.path.join(save_path, 'StoredFeatures')

    # setting model parameters
    win_len_samples = 2 ** 14
    win_overlap_samples = 2 ** 13  # 1/2 overlap
    f_s = float(1e3)  # Hz
    persistence_time = 200 * f_s / (win_len_samples - win_overlap_samples) + 1    # 200 seconds



    preictal_time = 5 * 60  # minutes times seconds, the amount of time prior to seizure onset defined as preictal
    postictal_time = 5 * 60  # minutes times seconds, the amount of time after seizure end defined as postictal
    include_awake = True
    include_asleep = True
    # include_asleep = False

    update_log_params(log_file, win_len_samples, win_overlap_samples, include_awake, include_asleep, f_s, patients,
                      preictal_time, postictal_time, persistence_time)

    # Changed persistence time to 200
    persistence_time = 200 * f_s/(win_len_samples - win_overlap_samples) + 1
    # persistence_time = 3.33 * 60  # minutes times seconds, the amount of time after a seizure prediction for which no alarm is raised
    preictal_time = 5 * 60  # minutes times seconds, the amount of time prior to seizure onset defined as preictal
    postictal_time = 5 * 60  # minutes times seconds, the amount of time after seizure end defined as postictal
    include_awake = True
    include_asleep = True



    # evaluate each patient
    for patient_index, patient_id in enumerate(patients):

        train_roc = []  # storing training ROC values, just for debugging

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths to be specific to each patient
        my_data_path = "/Users/stormrecluse/Desktop/Epilepsy 2018summer/Data"
        p_data_path = os.path.join(my_data_path, patient_id)

        print 'Retreiving stored raw data'
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(p_data_path, f_s, include_awake,
                                                                                  include_asleep, patient_id, None,
                                                                                  None, calc_train_local=True)
        number_files = len(all_files)

        print "Data filenames: ", data_filenames

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
            testing_file_name = data_filenames[i]
            testing_file_idx = i
            cv_file_names = data_filenames[:i] + data_filenames[i + 1:]
            cv_file_idxs = range(i) + range(i+1,number_files)

            cv_train_files = all_files[:i] + all_files[i + 1:]
            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            print '\tEntering offline training'

            # start training classifiers for each channel for majority voting
            my_svms, threshold, adapt_rate  = offline_training(cv_file_type, cv_file_names,  cv_file_idxs, cv_seizure_times, cv_train_files, win_len_samples, win_overlap_samples, f_s, i, patient_id,
                                      persistence_time, preictal_time, postictal_time,  train_roc)


            print'\tEntering online testing'
            # computing prediction on testing file for this fold

            print '\tCalculating testing features locally'
            # determine how many samples, windows, and channels we have in this test file

            total_test_samples = testing_file.shape[0]

            # load test file
            test_key = str(testing_file_idx) + "_" + data_filenames[testing_file_idx].split("/")[-1]

            # gets the test file for all channels
            test_sc = get_sc_features_all_channels(patient_id, test_key, win_len_samples = win_len_samples, win_overlap_samples = win_overlap_samples, num_channels = 6)

            test_features = test_sc  # should be list of (6, n_windows, 96)  # for loop to process each window in the test file
            # initializing for computing performance metrics
            num_test_windows = test_features.shape[1]
            full_file_decision = np.zeros(num_test_windows)
            full_file_outlierfraction = np.zeros(num_test_windows)
            full_file_labels = np.zeros(num_test_windows)
            vector_testing_labels = np.zeros(adapt_rate)
            alarm_timer = 0
            label_index = 0


            for index in np.arange(num_test_windows):
                # getting the single window of data for this iteration of the for loop
                feature = test_features[:, index, :]

                decision, test_outlier_fraction, vector_testing_labels, label_index, alarm_timer, testing_label = online_testing_all_channels(feature,f_s,i, persistence_time,label_index,
                                                                                                                  alarm_timer,my_svms,
                                                                                                                  threshold, adapt_rate,
                                                                                                                    vector_testing_labels)

                # storing the outlier fraction and decision for calculating performance metrics and visualization
                full_file_decision[index] = decision
                full_file_labels[index] = testing_label
                full_file_outlierfraction[index] = test_outlier_fraction



            true_test_labels = label_classes_samples(num_test_windows, preictal_time, postictal_time, win_len_samples,
                                  win_overlap_samples, seizure_times[i], file_type[i], f_s=f_s)

            # using outputs from test file to compute performance metrics
            print'\tCalculating performance stats'

            print "\tFile Type: ", file_type[i]

            # convert from units of windows to units of samples
            test_decision_sample, test_decision_sample_indices = window_to_samples_insamples(full_file_decision, win_len_samples, win_overlap_samples, f_s)
            test_label_sample, test_label_sample_indices = window_to_samples_insamples(full_file_labels, win_len_samples, win_overlap_samples, f_s)
            # find performance metrics for this fold of cross validation
            prediction_sensitivity[i], detection_sensitivity[i], latency[i], fp[i], times[i] = performance_stats(
                test_decision_sample, seizure_times[i], f_s, preictal_time, (win_len_samples * 1.0)/f_s, (win_overlap_samples * 1.0)/f_s)

            print "seizure time: ", seizure_times[i]

            # print the performance metrics and visualize the algorithm output on a graph
            print '\tPrediction sensitivity = ', prediction_sensitivity[i], 'Detection sensitivity = ', \
            detection_sensitivity[i], 'Latency = ', latency[i], 'FP = ', fp[i], 'Time = ', times[i]
            # viz_single_outcome(test_decision_sample, test_outlier_frac_sample, testing_file[:,0], seizure_times[i], threshold, i, patient_id, f_s)
            # outlier_fraction_samples, outlier_sample_indices = window_to_samples_insamples(outlier_fraction, win_len_samples, win_overlap_samples, f_s)
            outlier_fraction_samples, outlier_sample_indices = window_to_samples_insamples(full_file_outlierfraction, win_len_samples, win_overlap_samples, f_s)

            # finding the time indices corresponding to seizures
            if seizure_times[i]!= None:
                seizure_start_sample = seizure_times[i][0] * f_s
                seizure_end_sample = seizure_times[i][1] * f_s

            # number of samples corresponding to the chunks (short chunks at the end are ignored)
            num_samples = test_decision_sample.shape[0]

            # plot all alarms and actual seizure
            if plot:
                channel_names = choose_best_channels(patient_id= patient_id, seizure = file_type[i] is 'ictal', filename= testing_file_name)
                testing_file = notch_filter_data(testing_file, 500)
                # plot the time series
                fig = plt.figure(1)
                plt.subplots_adjust(left=0.2)

                for channel_idx in range(6):
                    plot_single_channel(fig, channel_idx, testing_file, seizure_times[i], seizure_start_sample,
                                        seizure_end_sample, test_label_sample_indices, full_file_labels, channel_names,
                                        num_samples)

                # plot the SVM outputs
                ax7 = fig.add_subplot(9, 1, 7)
                ax7.plot(test_label_sample , color = 'g')
                ax7.set_ylabel("SVM")
                if seizure_times[i] != None:
                    ax7.axvline(x=seizure_start_sample, color='r')
                    ax7.axvline(x=seizure_end_sample, color='r')

                # find all the raised alarms
                # find indices corresponding to alarms for marking
                marker_on = []  # set of times to put markers on
                for decision_idx in range(num_samples):
                    if test_decision_sample[decision_idx] == 1:
                        marker_on.append(decision_idx)


                # plot outlier fraction (unit: seconds) and decision
                ax8 = fig.add_subplot(9, 1, 8)
                ax8.plot(np.linspace(0, outlier_fraction_samples.shape[0] *1.0/1000, outlier_fraction_samples.shape[0]), outlier_fraction_samples, color = 'k',
                         marker= "*", markersize= 10, markevery= marker_on)
                ax8.set_ylabel("O.F.")
                ax8.set_xlabel("time/s")
                ax8.set_ylim([0, 1])
                if seizure_times[i] != None:
                    ax8.axvline(x=seizure_start_sample *1.0/1000, color='r')
                    ax8.axvline(x=seizure_end_sample *1.0/1000, color='r')
                ax8.axhline(y = threshold, color = 'b')

                fig.savefig(patient_id + '_' +  str(i) + '_'
                            + testing_file_name.split('/')[-1] + '_scattering_PV.png')
                plt.close(fig)

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
        update_log_stats(log_file, patient_id, prediction_sensitivity, detection_sensitivity, latency, fp, times)


if __name__ == '__main__':
    # create paths to the data folder
    to_data = os.path.dirname(os.getcwd())
    data_path = os.path.join(to_data, 'Data')
    save_path = os.path.join(to_data, 'Features')
    features_path = os.path.join(save_path, 'StoredFeatures')

    log_file = os.path.join(save_path, 'log_file_PV.txt')


    parent_function()
