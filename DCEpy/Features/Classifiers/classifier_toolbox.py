"""
Some basic tools shared by most classification pipelines. Includes:
(1)reduce_channels
(2)build_feature_dict
(3)label_classes
(4)viz_labels
(5)a bunch of scoring functions for the GridSearch Pipeline. Reference: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import matthews_corrcoef,hinge_loss,hamming_loss,jaccard_similarity_score,fbeta_score,cohen_kappa_score

def reduce_channels(all_files_old,chosen_channels):

    """
    :param all_files_old: original files
    :param chosen_channels: list of selected channels
    :return:
    """
    all_files=[]
    for file in all_files_old:
        all_files.append(file[:, chosen_channels])
    return all_files



def build_feature_dict(data_filenames, all_files,feat_func, win_len=2000, win_overlap=1000,*args):

    """
    builds single feature dictionary for a specific patient.

    :param patient_id:
    :param data_filenames: a list of strings
    :param all_files: a list of raw data for the patient
    :param win_len:
    :param win_overlap:
    :param feat_func: takes file, window length and window overlap as inputs and computes single feature on file

    :return: dictionary that maps each file to the computed feature matrices
    """

    feature_dict = {}

    file_num = len(all_files)

    for i in range(file_num):
        file= all_files[i]
        feature_dict[data_filenames[i]] = feat_func(file,win_len,win_overlap)  # feat_func processes all windows, returns np matrix with size (num_windows,num_features)


    return feature_dict


def label_classes(data,data_seizure_time,pre_ictal_lim=5*60,post_ictal_lim=10*60,win_len_seconds=2.0, win_overlap_seconds=1.0):
    """

    :param data: feature data for one file with shape (number of windows, number of features)
    :param data_seizure_time: None if interictal; tuple if ictal file
    :return: np array

    +1 if ourlier
    -1 if interictal
    """

    n,_= data.shape   # n is number of windows
    labels = np.ones(n)


    if data_seizure_time is not None:

        seizure_start_time = data_seizure_time[0]
        seizure_end_time = data_seizure_time[1]

        # seizure start window
        if seizure_start_time <win_len_seconds: # seizure starts in the 0th window
            seizure_start_window = 0
        else:
            seizure_start_window = int((seizure_start_time - win_len_seconds) / (win_len_seconds - win_overlap_seconds) + 1)

        # seizure end window
        if seizure_end_time<win_len_seconds:   # seizure ends in the 0th window
            seizure_end_window = 0
        else:
            seizure_end_window = int((seizure_end_time - win_len_seconds) / (win_len_seconds - win_overlap_seconds) + 1)

        # in case the seizure end window is larger than the max window index
        # (since if the ending does not have enough samples for a window, it will not be considered in the windowing function)
        if seizure_end_window>n-1:
            seizure_end_window=n-1

        # label the seizure period
        labels[seizure_start_window:seizure_end_window+1] = -np.ones(seizure_end_window+1 - seizure_start_window)

        # label the preictal(and interictal period if that exists) period of ictal file
        if seizure_start_time > pre_ictal_lim:  # if there is a long period before seizure onset
            pre_ict_length = pre_ictal_lim  # in seconds
            pre_ict_start = seizure_start_time - pre_ict_length
            pre_ict_start_win = int((pre_ict_start - win_len_seconds) / (win_len_seconds - win_overlap_seconds)+1)

            if pre_ict_start_win >= 0:
                labels[pre_ict_start_win:seizure_start_window] = -np.ones(seizure_start_window - pre_ict_start)
                labels[:pre_ict_start_win] = np.ones(pre_ict_start_win)

        else:
            labels[:seizure_start_window] = -np.ones(seizure_start_window)


        # label postictal period
        end_period_secs = (n - seizure_end_window) * (win_len_seconds - win_overlap_seconds)

        if end_period_secs > post_ictal_lim:
            # print "end period",end_period_secs
            post_ict_end = seizure_end_time + post_ictal_lim
            post_ict_end_win = int((post_ict_end - win_len_seconds) / (win_len_seconds - win_overlap_seconds)+1)

            if post_ict_end_win > n-1:
                post_ict_end_win = n-1

            labels[seizure_end_window+1:post_ict_end_win+1] = -np.ones(post_ict_end_win - seizure_end_window)
            labels[post_ict_end_win+1:] = np.ones(n-1 - post_ict_end_win)

        else:
            labels[seizure_end_window+1:] = -np.ones(n-1 - seizure_end_window)


    return labels



def viz_labels(labels,testing_file_name,test_data_seizure_time=None,f_s=1000,win_len_secs=2.0,win_overlap_secs=1.0):

    """
    :param labels: a list of labels
    :param testing_file_name: string. Name of the test file
    :return:
    plots labels over time
    """

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


def mat_corrcoeff(estimator, X_test,y_test):

    """Matthew Correlation coefficient as objective function."""

    y_predicted = estimator.predict(X_test)

    score = matthews_corrcoef(y_test,y_predicted)

    return score


def h_loss(estimator, X_test,y_test):
    "hinge loss"

    y_predicted = estimator.predict(X_test)

    score = -hinge_loss(y_test, y_predicted)

    return score


def ham_loss(estimator, X_test,y_test):
    "hamming loss"

    y_predicted = estimator.predict(X_test)

    score = hamming_loss(y_test, y_predicted)

    return score


def jss(estimator, X_test, y_test):
    "Jaccard similarity coefficients"

    y_predicted = estimator.predict(X_test)

    score = jaccard_similarity_score(y_test, y_predicted)

    return score

def f_beta(estimator, X_test, y_test):
    "f beta score"

    y_predicted = estimator.predict(X_test)

    score = fbeta_score(y_test, y_predicted,beta=2)  # based on Park et al (2011)

    return score


def chs(estimator, X_test, y_test):
    y_predicted = estimator.predict(X_test)

    score = cohen_kappa_score(y_test, y_predicted)  # based on Park et al (2011)

    return score

