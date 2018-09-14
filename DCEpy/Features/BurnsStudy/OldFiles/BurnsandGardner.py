import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
from DCEpy.Features.AnalyzePatient import analyze_patient_raw
from scipy.signal import csd
from scipy.signal import welch
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score

from DCEpy.Features.feature_functions import energy_features
from DCEpy.General.DataInterfacing.edfread import edfread


def update_list(original, update):
    count = 0
    for a in update:
        place = a[0] + count
        original = original[:place] + a[1] + original[place + 1:]
        count += len(a[1]) - 1
    return original

def analyze_patient_raw(data_path, f_s, include_awake, include_asleep, long_interictal, good_channels):

    # minutes per chunk (only for long interictal files)
    min_per_chunk = 15
    sec_per_min = 60

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    p_file = os.path.join(data_path, 'patient_pickle.txt')

    with open(p_file, 'r') as pickle_file:
        print("\tOpen Pickle: {}".format(p_file) + "...")
        patient_info = pickle.load(pickle_file)

    # add data file names
    data_filenames = list(patient_info['seizure_data_filenames'])
    seizure_times = list(patient_info['seizure_times'])
    file_type = ['ictal'] * len(data_filenames)
    seizure_print = [True] * len(data_filenames)  # mark whether is seizure

    if include_awake:
        data_filenames += patient_info['awake_inter_filenames']
        seizure_times += [None] * len(patient_info['awake_inter_filenames'])
        file_type += ['awake'] * len(patient_info['awake_inter_filenames'])
        seizure_print += [False] * len(patient_info['awake_inter_filenames'])

    if include_asleep:
        data_filenames += patient_info['asleep_inter_filenames']
        seizure_times += [None] * len(patient_info['asleep_inter_filenames'])
        file_type += ['sleep'] * len(patient_info['asleep_inter_filenames'])
        seizure_print += [False] * len(patient_info['asleep_inter_filenames'])

    data_filenames = [os.path.join(data_path, filename) for filename in data_filenames]

    # get data in numpy array
    num_channels = []
    all_files = []
    all_files_unfiltered = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []
    tmp_seizure_print = []

    print '\tGetting Data...'
    for i, seizure_file in enumerate(data_filenames):

        # this is for when we have inter-ictal files that are an hour long that has split it up into parts
        if long_interictal and not (file_type[i] is 'ictal'):
            #TODO: this code is hacky. make it better. I don't even know what it used to do in Gadner script
            #TODO: will have to fix filenames and types and stuff too
            # read data in
            # X, _, labels = edfread(seizure_file, good_channels=good_channels)
            X = np.ones((300000,150))

            long_file = X
            long_file_length = long_file.shape[0]

            sub_one = long_file[:int(long_file_length/3)]
            sub_two = long_file[int(long_file_length/3):int(long_file_length*2/3)]
            sub_three = long_file[int(long_file_length*2/3):]

            all_files.append(sub_one)
            all_files.append(sub_two)
            all_files.append(sub_three)

        else:
            print '\t\tSeizure file %d reading...\n' % (i + 1),
            # read data in
            X, _, labels = edfread(seizure_file, good_channels=good_channels)
            # X = np.ones((300000,150))

            n, p = X.shape
            num_channels.append(p)

            all_files.append(X)  # add raw data to files


    # update temporary stuff
    data_filenames = update_list(data_filenames, tmp_data_filenames)
    file_type = update_list(file_type, tmp_file_type)
    seizure_times = update_list(seizure_times, tmp_seizure_times)
    seizure_print = update_list(seizure_print, tmp_seizure_print)

    # double check that the number of channels matches across data
    if len(set(num_channels)) == 1:
        num_channels = num_channels[0]
        gt1 = num_channels > 1
        print '\tThere ' + 'is ' * (not gt1) + 'are ' * gt1 + str(num_channels) + ' channel' + 's' * gt1
    else:
        print 'Channels: ' + str(num_channels)
        print 'There are inconsistent number of channels in the raw edf data'
        sys.exit('Error: There are different numbers of channels being used for different seizure files...')

    # double check that no NaN values appear in the features
    for X, i in enumerate(all_files):
        if np.any(np.isnan(X)):
            print 'There are NaN in raw data of file', i
            sys.exit('Error: Uh-oh, NaN encountered while extracting features')

    return all_files, data_filenames, file_type, seizure_times, seizure_print

def window_to_samples(window_array, win_len, win_overlap, f_s):

    num_windows = np.size(window_array)
    time_indices = np.arange(num_windows)*(win_len-win_overlap) + win_len
    sample_indices = time_indices*f_s
    sample_array = np.zeros(max(sample_indices))

    for i in np.arange(0,num_windows-1):
        sample_array[sample_indices[i]:sample_indices[i+1]] = window_array[i]

    return sample_array

def choose_best_channels(patient_id):
    #TODO: experiement with other ways of choosing channels. Right now, the best method is using Rakesh's model

    # good_channel_dict = {'TS041':[1,2,3,14,15],'TS039':[79,80,81,96,97,98], 'TA023':[]}
    good_channel_dict = {'TS041':['LAH2','LAH3','LAH4','LPH1','LPH2'],'TS039':['RAH1','RAH2','RAH3','RPH2','RPH3','RPH4'],'TA023':['MST1','MST2','HD1','TP1']}
    dimensions_to_keep = good_channel_dict[patient_id]

    return dimensions_to_keep

def construct_coherency_matrix(X, f_s, freq_band):

    n,p = X.shape

    # initialize the adjacency matrix
    A = np.zeros((p,p))

    #TODO: play with nperseg and noverlap of the csd and welch methods below
    # construct adjacency matrix
    for i in range(p):
        for j in range(i+1,p):
            fxy, Pxy = csd(X[:,i], X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)
            fxx, Pxx = welch(X[:,i], fs = f_s, nperseg = 1000, noverlap = 500)
            fyy, Pyy = welch(X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)
            Pxy_band = np.mean([Pxy[n] for n in xrange(len(Pxy)) if fxy[n] <= freq_band[1] and fxy[n] >= freq_band[0]])
            Pxx_band = np.mean([Pxx[n] for n in xrange(len(Pxx)) if fxx[n] <= freq_band[1] and fxx[n] >= freq_band[0]])
            Pyy_band = np.mean([Pyy[n] for n in xrange(len(Pyy)) if fyy[n] <= freq_band[1] and fyy[n] >= freq_band[0]])
            c = abs(Pxy_band)**2/(Pxx_band*Pyy_band)
            A[i,j] = c # upper triangular part
            A[j,i] = c # lower triangular part

    # return adjacency matrix
    return A

def build_coherency_array(raw_data, win_len, win_ovlap, f_s, freq_band):

    data = raw_data

    n, p = data.shape

    win_len = win_len * f_s
    win_ovlap = win_ovlap * f_s

    num_windows = int( math.floor( float(n) / float(win_len - win_ovlap)) )
    coherency_array = np.zeros((p,p,num_windows))

    for index in np.arange(num_windows):
        start = index*(win_len - win_ovlap)
        end = min(start+win_len, n)
        window_of_data = data[start:end,:] # windowed data
        coherency_array[:,:,index] = construct_coherency_matrix(window_of_data, f_s, freq_band)

    return coherency_array

def find_normalizing_coherency(matrix):
    mean_mat = np.mean(matrix, axis=2)
    std_mat = np.std(matrix, axis=2)
    return mean_mat, std_mat

def transform_coherency(coherencies_list, mean, std):
    #TODO: try with and without normalizing: does it make a big difference?
    std[std == 0] = 0.0001
    num_files = len(coherencies_list)
    transformed_coherencies = []
    for j in xrange(num_files):
        coherency_matrices_one_file = coherencies_list[j]
        num_windows = coherency_matrices_one_file.shape[2]
        for i in xrange(num_windows):
            matrix = coherency_matrices_one_file[:,:,i].copy()
            matrix -= mean
            matrix = np.divide(matrix, std)
            matrix = np.divide(np.exp(matrix), 1 + np.exp(matrix))
            coherency_matrices_one_file[:,:,i] = matrix
        transformed_coherencies += [coherency_matrices_one_file]

    return transformed_coherencies

def find_evc(list_of_matrices):
    num_files = len(list_of_matrices)
    centrality_all_files = []
    for j in xrange(num_files):
        matrix = list_of_matrices[j]
        centrality = np.zeros((matrix.shape[2],matrix.shape[1]))
        for i in xrange(matrix.shape[2]):
            sub_matrix = matrix[:,:,i].copy()
            G = nx.Graph(sub_matrix)
            evc = nx.eigenvector_centrality(G, max_iter=500)
            #TODO: change this if you want, such that order stays the same. Not urgent.
            centrality[i,:] = np.asarray(evc.values())
        centrality_all_files += [centrality]
    return centrality_all_files

def create_outlier_frac(predicted_labels, adapt_rate):

    tot_obs = predicted_labels.size
    out_frac = np.zeros(tot_obs)

    for i in np.arange(adapt_rate,tot_obs):
        out_frac[i] = np.mean(predicted_labels[i-adapt_rate:i])

    return out_frac

def create_decision(test_outlier_fraction, threshold, pred_time):

    #TODO: get everything in the right units of time, work out f_s and win_len and win_ovlap. for ex this should be in windows, currently is in time
    decision = np.sign(test_outlier_fraction-threshold)
    decision[decision==0] = 1

    for i in xrange(len(decision)):
        if decision[i]==1:
            decision[i+1:i+pred_time] = -1
            i += pred_time

    return decision

def performance_stats(decision, seizure_time, f_s, pred_time):

    # if inter-ictal file
    if seizure_time is None:

        # get the amount of time that is passed, false positive
        time = float(decision.size) / float(f_s * 60. * 60.)
        FP = float(np.size(decision[decision>0])) / float(f_s)
        prediction_sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        detection_sensitivity = np.nan
        latency = np.nan # latency is meaningless since there is no seizure

    else:

        # start time, end time, etc
        seizure_start = int(f_s * seizure_time[0])
        seizure_end = int(f_s* seizure_time[1])

        false_positive_range = int(max(0, seizure_start - pred_time*f_s))
        false_positive_data = np.copy(decision[:false_positive_range])

        # initialize time and FP
        time = float(false_positive_range) / float(f_s * 60. * 60.)
        FP = float(np.size(false_positive_data[false_positive_data > 0])) / float(f_s)

        if not np.any(decision[false_positive_range:seizure_end] > 0):
            prediction_sensitivity = 0.0 # seizure not detected
            detection_sensitivity = 0.0
        elif not np.any(decision[false_positive_range:seizure_start] > 0):
            prediction_sensitivity = 0.0 # seizure detected late
            detection_sensitivity = 1.0
        elif np.any(decision[false_positive_range:seizure_start]):
            prediction_sensitivity = 1.0 # seizure detected early
            detection_sensitivity = 1.0 #TODO: this detection is not quite completely accurate

        # compute latency
        if np.any(decision[false_positive_range:seizure_end] > 0):
            detect_time = np.argmax(decision[false_positive_range:seizure_end] > 0) + false_positive_range
            latency = float(detect_time - seizure_start) / float(f_s) # (in seconds)
        else:
            latency = np.nan

    # put time and FP as nan if there was no time
    if time <= 0.0:
        time = np.nan
        FP = np.nan

    return prediction_sensitivity, detection_sensitivity, latency, FP, time

def viz_single_outcome(decision, out_frac, raw_ieeg, test_times, thresh, test_index, patient_id, f_s):

    fig, axes = plt.subplots(3, sharex=True)

    axes[0].plot(raw_ieeg)
    axes[0].set_title('Raw iEEG signal', size=14)
    axes[0].set_ylabel('Voltage', size=10)
    axes[0].set_yticklabels([])
    axes[0].set_yticks([])

    axes[1].plot(out_frac)
    axes[1].set_title('Outlier fraction', size=14)
    axes[1].set_ylabel('Liklihood of upcoming seizure', size=10)
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[1].set_ylim(ymin=0, ymax=1)
    if test_times is not None:
        axes[1].axvline(x=test_times[0]*f_s,lw=2,c='r')
        axes[1].axvline(x=test_times[1]*f_s,lw=2,c='r')
    axes[1].axhline(y=thresh,lw=2,c='k')

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

def viz_many_outcomes(all_outlier_fractions, seizure_times, patient_id, threshold, test_index):

    fig, axes = plt.subplots(len(all_outlier_fractions))
    for i in xrange(len(all_outlier_fractions)):
        this_out_frac = np.asarray(all_outlier_fractions[i])
        axes[i].plot(this_out_frac)
        axes[i].set_ylim(ymin=0, ymax=1)
        if seizure_times[i] is not None:
            axes[i].axvline(x=seizure_times[i][0], lw=2, c='r')
        axes[i].axhline(y=threshold, lw=2, c='k')
    # plt.show()
    plt.savefig('all_Patient_{}_file_{}'.format(patient_id,test_index))
    plt.clf()

def create_chronological_data_sets(list_train_data, filetype, seizure_times, f_s):

    labels_all_files = []
    file_num = len(list_train_data)
    data_all_files = []

    num_ictal_points = 0
    num_inter_points = 0

    for index in xrange(file_num):
        if filetype[index] is 'ictal':
            #TODO: play around with different class definitions. Right now I am using entire ictal file as a class... try only preictal, only ictal, only preictal+ictal, etc...

            # list_train_data[index] = list_train_data[index][0:seizure_times[index][1]]
            # num_ictal_points+=list_train_data[index].shape[0]
            # data_all_files+=[list_train_data[index]]
            labels_all_files+=[np.ones(list_train_data[index].shape[0])]

        else:
            # num_inter_points+=list_train_data[index].shape[0]
            # data_all_files+=[list_train_data[index]]
            labels_all_files+=[np.ones(list_train_data[index].shape[0])*-1]

    # if num_ictal_points<num_inter_points:
    #     ratio = float(num_ictal_points)/float(num_inter_points)
    #     for index in xrange(file_num):
    #         if filetype[index] is 'ictal':
    #             labels_all_files+=[np.ones(list_train_data[index].shape[0])]
    #             data_all_files+=[list_train_data[index]]
    #         else:
    #             indices = np.random.permutation(list_train_data[index].shape[0])[:int(list_train_data[index].shape[0]*ratio)]
    #             data_all_files+=[list_train_data[index][indices,:]]
    #             labels_all_files+=[np.zeros(indices.size)]
    #         print labels_all_files[index].shape
    #
    # else:
    #     ratio = float(num_inter_points)/float(num_ictal_points)
    #     for index in xrange(file_num):
    #         if filetype[index] is 'ictal':
    #             indices = np.random.permutation(list_train_data[index].shape[0])[:int(list_train_data[index].shape[0]*ratio)]
    #             data_all_files+=[list_train_data[index][indices,:]]
    #             labels_all_files+=[np.zeros(indices.size)]
    #         else:
    #             labels_all_files+=[np.ones(list_train_data[index].shape[0])]
    #             data_all_files+=[list_train_data[index]]
    #         print labels_all_files[index].shape


    return list_train_data, labels_all_files

def get_stats(predicted_labels, label_set):
    # initialize all metrics equal 0
    TP = 0 # true positives
    FP = 0 # false positives
    TN = 0 # true negatives
    FN = 0 # false negatives

    #TODO: are there other stats that would be important in training a seizure prediction model?
    # for each predicted label, add 1 to each metric according to its if criteria
    for i in xrange(len(predicted_labels)):
        #if label_set[i] == predicted_labels[i] == 1:
        if label_set[i] == 1 and predicted_labels[i] >= 0:
            TP += 1
        elif label_set[i] == -1 and predicted_labels[i] >= 0:
            FP += 1
        elif label_set[i] == -1 and predicted_labels[i] < 0:
            TN += 1
        elif label_set[i] == 1 and predicted_labels[i] < 0:
            FN += 1

    return TP, FP, TN, FN

def score_function(predicted_labels, actual_labels, beta):

    TP, FP, TN, FN = get_stats(predicted_labels, actual_labels)

    # calculate score
    #TODO: is this the best scoring function for our purposes? We could try Gardners, or our own creation. This model does not support negative latencies
    # score = float(((1+beta**2)*float(TP))/((1+beta**2)*float(TP)+beta**2*float(FN)+float(FP))) # calculate f2, a scoring criteria according to Park et al.
    # return score to parameter_tuning

    # if (TP+FN==0):
    #     score = beta*float(TN)/float(TN+FP)
    # elif (TN+FP==0):
    #     score = float(TP)/float(TP+FN)
    # else:
    #     print '\t\t\tSomething has gone wrong in scoring function'

    print 'true negative perentage', float(TN)/float(TN+FP)
    print 'true positive percentage', beta*float(TP)/float(TP+FN)
    score =  float(TN)/float(TN+FP) + beta*float(TP)/float(TP+FN)
    return score

def parameter_tuning_csvm(feature_set, label_set, num_ictal, num_inter):

    # create the cost-sensitive svm by using the SGD Classifier and setting the loss to hinger and class_weight to balanced.
    cssvm = svm.SVC(kernel = 'rbf', class_weight='balanced')

    # initialize output
    best_score = [0, 0, 0]

    #TODO: is this the most effective method for parameter tuning?
    #Grid search using 2 forloops
    for myC in np.arange(.3,2,.25): # for each penalty
        print '\t\tNew round of testing on new nu...'
        # for mygamma in np.hstack((np.arange(.001,.3,.005),np.arange(.3,1,.10),np.arange(1,5,.5))): # for each l1_ratio
        for mygamma in np.arange(.001,.1,.01):
            print myC, mygamma

            cssvm.set_params(gamma=mygamma, C=myC) # set parameters of SGDClassifier
            score = np.zeros(len(feature_set))

            for i in xrange(len(feature_set)):
                cv_test_data = feature_set[i]
                cv_test_labels = label_set[i]
                cv_train_file_set = feature_set[:i] + feature_set[i+1:]
                cv_train_label_set = label_set[:i] + label_set[i+1:]
                cv_train_data = np.vstack((cv_train_file_set[n] for n in xrange(len(cv_train_file_set))))
                cv_train_labels = np.hstack((cv_train_label_set[n] for n in xrange(len(cv_train_label_set))))

                cssvm.fit(cv_train_data, cv_train_labels) # fit training set
                predicted_labels = cssvm.predict(cv_test_data) # predict on validate set
                #TODO: should I score my model on ONLY preictal data? I'm currently training on the entire seizure file

                score[i] = score_function(predicted_labels, cv_test_labels, beta=1) # get score of the model by checking with validate labels

            avg_sensitivity = np.mean(score[:num_ictal])
            avg_precision = np.mean(score[num_ictal:])
            avg_score = avg_sensitivity + avg_precision

            if best_score[0] < avg_score: # if the score is greater than previous score
                print 'best'
                best_score[0] = avg_score # store score in 0 index of best_score
                best_score[1] = myC # store penalty type in 1 index of best_score
                best_score[2] = mygamma # store l1_ratio number in 2 index of best_score

    # best_score = [1,4,.05]
    # return best score and its corresponding parameters to use in cost-sensitive SVM
    return best_score

def tune_pi(best_clf, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, pred_time, verbose=0):

    all_predicted_labels = []
    for file in (cv_test_files):
        pred_labels = best_clf.predict(file)
        pred_labels[pred_labels==1]=0
        pred_labels[pred_labels==-1]=1
        all_predicted_labels += [pred_labels]

    best_score = -5000

    if verbose==1:
        threshold_range = np.arange(.2,1,.02)
        adapt_range = np.arange(30,61,10)
    else:
        threshold_range = np.arange(.3,1,.1)
        adapt_range = np.arange(30,61,10)

    for mythresh in threshold_range:
        for myadapt in adapt_range:
            pred_s = np.zeros(len(cv_test_files))
            det_s = np.zeros(len(cv_test_files))
            latency = np.zeros(len(cv_test_files))
            fp = np.zeros(len(cv_test_files))
            time = np.zeros(len(cv_test_files))
            my_fracs = []
            for i,predicted_labels in enumerate(all_predicted_labels):
                outlier_fraction = create_outlier_frac(predicted_labels, myadapt)
                my_fracs+=[outlier_fraction]
                decision = create_decision(outlier_fraction, mythresh, pred_time)
                decision_sample = window_to_samples(decision, win_len, win_overlap, f_s)
                pred_s[i], det_s[i], latency[i], fp[i], time[i] = performance_stats(decision_sample, cv_seizure_times[i], f_s, pred_time)
            score,sens,fpr = gardner_score(pred_s, latency, fp, time)
            if score > best_score:
                best_score = score
                threshold = mythresh
                adapt_rate = myadapt
    print'\t\t\tBest parameters are: Threshold = ', threshold, 'Adaptation rate = ', adapt_rate
    return threshold, adapt_rate

def mean_tune_pi(best_clf, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, pred_time, verbose=0):

    adapt_rate = 40
    all_predicted_labels = []
    mean = []
    for loc, file in enumerate(cv_test_files):
        if cv_file_type[loc] == 'ictal':
            pred_labels = best_clf.predict(file)
            pred_labels[pred_labels==1]=0
            pred_labels[pred_labels==-1]=1
            preictal_labels = pred_labels[:cv_seizure_times[loc][0]]
            mean += [np.mean(preictal_labels)]
    threshold = np.mean(mean)

    return threshold, adapt_rate

def loocv_nusvm_pi(best_clf, nu, gamma, cv_labels, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, pred_time, verbose=0):

    best_score = -5000

    if verbose==1:
        threshold_range = np.arange(.2,1,.02)
        adapt_range = np.arange(20,50,10)
    else:
        threshold_range = np.arange(.3,1,.1)
        adapt_range = np.arange(30,31,10)

    model = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)

    for mythresh in threshold_range:
        for myadapt in adapt_range:
            pred_s = np.zeros(len(cv_test_files))
            det_s = np.zeros(len(cv_test_files))
            latency = np.zeros(len(cv_test_files))
            fp = np.zeros(len(cv_test_files))
            time = np.zeros(len(cv_test_files))

            for i, file in enumerate(cv_test_files):
                train_data = np.vstack((cv_test_files[k] for k in xrange(len(cv_test_files))if k != i))
                # train_labels = np.hstack(cv_labels[:i]+cv_labels[i+1:])

                test_data = cv_test_files[i]

                model.fit(train_data)
                predicted_labels = model.predict(test_data)

                outlier_fraction = create_outlier_frac(predicted_labels, myadapt)

                decision = create_decision(outlier_fraction, mythresh, pred_time)
                decision_sample = window_to_samples(decision, win_len, win_overlap, f_s)
                pred_s[i], det_s[i], latency[i], fp[i], time[i] = performance_stats(decision_sample, cv_seizure_times[i], f_s, pred_time)
            score,sens,fpr = gardner_score(pred_s, latency, fp, time)
            if score > best_score:
                best_score = score
                threshold = mythresh
                adapt_rate = myadapt
    # print'\t\t\tBest parameters are: Threshold = ', threshold, 'Adaptation rate = ', adapt_rate
    return threshold, adapt_rate

def parameter_tuning_oneclasssvm(data_list, label_list, file_types, seizure_times):

    # nu_range = [.001, .005, .01, .02, .04, .05, .08, .1]
    # gamma_range = [.00001, .00005, .000075, .0001,  .0005, .00075, .001]
    # nu_range = [.001,.01]
    # gamma_range=[.0001,.001,.01]
    # create the cost-sensitive svm by using the SGD Classifier and setting the loss to hinger and class_weight to balanced.
    # nu_range = [.001, .0025, .005, .0075, .01, .015, .02, .03, .04, .05, .075, .1]
    # gamma_range = [.00001, .000025, .00005, .000075, .0001, .0002, .0003, .0004, .0005, .0006, .00075, .0009, .001, .002, .003, .004, .005, .006, .0075, .0085, .01, .015, .02, .03, .05, .075, .1, .3, .5, .75, 1, 2.5, 5]
    nu_range = [.001,.005,.01, .05]
    gamma_range = [.0001, .001,.05,.5,1,5]

    cssvm = svm.OneClassSVM(kernel = 'rbf')

    # initialize output
    best_score = [-1000, 0, 0]
    max_total_distance = -1

    #TODO: is this the most effective method for parameter tuning?
    #Grid search using 2 forloops
    for mynu in nu_range: # for each penalty
        # for mygamma in np.hstack((np.arange(.001,.3,.005),np.arange(.3,1,.10),np.arange(1,5,.5))): # for each l1_ratio
        for mygamma in gamma_range:
            # cssvm.set_params(gamma=mygamma, nu=mynu) # set parameters of SGDClassifier
            # # score = np.zeros(len(data_list))
            # pred_s = np.zeros(len(data_list))
            # det_s = np.zeros(len(data_list))
            # latency = np.zeros(len(data_list))
            # fp = np.zeros(len(data_list))
            # time = np.zeros(len(data_list))
            #
            # training_data = np.vstack(data_list[n] for n in xrange(len(file_types)) if file_types[n] is not 'ictal')
            # cssvm.fit(training_data)
            # mythresh, myadapt = tune_pi(cssvm,file_types,seizure_times,data_list,3.0,2.0,1000,210,verbose=0)
            #
            # for i in xrange(len(data_list)):
            #
            #     cv_test_data = data_list[i]
            #     cv_test_times = seizure_times[i]
            #
            #     cv_train_data = data_list[:i] + data_list[i+1:]
            #     cv_train_times = seizure_times[:i] + seizure_times[i+1:]
            #     cv_train_file_types = file_types[:i] + file_types[i+1:]
            #
            #     # training_data = np.vstack(cv_train_data[p] for p in xrange(len(cv_train_file_types)) if cv_train_file_types[p] is not 'ictal')
            #     # cssvm.fit(training_data) # fit training set
            #     #TODO: should I score my model on ONLY preictal data? I'm currently training on the entire seizure file
            #
            #     # mythresh, myadapt = tune_pi(cssvm, cv_train_file_types, cv_train_times, cv_train_data, win_len=3.0, win_overlap=2.0, f_s=1000, pred_time=210)
            #
            #     predicted_labels = cssvm.predict(cv_test_data)
            #     predicted_labels[predicted_labels==1] = 0
            #     predicted_labels[predicted_labels==-1] = 1
            #
            #     outlier_fraction = create_outlier_frac(predicted_labels, myadapt)
            #
            #     decision = create_decision(outlier_fraction, mythresh, pred_time=210)
            #     decision_sample = window_to_samples(decision, win_len=3.0, win_overlap=2.0, f_s=1000)
            #     pred_s[i], det_s[i], latency[i], fp[i], time[i] = performance_stats(decision_sample, cv_test_times, f_s=1000, pred_time=210)
            #
            # score,s,fpr = gardner_score(pred_s, latency, fp, time)

            thismodel2 = svm.OneClassSVM(nu=mynu,gamma=mygamma,kernel='rbf')
            thismodel2.fit(np.vstack(data_list[n] for n in xrange(len(file_types)) if file_types[n] is not 'ictal'))
            # th,ad = tune_pi(thismodel2,file_types,seizure_times,data_list,3.0,2.0,1000,210)

            num_ictal = 0
            num_inter = 0
            for type in file_types:
                if type is 'ictal':
                    num_ictal += 1
                else:
                    num_inter += 1

            mean_preict = np.zeros(num_ictal)
            mean_interict = np.zeros(num_inter)

            prespot=0
            interspot=0
            for loc, myfile in enumerate(data_list):
                labs = thismodel2.predict(myfile)
                labs[labs==1]=0
                labs[labs==-1]=1
                # out_frac = create_outlier_frac(labs,ad)
                out_frac = labs
                if file_types[loc] is 'ictal':
                    times = seizure_times[loc]
                    mean_preict[prespot] = np.median(out_frac[:times[0]]) + np.mean(out_frac[:times[0]])
                    prespot+=1
                    # if mini > min_preict:
                    #     min_preict = mini
                else:
                    mean_interict[interspot] = np.median(out_frac) + np.mean(out_frac)
                    interspot+=1
                    # if maxi < max_interict:
                    #     max_interict = maxi
            mean_distance = np.mean(mean_preict) - np.mean(mean_interict)

            score = mean_distance

            # if best_score[0] == score:
            #     if max_distance > max_total_distance:
            #         max_total_distance = max_distance
            #         best_score[0] = score # store score in 0 index of best_score
            #         best_score[1] = mynu # store penalty type in 1 index of best_score
            #         best_score[2] = mygamma # store l1_ratio number in 2 index of best_score
            #         my_fracs = []
            #         thismodel = svm.OneClassSVM(nu=mynu,gamma=mygamma,kernel='rbf')
            #         print 'On the validation process, the stats were: '
            #
            #         print pred_s
            #         print det_s
            #         print latency
            #         print fp
            #         print time
            #         my_fracs = []
            #         thismodel.fit(np.vstack(data_list[n] for n in xrange(len(file_types)) if file_types[n] is not 'ictal'))
            #         th,ad = tune_pi(thismodel,file_types,seizure_times,data_list,3.0,2.0,1000,210,verbose=0)
            #         print 'Visualizing final chosen model, nu, gamma, th'
            #         for myfile in enumerate(data_list):
            #             labs = thismodel.predict(myfile[1])
            #             labs[labs==1]=0
            #             labs[labs==-1]=1
            #             my_fracs+=[create_outlier_frac(labs,ad)]
            #         viz_many_outcomes(my_fracs, seizure_times, 'i',th,1)

            if best_score[0] < score: # if the score is greater than previous score
                best_score[0] = score # store score in 0 index of best_score
                best_score[1] = mynu # store penalty type in 1 index of best_score
                best_score[2] = mygamma # store l1_ratio number in 2 index of best_score
                my_fracs = []
                thismodel = svm.OneClassSVM(nu=mynu,gamma=mygamma,kernel='rbf')
                # print 'On the validation process, the stats were: '

                # print pred_s
                # print det_s
                # print latency
                # print fp
                # print time
                # my_fracs = []
                thismodel.fit(np.vstack(data_list[n] for n in xrange(len(file_types)) if file_types[n] is not 'ictal'))
                # th,ad = tune_pi(thismodel,file_types,seizure_times,data_list,3.0,2.0,1000,210,verbose=0)
                # print 'Visualizing final chosen model, nu, gamma, th'
                # for myfile in enumerate(data_list):
                #     labs = thismodel.predict(myfile[1])
                #     labs[labs==1]=0
                #     labs[labs==-1]=1
                #     my_fracs+=[create_outlier_frac(labs,ad)]
                # viz_many_outcomes(my_fracs, seizure_times, 'i',th,1)

    # best_clf = svm.OneClassSVM(nu=best_score[1], kernel='rbf', gamma=best_score[2])
    # best_clf.fit(np.vstack(data_list[p] for p in xrange(len(file_types)) if file_types[p] is not 'ictal'))

    # return best score and its corresponding parameters to use in cost-sensitive SVM
    return thismodel, best_score[1], best_score[2]

def grid_search_oneclasssvm(train_data, score_data, score_labels, vizfiles):

    # nu_range = [.001, .005, .01, .02, .04, .05, .08, .1]
    # gamma_range = [.00001, .00005, .000075, .0001,  .0005, .00075, .001]
    # nu_range = [.001,.01]
    # gamma_range=[.0001,.001,.01]
    # create the cost-sensitive svm by using the SGD Classifier and setting the loss to hinger and class_weight to balanced.
    # nu_range = [.001, .0025, .005, .0075, .01, .015, .02, .03, .04, .05, .075, .1]
    # gamma_range = [.00001, .000025, .00005, .000075, .0001, .0002, .0003, .0004, .0005, .0006, .00075, .0009, .001, .002, .003, .004, .005, .006, .0075, .0085, .01, .015, .02, .03, .05, .075, .1, .3, .5, .75, 1, 2.5, 5]
    nu_range = [.001,.005,.01, .05]
    gamma_range = [.0001,.0005,.001,.005,.05,.1,.5,1,2,3,4,5]

    cssvm = svm.OneClassSVM(kernel = 'rbf')

    # initialize output
    best_score = [-1000, 0, 0]

    for mynu in nu_range: # for each penalty
        for mygamma in gamma_range:
            cssvm.set_params(gamma=mygamma, nu=mynu) # set parameters of SGDClassifier
            cssvm.fit(train_data)

            predicted_labels = cssvm.predict(score_data)
            score = fbeta_score(score_labels,predicted_labels,beta=.1)
            # score = score_function(predicted_labels,score_labels,beta=.05)

            print score

            if best_score[0] < score: # if the score is greater than previous score
                best_score[0] = score # store score in 0 index of best_score
                best_score[1] = mynu # store penalty type in 1 index of best_score
                best_score[2] = mygamma # store l1_ratio number in 2 index of best_score

                plt.plot(predicted_labels,lw=.2)
                plt.plot(score_labels,lw=3)
                plt.ylim(ymin=-1.5,ymax=1.5)
                plt.show()
                print 'HIGHEST SCORE'
            # outfracs= []
            # for file in vizfiles:
            #     labs = cssvm.predict(file)
            #     labs[labs==1]=0
            #     labs[labs==-1]=1
            #     outfracs += [create_outlier_frac(labs, 1)]
            # viz_many_outcomes(outfracs,[None,None,None,None,None],'ts',1,0)

    print 'Nu = ',best_score[1],'Gamma = ',best_score[2]
    best_clf = svm.OneClassSVM(nu=best_score[1],gamma=best_score[2])
    best_clf.fit(train_data)
    # return best score and its corresponding parameters to use in cost-sensitive SVM
    return best_clf

def parameter_tuning_nusvm(data_list, label_list, file_types, seizure_times):
    # nu_range = np.arange(.001,.2,.001)
    # gamma_range = [.00001, .00005, .000075, .0001,  .00025, .0005, .00075, .001]
    # nu_range = [.001, .005, .01, .02, .04, .05, .08, .1]
    # gamma_range = [.00001, .00005, .000075, .0001,  .0005, .00075, .001]
    # nu_range = [.001, .0025, .005, .0075, .01, .015, .02, .03, .04, .05, .075, .1]
    # gamma_range = [.00001, .000025, .00005, .000075, .0001, .0002, .0003, .0004, .0005, .0006, .00075, .0009, .001, .002, .003, .004, .005, .006, .0075, .0085, .01, .015, .02, .03, .05, .075, .1]
    nu_range = [.001,.005,.01,.05,.1,.5,1,3,5]
    gamma_range=[.00005,.0001,.005,.001,.005,.01]
    # create the cost-sensitive svm by using the SGD Classifier and setting the loss to hinger and class_weight to balanced.
    # cssvm = svm.SVC(kernel = 'rbf')

    # initialize output
    best_score = [-1000, 0, 0]
    max_total_distance = -1+9

    #TODO: is this the most effective method for parameter tuning?
    #Grid search using 2 forloops
    for mynu in nu_range: # for each penalty
        print mynu
        # for mygamma in np.hstack((np.arange(.001,.3,.005),np.arange(.3,1,.10),np.arange(1,5,.5))): # for each l1_ratio
        for mygamma in gamma_range:
            # cssvm.set_params(gamma=mygamma, nu=mynu) # set parameters of SGDClassifier
            # # score = np.zeros(len(data_list))
            # pred_s = np.zeros(len(data_list))
            # det_s = np.zeros(len(data_list))
            # latency = np.zeros(len(data_list))
            # fp = np.zeros(len(data_list))
            # time = np.zeros(len(data_list))
            #
            # training_data = np.vstack(data_list[n] for n in xrange(len(file_types)) if file_types[n] is not 'ictal')
            # cssvm.fit(training_data)
            # mythresh, myadapt = tune_pi(cssvm,file_types,seizure_times,data_list,3.0,2.0,1000,210,verbose=0)
            #
            # for i in xrange(len(data_list)):
            #
            #     cv_test_data = data_list[i]
            #     cv_test_times = seizure_times[i]
            #
            #     cv_train_data = data_list[:i] + data_list[i+1:]
            #     cv_train_times = seizure_times[:i] + seizure_times[i+1:]
            #     cv_train_file_types = file_types[:i] + file_types[i+1:]
            #
            #     # training_data = np.vstack(cv_train_data[p] for p in xrange(len(cv_train_file_types)) if cv_train_file_types[p] is not 'ictal')
            #     # cssvm.fit(training_data) # fit training set
            #     #TODO: should I score my model on ONLY preictal data? I'm currently training on the entire seizure file
            #
            #     # mythresh, myadapt = tune_pi(cssvm, cv_train_file_types, cv_train_times, cv_train_data, win_len=3.0, win_overlap=2.0, f_s=1000, pred_time=210)
            #
            #     predicted_labels = cssvm.predict(cv_test_data)
            #     predicted_labels[predicted_labels==1] = 0
            #     predicted_labels[predicted_labels==-1] = 1
            #
            #     outlier_fraction = create_outlier_frac(predicted_labels, myadapt)
            #
            #     decision = create_decision(outlier_fraction, mythresh, pred_time=210)
            #     decision_sample = window_to_samples(decision, win_len=3.0, win_overlap=2.0, f_s=1000)
            #     pred_s[i], det_s[i], latency[i], fp[i], time[i] = performance_stats(decision_sample, cv_test_times, f_s=1000, pred_time=210)
            #
            # score,s,fpr = gardner_score(pred_s, latency, fp, time)

            thismodel2 = svm.SVC(C=mynu,gamma=mygamma,kernel='rbf', class_weight='balanced')
            thismodel2.fit(np.vstack(data_list), np.hstack(label_list))
            # th,ad = tune_pi(thismodel2,file_types,seizure_times,data_list,3.0,2.0,1000,210)

            num_ictal = 0
            num_inter = 0
            for type in file_types:
                if type is 'ictal':
                    num_ictal += 1
                else:
                    num_inter += 1

            mean_preict = np.zeros(num_ictal)
            mean_interict = np.zeros(num_inter)

            prespot=0
            interspot=0
            for loc, myfile in enumerate(data_list):
                labs = thismodel2.predict(myfile)
                labs[labs==1]=0
                labs[labs==-1]=1
                # out_frac = create_outlier_frac(labs,ad)
                out_frac = labs
                if file_types[loc] is 'ictal':
                    times = seizure_times[loc]
                    mean_preict[prespot] = np.median(out_frac[:times[0]]) + np.mean(out_frac[:times[0]])
                    prespot+=1
                    # if mini > min_preict:
                    #     min_preict = mini
                else:
                    mean_interict[interspot] = np.median(out_frac) + np.mean(out_frac)
                    interspot+=1
                    # if maxi < max_interict:
                    #     max_interict = maxi
            mean_distance = np.mean(mean_preict) - np.mean(mean_interict)

            score = mean_distance

            # if best_score[0] == score:
            #     if max_distance > max_total_distance:
            #         max_total_distance = max_distance
            #         best_score[0] = score # store score in 0 index of best_score
            #         best_score[1] = mynu # store penalty type in 1 index of best_score
            #         best_score[2] = mygamma # store l1_ratio number in 2 index of best_score
            #         my_fracs = []
            #         thismodel = svm.OneClassSVM(nu=mynu,gamma=mygamma,kernel='rbf')
            #         print 'On the validation process, the stats were: '
            #
            #         print pred_s
            #         print det_s
            #         print latency
            #         print fp
            #         print time
            #         my_fracs = []
            #         thismodel.fit(np.vstack(data_list[n] for n in xrange(len(file_types)) if file_types[n] is not 'ictal'))
            #         th,ad = tune_pi(thismodel,file_types,seizure_times,data_list,3.0,2.0,1000,210,verbose=0)
            #         print 'Visualizing final chosen model, nu, gamma, th'
            #         for myfile in enumerate(data_list):
            #             labs = thismodel.predict(myfile[1])
            #             labs[labs==1]=0
            #             labs[labs==-1]=1
            #             my_fracs+=[create_outlier_frac(labs,ad)]
            #         viz_many_outcomes(my_fracs, seizure_times, 'i',th,1)

            if best_score[0] < score: # if the score is greater than previous score
                best_score[0] = score # store score in 0 index of best_score
                best_score[1] = mynu # store penalty type in 1 index of best_score
                best_score[2] = mygamma # store l1_ratio number in 2 index of best_score
                my_fracs = []
    thismodel = svm.SVC(C=best_score[1],gamma=best_score[2],kernel='rbf', class_weight='balanced')
    # print 'On the validation process, the stats were: '

    # print pred_s
    # print det_s
    # print latency
    # print fp
    # print time
    # my_fracs = []
    thismodel.fit(np.vstack(data_list), np.hstack(label_list))
    # th,ad = tune_pi(thismodel,file_types,seizure_times,data_list,3.0,2.0,1000,210,verbose=0)
    # print 'Visualizing final chosen model, nu, gamma, th'
    # for myfile in enumerate(data_list):
    #     labs = thismodel.predict(myfile[1])
    #     labs[labs==1]=0
    #     labs[labs==-1]=1
    #     my_fracs+=[create_outlier_frac(labs,ad)]
    # viz_many_outcomes(my_fracs, seizure_times, 'i',th,1)

    # best_clf = svm.OneClassSVM(nu=best_score[1], kernel='rbf', gamma=best_score[2])
    # best_clf.fit(np.vstack(data_list[p] for p in xrange(len(file_types)) if file_types[p] is not 'ictal'))

    # return best score and its corresponding parameters to use in cost-sensitive SVM
    return thismodel, best_score[1], best_score[2]

def grid_tune_svm(train_data, train_labels):

    # define the grid
    # Options: different ranges, more parameters
    # nu_range = [.001, .002, .004, .005, .008, .01, .015, .02, .04, .05, .08, .1]
    # gamma_range = [.00001, .00005, .0001, .0005, .00075, .001, .002, .003, .004, .005, .006, .008, .01, .05, .1, .2, .5]
    # nu_range = [.001,.005,.01, .05]
    # gamma_range = [.0001, .001,.05,.5,1,5]

    nu_range = [2,6,12]
    gamma_range = [.001,.002,.005]
    grid = [{"C": nu_range, "gamma": gamma_range}]


    # define the scoring function(for model evaluation)
    # Options see: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # scoring = classifier_toolbox.f_beta
    scoring = 'f1'

    #initiate nu-svc object
    svr = svm.SVC(class_weight='balanced',kernel='rbf')

    # initiate grid search object
    clf = GridSearchCV(svr,grid,scoring=scoring)
    clf.fit(train_data,train_labels)
    # plt.plot(train_labels)
    # plt.plot(clf.predict(train_data))
    # plt.ylim(ymin=-1.5,ymax=1.5)
    # plt.show()
    print "\t\t\tThe best parameters are: ", clf.best_params_
    # # print "\t\t\t\tThis set of parameters receive a score of: ", clf.best_score_

    return clf.best_estimator_

def gardner_score(sensitivity, latency, fp, time, pred_time=500):

    #TODO: ictal files are doing weird things for sensitivity, latency
    # get aggregate statistics on performance metrics
    S = np.nanmean(sensitivity)
    FPR = float(np.nansum(fp)) / float(np.nansum(time))

    #TODO: means? Medians?
    # if any seizure is detected
    if np.any(np.isfinite(latency)):
        mu = np.nanmean(latency)
        detected = latency[np.isfinite(latency)]
        EDF = float(np.where(detected < 0)[0].size) / float(detected.size)

    else:
        mu = 500.0 # else give terrible values
        EDF = 0.0

    # objective function
    # desired_latency = -15.0
    # alpha1 = 250*S-10*(1-np.sign(S-0.75))
    # alpha2 = 20*EDF
    # alpha3 = -15*FPR-30*(1-np.sign(allowable_fpr-FPR))
    # alpha4 = min(30.0*(mu / desired_latency),30)
    # score = alpha1+alpha2+alpha3+alpha4

    alpha1 = 100*S

    beta = 10
    allowable_fpr = 6.0
    random_fpr = float(3600)/float(pred_time)
    if FPR >= random_fpr:
        alpha2 = -100
    else:
        alpha2 = min(float(-beta*FPR)/allowable_fpr, float((-100-beta)*(FPR-random_fpr))/float(random_fpr-allowable_fpr)-100)

    desired_latency = -30
    alpha3 = min(30,30.0/float(desired_latency)*mu)

    score = alpha1+alpha2+alpha3

    return score, S, FPR

def plot_params(predicted_training_labels, file_type):
    try_adapt_rate = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    num_ictal = len([file_type[i] for i in xrange(len(file_type)) if file_type[i] is 'ictal'])
    num_inter = len([file_type[i] for i in xrange(len(file_type)) if file_type[i] is not 'ictal'])

    ictal_distances = [None] * len(try_adapt_rate)
    inter_distances = [None] * len(try_adapt_rate)
    plot_thresholds = [None] * len(try_adapt_rate)

    for j, adapt_rate in enumerate(try_adapt_rate):
        ictal_thresh = [None] * num_ictal
        inter_thresh = [None] * num_inter

        ic = 0
        inter = 0
        for i, (labels, types) in enumerate(zip(predicted_training_labels, file_type)):
            out_frac = create_outlier_frac(labels, adapt_rate)
            if types is 'ictal':
                thresh = np.median(out_frac)
                ictal_thresh[ic] = thresh
                ic += 1
            elif types is not 'ictal':
                thresh = np.median(out_frac)
                inter_thresh[inter] = thresh
                inter += 1
        thresholds = [np.median(ictal_thresh), np.median(inter_thresh)]
        plot_threshold = np.mean(thresholds) # int

        ictal_distance = [None] * num_ictal
        inter_distance = [None] * num_inter
        ic = 0
        inter = 0
        for i, (labels, types) in enumerate(zip(predicted_training_labels, file_type)):
            out_frac = create_outlier_frac(labels, adapt_rate)
            if types is 'ictal':
                distance = min(out_frac) - plot_threshold
                ictal_distance[ic] = distance
                ic += 1
            elif types is not 'ictal':
                distance = max(out_frac) - plot_threshold
                inter_distance[inter] = distance
                inter += 1
        min_ictal_distance = min(ictal_distance)
        max_inter_distance = max(inter_distance)

        plot_thresholds[j] = plot_threshold
        ictal_distances[j] = min_ictal_distance
        inter_distances[j] = max_inter_distance

    bad_ictal_indices = [i for i in xrange(len(ictal_distances)) if ictal_distances[i] < 0]
    bad_inter_indices = [i for i in xrange(len(inter_distances)) if inter_distances[i] > 0]

    bad_indices = np.concatenate([bad_ictal_indices, bad_inter_indices])
    bad_indices = list(set(bad_indices))

    if len(bad_indices) > 0 and len(bad_indices) < len(try_adapt_rate):
        print 'Getting rid of bad indices'
        plot_thresholds = np.delete(plot_thresholds, bad_indices)
        ictal_distances = np.delete(ictal_distances, bad_indices)
        inter_distances = np.delete(inter_distances, bad_indices)

        total_distance = np.subtract(ictal_distances, inter_distances)
        max_distance = max(total_distance)
        total_distance = list(total_distance)
        ideal_index = total_distance.index(max_distance)
        adapt_rate = try_adapt_rate[ideal_index]
        ideal_threshold = plot_thresholds[ideal_index]


    elif len(bad_indices) == 0:
        print 'no bad indices'
        total_distance = np.subtract(ictal_distances, inter_distances)
        max_distance = max(total_distance)
        total_distance = list(total_distance)
        ideal_index = total_distance.index(max_distance)
        adapt_rate = try_adapt_rate[ideal_index]
        ideal_threshold = plot_thresholds[ideal_index]

    elif len(bad_indices) == len(try_adapt_rate):
        print 'All bad indices'
        total_distance = np.abs(np.subtract(ictal_distances, inter_distances))
        min_distance = min(total_distance)
        total_distance = list(total_distance)
        ideal_index = total_distance.index(min_distance)
        adapt_rate = try_adapt_rate[ideal_index]
        ideal_threshold = plot_thresholds[ideal_index]

    return ideal_threshold, adapt_rate

def training(feature_set, label_set, file_types):

    num_ictal=0
    num_inter=0
    for type in file_types:
        if type is 'ictal':
            num_ictal+=1
        else:
            num_inter+=1

    # best_score = parameter_tuning_csvm(feature_set, label_set, num_ictal, num_inter)
    # print '\tBest score = ', best_score[0], ' Found C = ', best_score[1], ' Found gamma = ', best_score[2]
    # cssvm = svm.SVC(C = best_score[1], kernel = 'rbf', gamma = best_score[2], class_weight='balanced')
    # cssvm.fit(np.vstack((feature_set[n] for n in xrange(len(feature_set)))), np.hstack((label_set[m] for m in xrange(len(label_set)))))

    # best_score = parameter_tuning_nusvm(feature_set, label_set, num_ictal, num_inter)
    # print '\tBest score = ', best_score[0], ' Found nu = ', best_score[1], ' Found gamma = ', best_score[2]
    # cssvm = svm.NuSVC(nu = best_score[1], kernel = 'rbf', gamma = best_score[2], class_weight='balanced')
    # cssvm.fit(np.vstack((feature_set[n] for n in xrange(len(feature_set)))), np.hstack((label_set[m] for m in xrange(len(label_set)))))

    cssvm = grid_tune_svm(feature_set,label_set)

    #TODO: does kalman expect consecutive inputs? as in nonrandom windows? would this make the filter more effective?
    #TODO: Take advantage of Kalman's abilities
    kf = []

    return cssvm, kf

def offline_training(cv_file_names, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, freq_band, test_index, patient_id, pred_time):


    load_data = sio.loadmat('evc_training{}_{}.mat'.format(patient_id[-2:],test_index))
    training_evc_cv_files = load_data.get('data')[0]

    print '\t\tTraining the classification model'
    training_evc_cv_files = np.ndarray.tolist(training_evc_cv_files)
    print'\t\tGardner intercept'
    training_labels = []
    training_data = []
    for loc, type in enumerate(cv_file_type):
        gardner_matrix = np.zeros((training_evc_cv_files[loc].shape[0],cv_test_files[loc].shape[1]*3))
        for col in xrange(cv_test_files[loc].shape[1]):
            one_channel_data = cv_test_files[loc][:,col]
            gardner_matrix[:,col:col+3] = energy_features(patient_id,'', [],one_channel_data[:,None] , win_len=3,win_overlap=2)
        if type is 'ictal':
            training_data += [np.hstack((training_evc_cv_files[loc][:,:],gardner_matrix))]
            training_labels += [-1*np.ones(training_data[loc].shape[0])]
        else:
            training_data += [np.hstack((training_evc_cv_files[loc][:,:],gardner_matrix))]
            training_labels += [np.ones(training_data[loc].shape[0])]
    print'\t\tdone gardnering'
    # best_clf = grid_search_oneclasssvm(np.vstack(training_evc_cv_files[n] for n in xrange(len(cv_file_type)) if cv_file_type[n] is not 'ictal'), np.vstack(training_data), np.hstack(training_labels),training_evc_cv_files)
    # best_clf, best_nu, best_gamma = parameter_tuning_oneclasssvm(training_evc_cv_files, training_labels, cv_file_type, cv_seizure_times)
    # best_clf, best_nu, best_gamma = parameter_tuning_nusvm(training_evc_cv_files, training_labels, cv_file_type, cv_seizure_times)
    best_clf = grid_tune_svm(np.vstack(training_evc_cv_files),np.hstack(training_labels))

    print '\t\tTraining the decision model'
    threshold, adapt_rate = tune_pi(best_clf, cv_file_type, cv_seizure_times, training_evc_cv_files, win_len, win_overlap, f_s, pred_time, verbose=1)
    # threshold, adapt_rate = loocv_nusvm_pi(best_clf, best_nu, best_gamma, training_labels, cv_file_type, cv_seizure_times, training_evc_cv_files, win_len, win_overlap, f_s, pred_time, verbose=1)
    # threshold, adapt_rate = mean_tune_pi(best_clf, cv_file_type, cv_seizure_times, training_evc_cv_files, win_len, win_overlap, f_s, pred_time, verbose=1)

    # print'\t\t\tBest parameters are: Threshold = ', threshold, 'Adaptation rate = ', adapt_rate
    # threshold = 1
    # adapt_rate = 40
    mean_coherency_matrix = []
    sd_coherency_matrix = []

    return mean_coherency_matrix, sd_coherency_matrix, best_clf, threshold*.95, adapt_rate, training_evc_cv_files

def online_testing(test_data, mean_mat, std_mat, cs_svm, threshold, adapt_rate, win_len, win_overlap, f_s, freq_band, test_index, patient_id, pred_time):

    # print'\t\tBuilding coherency matrices'
    # test_coherency_matrices = [build_coherency_array(test_data, win_len, win_overlap, f_s, freq_band)]
    # print'\t\tTransforming coherency matrices'
    # transformed_test_coherency_matrices = transform_coherency(test_coherency_matrices, mean_mat, std_mat)
    # print'\t\tFinding eigenvec centrality'
    # test_evc = find_evc(transformed_test_coherency_matrices)
    # test_evc = np.asarray(test_evc)
    # test_evc = test_evc[0,:,:]
    # sio.savemat('evc_testing{}_{}.mat'.format(patient_id[-2:],test_index), {'data':test_evc})

    load_test_data = sio.loadmat('evc_testing{}_{}.mat'.format(patient_id[-2:],test_index))
    testing_evc_cv_files = load_test_data.get('data')
    testing_data = np.vstack(testing_evc_cv_files)

    print '\t\tPredicting test labels'
    testing_labels = cs_svm.predict(testing_data) # predict with test set
    testing_labels[testing_labels==1] = 0
    testing_labels[testing_labels==-1] = 1

    print '\t\tCreating outlier fraction'
    test_outlier_fraction = create_outlier_frac(testing_labels, adapt_rate)

    print '\t\tCreating prediciton decision'
    decision = create_decision(test_outlier_fraction, threshold, pred_time)

    return decision, test_outlier_fraction

def parent_function():

    # parameters -- sampling data

    win_len = 3.0  # in seconds
    win_overlap = 2.0  # in seconds
    f_s = float(1e3)  # sampling frequency

    #TODO: more patients, in particular patient 23
    patients = ['TS039', 'TS041']
    long_interictal = [False, False]
    include_awake = True
    #TODO: try with asleep data
    include_asleep = False
    freq_band = [30,40]
    pred_time = 500

    # get the paths worked out
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')

    for patient_index, patient_id in enumerate(patients):

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)
        print 'Retreiving stored raw data'
        good_channels = choose_best_channels(patient_id)
        all_files, data_filenames, file_type, seizure_times, seizure_print = analyze_patient_raw(p_data_path, f_s, include_awake, include_asleep, long_interictal[patient_index], good_channels)

        file_num = len(all_files)
        prediction_sensitivity = np.zeros(file_num)
        detection_sensitivity = np.zeros(file_num)
        latency = np.zeros(file_num)
        fp = np.zeros(file_num)
        time = np.zeros(file_num)

        # cross-validation
        for i in [0,1,2,3,4,5]:

            # set up test files, seizure times, etc. for this k-fold
            print '\nCross validations, k-fold %d of %d ...' % (i+1, file_num)
            testing_file = all_files[i]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_test_files = all_files[:i] + all_files[i+1:]
            cv_file_type = file_type[:i] + file_type[i+1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i+1:]

            print '\tEntering offline training'
            mean_mat, std_mat, cs_svm, threshold, adapt_rate, traindat = offline_training(cv_file_names, cv_file_type, cv_seizure_times, cv_test_files, win_len, win_overlap, f_s, freq_band, i, patient_id, pred_time)

            #START HERE TOMORROW; fix gardner c/adapt tuning on interictal files
            # threshold, adapt_rate = plot_params(all_labs, cv_file_type)

            print'\tEntering online testing'
            test_decision_windows, test_outlier_frac_windows = online_testing(testing_file, mean_mat, std_mat, cs_svm, threshold, adapt_rate, win_len, win_overlap, f_s, freq_band, i, patient_id, pred_time)
            print'\tEnded online testing'

            print'\tPerformance stats & visualization'
            test_decision_sample = window_to_samples(test_decision_windows, win_len, win_overlap, f_s)
            test_outlier_frac_sample = window_to_samples(test_outlier_frac_windows, win_len, win_overlap, f_s)
            prediction_sensitivity[i], detection_sensitivity[i], latency[i], fp[i], time[i] = performance_stats(test_decision_sample, seizure_times[i], f_s, pred_time)
            print '\tPrediction sensitivity = ', prediction_sensitivity[i], 'Detection sensitivity = ', detection_sensitivity[i], 'Latency = ', latency[i], 'FP = ', fp[i], 'Time = ', time[i]

            all_fracs = []
            for file in traindat:
                labels = cs_svm.predict(file)
                labels[labels==1]=0
                labels[labels==-1]=1
                all_fracs += [create_outlier_frac(labels, adapt_rate)]
            viz_many_outcomes(all_fracs, cv_seizure_times, patient_id, threshold, i)

            viz_single_outcome(test_decision_sample, test_outlier_frac_sample, testing_file[:,0], seizure_times[i], threshold, i, patient_id, f_s)

        fpr = float(np.nansum(fp)) / float(np.nansum(time))
        print 'Mean predition sensitivity = ', np.nanmean(prediction_sensitivity), 'Mean detection sensitivity = ', np.nanmean(detection_sensitivity), 'Mean latency = ', np.nanmean(latency), 'Mean FPR = ', fpr
        print 'Median prediction sensitivity = ', np.nanmedian(prediction_sensitivity), 'Median detection sensitivity = ', np.nanmedian(detection_sensitivity), 'Median latency = ', np.nanmedian(latency)

if __name__ == '__main__':
    #TODO: train/test cv for c/adapt, color grid for params, chain tuner, try changing nu and gamma range, view every output of nu gamma loop
    parent_function()
