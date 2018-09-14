import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import signal
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans
from DCEpy.Features.Spectral_Energy import spectral_energy
import itertools

# from edfread import edfread
import array
import random
#import feature functions
from feature_functions import burns_features
from feature_functions import energy_features
from feature_functions import spectral_energy_features
from AnalyzePatient import analyze_patient_raw
from AnalyzePatient import analyze_patient_denoise
from AnalyzePatient import analyze_patient_noise



def random_forest_feature_selector(feature_matrix, labels, feature_names):
    rf = RandomForestClassifier()
    rf.fit(feature_matrix, labels)
    print "\t\tFeatures sorted by their score according to random forests & hand labeled data:"
    print "\t\t", sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)

def recursive_feature_selection(feature_matrix, labels, feature_names):
    estimator = SVC(kernel="linear")
    selector = RFECV(estimator, step=1, cv=3)
    selector = selector.fit(feature_matrix, labels)
    print "\t\tFeatures sorted by their score according to recursive feature selection & hand labeled data:"
    print "\t\t", sorted(zip(map(lambda x: round(x, 4), selector.ranking_), feature_names), reverse=False)

def cluster(data_matrix, num_data_clusters):
    print '\t\tClustering features using kmeans'
    clusters = KMeans(num_data_clusters)  # make cluster class
    clusters.fit(data_matrix)  # train model
    labels = clusters.predict(data_matrix)  # use trained model to label clusters
    inertia = clusters.inertia_
    return inertia, labels

def plot_cluster_assignments(labels, feature_names):
    plt.plot(labels)
    plt.show()

def hand_label(num_data_windows, seizure_times, win_len, win_overlap, f_s, file_type, preictal_length):
    # 0: interictal awake, 1: preictal, 2: ictal, 3: postictal, 4 intericatal asleep

    labels = np.empty([num_data_windows])

    if file_type is "ictal":
        seizure_start_time = seizure_times[0] * f_s
        seizure_end_time = seizure_times[1] * f_s
        preictal_start_time = (seizure_start_time - preictal_length*f_s)

        seizure_start_window = int(seizure_start_time / (win_len - win_overlap))
        seizure_end_window = int(seizure_end_time / (win_len - win_overlap))
        preictal_start_window = int(preictal_start_time / (win_len - win_overlap))

        labels[:preictal_start_window] = [0] * preictal_start_window
        labels[preictal_start_window:seizure_start_window] = [1] * (seizure_start_window - preictal_start_window)
        labels[seizure_start_window:seizure_end_window] = [2] * (seizure_end_window - seizure_start_window)
        labels[seizure_end_window:] = [3] * (num_data_windows - seizure_end_window)

    elif file_type is "awake":
        labels = [0] * num_data_windows

    elif file_type is "sleep":
        labels = [4] * num_data_windows

    else:
        print "Uh-oh! In attempting to label the data, the file type is not recognized."

    return labels

def unsupervised_label(raw_data):
    psd_features = spectral_energy.feature_selection(raw_data)
    unsupervised_feature_labels = spectral_energy.clustering(psd_features)
    return unsupervised_feature_labels

def stack_dict(dictionary, keys):
    for i,feature_of_interest in enumerate(keys):
        if feature_of_interest == 'Zhang':
            dictionary[feature_of_interest] = np.repeat(dictionary[feature_of_interest],4,axis=0)
        if feature_of_interest == 'Gardner':
            dictionary[feature_of_interest] = np.delete(dictionary[feature_of_interest],[0,1],axis=0)
        if feature_of_interest == 'Burns':
            dictionary[feature_of_interest] = np.delete(dictionary[feature_of_interest],[0,1,2,3],axis=0)
    stacked_dict = np.hstack([dictionary[feature_name] for feature_name in keys])
    names = []
    for i in xrange(len(dictionary)):
        datamat = dictionary[keys[i]]
        names += [keys[i] + str(dim) for dim in xrange(datamat.shape[1])]
    return stacked_dict, names

def build_feature_dict(patient_id, seizure_file_name, raw_data, raw_data_one_channel, feature_names, feature_func_dict,
                       win_len, win_overlap):

    feat_matrices = {}
    for i, feature_of_interest in enumerate(feature_names):
        print "\t\tFinding features for ", feature_of_interest
        X_feat = feature_func_dict[feature_of_interest](patient_id,seizure_file_name,raw_data,raw_data_one_channel,win_len,
                                           win_overlap)  # extract
        feat_matrices[feature_of_interest] = X_feat

    return feat_matrices

def reduce_channels(data, channels):
    reduced_data = data[:, channels]
    return reduced_data

def update_list(original, update):
    count = 0
    for a in update:
        place = a[0] + count
        original = original[:place] + a[1] + original[place + 1:]
        count += len(a[1]) - 1
    return original

if __name__ == '__main__':

    # parameters -- sampling data
    win_len = 1.0  # in seconds
    win_overlap = 0.5  # in seconds
    f_s = float(1e3)  # sampling frequency
    win_len = int(win_len * f_s)
    win_overlap = int(win_overlap * f_s)

    patients = ['TS041']
    long_interictal = [False]
    include_awake = False
    include_asleep = False
    chosen_channels = [107]
    feature_names = ['Zhang']
    feature_func_dict = {'Burns': burns_features, 'Gardner': energy_features, 'Zhang': spectral_energy_features}
    num_data_clusters = 4
    preictal_length = 30  # seconds

    # get the paths worked out
    to_data = os.path.dirname(os.path.dirname(os.getcwd()))
    data_path = os.path.join(to_data, 'data')

    for k, patient_id in enumerate(patients):

        print "---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)

        # analyze the patient, write to the file
        # options: analyze_patient_raw; analyze_patient_denoise; analyze_patient_noise
        all_files, data_filenames, file_type, seizure_times, seizure_print = analyze_patient_raw(p_data_path, f_s,
                                                                                             include_awake,
                                                                                             include_asleep,
                                                                                             long_interictal[k])
        file_num = len(all_files)

        # cross-validation
        for i in xrange(file_num):

            X_test_raw = all_files[i]
            X_test_one_channel = reduce_channels(X_test_raw, chosen_channels)

            # set up test files, seizure times, etc. for this k-fold
            print 'Test on seizure file %d of %d ...' % (i + 1, file_num)

            cv_test_files = all_files[:i] + all_files[i + 1:]
            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            # collect interictal files
            print '\tBuilding dictionary of features matrices'
            feat_dict = build_feature_dict(patient_id, data_filenames[i], X_test_raw, X_test_one_channel, feature_names,
                                           feature_func_dict, win_len, win_overlap)

            print '\tBeginning unsupervised feature selection...'

            all_features_inertias = np.empty(len(feature_names))
            for j in xrange(len(feature_names)):
                print '\tClustering data for feature: ', feature_names[j]
                data_matrix = feat_dict[feature_names[j]]
                inertia, labels = cluster(data_matrix, num_data_clusters)
                all_features_inertias[j] = inertia
                plot_cluster_assignments(labels, feature_names[j])
                # TODO variance, gradient; add new features, add weights features;

            print '\tFeatures evaluated: ', feature_names
            print '\tCorresponding inertias ', all_features_inertias
            print '\tChosen feature using unsupervised learning, minimizing kmeans inertia: ', feature_names[
                np.argmax(inertia)]

            print '\tBeginning supervised feature selection, hand labels...'
            num_data_windows = data_matrix.shape[0]
            full_feature_matrix, names = stack_dict(feat_dict, feature_names)

            # print '\tHand labeling data'
            # hand_seizure_labels = hand_label(num_data_windows, seizure_times[i], win_len, win_overlap, f_s,
            #                                  file_type[i], preictal_length)
            #
            # print '\tUsing random forests for feature selection'
            # random_forest_feature_selector(full_feature_matrix, hand_seizure_labels, names)
            #
            # print '\tUsing recursive feature selection'
            # recursive_feature_selection(full_feature_matrix, hand_seizure_labels, names)

            print '\tBeginning supervised feature selection, unsupervised labels...'

            print '\tUnsupervised labeling of data'
            unsupervised_feature_labels = unsupervised_label(X_test_raw)

            # print '\tUsing random forests for feature selection'
            # random_forest_feature_selector(full_feature_matrix, list(itertools.chain.from_iterable(itertools.repeat(label, 4) for label in unsupervised_feature_labels)), names)


            # print '\tUsing recursive feature selection'
            # recursive_feature_selection(full_feature_matrix, unsupervised_feature_labels, names)
