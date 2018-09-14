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
from sklearn.decomposition import PCA
from DCEpy.Features.Spectral_Energy import spectral_energy
import itertools

# from edfread import edfread
import array
import random
#import feature functions
from DCEpy.Features.feature_functions import burns_features
from DCEpy.Features.feature_functions import energy_features
from DCEpy.Features.feature_functions import spectral_energy_features
from DCEpy.Features.AnalyzePatient import analyze_patient_raw
from DCEpy.Features.AnalyzePatient import analyze_patient_denoise
from DCEpy.Features.AnalyzePatient import analyze_patient_noise

def collect_windows(full_dict, all_file_names, key):
    num_training_windows = 2000
    all_training_data = np.vstack([full_dict[file_name][key] for file_name in all_file_names])
    if all_training_data.shape[0]>num_training_windows:
        training_data = all_training_data[np.random.randint(all_training_data.shape[0],size=num_training_windows), :]
    else:
        training_data = all_training_data
    return training_data

def random_forest_feature_selector(feature_matrix, labels, feature_names):
    rf = RandomForestClassifier()
    rf.fit(feature_matrix, labels)
    print "\t\tFeatures sorted by their score according to random forests & hand labeled data:"
    print "\t\t", sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)
    return rf.feature_importances_, feature_names

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

def plot_cluster_assignments(labels):
    plt.plot(labels)
    plt.show()

def hand_label(num_data_windows, seizure_times, win_len, win_overlap, f_s, file_type, preictal_length):
    # 0: interictal awake, 1: preictal, 2: ictal, 3: postictal, 4 intericatal asleep

    labels = np.empty([num_data_windows])

    if file_type is "ictal":
        seizure_start_time = seizure_times[0]
        seizure_end_time = seizure_times[1]
        preictal_start_time = (seizure_start_time - preictal_length)

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

def hand_label_seizuretimes(datamat, file_type, seizure_times, f_s=1000, win_len=1, win_overlap=0.5):
    n,p = datamat.shape
    labels = np.zeros(n) #label is 0 if not seizure

    if file_type is "ictal":

        seizure_start_time = seizure_times[0]
        seizure_end_time = seizure_times[1]

        seizure_start_window = int(seizure_start_time / (win_len - win_overlap))
        seizure_end_window = int(seizure_end_time / (win_len - win_overlap))

        labels[0:seizure_start_window] = np.ones(seizure_start_window)

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
        # below is needed for fasai code, where testing/training are not separated
        # if feature_of_interest == 'Zhang':
        #     X_feat = X_feat[:,:,0]
        feat_matrices[feature_of_interest] = X_feat

    return feat_matrices

def anomaly_detection_viz(feature_names, training_data_dict, testing_feat_dict, seizure_times):

    myrange = np.arange(0.05,.6,.05)
    fig, axes = plt.subplots(5,2)
    row=0
    col=0

    for mynu in myrange:

        for featcount, feature_of_interest in enumerate(feature_names):

            clf = svm.OneClassSVM(kernel='rbf', nu=mynu)
            clf.fit(training_data_dict[feature_of_interest])
            novelty_seq = clf.predict(testing_feat_dict[feature_of_interest])
            axes[row, col].plot(novelty_seq)
            axes[row, col].set_ylim(ymin=-1.5, ymax=1.5)
            relevant_times = seizure_times
            if relevant_times is not None:
                axes[row, col].axvline(x=relevant_times[0]*2, lw=3, color='r')
            axes[row, col].set_title('%f'%mynu)

        row+=1
        if row>4:
            row=0
            col=1
    plt.show()

def outlier_frac_viz(training_data_dict, testing_feat_dict, feature_names):

    clf = svm.OneClassSVM(kernel='rbf', nu=0.05)
    clf.fit(training_data_dict['Burns'])
    novelty_seq = clf.predict(testing_feat_dict['Burns'])
    z = np.copy(novelty_seq)
    z[np.where(novelty_seq==-1)] = 1 # anomalous
    z[np.where(novelty_seq==1)] = 0 # not anomalous
    n,p = testing_feat_dict['Burns'].shape


    myrange = np.arange(20,70,5)

    fig, axes = plt.subplots(5,2)
    row=0
    col=0

    for myadapt in myrange:

        for featcount, feature_of_interest in enumerate(feature_names):
            out_frac = np.zeros(novelty_seq.size)

            for iii in np.arange(myadapt,n):
                out_frac[iii] = np.mean(z[iii-myadapt:iii])

            axes[row, col].plot(out_frac)
            axes[row, col].set_ylim(ymin=0, ymax=1)
            relevant_times = seizure_times[i]
            if relevant_times is not None:
                axes[row, col].axvline(x=relevant_times[0]*2, lw=3, color='r')
            axes[row, col].set_title('%f'%myadapt)

        row+=1
        if row>4:
            row=0
            col=1
    plt.show()

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

def write_file(file_path, X):
    print '\t\tWriting energy file'
    n,p = X.shape
    f = open(file_path, 'w')
    for i in range(n):
        str = ''
        for j in range(p):
            str += "%.6f," %(X[i,j])
        f.write(str[:-1] + "\n")
    f.close()
    return

def read_file(file_path):

    reader = csv.reader(open(file_path, 'rb'))
    X = []
    for i,row in enumerate(reader):
        X.append(np.array([float(b) for b in row]))
    X = X.asarray

    return X

if __name__ == '__main__':

    # parameters -- sampling data
    win_len = 1.0  # in seconds
    win_overlap = 0.5  # in seconds
    f_s = float(1e3)  # sampling frequency
    win_len = int(win_len * f_s)
    win_overlap = int(win_overlap * f_s)

    patients = ['TS041']
    long_interictal = [False]
    include_awake = True
    include_asleep = False
    chosen_channels = [1]
    feature_names = ['Burns']
    feature_func_dict = {'Burns': burns_features, 'Gardner': energy_features, 'Zhang': spectral_energy_features}
    num_data_clusters = 4
    preictal_length = 30  # seconds

    # get the paths worked out
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')

    for k, patient_id in enumerate(patients):

        print "---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)

        # analyze the patient, write to the file
        all_files, data_filenames, file_type, seizure_times, seizure_print = analyze_patient_raw(p_data_path, f_s,
                                                                                             include_awake,
                                                                                             include_asleep,
                                                                                             long_interictal[k])
        file_num = len(all_files)
        all_feature_dicts = {}
        for j in xrange(file_num):
            print '\tBuilding feature dictionary for file',j
            all_feature_dicts[data_filenames[j]] = build_feature_dict(patient_id, data_filenames[j], all_files[j], reduce_channels(all_files[j],chosen_channels), feature_names, feature_func_dict, win_len, win_overlap)
        all_dims_to_keep = {}

        # cross-validation
        for i in xrange(file_num):

            # set up test files, seizure times, etc. for this k-fold
            print '\nCross validations, k-fold %d of %d ...' % (i + 1, file_num)

            testing_file_name = data_filenames[i]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            seizure_labels = {}
            for spot, filename in enumerate(cv_file_names):
                seizure_labels[filename] = hand_label_seizuretimes(all_feature_dicts[filename]['Burns'], cv_file_type[spot], cv_seizure_times[spot])

            all_training_data = np.vstack([all_feature_dicts[training_file]['Burns'] for training_file in cv_file_names])
            all_training_labels = np.hstack([seizure_labels[training_file] for training_file in cv_file_names])

            feature_names = np.arange(0,134,1)
            print '\tRandom forests'
            importances, dimensions = random_forest_feature_selector(all_training_data, all_training_labels, feature_names)

            important_features = sorted(zip(map(lambda x: round(x, 4), importances), dimensions), reverse=True)

            last_dimension = np.where(np.cumsum(np.flipud(np.sort(importances)))>.75)[0][0]
            dimensions_to_keep = []

            for m in xrange(last_dimension):
                mytuple = important_features[m]
                dimensions_to_keep += [mytuple[1]]
            all_dims_to_keep[i] = dimensions_to_keep

        print set(all_dims_to_keep[0]).intersection(all_dims_to_keep[1]).intersection(all_dims_to_keep[2]).intersection(all_dims_to_keep[3]).intersection(all_dims_to_keep[4]).intersection(all_dims_to_keep[5])

            # print '\tRecursive feature selector'
            # recursive_feature_selection(all_training_data, all_training_labels, feature_names)
