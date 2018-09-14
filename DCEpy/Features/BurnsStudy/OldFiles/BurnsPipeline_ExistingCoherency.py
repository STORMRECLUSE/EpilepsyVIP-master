import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
from DCEpy.Features.AnalyzePatient import analyze_patient_raw
from scipy.signal import csd
from scipy.signal import welch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from DCEpy.Features.feature_functions import burns_features
from DCEpy.Features.feature_functions import energy_features
from DCEpy.Features.feature_functions import spectral_energy_features


def build_offline_feature_dict(patient_id, data_filenames, all_files, win_len, win_overlap):

    feat_matrices = {}

    for i, filename in enumerate(data_filenames):
        X_feat = burns_features(patient_id, filename)
        feat_matrices[filename] = X_feat.copy()

    return feat_matrices

def hand_label_seizuretimes(data_mat_dict, filetypes, seizuretimes, filenames, win_len, win_overlap):

    label_dict = {}

    for i, filename in enumerate(filenames):
        datamat = data_mat_dict[filename]

        n,p = datamat.shape
        labels = np.zeros(n) #label is 0 if not seizure

        file_seizure_times = seizuretimes[i]

        if filetypes[i] is "ictal":

            seizure_start_time = file_seizure_times[0]
            seizure_end_time = file_seizure_times[1]

            seizure_start_window = int(seizure_start_time / (win_len - win_overlap))
            seizure_end_window = int(seizure_end_time / (win_len - win_overlap))

            #TODO: Figure out what labeling method works best for random forests
            labels[0:seizure_start_window] = np.ones(seizure_start_window)

        label_dict[filename] = labels

    return label_dict

def random_forest_feature_selector(feature_matrix, labels, feature_names):

    rf = RandomForestClassifier()
    rf.fit(feature_matrix, labels)
    # print "\t\tFeatures sorted by their score according to random forests & hand labeled data:"
    # print "\t\t", sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)
    return rf.feature_importances_, feature_names

def choose_best_channels(data_dict, label_dict, filenames):

    num_training_files = len(filenames)
    all_dims_to_keep = {}

    for i in xrange(num_training_files):

        data = np.vstack([data_dict[key] for key in filenames[:i]+filenames[i+1:]])
        labels = np.hstack([label_dict[key2] for key2 in filenames[:i]+filenames[i+1:]])

        num_feats = len(labels)
        feature_names = np.arange(0,num_feats,1)
        #TODO: is there a better feature selector than RF?
        print '\tRandom forests feature selection'
        importances, dimensions = random_forest_feature_selector(data, labels, feature_names)

        important_features = sorted(zip(map(lambda x: round(x, 4), importances), dimensions), reverse=True)

        last_dimension = np.where(np.cumsum(np.flipud(np.sort(importances)))>.85)[0][0]
        dimensions_to_keep = []

        for m in xrange(last_dimension):
            mytuple = important_features[m]
            dimensions_to_keep += [mytuple[1]]

        all_dims_to_keep[i] = dimensions_to_keep

    intersectional_set = set(all_dims_to_keep[0]).intersection(all_dims_to_keep[1])

    for j in xrange(num_training_files-2):
        intersectional_set = intersectional_set.intersection(all_dims_to_keep[j+2])

    return intersectional_set

def construct_coherency_matrix(X, f_s, freq_band):

    n,p = X.shape

    # initialize the adjacency matrix
    A = np.zeros((p,p))

    # construct adjacency matrix
    for i in range(p):
        for j in range(i+1,p):
            #TODO: try methods other than coherence to build graph
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

def build_coherency_array(raw_data, chosen_channels, win_len, win_ovlap, f_s, choose_random, num_desired_windows, freq_band):

    data = raw_data[:,np.array(list(chosen_channels))].copy()

    n, p = data.shape

    win_len = win_len * f_s
    win_ovlap = win_ovlap * f_s

    num_windows = int( math.floor( float(n) / float(win_len - win_ovlap)) )

    #TODO: could train on a larger data set
    if choose_random and num_desired_windows<num_windows:
        coherency_array = np.zeros((p,p,num_desired_windows))
        window_indices = np.random.choice(num_windows, num_desired_windows, replace=False)
    else:
        coherency_array = np.zeros((p,p,num_windows))
        window_indices = np.arange(num_windows)

    # go through each window
    for spot, index in enumerate(window_indices):
        print'\t\tCoherency matrix for window', spot
        # isolate time window from seizure data
        start = index*(win_len - win_ovlap)
        end = min(start+win_len, n)
        window_of_data = data[start:end,:] # windowed data
        coherency_array[:,:,spot] = construct_coherency_matrix(window_of_data, f_s, freq_band)

    return coherency_array

def find_normalizing_coherency(matrix):
    mean_mat = np.mean(matrix, axis=2)
    std_mat = np.std(matrix, axis=2)
    return mean_mat, std_mat

def transform_coherency(matrices, mean, std):
    #TODO: try with and without normalizing: does it make a big difference?
    std[std == 0] = 0.001

    if np.ndim(matrices)>2:
        num_matrices = matrices.shape[2]
        for i in xrange(num_matrices):
            matrix = matrices[:,:,i].copy()
            matrix -= mean
            matrix = np.divide(matrix, std)
            matrix = np.divide(np.exp(matrix), 1+ np.exp(matrix))
            matrices[:,:,i] = matrix
    else:
        matrices -= mean
        matrices = np.divide(matrices, std)
        matrices = np.divide(np.exp(matrices), 1+ np.exp(matrices))

    return matrices

def find_evc(matrix):
    #TODO: try other methods than eigenvector centrality for represnting graph
    if np.ndim(matrix)>2:
        centrality = np.zeros((matrix.shape[2],matrix.shape[1]))
        for i in xrange(matrix.shape[2]):
            print'\t\tFinding eigenvector centrality for window', i
            sub_matrix = matrix[:,:,i].copy()
            G = nx.Graph(sub_matrix)
            evc = nx.eigenvector_centrality(G)
            centrality[i,:] = np.asarray(evc.values())
    else:
        G = nx.DiGraph(matrix)
        evc = nx.eigenvector_centrality(G)
        centrality = np.asarray(evc.values())
    return centrality

def train_nu_gamma(trainingictal):
    nu = 0.05
    gamma = 1/float(135)
    return nu, gamma

def train_oneclass_svm(traindat, nu, gamma):
    clf = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    clf.fit(traindat)
    return clf

def train_decision_params(nu, gamma, svm_model, ictal_training_data):
    adapt_rate = 35
    C = 0.8
    return adapt_rate, C

def create_outlier_frac(testdat, model, adaptrate):

    novelty_seq = model.predict(testdat)
    z = np.copy(novelty_seq)
    z[np.where(novelty_seq==-1)] = 1 # anomalous
    z[np.where(novelty_seq==1)] = 0 # not anomalous
    n,p = testdat.shape
    out_frac = np.zeros(novelty_seq.size)
    for i in np.arange(adaptrate,n):
        out_frac[i] = np.mean(z[i-adaptrate:i])
    return out_frac

def outlier_frac_viz(traindat, testdat, seizure_times):

    # mynu = [.05]
    #
    # n,p,k = testdat.shape
    # all_metrics = np.zeros((k,13))
    #
    # for i in xrange(k):
    #     print i
    #     graph = nx.Graph(testdat[:,:,i])
    #     all_metrics[i,:] = graph_metric.large_graph_metric(graph, weight=1, thresh=0.1)
    #
    # numwins, nummets = all_metrics.shape
    #
    # fig, axes = plt.subplots(4,4)
    # row=0
    # col=0
    # print'\tEntering plotting process'
    # for metric in xrange(nummets):
    #
    #     axes[row, col].plot(all_metrics[:,metric])
    #     axes[row, col].set_ylim(ymin=min(all_metrics[:,metric])-1, ymax = max(all_metrics[:,metric])+1)
    #     relevant_times = seizure_times
    #     if relevant_times is not None:
    #         axes[row, col].axvline(x=relevant_times[0], lw=3, color='r')
    #     axes[row, col].set_title('%f'%metric)
    #     row+=1
    #     if row==4:
    #         row=0
    #         col+=1
    #
    # plt.show()

    #
    # adapts = np.arange(10,200,20)
    # fig, axes = plt.subplots(5,2)
    # row=0
    # col=0
    # print'\tEntering plotting process'
    #
    # n,p,k = testdat.shape
    # measure = np.zeros(k)
    #
    # for window in xrange(k):
    #     G = nx.Graph(testdat[:,:,window])
    #     rm_edges = []
    #     for edge in G.edges_iter(data=True):
    #         if edge[2]['weight'] < .1:
    #             rm_edges.append([edge[0],edge[1]])
    #     H = G.copy()
    #     H.remove_edges_from(rm_edges)
    #     measure[window] = float(sum(len(H[u]) for u in H)) / len(H)
    #
    # for adaptrate in adapts:
    #     variance = np.zeros(k)
    #     for iii in np.arange(adaptrate,k):
    #         variance[iii] = np.std(measure[iii-adaptrate:iii])
    #
    #     axes[row, col].plot(variance)
    #     # axes[row, col].set_ylim(ymin=0, ymax=0.02)
    #     relevant_times = seizure_times
    #     if relevant_times is not None:
    #         axes[row, col].axvline(x=relevant_times[0], lw=3, color='r')
    #     axes[row, col].set_title('%f'%adaptrate)
    #
    #     row+=1
    #     if row>4:
    #         row=0
    #         col=1
    # plt.show()

    # mygamma = [.0005, .00075, .001, .0015, .002, .005]
    # mynu = [.03, .04, .05, .06, .07]
    # myadapt = 45
    #
    # row=0
    # col=0
    # fig, axes = plt.subplots(6,5)
    # print'\tEntering plotting process'
    #
    # for mynus in mynu:
    #     for mygammas in mygamma:
    #
    #         clf = svm.OneClassSVM(kernel='rbf', nu=mynus, gamma=mygammas)
    #         clf.fit(traindat)
    #         novelty_seq = clf.predict(testdat)
    #         z = np.copy(novelty_seq)
    #         z[np.where(novelty_seq==-1)] = 1 # anomalous
    #         z[np.where(novelty_seq==1)] = 0 # not anomalous
    #         n,p = testdat.shape
    #
    #         out_frac = np.zeros(novelty_seq.size)
    #
    #         for iii in np.arange(myadapt,n):
    #             out_frac[iii] = np.mean(z[iii-myadapt:iii])
    #
    #         axes[row, col].plot(out_frac)
    #         axes[row, col].set_ylim(ymin=0, ymax=1)
    #         relevant_times = seizure_times
    #         if relevant_times is not None:
    #             axes[row, col].axvline(x=relevant_times[0], lw=3, color='r')
    #         # axes[row, col].set_title('nu = %f, gamma = %f'%(float(mynus),float(mygammas)))
    #
    #         row+=1
    #
    #     row=0
    #     col+=1
    #
    # plt.show()

    # myadapts = np.arange(10,110,10)
    # fig, axes = plt.subplots(5,2)
    # row=0
    # col=0
    # mynu = .07
    # mygamma = .001
    # n,p = testdat.shape
    # for adaptrate in myadapts:
    #
    #     clf = svm.OneClassSVM(kernel='rbf', nu=mynu, gamma=mygamma)
    #     clf.fit(traindat)
    #     novelty_seq = clf.predict(testdat)
    #     z = np.copy(novelty_seq)
    #     z[np.where(novelty_seq==-1)] = 1 # anomalous
    #     z[np.where(novelty_seq==1)] = 0 # not anomalous
    #     n,p = testdat.shape
    #
    #     out_frac = np.zeros(novelty_seq.size)
    #
    #     for iii in np.arange(adaptrate,n):
    #         out_frac[iii] = np.mean(z[iii-adaptrate:iii])
    #
    #     axes[row, col].plot(out_frac)
    #     axes[row, col].set_ylim(ymin=0, ymax=1)
    #     relevant_times = seizure_times
    #     if relevant_times is not None:
    #         axes[row, col].axvline(x=relevant_times[0], lw=3, color='r')
    #
    #     axes[row, col].set_title('%f'%adaptrate)
    #
    #     row+=1
    #     if row>4:
    #         row=0
    #         col=1
    # plt.show()


    adaptrate = 50
    mynu = .005
    mygamma = .000001
    n,p = testdat.shape

    clf = svm.OneClassSVM(kernel='rbf', nu=mynu, gamma=mygamma)
    print traindat.shape
    clf.fit(traindat)
    novelty_seq = clf.predict(testdat)
    z = np.copy(novelty_seq)
    z[np.where(novelty_seq==-1)] = 1 # anomalous
    z[np.where(novelty_seq==1)] = 0 # not anomalous
    n,p = testdat.shape

    out_frac = np.zeros(novelty_seq.size)

    for iii in np.arange(adaptrate,n):
        out_frac[iii] = np.mean(z[iii-adaptrate:iii])

    plt.figure()
    plt.plot(out_frac)
    plt.ylim(ymin=0, ymax=1)
    relevant_times = seizure_times
    if relevant_times is not None:
        plt.axvline(x=relevant_times[0], lw=3, color='r')

    plt.show()

def viz_single_outcome(outlier_fraction, thresh, test_times):
    plt.plot(outlier_fraction)
    if test_times is not None:
        plt.axvline(x=test_times[0]*2, lw=3, color='r')
    plt.axhline(y=thresh, lw=2, color='k')
    plt.show()

def parent_function():
    # parameters -- sampling data
    win_len = 1.0  # in seconds
    win_overlap = 0.5  # in seconds
    f_s = float(1e3)  # sampling frequency

    patients = ['TS039']
    long_interictal = [False]
    include_awake = True
    include_asleep = False
    chosen_channels = [1]
    feature_names = ['Burns']
    feature_func_dict = {'Burns': burns_features, 'Gardner': energy_features, 'Zhang': spectral_energy_features}
    num_data_clusters = 4
    preictal_length = 30  # seconds
    num_training_windows = 1000
    bands = np.asarray([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])
    rstat_window_len=2500
    rstat_window_interval=1500

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

        # relevant_freq_bands = fasai's code


        file_num = len(data_filenames)
        files = [0,1,2,3,4,5]
        # cross-validation
        adapts = np.arange(10,200,20)
        fig, axes = plt.subplots(5,2)

        for i in files:

            # set up test files, seizure times, etc. for this k-fold
            print '\nCross validations, k-fold %d of %d ...' % (i + 1, file_num)

            testing_file_name = data_filenames[i]
            cv_test_files = all_files[:i] + all_files[i+1:]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            testing_info = sio.loadmat('testing39ten_%d.mat'%i)
            training_info = sio.loadmat('training39ten_%d.mat'%i)

            test_evc = testing_info.get('test_evc')
            train_evc = training_info.get('training_interictal_evc')

            # samples, channels = train_evc.shape
            # all_channels_ictal_variance = np.zeros()
            # best_channels = [0,2,3,4,10,11,12,13,14,16,19,21,22,23,29]
            # test_evc = test_evc[:,best_channels]
            # train_evc = train_evc[:,best_channels]
            # mean = training_info.get('mean')
            # std = training_info.get('std')
            #
            # traindata = transform_coherency(train_evc, mean, std)
            # testdata = transform_coherency(test_evc, mean, std)

            # training_coh = training_info.get('training_coherency')
            # testing_coh = testing_info.get('test_coherency')
            #
            # mean_coherency_matrix = training_info.get('mean')
            # sd_coherency_matrix = training_info.get('std')
            #
            # transformed_train_coherency_matrices = transform_coherency(training_coh, mean_coherency_matrix, sd_coherency_matrix)
            # transformed_test_coherency_matrices = transform_coherency(testing_coh, mean_coherency_matrix, sd_coherency_matrix)
            #
            # training_centrality = find_evc(transformed_train_coherency_matrices)
            # testing_evc = find_evc(transformed_test_coherency_matrices)

            outlier_frac_viz(train_evc, test_evc, seizure_times[i])


        #     print'\tEntering plotting process'
        #
        #     n,p,k = testdat.shape
        #     measure = np.zeros(k)
        #
        #     for window in xrange(k):
        #         G = nx.Graph(testdat[:,:,window])
        #         rm_edges = []
        #         for edge in G.edges_iter(data=True):
        #             if edge[2]['weight'] < .1:
        #                 rm_edges.append([edge[0],edge[1]])
        #         H = G.copy()
        #         H.remove_edges_from(rm_edges)
        #         measure[window] = float(sum(len(H[u]) for u in H)) / len(H)
        #     row=0
        #     col=0
        #     for adaptrate in adapts:
        #         variance = np.zeros(k)
        #         for iii in np.arange(adaptrate,k):
        #             var = np.var(measure[iii-adaptrate:iii])
        #             if var == 0 :
        #                 variance[iii] = 100
        #             else:
        #                 variance[iii] = 1/float(var)
        #         if file_type[i] is 'ictal':
        #             axes[row, col].plot(variance,color='b', lw=2)
        #         else:
        #             axes[row, col].plot(variance,color='k',lw=3)
        #         axes[row, col].set_ylim(ymin=0, ymax=1000)
        #         relevant_times = seizure_times[i]
        #         if relevant_times is not None:
        #             axes[row, col].axvline(x=relevant_times[0], lw=3, color='r')
        #         axes[row, col].set_title('%f'%adaptrate)
        #
        #         row+=1
        #         if row>4:
        #             row=0
        #             col=1
        # plt.show()

if __name__ == '__main__':
    parent_function()
