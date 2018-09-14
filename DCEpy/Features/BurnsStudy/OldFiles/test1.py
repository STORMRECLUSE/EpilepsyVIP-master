# testing with TS041

import sys

#Chris
sys.path.append('/home/chris/Documents/Rice/senior/EpilepsyVIP')

# #Emily
# sys.path.append('C:\Users\User\Documents\GitHub\EpilepsyVIP')

from DCEpy.Features.GardnerStudy import edfread
from DCEpy.Features.Preprocessing import preprocess
from DCEpy.Features.Graphs.build_network import build_network
from DCEpy.ML import gap_stat
from DCEpy.Features.BurnsStudy import rstat_42 as rstat
import numpy as np
import networkx as nx
from scipy import signal, cluster
import math

# TODO:
#   - change coherence to take in the right shape
#   - normalize the eigen vector (see Burns paper)
#       might actually involve processing on the connectivity matrix

def burns(all_files, ictal_interval, inter_interval):
    """
    Input:
        - array of files for the patient
            0 position contains ictal data for band picking
            1 position contains inter ictal data for band picking
        - interval [a, b] for ictal data for band picking
        - interval [c, d] for inter ictal data for band picking
         (ensure that b-a = d-c)
    Output:
        - list of states and the centers of the clusters
        (might change this a bit)
    """

    # testing with two files of TS039
    #test_file = 'CA1353FN_1-1_small.edf'
    #[test_patient1, annotations1, labels1] = edfread.edfread(test_file)

    #inter_file = 'C:\Users\User\Documents\EpilepsySeniorDesign\Burns\CA00100D_1-1+.edf'
    #ictal_file = 'C:\Users\User\Documents\EpilepsySeniorDesign\Burns\DA00101L_1-1+.edf'

    #[inter_data, annotations1, labels1] = edfread.edfread(inter_file)
    #[ictal_data, annotations2, labels2] = edfread.edfread(ictal_file)

    #y_inter = preprocessing(inter_data)
    #y_ictal = preprocessing(ictal_data)

    # create list to hold data and sampling frequency
    all_data = []
    fs = 1000

    # load and preprocess
    for file in all_files:
        [data, annotations, labels] = edfread.edfread(file)
        all_data.append(preprocessing(data))
    print('Data is loaded and preprocessed')
    
    # Find band for r statistic 
    y_ictal = all_data[0][ictal_interval[0]:ictal_interval[1],0:1]
    y_inter = all_data[1][inter_interval[0]:inter_interval[1],0:1]
    bands = np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]]) # possible bands
    band  = rstat.calc_rstat(y_ictal, y_inter, fs, bands);
    print('Band selected is: ' + str(band))
    
    # band pass filter given band
    band_norm = [(1.0*band[0])/(fs/2.0), (1.0*band[1])/(fs/2.0)] # normalize the band
    filt_order = 3
    b, a = signal.butter(filt_order, band_norm, 'bandpass') # design filter   
    num_files = len(all_data)
    for j in range(num_files):
        all_data[j] = signal.filtfilt(b,a,all_data[j],axis=0) # filter the data
    print 'Done filtering'

    # list to hold eigenvectors
    evc = []

    # for each file given
    for file_data in all_data:

        # get data shape
        n, m = file_data.shape
        print('Data has size '+str(file_data.shape))

        # determine edges to be used
        connections = range(m)
        weightType = 'coherence'

        # go through each window and create a coherence graph
        num_windows = int(math.floor((1.0*n)/1000)-3)
        for i in range(0, num_windows):

            # get window
            col1 = i*1000
            col2 = col1 + 3000
            window = file_data[:,col1:col2]

            # build coherence graph
            G = build_network(window, connections, weightType)

            # get eigenvector centrality
            try:
                current_evc = nx.eigenvector_centrality(G, weight=weightType)
                current_evc_ar = np.empty(m) # dictionary to array
                for i in range(m):
                    current_evc_ar[i] = current_evc[i]
                evc.append(current_evc_ar)
            except:
                num_exc += 1
                print ("Eigenvector Centrality not found/did not converge")
    print('Finished computing eigenvector centrality')

    # convert into a numpy array
    evcs = np.array(evc)

    # choose k by gap statistic
    K = np.arange(20)
    n_tests = 10
    k, min_gap = gap_stat.gap_statistic(evcs, K, n_tests)
    print('Gap Statistic chose k=' + str(k))

    # cluster the eigenvectors
    [centroids, labels] = cluster.vq.kmeans2(evcs, k)
    return centroids, labels

# Need to import preprocess file so this will work
def preprocessing(data):
    
    fs = 1000 # sample rate, Hz

    data_filt = preprocess.notch(data, 55.0, 65.0, fs)

    data_filt = preprocess.normalize(data_filt)

    return data_filt


def coherence_network(data, nodes_used):
    """
    Creates and returns a NetworkX graph using coherence as edge weights.

    Input:
        data: an n x s array, n is number of channels
                              s is the number of samples in the recording
        nodes_used: a list of integers 1...n that correspond to the electrodes
                    we want to consider in building our graph
                    if left empty, all channels will be used

    Output:
        G: a NetworkX graph with channels as nodes and coherence between the
            channels as edge weights

    """

    n, m = data.shape

    # if empty, use all of the channels
    if nodes_used == []:
        nodes_used = range(0, n);

    edges = []
    G=nx.Graph()

    for i in nodes_used:
        G.add_node(i)
        for j in nodes_used:
            cxy, f = mplm.cohere(data[i], data[j])
            c = np.mean(cxy) 
            edges.append((i, j, {'weight': c})) 

    G.add_edges_from(edges)

    return G


def find_cluster(X, centroids):
    """
    Using the centroids from k-means
    take in an array of data, X, (where each row is a data point)
    and make an array of point to cluster number
    """

    labels = []

    for pt in range(X.shape[0]):
        cr_ar = np.asarray(centroids);
        dist_2 = np.sum((cr_ar - X[pt,:])**2, axis=1)
        lbl = np.argmin(dist_2)
        labels.append(lbl)

    return labels
