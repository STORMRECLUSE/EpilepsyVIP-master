__author__ = 'Chris'


from scipy.signal import coherence
import multiprocessing
import math, codecs, json, os
import numpy as np
from DCEpy.General.DataInterfacing.edfread import edfread
# from edfread import edfread
import networkx as nx
import sys
import pickle
'''
coherence_graph()
Given a time window of iEEG data, this function returns
a matrix of coherence computations

Inputs
    X   ==> numpy array that is the window of the data
    fs  ==> the sampling frequency

Outputs
    A   ==> coherence matrix
'''
def coherence_graph(X, fs):

    # get the data shape
    n,p = X.shape

    # initialize the adjacency matrix
    A = np.zeros((p,p))

    # construct adjacency matrix
    for i in range(p):
        for j in range(i+1,p):
            f, cxy = coherence(X[:,i], X[:,j], fs=fs)
            c = np.mean(cxy)
            A[i,j] = c # upper triangular part
            A[j,i] = c # lower triangular part
            if np.isnan(c):
                name = multiprocessing.current_process().name
                print '\t\t(' + str(i) + ',' + str(j) + ") is nan in " + name

    # return adjacency matrix
    return A

'''
coherence_worker()
The worker function, involked in a process. Given a small chunk
of iEEG data, this function breaks it into windows and computes
coherence matrices, which are placed into a queue.

Inputs
    X           ==> numpy array that is the chunk of the data
    start_in    ==> index of the first matrix computed (in the larger chunk)
    win_len     ==> the window length in samples
    win_overlap ==> the overlap of windows in samples
    fs          ==> the sampling frequency
    out_q       ==> queue where results are stored
'''
def coherence_worker(X, start_ind, win_len, win_overlap, fs, out_q):
    """ The worker function, invoked in a process. X is the chunk
        of seizure data to be processed with other parameters.
        The resulting adjacency matrices are placed in
        a dictionary that's pushed to a queue.
    """

    name = multiprocessing.current_process().name


    # get data dimensions
    n, p = X.shape
    ind = start_ind
    # print '\tStarting coherence calculations in ' + name

    # determine number of windows and declare output dictionary
    num_windows = int( math.ceil( float(n) / float(win_len - win_overlap)) )
    outdict = {}

    # go through each window
    for i in range(num_windows):

        # isolate time window from seizure data
        start = i*(win_len - win_overlap)
        end = min(start+win_len, n)
        X_w = X[start:end,:] # windowed data

        # compute coherence adjacency matrix

        outdict[ind] = coherence_graph(X_w, fs)
        ind += 1

    out_q.put(outdict)
    return

'''
mp_coherence()
Given a chunk from an edf file, this function
    (1) breaks it into smaller chunks
    (2) in parallel, computes coherence of the smaller chunks
    (3) combines the coherence computations and saves as json

Inputs
    X           ==> numpy array that is the chunk of the data
    win_len     ==> the window length in samples
    win_overlap ==> the overlap of windows in samples
    fs          ==> the sampling frequency
    nprocs      ==> number of processors in parallel computation
    save_path   ==> the file path (w/o .json) that you would like to save
                        json files to (see code for exact use)
                        ex: './dir1/dir2/coherence_graphs'
'''
def mp_coherence(X, win_len, win_overlap, fs, nprocs, save_path):

    # get data dimensions
    n,p = X.shape

    # Each process will get 'chunksize' samples and
    # a queue to put his out dict into
    out_q = multiprocessing.Queue()
    win_starts = np.arange(0, n, win_len-win_overlap) # start times for all the windows
    chunk_size = int(math.ceil(len(win_starts) / float(nprocs)))
    procs = []
    win_ind = 0 # keep track of window indices

    for i in range(nprocs):

        # extract the time chunk
        start = win_starts[i*chunk_size] # the index of the fist sample for this chunk
        try:
            end = win_starts[(i+1)*chunk_size]
        except IndexError:
            end = n

        X_c = X[start:end,:]

        # create thread
        p = multiprocessing.Process(
            target=coherence_worker,
            args=(X_c, win_ind, win_len, win_overlap, fs, out_q)
        )
        procs.append(p)
        p.start()

        # update window indices
        win_ind += chunk_size

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    # collect results in a list
    num_graphs = int(math.ceil( n / float(win_len- win_overlap) ))
    # co_list = [0] * num_graphs # list of coherency adjacency matrices
    evc_list = {}
    for i in range(num_graphs):
        # co_list[i] = resultdict[i].tolist()
        coh_graph = resultdict[i]
        G = nx.DiGraph(coh_graph)
        centrality = nx.eigenvector_centrality(G)
        evc_list[i] = centrality

    # save list of coherency adjacency matrices to json
    print 'Saving chunk as a json file'
    json.dump(evc_list, codecs.open(save_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

    return

'''
parallel_coherence()
Given a edf file, this function
    (1) cuts it into 10 minute chunks
    (2) passes the 10 minutes chunks to another function to
            save coherence computations as json files

Inputs
    file_path   ==> the file path of the .edf file to be read
    win_len     ==> the window length in samples
    win_overlap ==> the overlap of windows in samples
    fs          ==> the sampling frequency
    nprocs      ==> number of processors in parallel computation
    save_path   ==> the file path (w/o .json) that you would like to save
                        json files to (see code for exact use)
                        ex: './dir1/dir2/coherence_graphs'
'''
def parallel_coherence(file_path, win_len, win_overlap, fs, nprocs, save_path):

    # these are hard-coded parameters to
    # chunk up data into 10 min chunks
    min_per_chunk = 10
    sec_per_min = 60

    i = 0
    while True:

        # get chunk start and end times
        start = i * sec_per_min * min_per_chunk
        end = (i+1) * sec_per_min * min_per_chunk + float(win_overlap) / fs

        # get the chunk
        try:

            # extract the chunk
            print 'Extracting the ' + str(i) + ' chunk'
            X_chunk, _, labels = edfread(file_path, rec_times = [start, end])

            print 'Printing the labels:'
            for i in range(len(labels)):
                print '\tChannel ' + str(i) + ' is ' + labels[i]

            # compute coherence for this chunk and save json file
            this_save_path = save_path + ".json"
            mp_coherence(X_chunk, win_len, win_overlap, fs, nprocs, this_save_path)

            # if less than an entire chunk was read, then this is the last one!
            if X_chunk.shape[0] < sec_per_min * min_per_chunk:
                break

        except ValueError:

            break # the start was past the end!

    return

if __name__ == '__main__':

    include_awake = False
    include_asleep = False

    patients = ['TS039']
    long_interictal = [False]

    # get the paths worked out
    to_data = '/scratch/smh10/DICE'
    # to_data = os.path.dirname(os.path.dirname(os.getcwd()))
    data_path = os.path.join(to_data, 'data')

    # specify other parameters
    fs = 1000
    win_len = 1 * fs
    win_overlap = int(0.5 * fs)
    nprocs = 8

    for i, patientID in enumerate(patients):

        # specify data paths
        if not os.path.isdir(data_path):
            sys.exit('Error: Specified data path does not exist')

        p_file = os.path.join(data_path, patientID, 'patient_pickle.txt')

        with open(p_file,'r') as pickle_file:
            print("Open Pickle: {}".format(p_file)+"...\n")
            patient_info = pickle.load(pickle_file)

        # add data file names
        data_filenames = patient_info['seizure_data_filenames']
        seizure_times = patient_info['seizure_times']
        file_type = ['ictal'] * len(data_filenames)
        seizure_print = [True] * len(data_filenames)      # mark whether is seizure

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


        print 'Getting Data...'
        for i, seizure_file in enumerate(data_filenames):

            seizureID = seizure_file[:-4]
            # update paths specific to each patient
            p_file_path = os.path.join(to_data, 'data', patientID, seizureID) + '.edf'
            p_save_path = os.path.join(to_data, 'data', patientID, seizureID) + 'eigenvec_centrality'
            parallel_coherence(p_file_path, win_len, win_overlap, fs, nprocs, p_save_path)



