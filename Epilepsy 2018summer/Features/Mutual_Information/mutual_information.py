from scipy import special, signal
from preprocessing import notch_filter_data, get_freq_bands
from Features.ictal_inhibitors_2018 import analyze_patient_raw
import numpy.random as nr
import scipy.spatial as spatial
import numpy as np
import os, sys
import pickle
import scipy
import math

def mutual_information_ksg2004_XY(x, y, k = 3, seed = 1, threshold = True ):
    """" Mutual information between x and y in nats. Implementation by Rakesh. """

    # tol - small noise to break degeneracy, see doc.
    tol = 1e-10
    assert len(x) == len(y)
    assert k <= len(x) - 1  # to ensure k^th nearest neighbor exists for each point in the x and y

    nr.seed(seed = seed)
    x = [list(p + tol * nr.rand(len(x[0]))) for p in x]
    y = [list(p + tol * nr.rand(len(y[0]))) for p in y]

    data = [list(x[ii] + y[ii]) for ii in range(len(x))]

    # Build a kd-tree data structure for the data in joint space and in marginal subspaces, to
    # ensure faster k-nn queries
    xy_tree = spatial.cKDTree(data)
    x_tree = spatial.cKDTree(x)
    y_tree = spatial.cKDTree(y)
    # Compute the distance between each data point and its kth nearest neighbor
    xy_dist = [xy_tree.query(elem, k=k +1, p=float('inf'))[0][k] for elem in data]
    x_num = [len(x_tree.query_ball_point(x[ii], xy_dist[ii] - 1e-15, p=float('inf'))) for ii in
             range(len(x))]
    y_num = [len(y_tree.query_ball_point(y[ii], xy_dist[ii] - 1e-15, p=float('inf'))) for ii in
             range(len(y))]

    mi_est = special.digamma(k) + special.digamma(len(x)) - (sum(special.digamma( x_num))
                                                             + sum(special.digamma( y_num))) / len(x)

    if threshold:
        if mi_est < 0:
            mi_est = 0

    return mi_est



def mi_in_frequency_channel(dXband, dYband, freqband):
    nf = dXband.shape[1]  # need to check shape
    mi_band = []
    for i in range(nf):
        real_dXi = np.hstack((np.real(dXband[:, [i]]), np.imag(dXband[:, [i]])))
        real_dYi = np.hstack((np.real(dYband[:, [i]]), np.imag(dYband[:, [i]])))
        # mi_band.append(mutual_information((real_dXi, real_dYi), k=3))
        mi_band.append(mutual_information_ksg2004_XY(real_dXi, real_dYi, k = 3))
    return np.array(mi_band)


def cross_channel_mutual_information_in_frequency(X, Y, resolution = 5, freqbands = [[1, 100]],  f_s = 1000):
    """
    Computes mutual information between two signals for the selected frequency bands.
    X: Ns*n matrix, where Ns is the number of windows per chunk and n is the length of a window.
       each signal can be from a single channel or the average of all channels.
    Y:  Ns*n matrix, where Ns is the number of windows per chunk and n is the length of a window.
       each signal can be from a single channel or the average of all channels.
    Nf: DFT length.
    fhigh: max frequency. Should be less than 500Hz.
    """
    Nf = f_s / resolution
    dX = np.fft.fft(X, n = Nf, axis = 1)    # Ns by Nf matrix     We want to extract only the 0~ 300 Hz content with frequency resolution = 10Hz, so 30 points
    dY = np.fft.fft(Y, n = Nf, axis = 1)
    frequencies = np.fft.fftfreq(Nf) * f_s
    list_mi_bands = []

    for freqband in freqbands:
        # select the FFT within a frequency band
        freq_indices = np.where((freqband[0] <= frequencies) & (frequencies <= freqband[1]))[0]

        dXband = dX[:, freq_indices]
        dYband = dY[:, freq_indices]

        # compute cross channel MI in frequency
        mi_band = mi_in_frequency_channel(dXband, dYband, freqband)
        list_mi_bands.append(np.mean(mi_band))

    print "MI of the band: ", list_mi_bands
    return list_mi_bands


def cross_channel_MI(X_chunk, freqbands = [[1, 100]], Nf = 500, f_s = 1000):
    """
    Returns cross channel mutual information for all frequency bands. Should be
    (number of channels, number of channels, number of frequency bands)
    """
    # for every two channels
    N, Ns, p = X_chunk.shape
    cmi_graph = np.zeros(shape=(p, p, len(freqbands)))

    for i in xrange(p):
        for j in xrange(i, p):
            if i == j:
                cmi_graph[i,j,:] = np.zeros(shape = (len(freqbands), ))
            else:
                X_channel_i = X_chunk[:, :, i]
                X_channel_j = X_chunk[:, :, j]
                cmi_graph[i, j, :] = cross_channel_mutual_information_in_frequency(X_channel_i, X_channel_j, freqbands=freqbands)
                cmi_graph[j, i, :] = cmi_graph[i, j, :]

    list_mis_allbands = []
    for i in range(len(freqbands)):
        list_mis_allbands.append(cmi_graph[:,:,i])

    # shape should be number of (frequencies, number of channels, number of channels)
    return np.array(list_mis_allbands)


def window_data(data, win_len, win_ovlap, f_s = 1000):
    n = data.shape[0]
    all_windowed_data = []

    # getting window information in units of samples
    win_len = win_len * f_s
    win_ovlap = win_ovlap * f_s

    # computing the number of windows from the given data in units of samples
    num_windows = int(math.floor(float(n) / float(win_len - win_ovlap)))

    # compute the coherency matrix for each window of data in the file
    for index in np.arange(num_windows):
        # find start/end indices for window within file
        start = index * (win_len - win_ovlap)
        end = min(start + win_len, n)

        # get window of data
        if end <= n:
            window_of_data = data[int(start):int(end), :]     # multidimensional
            # print "windowed data shape : ", window_of_data.shape
            if window_of_data.shape[0] < win_len:                      # get rid of short windows in the end
                break
            all_windowed_data.append(window_of_data)
    return np.array(all_windowed_data)


# new
def extract_CMI_whole_file(X, filename, patient_id, i, freqbands = [[1, 100]],  save_file = False, chunk_len = 300, chunk_ovlp = 270, mi_win_len = 0.25, mi_win_overlap = 0):
    """
    If you want to extract MI features for a file, use this one.

    """
    # calculate features for 5 minutes (300s)
    mi_list = []

    # window data
    chunked_data = window_data(X, win_len= chunk_len, win_ovlap= chunk_ovlp)
    print "original data shape: ", X.shape
    print "chunked data shape: ", chunked_data.shape

    # compute MI information for each chunk
    for chunk_idx in range(0, chunked_data.shape[0]):
        chunk = chunked_data[chunk_idx, :, :]
        windowed_chunk =  window_data(chunk, win_len = mi_win_len, win_ovlap = mi_win_overlap)
        print "windowed chunk shape: ", windowed_chunk.shape
        # compute cross channel
        mi_all_freqs = cross_channel_MI(windowed_chunk, freqbands=freqbands , Nf = 500, f_s = 1000)
        mi_list.append(mi_all_freqs)

    print "\t MI matrix shape (should be 6*6*6): ", mi_all_freqs.shape

    if save_file == True:
        index = filename.rfind('/')
        truncated_filename = filename[index + 1:]
        if i == 0:
            # scipy.io.savemat(patient_id + "corrected_CMI.mat", {str(i) + "_" + truncated_filename: mi_list})
            scipy.io.savemat(patient_id + "CMI_5m.mat", {str(i) + "_" + truncated_filename: mi_list})

        else:
            # mat_dict = scipy.io.loadmat(patient_id + "corrected_CMI.mat")
            mat_dict = scipy.io.loadmat(patient_id + "CMI_5m.mat")

            mat_dict[str(i) + "_" + truncated_filename] = mi_list
            # scipy.io.savemat(patient_id + "corrected_CMI.mat", mat_dict)
            scipy.io.savemat(patient_id + "CMI_5m", mat_dict)
            print "\tfeatures for this file is saved!"
        print "features for this patient is saved!"
    return np.array(mi_list)


def get_MIIF_features():
    """
    Generates features for all patients locally.
    """
    home_path = "/Users/TianyiZhang/Desktop/PatientData"
    # home_path = "/Volumes/Brain_cleaner/Seizure Data/data"
    win_len = 300  # seconds
    win_overlap = 270  # seconds
    f_s = float(1e3)  # Hz
    freqband_names, freq_bands = get_freq_bands()               # get all frequency bands
    # these patients has well-chosen focus channels (see Patient Data Summary)
    patients = ["TS057"]

    for patient_id in patients:

        data_path = os.path.join(home_path, patient_id)
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(data_path=data_path, f_s=f_s,
                                                                                  include_awake=True, include_asleep=True,
                                                                                  patient_id=patient_id, win_len=win_len,
                                                                                  win_overlap=win_overlap,
                                                                                  calc_train_local=True)
        print "data filenames: ", data_filenames


        print data_filenames

        print "================= extracting features for patient: " + patient_id + " ================="
        for i in range(len(all_files)):
            X = all_files[i]
            filename = data_filenames[i]
            X = notch_filter_data(X, 500)
            print "processing file: ", i
            mi_all_mat = extract_CMI_whole_file(X, filename, patient_id, i, freqbands = freq_bands, save_file=True, chunk_len=win_len, chunk_ovlp=win_overlap)
            print mi_all_mat.shape


def get_seizure_time(patient_id, filename):
    home_path = "/Users/TianyiZhang/Desktop/PatientData"
    # home_path = "/Volumes/Brain_cleaner/Seizure Data/data"
    data_path = os.path.join(home_path, patient_id)

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

    return seizure_times[data_filenames.index(filename)]


def get_seizure_windows(seizure_time, win_len, win_overlap):
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

    return seizure_start_window, seizure_end_window



