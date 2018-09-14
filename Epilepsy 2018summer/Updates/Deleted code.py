
# Ictal inhibitor
"""
precompute_fft_and_scale()

Note: this is a helper function that is a result of breaking down the Python code for C implementation; the details
of how this function works are not important, just that it is used to help compute coherence.

Purpose: the helper function that creates the fft window output and scale needed
for online_testing()

Inputs:
    window_of_data: a single window of raw iEEG data, rows represent samples and cols represent channels
    nperseg: number of fft samples per window

Outputs:
    channel_fft: the fft for each channel of input data
    scale: parameter used for fft
"""


def precompute_fft_and_scale(window_of_data, f_s, nperseg, noverlap):
    nfft = 1024
    detrend_func = 'constant'
    hann_window = create_hann_window(nperseg)
    scale = 0

    for entry_in_win in hann_window:
        entry_in_win_multi = entry_in_win * entry_in_win
        scale = scale + entry_in_win_multi

    channel_fft = np.zeros(shape=[6, 5, 1024], dtype=np.complex_)
    n, p = window_of_data.shape
    for i in range(p):
        fft_output = _fft_helper_new(window_of_data[:, i], hann_window, detrend_func, nperseg, noverlap, nfft)
        channel_fft[i, :, :] = fft_output

    return channel_fft, scale

"""
viz_single_outcome

Purpose: to visualize iEEG data, the outlier fraction, and the seizure prediction decision on one plot

Inputs:
    decision: one dimensional ndarray in units of samples indicating whether or not a seizure is predicted
    out_frac: one dimensional ndarray in units of samples containing the outlier fraction
    raw_ieeg: one dimensional ndarray in units of samples containing the iEEG data from the .edf file
    test_times: tuple containing the start and end time of the seizure, or None if interictal file
    test_index: int indicating which patient file we are visualizing
    patient_id: list indicating patient name
    f_s: int indicating sampling frequency
    show_fig: if show_fig is 1, the plot will show; if show_fig is 0, the plot will save to the current working directory

Outpus:
    No output. The graph will either be visualized or be saved; graph contains three subplots: the raw iEEG data,
    the outlier fraction, and the decision rule. All share an x axis
"""


def viz_single_outcome(decision, out_frac, raw_ieeg, test_times, thresh, test_index, patient_id, f_s, show_fig=0):
    # initialize the subplots
    fig, axes = plt.subplots(3, sharex=True, figsize=(9, 5))

    number_samples = out_frac.shape[0]
    number_seconds = float(number_samples) / float(f_s)

    # plot the raw iEEG signal
    axes[0].plot(raw_ieeg)
    axes[0].set_title('Raw iEEG signal', size=14)
    axes[0].set_ylabel('Voltage', size=10)
    axes[0].set_yticklabels([])
    axes[0].set_yticks([])
    if test_times is not None:
        axes[0].axvline(x=test_times[0] * f_s, lw=2, c='r')
        axes[0].axvline(x=test_times[1] * f_s, lw=2, c='r')
        axes[0].set_xticks([test_times[0] * f_s, test_times[1] * f_s, number_samples - 1])
        axes[0].set_xticklabels(['Seizure \n onset', 'Seizure \n end', number_seconds], size=10)

    # plot the outlier fraction and mark the threshold and seizure times
    axes[1].plot(out_frac)
    axes[1].set_title('Likelihood of upcoming seizure', size=14)
    axes[1].set_ylabel('Likelihood', size=10)
    axes[1].set_ylim(ymin=0, ymax=1)
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    if test_times is not None:
        axes[1].axvline(x=test_times[0] * f_s, lw=2, c='r')
        axes[1].axvline(x=test_times[1] * f_s, lw=2, c='r')
        axes[1].set_xticks([test_times[0] * f_s, test_times[1] * f_s, number_samples - 1])
        axes[1].set_xticklabels(['Seizure \n onset', 'Seizure \n end', str(number_seconds) + ' sec'], size=10)
    axes[1].axhline(y=thresh, lw=2, c='k')

    # plot the seizure prediction decision
    axes[2].plot(decision)
    axes[2].set_title('Prediction Decision', size=14)
    axes[2].set_xlabel('Samples', size=10)
    axes[2].set_yticks([-1, 1])
    axes[2].set_yticklabels(['No seizure', 'Seizure'], size=10)
    axes[2].set_ylim(ymin=-1.5, ymax=1.5)

    fig.suptitle('Patient {}, file {}'.format(patient_id, test_index), size=16)
    fig.subplots_adjust(hspace=.8)

    if show_fig:
        plt.show()
    else:
        plt.savefig('Patient_{}_single_file_{}'.format(patient_id, test_index))
        plt.clf()
        plt.close()

    return


"""
viz_many_outcomes()

Purpose: to visualize all of the outlier fractions of the training data

Inputs:
    all_outlier_fractions: list of one dimensional ndarrays containing many files' outlier fractions
    seizure_times: list of tuples indicating start and end time of seizure or None for interictal files
    patient_id: string containing the patient name
    threshold: list of ints indicating the threshold above which to flag a seizure for each file
    test_index: int indicating which file currently being tested
    f_s: sampling frequency
    show_fig: if show_fig is 1, the plot will show; if show_fig is 0, the plot will save to the current working directory

Outputs:
    No output. The plot will either be visualized or saved. The single plot will contain subplots of all given
    outlier fractions.
"""


def viz_many_outcomes(all_outlier_fractions, seizure_times, patient_id, threshold, f_s, show_fig=1):
    # initialize subplots
    fig, axes = plt.subplots(len(all_outlier_fractions))

    # visualize each outlier fraction
    for i in xrange(len(all_outlier_fractions)):
        this_out_frac = np.asarray(all_outlier_fractions[i])
        axes[i].plot(this_out_frac)
        axes[i].set_ylim(ymin=0, ymax=1)

        # mark the seizure times and determined threshold
        if seizure_times[i] is not None:
            axes[i].axvline(x=seizure_times[i][0] * f_s, lw=2, c='r')
            axes[i].axvline(x=seizure_times[i][1] * f_s, lw=2, c='r')
        axes[i].axhline(y=threshold[i], lw=2, c='k')

    if show_fig:
        plt.show()
    else:
        plt.savefig('Patient_{}_all_files'.format(patient_id))
        plt.clf()
        plt.close()

    return

"""
update_log_params()

Purpose: to record the parameters of this run

Inputs:
    log_file: path to the log file
    win_len: float indicating window length
    win_overlap: float indicating window overlap
    include_awake: boolean specifying whether or not to include awake data in the training
    include_asleep: boolean specifying whether or not to include asleep data in the training
    f_s: int indicating sampling frequnecy
    patients: list holding strings of patient names

Outputs:
    None. The log file will be saved as a .txt in the save_path specificed in parent_function()
"""


def update_log_params(log_file, win_len, win_overlap, include_awake, include_asleep, f_s, patients, freq_band,
                      preictal_time, postictal_time, persistence_time):
    f = open(log_file, 'w')

    # write the first few lines
    f.write("Results file for Burns Pipeline\n")

    # write the parameters
    f.write('Parameters used for this test\n================================\n')
    f.write('Feature used is Burns Features\n')
    f.write('Window Length \t%.3f\nWindow Overlap \t%.3f\n' % (win_len, win_overlap))
    f.write('Sampling Frequency \t%.3f\n' % f_s)
    f.write('Awake Times are ' + (not include_awake) * 'NOT ' + ' included in training\n')
    f.write('Asleep Times are ' + (not include_asleep) * 'NOT ' + ' included in training\n\n')
    f.write('Frequency band is {} to {}\n'.format(freq_band[0], freq_band[1]))
    f.write('Preictal time is {}\n'.format(preictal_time / f_s))
    f.write('Postictal time is {}\n'.format(postictal_time / f_s))
    f.write('Prediction/persistance time is {}\n\n'.format(persistence_time / f_s))

    # write the patients
    f.write('Patients are ' + " ".join(patients) + "\n\n")
    f.close()
    return


# Features.Mutual_Information.centralities

def betweenness_centrality(coherency_matrix):
    # and compute the eigenvector centrality for each window's graph
    sub_matrix = coherency_matrix.copy()
    # cast to a graph type
    G = nx.Graph(sub_matrix)
    try:
        # compute the eigenvector centrality
        evc = nx.betweenness_centrality(G)
        centrality = np.asarray(evc.values())
    except:
        # check to see if convergence criteria failed and if so, return an EVC of all ones
        centrality = np.ones(coherency_matrix.shape[1]) / float(coherency_matrix.shape[1])
        print"Convergence failure in EVC; bad vector returned"
    return centrality

def compute_hubs(coherency_matrix):
    # and compute the eigenvector centrality for each window's graph
    sub_matrix = coherency_matrix.copy()
    # cast to a graph type
    G = nx.Graph(sub_matrix)
    try:
        # compute the eigenvector centrality
        evc, a = nx.hits(G, max_iter=500)
        centrality = np.asarray(evc.values())
    except:
        # check to see if convergence criteria failed and if so, return an EVC of all ones
        centrality = np.ones(coherency_matrix.shape[1]) / float(coherency_matrix.shape[1])
        print"Convergence failure in EVC; bad vector returned"
    return centrality

def compute_authorities(coherency_matrix):
    # and compute the eigenvector centrality for each window's graph
    sub_matrix = coherency_matrix.copy()
    # cast to a graph type
    G = nx.Graph(sub_matrix)
    try:
        # compute the eigenvector centrality
        evc, a = nx.hits(G, max_iter=500)
        centrality = np.asarray(a.values())
    except:
        # check to see if convergence criteria failed and if so, return an EVC of all ones
        centrality = np.ones(coherency_matrix.shape[1]) / float(coherency_matrix.shape[1])
        print"Convergence failure in EVC; bad vector returned"
    return centrality

#mi_sn_ictal_inhibitors

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

def extract_features(X, filename, patient_id, i, chunk_len = 60, chunk_overlap = 50, win_len = 3.0, win_overlap = 2.0, f_s = 1000, plot_features = False):
    # X should be 3d: number of chunks, number of samples within the chunk, number of channels
    
    # Extract MI features
    windowed_X = window_data(X, win_len = win_len, win_ovlap= win_overlap)
    mi_mat = cross_channel_MI(windowed_X, freqbands=[[0,100]])[0]   # shape should be number of chunks * Nf * Nf
    nf, _ = mi_mat.shape
    if plot_features:
        # color plot MI
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        # fig, ax = plt.subplots(2)
        p = ax0.pcolormesh(mi_mat)       # 336*10/60 = 55 min
        fig.colorbar(p)
    
    mi_mat = mi_mat.reshape((1, nf*nf))
    return mi_mat


def test_transform_MI(coherency_matrix, mean, std, num_channels):
    # step through each value in the coherence matrix
    band_num = coherency_matrix.shape[0]
    for band in range(band_num):
        for row in np.arange(num_channels):
            for col in np.arange(num_channels):
                # normalize
                coherency_matrix[band, row, col] -= mean[band, row, col]
                coherency_matrix[band, row, col] = float(coherency_matrix[band, row, col]) / float(std[band, row, col])
                # transform
                coherency_matrix[band, row, col] = (2.7182818284590452353602874713527 ** coherency_matrix[band, row, col])
                denominator = coherency_matrix[band, row, col] + 1
                coherency_matrix[band, row, col] = coherency_matrix[band, row, col] / float(denominator)
    return coherency_matrix

#Features.Mutual_Information.mutual_information

def mutual_information_in_frequency(X, resolution = 10, fhigh = 300, fs = 1000):
    """
        X: Ns*n matrix, where Ns is the number of windows per chunk and n is the length of a window.
        each signal can be from a single channel or the average of all channels.
        Nf: DFT length.
        fhigh: max frequency. Should be less than 500Hz.
        """
    Nf = fs/resolution
    dX = np.fft.fft(X, n = Nf, axis = 1)    # Ns by Nf matrix     We want to extract only the 0~ 300 Hz content with frequency resolution = 10Hz, so 30 points
    nf = int(fhigh / resolution)
    dX = dX[:, :nf]   # we only need up to 300Hz
    
    # test
    freqs =   np.fft.fftfreq(n = Nf)
    print "all frequencies: ", freqs
    mi_in_frequency = np.zeros(shape=(nf, nf)) # nf by nf matrix
    
    for i in xrange(nf):
        for j in xrange(i, nf):
            real_dXi = np.hstack((np.real(dX[:, [i]]), np.imag(dX[:, [i]])))
            real_dXj = np.hstack((np.real(dX[:, [j]]), np.imag(dX[:, [j]])))
            # mi_in_frequency[i, j] = mutual_information((real_dXi, real_dXj), k=3)
            
            mi_est = mutual_information_ksg2004_XY(real_dXi, real_dXj)
            if mi_est < 0 or mi_est > 1:
                print "frequencies: ", freqs[i]*fs, freqs[j]*fs
                print "mi estimation: ", mi_est
        
            mi_in_frequency[i, j] = mutual_information_ksg2004_XY(real_dXi, real_dXj)
            
            
            if j > i:
                mi_in_frequency[j,i] = mi_in_frequency[i,j]
    if i == j:
        mi_in_frequency[i,j] = 0
    return mi_in_frequency     # should be a 30 by 30 matrix


# get_MIIF_features()


# A list of patients and files of interest
# patient       filename                        filetype      seizure time(seconds)
# TA023         'DA001020_1-1+.edf'             long seizure   (56*60 + 52,57*60 + 52)
# TA510         'CA1353FG_1-1.edf'              long seizure   (26*60 + 28,26*60 + 52)
#               'CA1353FL_1-1.edf'              long seizure   (26*60 + 6,26*60 + 36)
#               'CA1353FN_1-1.edf'              long seizure   (55*60 ,  55*60 + 32)
#               'CA1353FQ_1-1.edf'              long seizure   (27*60 + 8,27*60 + 50)
# TA511         'CA129255_1-1.edf'              long seizure   (20 * 60 + 43, 21 * 60 + 19)
#               'CA12925J_1-1.edf'              long seizure   (53 * 60 + 48, 54 * 60 + 24)
#               'CA12925N_1-1.edf'              long seizure    (10*60 + 28,11*60 + 13)






# print dict.keys()

# seizure = 1

#
# win_len = 300  # seconds
# win_overlap = 270  # seconds
# f_s = float(1e3)  # Hz
# # h_path = "/Volumes/Brain_cleaner/Seizure Data/data"
# h_path = "/Desktop/PatientData"
# d_path = os.path.join(h_path, patient_id)
#
#
# dict = scipy.io.loadmat(os.path.join("/Users/TianyiZhang/Documents/EpilepsyVIP/DCEpy/Features/BurnsStudy/MI_features", patient_id + "CMI_5m_30shift.mat"))
# print "keys: ", dict.keys()
# data_matrix = dict['0_DA00100A_1-1+.edf']
# if seizure:
#     seizure_time = get_seizure_time(patient_id = patient_id, filename =filename)
# else :
#     seizure_time = None
#
#
#
# # visualize MI for a specific frequency. Refer to get_frequency_bands
# # # "theta", "alpha", "beta", "gamma", "high", "very high"
# band_name = "beta"
# all_band_names, bands = get_freq_bands()
# band_index = all_band_names.index(band_name)
#
#
# title = "Katz Centrality for Interictal File (" + band_name + ")"
# if seizure_time!= None:
#     seizure_start_window, seizure_end_window = get_seizure_windows(seizure_time, win_len, win_overlap)
#     print "seizure time: ", seizure_time
#     print "seizure windows: ", seizure_start_window, seizure_end_window
#     print data_matrix.shape
#
# # ==================== dimensionality reduction and plot time series ==================================
# # extract MI for each band
# # "theta", "alpha", "beta", "gamma", "high"
# band_data = data_matrix[:, band_index, :, :]
# #
# # plot reduced features for all bands
# # eigen values, eigen value centrality, katz centrality, pagerank centrality
# reduced = []
# for i in range(0, band_data.shape[0]):
#     print band_data.shape
#     sample = band_data[i, :, :]
#     print "shape of sample:", sample.shape
#     reduced.append(compute_katz(sample))
#     # reduced.append(pagerank_centrality(sample))
#     # reduced.append(eigen(sample))
#     # reduced.append(compute_eigen_centrality(sample))
# reduced = np.array(reduced)   # shape should be samples * 6
# dim1 = reduced
# dim1 = reduced[:, 0]
# dim2 = reduced[:, 1]
# dim3 = reduced[:, 2]
# dim4 = reduced[:, 3]
# dim5 = reduced[:, 4]
# dim6 = reduced[:, 5]
# xs = range(1, 1 + band_data.shape[0])    # num window
# plt.plot(xs, dim1)
# plt.plot(xs, dim2)
# plt.plot(xs, dim3)
# plt.plot(xs, dim4)
# plt.plot(xs, dim5)
# plt.plot(xs, dim6)
# plt.xlabel('Number of Windows')
# plt.title(title)
#
# # plot seizure time
# if seizure_time!= None:
#     plt.axvline(x = seizure_start_window, label = 'seizure start', ls = "dashed")
#     plt.axvline(x = seizure_end_window, label = 'seizure end', ls = "dashed")
# # plt.legend(['1st Dimension', '2nd Dimension', '3rd Dimension', '4th Dimension', '5th Dimension','6th Dimension'], loc='upper left')
# plt.show()
#
#
#
#
#
# # ==================== plot graph features for all frequency bands ====================================
#
# # "theta", "alpha", "beta", "gamma", "high"
# # p = ax.pcolormesh(data_matrix[23, 0, :, :])       # 340*10/60 = 56 min
#
#
# #
# import matplotlib.ticker as ticker
# # fig, axes = plt.subplots(nrows=2, ncols=3)      # 36
# # image_idx =30
# # #
# #
# # print data_matrix[image_idx, 0, :, :]
# # vmin = 0
# # vmax = 1
#
# # "theta"
# # p0 = axes[0, 0].pcolormesh(data_matrix[image_idx, 0, :, :], vmin= vmin, vmax=vmax)
# # axes[0, 0].set_xticklabels('')
# # axes[0, 0].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# # axes[0, 0].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # # axes[0, 0].set_xlabel('channels')
# # # axes[0, 0].set_ylabel('channels')
# # axes[0, 0].set_title('Theta')
# # print "eigen:  ", eigen(data_matrix[image_idx, 0, :, :])
# # print "katz: ",compute_katz(data_matrix[image_idx, 0, :, :])
# # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 0, :, :])
# # # print "subgraph: ", betweenness_centrality(data_matrix[image_idx, 0, :, :])
# # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 0, :, :])
# # print "\n"
# # fig.colorbar(p0)
# #
# #
# # p1 = axes[0, 1].pcolormesh(data_matrix[image_idx, 1, :, :], vmin=vmin, vmax=vmax)
# # axes[0, 1].set_title('Alpha')
# # axes[0, 1].set_xticklabels('')
# # axes[0, 1].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# # axes[0, 1].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # # axes[0, 1].set_xlabel('channels')
# # # axes[0, 1].set_ylabel('channels')
# # print "eigen:  ", eigen(data_matrix[image_idx, 1, :, :])
# # print "katz: ",compute_katz(data_matrix[image_idx, 1, :, :])
# # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 1, :, :])
# # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 1, :, :])
# # print "\n"
# # # fig.colorbar(p1)
# #
# #
# # p2 = axes[0, 2].pcolormesh(data_matrix[image_idx, 2, :, :], vmin=vmin, vmax=vmax)
# # axes[0, 2].set_title('Beta')
# # axes[0, 2].set_xticklabels('')
# # # axes[0, 2].set_xlabel('channels')
# # # axes[0, 2].set_ylabel('channels')
# # axes[0, 2].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# # axes[0, 2].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # print "eigen:  ", eigen(data_matrix[image_idx, 2, :, :])
# # print "katz: ",compute_katz(data_matrix[image_idx, 2, :, :])
# # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 2, :, :])
# # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 2, :, :])
# # print "\n"
# # # fig.colorbar(p2)
# #
# #
# # p3 = axes[1, 0].pcolormesh(data_matrix[image_idx, 3, :, :], vmin=vmin, vmax=vmax)
# # axes[1, 0].set_title('Gamma')
# # axes[1, 0].set_xticklabels('')
# # # axes[1, 0].set_xlabel('channels')
# # # axes[1, 0].set_ylabel('channels')
# # axes[1, 0].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# # axes[1, 0].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # print "eigen:  ", eigen(data_matrix[image_idx, 3, :, :])
# # print "katz: ",compute_katz(data_matrix[image_idx, 3, :, :])
# # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 3, :, :])
# # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 3, :, :])
# # print "\n"
# # # fig.colorbar(p3)
# #
# # p4 = axes[1, 1].pcolormesh(data_matrix[image_idx, 4, :, :], vmin=vmin, vmax=vmax)
# # axes[1, 1].set_title('High')
# # axes[1, 1].set_xticklabels('')
# # # axes[1, 1].set_xlabel('channels')
# # # axes[1, 1].set_ylabel('channels')
# # axes[1, 1].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# # axes[1, 1].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # print "eigen:  ", eigen(data_matrix[image_idx, 4, :, :])
# # print "katz: ",compute_katz(data_matrix[image_idx, 4, :, :])
# # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 4, :, :])
# # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 4, :, :])
# # print "\n"
# # # fig.colorbar(p4)
# # # #
# # # # # plt.subplot(2, 3, 6)
# # # # p5 = axes[1, 2].pcolormesh(data_matrix[image_idx, 5, :, :],vmin=vmin, vmax=vmax )
# # # # axes[1, 2].set_title('Very High')
# # # # print "eigen:  ", eigen(data_matrix[image_idx, 5, :, :])
# # # # print "katz: ",compute_katz(data_matrix[image_idx, 5, :, :])
# # # # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 5, :, :])
# # # # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 5, :, :])
# # # # print "\n"
# # # # # fig.colorbar(p5)
# # plt.show()
# # # #
# # #
# # #


# Preprocessing/preprocess

def normalize(data):
    """
        Normalizes the data to mean and scales the data by standard deviation
        Inputs:
        -------
        data = data to be normalized: Nxn numpy array,
        N = number of channels, n = number of samples
        Outputs:
        --------
        y_normal = normalized data Nxn numpy array,
        N = number of channels, n = number of samples
        """
    
    # subtract mean and divide by standard deviation to normalize and scale
    mean = np.mean(data,axis=0)
    std =  np.std(data, axis=0)
    y_normal = (data - mean)/std
    return y_normal


def downsample(data,fs,dsrate):
    """
        Downsamples data
        Inputs:
        -------
        data = the data to be downsampled: Nxn numpy array,
        N = number of channels
        n = number of samples
        fs = the sampling rate of data in Hz: int
        dsrate = the desired downsampled sampling rate in Hz: int
        Outputs:
        --------
        y_ds = the downsampled data
        """
    data = data.T
    
    interval = fs/dsrate
    N = data.shape[0]
    n = data.shape[1]
    y_ds = np.zeros((N,n/interval))
    
    # downsample the data by taking sample every interval
    for ch in range(0,N):
        j = 0
        for i in range(0,n):
            if i % interval == 0:
                y_ds[ch,j] = data[ch,i]
                j += 1

y_ds = y_ds.T
    return y_ds


def lpf(data,fs,fc,order,mode=False):
    
    """
        Applies a low pass filter to the input data
        
        Inputs:
        -------
        data = the data to be filtered: nxN numpy matrix,
        N = number of channels, n = number of samples
        order = order of the low pass filter: int
        fs =  sample rate of the data in Hz: int
        fc = cutoff frequency of the filter in Hz: int
        mode = whether zero-phase filtering is used: False = no, True = yes
        
        Outputs:
        --------
        y_lpf = filtered data: nxN numpy matrix,
        N = number of channels, n = number of samples
        """
    
    # Filter the data
    y_lpf = butter_lowpass_filter(data, fc, fs, order, mode)
    return y_lpf

def butter_bandpass(band,fs,order=5):
    nyq = fs/2.
    normalized_band = np.asarray(band)/nyq
    b,a = butter(order,normalized_band,btype='bandpass')
    return b,a

def butter_bandpass_filter(data,band,fs,order=5,mode=False):
    b, a = butter_bandpass(band, fs, order=order)
    if mode:
        y_lpf = filtfilt(b, a, data, axis=0)
    else:
        y_lpf= lfilter(b, a, data, axis=0)
    return y_lpf

def bandpass(data,fs,band,order=5,mode=False):
    
    return butter_bandpass_filter(data,band,fs,order=order,mode=mode)


def box_filter(data,filter_len,axis=0):
    '''
        Performs a box filter on the input data
        :param filter_len: length of the filter (adapt_rate)
        :param axis: along which axis?
        :return:
        '''
    return lfilter(np.ones(filter_len)/filter_len,[1],data,axis=axis)

def easy_seizure_preprocessing(x,fs, axis = 0,order = 5):
    
    """
        The preprocessing function:
        (1) applies 49-51Hz band-reject filter to remove the power line noise
        (2) applies .5Hz cutoff high pass filter to remove the dc component
        
        Your one-stop-shop for preprocessing!
        
        :param x: single channel time series with shape (number of samples,)
        :param fs: sampling frequency
        :return: preprocessed signal
        
        """
    # empirical maximum
    reject=(59,61)          # band-reject frequency range (59Hz,61Hz) to remove power line noise
    x = bandreject(x, reject, fs,axis = axis, order=order)
    cf = .5                 # 0.5Hz cutoff filter to remove dc
    x= high_pass(x,cf,fs,axis = axis,order=order)
    
    return x


# band reject filter
def bandreject(data,band,fs,axis = 0,order=5,mode=False):
    
    """
        Band reject filtering on given data.
        
        :param data: single channel time series
        :param fs: sampling frequency
        :param band: reject range, tuple/sequence with 2 elements
        :param order:
        :param mode:
        :return:
        """
    
    def butter_bandreject_filter(data, band, fs, order=5, mode=False):
        
        def butter_bandreject(band, fs, order=5):
            nyq = fs / 2.
            normalized_band = np.asarray(band) / nyq  # normalized band for digital filters
            b, a = butter(order, normalized_band, btype='bandstop')
            return b, a
        
        b, a = butter_bandreject(band, fs, order=order)
        
        if mode:
            y_lpf = filtfilt(b, a, data, axis=axis)
        else:
            y_lpf = lfilter(b, a, data, axis=axis)
        return y_lpf
    
    return butter_bandreject_filter(data,band,fs,order=order,mode=mode)
# from DCEpy.Features.BurnsStudy.rstat_42 import calc_rstat
# class RstatPreprocessor(object):
#     '''
#     A class that computes the r-statistic on a file
#     '''
#     def __init__(self,inter_file,ictal_file, seizure_times,fs=1000):
#         self.inter_file,self.ictal_file = inter_file,ictal_file
#         self.seizure_times = seizure_times
#         self.fs = fs
#
#
#     def _get_windows(self,good_channels = None,bad_channels=None,
#                         window_len = 20000):
#         all_ictal_data,_,_ = edfread(self.ictal_file,bad_channels=bad_channels,
#                                      good_channels=good_channels)
#
#         ictal_window_end = np.random.randint(self.fs*self.seizure_times[0] + window_len,
#                                               self.fs*self.seizure_times[1])
#
#         self.ictal_window = all_ictal_data[ictal_window_end-window_len:ictal_window_end,:]
#
#         all_inter_data,_,_ = edfread(self.ictal_file,bad_channels=bad_channels,
#                                      good_channels=good_channels)
#
#         inter_window_end = np.random.randint(window_len,np.size(all_inter_data,axis=0))
#
#         self.inter_window = all_inter_data[inter_window_end-window_len:inter_window_end,:]
#
#     def _compute_rstat(self,bands,window_len=2500,window_interval=1500,mode=0):
#         self.good_band = calc_rstat(self.ictal_window,self.inter_window,self.fs,bands,
#                    window_len=window_len,
#                    window_interval=window_interval,mode=mode)
#
#     def prepare_rstat(self,bands,good_channels=None,bad_channels=None,window_len=20000,
#                       rstat_win_len = 2500,rstat_win_interval=1500,mode=0):
#         self._get_windows(good_channels=good_channels,bad_channels=bad_channels,window_len=window_len)
#         self._compute_rstat(bands,window_len=rstat_win_len,window_interval=rstat_win_interval,mode=mode)
#
#     def optimal_bandpass(self,data,order=5,mode=0):
#         return bandpass(data,self.fs,self.good_band,order,mode)

