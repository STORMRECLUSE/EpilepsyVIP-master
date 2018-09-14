import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import signal
import scipy.io as sio
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans

def update_list(original, update):
    count = 0
    for a in update:
        place = a[0] + count
        original = original[:place] + a[1] + original[place + 1:]
        count += len(a[1]) - 1
    return original

def analyze_patient_raw(data_path, f_s=1e3, include_awake=True, include_asleep=False, long_interictal=False):

    # minutes per chunk (only for long interictal files)
    min_per_chunk = 15
    sec_per_min = 60

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    p_file = os.path.join(data_path, 'patient_pickle.txt')

    with open(p_file, 'r') as pickle_file:
        print("Open Pickle: {}".format(p_file) + "...\n")
        patient_info = pickle.load(pickle_file)

    # # add data file names
    data_filenames = patient_info['seizure_data_filenames']
    seizure_times = patient_info['seizure_times']
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
    good_channels = patient_info['good_channels']

    # band pass filter parameters
    # band = np.array([0.1, 100.])
    # band_norm = band / (f_s / 2.)  # normalize the band
    # filt_order = 3
    # b, a = signal.butter(filt_order, band_norm, 'bandpass')  # design filter

    # get data in numpy array
    num_channels = []
    all_files = []
    all_files_unfiltered = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []
    tmp_seizure_print = []

    print 'Getting Data...'
    for i, seizure_file in enumerate(data_filenames):

        # this is for when we have inter-ictal files that are an hour long that has split it up into parts
        if long_interictal and not (file_type[i] is 'ictal'):

            print '\tThis code has not been written'

        else:

            print '\tSeizure file %d reading...' % (i + 1),

            # read data in
            X, _, labels = edfread(seizure_file)

            all_files_unfiltered.append(X)
            n, p = X.shape
            num_channels.append(p)

            # good_channels_ind = []
            # labels = list(labels)
            # for channel in good_channels:
            #     good_channels_ind.append(labels.index(channel))

            # # filter data
            # print 'filtering...',
            # X = signal.filtfilt(b, a, X, axis=0)  # filter the data

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
        print 'There ' + 'is ' * (not gt1) + 'are ' * gt1 + str(num_channels) + ' channel' + 's' * gt1 + "\n"
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

def analyze_patient_noise(data_path, f_s=1e3, include_awake=True, include_asleep=False, long_interictal=False):

    # minutes per chunk (only for long interictal files)
    min_per_chunk = 15
    sec_per_min = 60

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    p_file = os.path.join(data_path, 'patient_pickle_mat.txt')

    with open(p_file, 'r') as pickle_file:
        print("Open Pickle: {}".format(p_file) + "...\n")
        patient_info = pickle.load(pickle_file)

    # add data file names
    data_filenames = patient_info['seizure_data_filenames']
    seizure_times = patient_info['seizure_times']
    file_type = ['ictal'] * len(data_filenames)
    seizure_print = [True] * len(data_filenames)  # mark whether is seizure

    if include_awake:
        data_filenames += patient_info['awake_inter_filenames']
        seizure_times += [None] * len(patient_info['awake_inter_filenames'])
        file_type += ['awake'] * len(patient_info['awake_inter_filenames'])
        seizure_print += [False] * len(patient_info['awake_inter_filenames'])

    if include_asleep:
        data_filenames += patient_info['asleep_inter_filenames']
        seizure_times += [None] + len(patient_info['asleep_inter_filenames'])
        file_type += ['sleep'] * len(patient_info['asleep_inter_filenames'])
        seizure_print += [False] * len(patient_info['asleep_inter_filenames'])

    data_filenames = [os.path.join(data_path, filename) for filename in data_filenames]
    good_channels = patient_info['good_channels']

    # band pass filter parameters
    # band = np.array([0.1, 100.])
    # band_norm = band / (f_s / 2.)  # normalize the band
    # filt_order = 3
    # b, a = signal.butter(filt_order, band_norm, 'bandpass')  # design filter

    # get data in numpy array
    num_channels = []
    all_files = []
    all_files_unfiltered = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []
    tmp_seizure_print = []

    print 'Getting Data...'
    for i, seizure_file in enumerate(data_filenames):

        # this is for when we have inter-ictal files that are an hour long that has split it up into parts
        if long_interictal and not (file_type[i] is 'ictal'):

            print '\tThis code has not been written'

        else:

            print '\tSeizure file %d reading...' % (i + 1),

            # read data in
            data_location = os.path.join(data_path,'noise')
            file_begin = os.path.join(data_location, os.path.splitext(os.path.basename(data_filenames[i]))[0])
            noise_file = file_begin + '_noise.mat'
            X = sio.loadmat(noise_file)
            X = X.get('all_noise')
            all_files_unfiltered.append(X)
            n, p = X.shape
            num_channels.append(p)

            # # filter data
            # print 'filtering...',
            # X = signal.filtfilt(b, a, X, axis=0)  # filter the data

            all_files.append(X)  # add feature to files


    # update temporary stuff
    data_filenames = update_list(data_filenames, tmp_data_filenames)
    file_type = update_list(file_type, tmp_file_type)
    seizure_times = update_list(seizure_times, tmp_seizure_times)
    seizure_print = update_list(seizure_print, tmp_seizure_print)

    # double check that the number of channels matches across data
    if len(set(num_channels)) == 1:
        num_channels = num_channels[0]
        gt1 = num_channels > 1
        print 'There ' + 'is ' * (not gt1) + 'are ' * gt1 + str(num_channels) + ' channel' + 's' * gt1 + "\n"
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

def analyze_patient_denoise(data_path, f_s=1e3, include_awake=True, include_asleep=False, long_interictal=False):

    # minutes per chunk (only for long interictal files)
    min_per_chunk = 15
    sec_per_min = 60

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    p_file = os.path.join(data_path, 'patient_pickle_mat.txt')

    with open(p_file, 'r') as pickle_file:
        print("Open Pickle: {}".format(p_file) + "...\n")
        patient_info = pickle.load(pickle_file)

    # add data file names
    data_filenames = patient_info['seizure_data_filenames']
    seizure_times = patient_info['seizure_times']
    file_type = ['ictal'] * len(data_filenames)
    seizure_print = [True] * len(data_filenames)  # mark whether is seizure

    if include_awake:
        data_filenames += patient_info['awake_inter_filenames']
        seizure_times += [None] * len(patient_info['awake_inter_filenames'])
        file_type += ['awake'] * len(patient_info['awake_inter_filenames'])
        seizure_print += [False] * len(patient_info['awake_inter_filenames'])

    if include_asleep:
        data_filenames += patient_info['asleep_inter_filenames']
        seizure_times += [None] + len(patient_info['asleep_inter_filenames'])
        file_type += ['sleep'] * len(patient_info['asleep_inter_filenames'])
        seizure_print += [False] * len(patient_info['asleep_inter_filenames'])

    data_filenames = [os.path.join(data_path, filename) for filename in data_filenames]
    # good_channels = patient_info['good_channels']

    # band pass filter parameters
    # band = np.array([0.1, 100.])
    # band_norm = band / (f_s / 2.)  # normalize the band
    # filt_order = 3
    # b, a = signal.butter(filt_order, band_norm, 'bandpass')  # design filter

    # get data in numpy array
    num_channels = []
    all_files = []
    all_files_unfiltered = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []
    tmp_seizure_print = []

    print 'Getting Data...'
    for i, seizure_file in enumerate(data_filenames):

        # this is for when we have inter-ictal files that are an hour long that has split it up into parts
        if long_interictal and not (file_type[i] is 'ictal'):

            print '\tThis code has not been written'

        else:

            print '\tSeizure file %d reading...' % (i + 1),

            # read data in
            data_location = os.path.join(data_path,'denoised')
            file_begin = os.path.join(data_location, os.path.splitext(os.path.basename(data_filenames[i]))[0])
            noise_file = file_begin + '_denoised_signal.mat'
            X = sio.loadmat(noise_file)
            X = X.get('all_signals')
            all_files_unfiltered.append(X)
            n, p = X.shape
            num_channels.append(p)

            # # filter data
            # print 'filtering...',
            # X = signal.filtfilt(b, a, X, axis=0)  # filter the data

            all_files.append(X)  # add feature to files


    # update temporary stuff
    data_filenames = update_list(data_filenames, tmp_data_filenames)
    file_type = update_list(file_type, tmp_file_type)
    seizure_times = update_list(seizure_times, tmp_seizure_times)
    seizure_print = update_list(seizure_print, tmp_seizure_print)

    # double check that the number of channels matches across data
    if len(set(num_channels)) == 1:
        num_channels = num_channels[0]
        gt1 = num_channels > 1
        print 'There ' + 'is ' * (not gt1) + 'are ' * gt1 + str(num_channels) + ' channel' + 's' * gt1 + "\n"
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