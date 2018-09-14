from preprocessing import re_reference, notch_filter_data, get_freq_bands
from DCEpy.Features.BurnsStudy.ictal_inhibitors_final import analyze_patient_raw
from DCEpy.Features.BurnsStudy.ictal_inhibitors_final import choose_best_channels
import numpy as np
import os, sys
import pickle
import time
import scipy
from DCEpy.Features.MFCWT.Croise_SCAT.main import transform
from DCEpy.General.DataInterfacing.edfread import edfread
import numpy as np
import matplotlib.pyplot as plt

DATA_HOME_PATH = "../../EpilepsyVIP/data/PatientData"    # folder where raw patient data is stored 
FEATURE_SAVE_PATH = "scattering_all_channels_all/" # folder where you want features to be stored 

def get_seizure_time(patient_id, filename):
    home_path = DATA_HOME_PATH
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
    seizure_data_filenames = list(patient_info['seizure_data_filenames'])
    seizure_times = list(patient_info['seizure_times'])

    if filename not in seizure_data_filenames:
        return None
    else:
        return seizure_times[seizure_data_filenames.index(filename)]



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


def get_preictal_win(seizure_times, win_len = 3 * 60, win_overlap = 150):
    # determine the time in seconds where preictal period begins
    preictal_time = 5 * 60      # in seconds
    seizure_start_time = seizure_times[0]

    if seizure_start_time > preictal_time:
        preictal_start_time = seizure_start_time - preictal_time
        # determine the time in windows where preictal period begins
        preictal_start_win = int((preictal_start_time - win_len) / (win_len - win_overlap) + 1)
        return preictal_start_win
    else:
        return 0


"""
Extract the scattering network features for all channels.
data_filesnames: data filenames with complete data_path included
save_P: If 1, the P feature are stored
save_V: If 1, the V features are stored
save_S1: If 1, the S1 features are stored.
"""

def extract_scattering_all_channels(patient_id, all_files, data_filenames, data_path, save_P = 1, save_V = 0, save_S1 = 0):
	save_path = FEATURE_SAVE_PATH
	# Define the transformation parameters 
    	window_size = 2**14
    	# representation scales
    	J1,Q1=  8,6
    	J2,Q2 = 8,4
	# Wavelet Family
    	family_names = ['Morlet','Paul','Gammatone']
    	#Wavelet Family parameter
    	family_params = [15,8,[6.,.5]]	
	tstart_total = time.time()
	for (i, filename) in enumerate(data_filenames):
		relative_filename = str(i) + "_" + filename.split("/")[-1]
    		# read data in
    		X = all_files[i]
		
		# iterate over each focal channel and store the features 
		num_channels = X.shape[1]
		for channel_idx in range(num_channels):
			print "\tComputing features for channel: ", channel_idx
			tstart_channel = time.time()
			# process a single channel
			channel_X = X[:, channel_idx]
			# notch filter 60Hz harmonics 
    			channel_X = notch_filter_data(channel_X, 500)
			S1_data = []
			P_data = []
			V_data = []
			n = channel_X.shape[0]
			# test:
			#n = 2**16
			t_idx = 0
			while (t_idx + window_size <= n): 
				sig = channel_X[t_idx:t_idx + window_size]
				# print "shape of the signal:", sig.shape
				# sig = np.reshape(sig, (sig.shape[0], ))
	 			S1, S2, P, V = transform(sig, window_size, family_names, family_params, J1, Q1, J2, Q2)
				S1_data.append(np.reshape(S1, (S1.shape[0], S1.shape[1])))
				P_data.append(np.reshape(P, (P.shape[0], P.shape[1])))
				V_data.append(np.reshape(V, (V.shape[0], V.shape[1])))
				t_idx = t_idx + window_size/2
			S1_data = np.array(S1_data)
			P_data = np.array(P_data)
			V_data = np.array(V_data)

			tend_channel = time.time()
			# print "Filename: ",relative_filename
			print "\t\ttime for this channel: ", tend_channel - tstart_channel
			print "\t\ttotal number of samples: ", X.shape


			# store the data based on the flags
			if i == 0:
				# new file 	
				if (save_P):
					scipy.io.savemat(save_path + patient_id + "_scatteringcoeffP_channel" + str(channel_idx) + ".mat", {relative_filename: P_data})
					print "\tP data size:", P_data.shape
				if (save_V):
                                        scipy.io.savemat(save_path + patient_id + "_scatteringcoeffV_channel" + str(channel_idx) + ".mat", {relative_filename: V_data})
                                        print "\tV data size:", V_data.shape
				if (save_S1):
					scipy.io.savemat(save_path + patient_id + "_scatteringcoeffS1_channel" + str(channel_idx) + ".mat", {relative_filename: S1_data})
					print "\tS1 data size: ", S1_data.shape
			else:
				if (save_P):
					# load copy of dictionary 
					P_dict = scipy.io.loadmat(save_path + patient_id + "_scatteringcoeffP_channel" + str(channel_idx) + ".mat")
					# update dictionary 
					P_dict[relative_filename] = P_data
					# save new dictionary
					scipy.io.savemat(save_path + patient_id + "_scatteringcoeffP_channel" + str(channel_idx) + ".mat", P_dict)
				if (save_V):
                                        V_dict = scipy.io.loadmat(save_path + patient_id + "_scatteringcoeffV_channel" + str(channel_idx) + ".mat")
                                        V_dict[relative_filename] = V_data
					scipy.io.savemat(save_path + patient_id + "_scatteringcoeffV_channel" + str(channel_idx) + ".mat", V_dict)
				if (save_S1):
					S_dict = scipy.io.loadmat(save_path + patient_id + "_scatteringcoeffS1_channel" + str(channel_idx) + ".mat")
					S_dict[relative_filename] = S1_data
					scipy.io.savemat(save_path + patient_id + "_scatteringcoeffS1_channel" + str(channel_idx) + ".mat", S_dict)

		tend_total = time.time()
		print "Total time spent on this patient: ", tend_total - tstart_total 


"""
Extracts scattering features for all channels for every patient.
"""
def extract_all_sc():
    # TODO: input patients here
	patients = ["TS057"]
	# data_path = "../EpilepsyVIP/data/PatientData"
	data_path = DATA_HOME_PATH
	f_s = 1000
	include_awake = 1
	include_asleep = 1 
	win_len_samples = 2 ** 14
	win_overlap_samples = 2 ** 13
	for patient_id in patients:
		print "Extracting Features for patient ", patient_id
		p_data_path = os.path.join(data_path, patient_id)
		all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(data_path = p_data_path, f_s = f_s, 
			include_awake = include_awake, include_asleep = include_asleep, patient_id = patient_id, win_len = win_len_samples * 1.0/f_s, 
			win_overlap = win_overlap_samples * 1.0/f_s, calc_train_local = 0, single_channel = 0
		)
		extract_scattering_all_channels(patient_id, all_files, data_filenames, data_path, save_P = 1, save_V = 1, save_S1 = 1)

extract_all_sc()
