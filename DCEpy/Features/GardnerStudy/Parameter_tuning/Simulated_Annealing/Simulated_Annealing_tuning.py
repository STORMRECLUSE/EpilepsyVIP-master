"""
Parameter Tuning with Simulated Annealing as an alternative to Genetic Algorithm.

Tunes pi for each channel.

"""

import numpy as np
# import matplotlib.pyplot as plt
from sklearn import svm
import scipy
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
# from edfread import edfread
import array
import random
import multiprocessing
import math

from DCEpy.Features.GardnerStudy.weighteddr_bestchannels_parallel import energy_features,collect_windows,init_one_svm,anomaly_detection,decision_rule,window_to_sample,get_stats,performance_stats,score_performance
from DCEpy.Features.GardnerStudy.weighteddr_bestchannels_parallel import create_decision_csv,create_energy_csv,create_param_file,create_filtered_csv,create_model_file,read_energy_file,write_energy_file,update_list,update_log


# name of the folder in data folder to be saved
NAME_FOR_SAVING_FOLDER = 'SA_tuning_test'
# SELECTED_CHANNELS = [4,5,6,101,133]
SELECTED_CHANNELS=[0,1,2]



def reverse_fitness_fun_single_channel(individual, X_train, test_files, win_len, win_overlap, f_s, seizure_time):


    # get parameters from individual
    nu = [individual[0]]
    gamma = [individual[1]]
    C = [individual[2]]
    adapt_rate = [individual[3]]
    T_per = [individual[4]]
    weight = [1]

    # get performance characteristics
    sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, weight, nu, gamma,
                                                       T_per[0], adapt_rate, win_len, win_overlap, f_s, seizure_time)

    # use the gardner objective function
    score = -score_performance(sensitivity, latency, FP, time)
    return score




def within_constraints_pi(x_new,MIN,MAX):
    for i in range(5):
        if x_new[i]<MIN[i] or x_new[i]>MAX[i]:
            return False
    return True



def parameter_tuning_SA(X_train, test_files, win_len, win_overlap, f_s, seizure_time, num_channels):

    """
    The function takes as arguments:
    1. X_train:   ndarray with training data, size n, 3, p
    2. test_files:  list of ndarray (size test_num); list of test data each of size n, 3, p
    3. win_len  : int;  length of window (samples)
    4. win_overlap : int; length of window overlap (samples)
    5. f_s : float; sampling frequency
    6. seizure_time : list of tuple (if ictal) / None (otherwise)
        tuples are (start time, end time) of ints (samples)


    and returns:???????????
    1.v for all channels
    2.g for all channels
    3.p for all channels

    4.weights for all channels()

    5.adapt_rate for all channels
    6.Tper for all channels

    """

    print "number of channels!",num_channels



    best_genes=[]
    best_fitnesses=[]



    for channel in range(num_channels):

        print '\t\tTuning pi for channel %d' % (channel+1)

        # Initialize random pi
        t_per_min = 10 * f_s
        t_per_max = 200 * f_s
        MIN = [0.02, 0.25, 0.3, 10, t_per_min]
        MAX = [.2, 10, 1, 100, t_per_max]
        v = random.uniform(MIN[0], MAX[0])
        g = random.uniform(MIN[1], MAX[1])
        p = random.uniform(MIN[2], MAX[2])
        N = random.uniform(MIN[3], MAX[3])
        T = random.uniform(MIN[4], MAX[4])
        bounds=zip(MIN,MAX)

        # use method L-BFGS-B because the problem is smooth and bounded
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

        pi_init = [v, g, p, N, T]

        num_windows = X_train.shape[0]
        para = X_train.shape[1]

        X_channel = np.ones([num_windows, para, 1])

        for i in range(num_windows):
            for j in range(para):
                X_channel[i][j][0] = X_train[i][j][channel]

        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
        results=scipy.optimize.basinhopping(func=lambda x: reverse_fitness_fun_single_channel(x, X_channel, test_files, win_len, win_overlap, f_s, seizure_time),
                              x0=pi_init,
                              minimizer_kwargs=minimizer_kwargs,
                              disp=True
                              )


        print "Xmin, Jmin,T at termination, feval , iters ,accept ,status :", results

        best_genes+=list(results.x)
        best_fitnesses+=-results.fun

        print "\t\t best parameters:", list(results.x)
        print "\t\t best fitness:", -results.fun

    # Find the parameters for the good channels
    v_s = best_genes[0::5]
    g_s = best_genes[1::5]
    p_s = best_genes[2::5]
    adapt_rates = best_genes[3::5]
    T_pers = best_genes[4::5]


    return [0,1,2], v_s, g_s, p_s, adapt_rates, T_pers, best_fitnesses





    # print "Tuning weights for all channels"
    # # weights = weights_tuning(best_genes_selected_channels, best_fitness_selected_channels, X_train[:,:,good_channels],
    # # [X[:, :, good_channels] for X in test_files], win_len,
    # #                          win_overlap, f_s, seizure_time, len(good_channels))
    # weights = weights_tuning(best_genes_selected_channels, best_fitness_selected_channels, X_train, test_files, win_len, win_overlap, f_s, seizure_time, X_train.shape[2])
    #
    # #will return v for all channels, g for all channels, p for all channels, adapt_rate for all channels, and Tper for all channels
    # return good_channels, v_s, g_s, p_s, adapt_rates, T_pers,weights, sum(best_fitness_selected_channels)


def loocv_testing(all_files, data_filenames, file_type, seizure_times, seizure_print, win_len, win_overlap, num_windows, f_s, save_path):

    file_num = len(all_files)

    # leave-one-out-cross-validation
    for i, X_test in enumerate(all_files):

        # set up test files, seizure times, etc. for this k-fold
        print 'Cross Validation on k-fold %d of %d ...' %(i+1, file_num)
        cv_test_files = all_files[:i] + all_files[i+1:]
        cv_file_type = file_type[:i] + file_type[i+1:]
        cv_seizure_times = seizure_times[:i] + seizure_times[i+1:]

        # collect interictal files
        print '\tCollecting Interictal File'
        inter_ictal = [cv_test_files[j] for j in range(len(cv_test_files)) if cv_file_type[j] is not "ictal"]
        X_train = collect_windows(inter_ictal, num_windows)

        # For testing only, comment when running on full data
        # X_train = X_train[:,:,0:3]a
        X_train = X_train[:,:,SELECTED_CHANNELS]

        num_channels = X_train.shape[2]

        # parameter tuning
        print '\tChannel SVM parameter tuning with genetic algorithm'
        good_channel_indices, nu, gamma, C, adapt_rate, T_per, all_fitness = parameter_tuning_SA(X_train, cv_test_files, win_len,
                                                                                   win_overlap, f_s, cv_seizure_times,
                                                                                   X_train.shape[2])


        print "\tgood channels: ",good_channel_indices
        print "\tfitnesses: ",all_fitness
        print "\tnu: ", nu
        print "\t gamma: ",gamma
        print "\t threshold", C
        print "\t adapt rate: ",adapt_rate
        print "\t T persistence", T_per




        # Truncate X_train, cv_test_files and X_test to only include the good channels
        new_X_train = X_train[:, :, good_channel_indices]
        new_cv_test_files = [X[:, :, good_channel_indices] for X in cv_test_files]
        new_X_test=X_test[:, :, good_channel_indices]

        num_good_channels = new_X_train.shape[2]

        # train SVMs for individual selected channel
        print '\tTraining the SVMs with optimal parameters'
        svm_list = []
        for j in range(num_good_channels):
            svm_list.append(init_one_svm(new_X_train[:,:,j], nu=nu[j], gamma=gamma[j]))


        # test result for each single channel on the test data
        print '\tAnomaly detection'
        n = X_test.shape[0]
        out_frac_total = np.empty((n,num_good_channels))

        for j in range(num_good_channels):
            print j
            _, out_frac_total[:,j] = anomaly_detection(new_X_test[:,:,j], svm_list[j], adapt_rate[j])
            decision=np.sign(out_frac_total[:,j]-C[j])

            sensitivity, latency,FP,time = get_stats(decision, T_per[j], seizure_times[i], f_s, win_len, win_overlap)


            print '\tFor channel ',j
            print '\t\tSensitivity %.2f\tLatency %.3f\tFP %.2f\tTime %.2f' %(sensitivity, latency, FP, time)
            if seizure_times[i]== None:
                print "\t\t No seizure!"
            else:
                print '\t\t The actual seizure happened on ', (seizure_times[i][0]/1000.0,seizure_times[i][1]/1000.0)

            print '\t\t Persistence times are ',[T/1000.0 for T in T_per], 'seconds'


        print '\tWriting data to file'
        file_begin = os.path.join(save_path, os.path.splitext(os.path.basename(data_filenames[i]))[0])


def analyze_patient(patient_id, data_path, save_path, log_file, win_len=1.0, win_overlap=0.5, num_windows=1000, f_s=1e3, include_awake=True, include_asleep=False, long_interictal=False):

    # minutes per chunk (only for long interictal files)
    min_per_chunk = 15
    sec_per_min = 60

    # reformat window length and overlap as indices
    win_len = int(win_len * f_s)
    win_overlap = int(win_overlap * f_s)

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    p_file = os.path.join(data_path, 'patient_pickle.txt')


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

    data_filenames = [os.path.join(data_path,filename) for filename in data_filenames]
    good_channels = patient_info['good_channels']

    # band pass filter parameters
    band = np.array([0.1,100.])
    band_norm = band / (f_s / 2.) # normalize the band
    filt_order = 3
    b, a = scipy.signal.butter(filt_order, band_norm, 'bandpass') # design filter

    # get data in numpy array
    num_channels = []
    all_files = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []
    tmp_seizure_print = []

    print 'Getting Data...'
    for i, seizure_file in enumerate(data_filenames):

        # check that we haven't already gotten energy statistics for this seizure file
        file_begin = os.path.join(save_path, os.path.splitext(os.path.basename(data_filenames[i]))[0])

        # this is for when we have inter-ictal files that are an hour long that has split it up into parts
        if long_interictal and not (file_type[i] is 'ictal'):

            # get all files in the save path
            all_files_in_dir = [os.path.join(save_path,fn) for fn in next(os.walk(save_path))[2]]
            # all_files_in_dir = os.listdir(save_path)


            # if the energy stats exist for this file(.es represents energy statistics)
            inter_ictal_files = [s for s in all_files_in_dir if s.startswith(file_begin) and s.endswith(".es")]

            # if inter ictal (energy) files found, read them
            if inter_ictal_files:

                tmp_data_filenames.append( (i, inter_ictal_files))
                tmp_file_type.append( (i, [file_type[i]] * len(inter_ictal_files)) )
                tmp_seizure_times.append( (i, [seizure_times[i]] * len(inter_ictal_files)) )
                tmp_seizure_print.append( (i, [seizure_print[i]] * len(inter_ictal_files)) )

                # read each of the interictal files
                print "\tSeizure file %d is long --reading energy statistics directly..." %(i+1),
                for j, file_name in enumerate(inter_ictal_files):
                    print "%d" %(j+1),
                    X_feat, p = read_energy_file(file_name)
                    all_files.append(X_feat)        #  feature vector X from all the inter_ictal_files saved to all_files
                    num_channels.append(p)
                print " "


            else:

                # read the file
                print '\tSeizure file %d is long...' %(i+1)
                j = 0
                while True:
                    # get chunk start and end times
                    start = j * sec_per_min * min_per_chunk
                    end = (j+1) * sec_per_min * min_per_chunk

                    # get the chunk
                    try:

                        # extract the chunk
                        print '\t\tChunk ' + str(j+1) + ' reading...',
                        X_chunk, _, labels = edfread(seizure_file, rec_times = [start, end])
                        n,p = X_chunk.shape
                        num_channels.append(p)
                        good_channels_ind = []
                        labels = list(labels)
                        for channel in good_channels:
                            good_channels_ind.append(labels.index(channel))

                        # compute feature vector
                        print 'filtering...',
                        X_chunk = scipy.signal.filtfilt(b,a,X_chunk,axis=0) # filter the data

                        # get feature vectors from windows -- energy statistics
                        print 'extracting features...',
                        n_windows = n / (win_len - win_overlap) - 1 # evaluates to floor( n / (L - O ) - 1 since ints
                        X_feat = np.empty((n_windows,3,p))
                        m = 0
                        for k in range(win_len, X_chunk.shape[0], win_len - win_overlap):
                            window = X_chunk[(k-win_len):k,:] # select window
                            f = energy_features(window) # extract energy statistics
                            X_feat[m,:,:] = f
                            m += 1
                        all_files.append(X_feat) # add feature to files

                        # save energy statistics file
                        print 'saving...'
                        es_file = file_begin + "_%d.es"%(j)
                        write_energy_file(es_file, X_feat)

                        # print to csv if this is a desired seizure file
                        if seizure_print[i]:
                            filtered_file = file_begin + '_%d_filtered.csv'%(j)
                            energy_file = file_begin + '_%d_energystats.csv'%(j)
                            create_filtered_csv(filtered_file, X_chunk, good_channels_ind)
                            create_energy_csv(energy_file, X_feat[:,:,good_channels_ind[0]])

                        # update count
                        j += 1
                        # if less than an entire chunk was read, then this is the last one!
                        if X_chunk.shape[0] < sec_per_min * min_per_chunk:
                            break
                    except ValueError:
                            print "no wait, that doesn't exist!"
                            break # the start was past the end!

                # store temporary stuff
                tmp_data_filenames.append( (i, [os.path.join(save_path,file_begin + "_%d.es"%(k)) for k in range(j)]) )
                tmp_file_type.append( (i, [file_type[i]] * j) )
                tmp_seizure_times.append( (i, [seizure_times[i]] * j) )
                tmp_seizure_print.append( (i, [seizure_print[i]] * j) )

        else:
            es_file = file_begin + ".es"
            if os.path.isfile(es_file):
                print "\tSeizure file %d --reading energy statistics directly" %(i+1)
                X_feat, p = read_energy_file(es_file)
                all_files.append(X_feat)
                num_channels.append(p)
            else:
                print '\tSeizure file %d reading...' %(i+1),

                # read data in
                X,_,labels = edfread(seizure_file)
                n,p = X.shape
                num_channels.append(p)
                good_channels_ind = []
                labels = list(labels)
                for channel in good_channels:
                    print "channel",channel
                    good_channels_ind.append(labels.index(channel))

                # filter data
                print 'filtering...',
                X = scipy.signal.filtfilt(b,a,X,axis=0) # filter the data

                # get feature vectors from windows -- energy statistics
                print 'extracting features...'
                n_windows = n / (win_len - win_overlap) - 1 # evaluates to floor( n / (L - O ) - 1 since ints
                X_feat = np.empty((n_windows,3,p))
                k = 0
                for j in range(win_len, X.shape[0], win_len - win_overlap):
                    window = X[(j-win_len):j,:] # select window
                    f = energy_features(window) # extract energy statistics
                    X_feat[k,:,:] = f
                    k += 1
                all_files.append(X_feat) # add feature to files

                # save energy statistics file
                write_energy_file(es_file, X_feat)

                # print to csv if this is a desired seizure file
                if seizure_print[i]:
                    filtered_file = file_begin + '_filtered.csv'
                    energy_file = file_begin + '_energystats.csv'
                    create_filtered_csv(filtered_file, X, good_channels_ind)
                    create_energy_csv(energy_file, X_feat[:,:,good_channels_ind[0]])


    # update temporary stuff
    data_filenames = update_list(data_filenames, tmp_data_filenames)
    file_type = update_list(file_type, tmp_file_type)
    seizure_times = update_list(seizure_times, tmp_seizure_times)
    seizure_print = update_list(seizure_print, tmp_seizure_print)

    # double check that the number of channels matches across data
    if len(set(num_channels)) == 1:
        num_channels = num_channels[0]
        gt1 = num_channels > 1
        print 'There ' + 'is '*(not gt1) + 'are '*gt1 + str(num_channels) + ' channel' + 's'*gt1+"\n"
    else:
        print 'Channels: ' + str(num_channels)
        sys.exit('Error: There are different numbers of channels being used for different seizure files...')

    # double check that no NaN values appear in the features
    for X in all_files:
        if np.any(np.isnan(X)):
            sys.exit('Error: Uh-oh, NaN encountered while extracting features')

    # leave one out cross validation, update log
    loocv_testing(all_files, data_filenames, file_type, seizure_times, seizure_print, win_len, win_overlap, num_windows, f_s, save_path)
    #update_log(log_file, patient_id, sensitivity, latency, FP, time)

    return

if __name__ == '__main__':

    # parameters -- sampling data
    win_len = 1.0 # in seconds
    win_overlap = 0.5 # in seconds
    num_windows = 1000 # number of windows to sample to build svm model
    f_s = float(1e3) # sampling frequency
    include_awake = True
    include_asleep = False


    patients = ['TS041']
    long_interictal = [False]

    # get the paths worked out
    # to_data = '/scratch/smh10/DICE'
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(to_data, 'data', NAME_FOR_SAVING_FOLDER)

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # create results file
    log_file = os.path.join(save_path, 'log_file.txt')
    f = open(log_file, 'w')

    # write the first few lines
    f.write('Ran on %s\n' %time.strftime("%c"))

    # write the parameters
    f.write('Parameters used for this test\n================================\n')
    f.write('Feature used is energy statistic\n')
    f.write('Window Length \t%.3f\nWindow Overlap \t%.3f\nNumber of training windows \t%d\n' %(win_len, win_overlap, num_windows))
    f.write('Sampling Frequency \t%.3f\n' % f_s)
    f.write('Awake Times are ' + (not include_awake)*'NOT ' + ' included in training\n')
    f.write('Asleep Times are ' + (not include_asleep)*'NOT ' + ' included in training\n\n')

    # write the patients
    f.write('Patients are ' + " ".join(patients) + "\n\n")
    f.close()

    for i, patient_id in enumerate(patients):

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)
        p_save_path = os.path.join(save_path, patient_id)

        print "---------------------------Analyzing patient ",patient_id,"----------------------------\n"

        # analyze the patient, write to the file
        analyze_patient(patient_id, p_data_path, p_save_path, log_file, win_len, win_overlap, num_windows, f_s, include_awake, include_asleep, long_interictal[i])
















