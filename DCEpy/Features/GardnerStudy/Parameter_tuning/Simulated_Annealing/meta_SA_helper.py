"""
Includes helper functions for meta-SA

Multichannel + weighted decision rule. Can be tested on different control parameters during meta-SA tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import signal
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
# from edfread import edfread
import array
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import multiprocessing
import math


from DCEpy.Features.GardnerStudy.weighteddr_bestchannels_parallel import create_decision_csv,create_energy_csv,create_model_file,create_param_file,create_filtered_csv,read_energy_file,write_energy_file,update_list,update_log,ga_worker,parameter_tuning_1
from DCEpy.Features.GardnerStudy.weighteddr_bestchannels_parallel import energy_features,collect_windows,init_one_svm,anomaly_detection,decision_rule,window_to_sample,get_stats,performance_stats,score_performance,fitness_fun_single_channel,fitness_fun_weights

# constants used in _main_ function
NAME_FOR_SAVING_FOLDER = 'energy_stats'
METHOD ='Gardners+SA'
# SELECTED_CHANNELS = [4,5,6,101,133]
SELECTED_CHANNELS=[0]


POP_SIZE=20


def plot_outlier(channel_idx, out_frac, C, file_path=None, FP_times=None, seizure_time=None, T_per=None, f_s=1e3):

    print "\tentered plot outlier"

    #create time vector
    n = out_frac.shape[0]

    # time = np.arange(n) / f_s
    # time = np.arange(n)

    # get figure
    fig = plt.figure(figsize=(7,4))


    # plot outlier fraction and threshold
    # time = np.linspace(0, float(n), out_frac.size)

    time=np.arange(0,float(n/2)+.5,0.5)
    # print time
    # print "time length",len(time)
    # print "length of outlier fraction",len(out_frac)
    # time = np.linspace(0, float(n), n)
    # print "how does time look like?",len(time)
    plt.plot(time, out_frac)
    # print out_frac


    # print "\t\t\tplotted outlier fraction"
    # plt.plot(time, C*np.ones(out_frac.size), 'm-')
    plt.plot(time, C * np.ones(n), 'm-')
    # print "\t\t\tplotted threshold"
    plt.ylim([0.0,1.1])
    plt.xlabel('Time (samples)')
    plt.title('Estimated Outlier Fraction for channel with index '+str(channel_idx))


    # plot seizure start and end lines
    if seizure_time!=None:
        start,end=seizure_time
        plt.axvline(start,color='r')
        plt.axvline(end, color='r')
        if T_per != None:
            if start-T_per>0:
                plt.axvline(start-T_per,color="b")


    # plot false positive lines
    if FP_times!=None:
        for FP in FP_times:
            plt.axvline(FP/1000.0,color='g')

    # either show or save figure
    if file_path is None:
        plt.show()
    else:
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)

    return


m



def within_constraints_pi(MIN, MAX):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if MIN[i] > child[i]:
                        child[i] = MIN[i]
                    if MAX[i] < child[i]:
                        child[i] = MAX[i]
            return offspring
        return wrapper
    return decorator


def within_constraints_weights(MIN):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if MIN[i] > child[i]:
                        child[i] = MIN[i]
                        #could change this; rn it makes a lot of weights 0 if they get mutated negative; could shift and normalize
            return offspring
        return wrapper
    return decorator


def init_weights(icls, best_fitnesses):

    # weight initializers
    smallest=min(best_fitnesses)
    fitness_above_one=best_fitnesses

    if smallest<0:
        add=1-smallest
        fitness_above_one=[fitness+add for fitness in best_fitnesses]

    # print "fitness above one",fitness_above_one
    tot = sum(fitness_above_one)
    weights_init=[float(fitness)/tot for fitness in fitness_above_one]

    ind = icls(np.random.normal(weights_init[spot], 0.005) for spot in range(len(weights_init)))

    return ind

"""
parameter_tuning()

OUTPUT:
    nu --> ndarray (size num_channels)
        parameter for each of the channel's SVM
    gamma --> ndarray (size num_channels)
        parameter for each of the channel's SVM
    C --> ndarray (size num_channels)
        parameter for each channel's novelty detection
    weights --> ndarray (size num_channels)
        optimal weighting for channel's alarm sequences
    N --> ndarray (size num_channels)
        adaptation rate for each channel; will be same for all channels
    T --> ndarray (size num_channels)
        persistance time for each channel; will be same for all channels
"""

def evolution(toolbox, pop, NGEN, prev_mean,next_mean, CXPB, MUTPB):

    # defining variables to keep track of best indivudals throughout species
    best_species_genes = tools.selBest(pop, 1)[0]
    best_species_value = best_species_genes.fitness.values
    best_gen = 0



    for g in range(NGEN):

        # print '\t\t\t-------Working on generation %d-------' % (g + 1)


        if abs(next_mean - prev_mean) >= 0.5:

            prev_mean = next_mean
            # Select the next generation's parents
            parents = toolbox.select(pop, len(pop))

            # Clone the parents and call them offspring: crossover and mutation will be performed below
            offspring = list(map(toolbox.clone, parents))

            # Apply crossover to children in offspring with probability CXPB
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation to children in offspring with probability MUTPB
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Find the fitnessess for all the children for whom fitness changed
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Offspring becomes the new population
            pop[:] = offspring

            # updating best species values
            if len(fitnesses) > 0:
                if max(fitnesses) > best_species_value:
                    best_species_genes = tools.selBest(pop, 1)[0]
                    best_species_value = best_species_genes.fitness.values
                    best_gen = g
                    # best_next_obj = max(fitnesses)



            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            next_mean = sum(fits) / length
            # print "\tMaximum fitness in the current population:  ", max(fits)
            # print "\tMean fitness in the current generation:  ", next_mean


    return best_species_genes,best_species_value,best_gen

#
# def ga_worker(X_train, test_files, win_len, win_overlap, f_s, seizure_time, num_channels, out_q, channel_start_ind,parameters):
#
#     """
#     The function takes as arguments:
#     1. X_train:   ndarray with training data, size n, 3, p
#     2. test_files:  list of ndarray (size test_num); list of test data each of size n, 3, p
#     3. win_len  : int;  length of window (samples)
#     4. win_overlap : int; length of window overlap (samples)
#     5. f_s : float; sampling frequency
#     6. seizure_time : list of tuple (if ictal) / None (otherwise)
#         tuples are (start time, end time) of ints (samples)
#
#
#     and returns:???????????
#     1.v for all channels
#     2.g for all channels
#     3.p for all channels
#
#     4.weights for all channels()
#
#     5.adapt_rate for all channels
#     6.Tper for all channels
#
#     """
#
#     #creating types
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)
#     toolbox = base.Toolbox()
#
#     #defining genes
#     #ranges are given by Gardner paper
#     t_per_min = 10*f_s
#     t_per_max = 200*f_s
#     MIN = [0.02, 0.25, 0.3, 10, t_per_min]
#     MAX = [.2, 10, 1, 100, t_per_max]
#     toolbox.register("attr_v", random.uniform, MIN[0], MAX[0])
#     toolbox.register("attr_g", random.uniform, MIN[1], MAX[1])
#     toolbox.register("attr_p", random.uniform, MIN[2], MAX[2])
#     toolbox.register("attr_N", random.uniform, MIN[3], MAX[3])
#     toolbox.register("attr_T", random.uniform, MIN[4], MAX[4])
#
#     #defining an individual as a group of the five genes
#     toolbox.register("individual", tools.initCycle, creator.Individual,
#                      (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N, toolbox.attr_T), 1)
#
#     #defining the population as a list of individuals
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.decorate("population", within_constraints_pi(MIN,MAX))
#
#
#     # register the crossover operator
#     # other options are: cxOnePoint, cxUniform (requires an indpb input, probably can just use CXPB)
#     # there are others, more particular than these options
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.decorate("mate", within_constraints_pi(MIN,MAX))
#
#     # register a mutation operator with a probability to mutate of 0.05
#     # can change: mu, sigma, and indpb
#     # there are others, more particular than this
#     mu = np.tile([0],5)
#     sigma = np.tile([.06, 3.25, .23, 30, 63.33*f_s],1)
#     indpb = np.tile([.02],5)
#     toolbox.register("mutate", tools.mutGaussian, mu=mu.tolist(), sigma=sigma.tolist(), indpb=indpb.tolist())
#     toolbox.decorate("mutate", within_constraints_pi(MIN,MAX))
#
#     # operator for selecting individuals for breeding the next generation
#     # other options are: tournament: randonly picks tournsize out of population, chosses fittest, and has that be
#     # a parent. continues until number of parents is equal to size of population.
#     # there are others, more particular than this
#     toolbox.register("select", tools.selTournament, tournsize=2)
#     # toolbox.register("select", tools.selRoulette)
#
#     # CXPB  is the probability with which two individuals are crossed
#     CXPB = parameters[0]
#
#     # MUTPB is the probability for mutating an individual
#     MUTPB = parameters[1]
#
#     # NGEN  is the number of generations until final parameters are picked
#     # NGEN = 5
#     NGEN = 10
#
#
#
#     # initiate lists to keep track of the best genes(best pi) and best fitness scores for everychannel
#     # best_genes_all_channels =[]
#     # best_fitness_all_channels=[]
#
#     store_ind = channel_start_ind
#     outdict = {}
#
#     for channel in range(num_channels):
#
#         print '\t\tTuning pi for channel %d' % (channel+1)
#
#         # create an initial population of size 20
#         pop = toolbox.population(n=POP_SIZE)
#
#         # find X_channel (the channel layer in X_train)
#         num_windows=X_train.shape[0]
#         para=X_train.shape[1]
#
#         X_channel=np.ones([num_windows,para,1])
#
#         for i in range(num_windows):
#             for j in range(para):
#                 X_channel[i][j][0]=X_train[i][j][channel]
#
#
#         # register the fitness function
#         toolbox.register("evaluate",lambda x: fitness_fun_single_channel(x, X_channel, test_files, win_len, win_overlap, f_s, seizure_time))
#
#         # find the fitness of every individual in the population
#         fitnesses = list(map(toolbox.evaluate, pop))
#
#         #assigning each fitness to the individual it represents
#         for ind, fit in zip(pop, fitnesses):
#             ind.fitness.values = fit
#
#
#         next_mean=1
#         prev_mean=0
#
#         #start evolution
#         best_species_genes,best_species_value,best_gen=evolution(toolbox, pop, NGEN, prev_mean,next_mean, CXPB, MUTPB)
#
#         print "\t\tThis channel has a fitness of: ", best_species_value[0]
#
#         outdict[store_ind,0] = np.asarray(best_species_genes)
#         outdict[store_ind,1] = np.asarray(best_species_value)
#
#         store_ind+=1
#
#     out_q.put(outdict)
#
#     return

def parameter_tuning_1(X_train, test_files, win_len, win_overlap, f_s, seizure_time, num_channels,control_parameters):

    nprocs = 1

    # get data dimensions
    n,p,k = X_train.shape

    # Each process will get 'chunksize' samples and
    # a queue to put his out dict into
    out_q = multiprocessing.Queue()
    channels_per_proc = int(math.ceil(k / float(nprocs)))

    procs = []

    print "\t\tEntered parameter_tuning_1, sending data to procs"
    for i in range(nprocs):

        # extract the time chunk
        start = i*channels_per_proc # the index of the first sample for this chunk
        if (i+1)*channels_per_proc <= k:
            end = (i+1)*channels_per_proc
        else:
            end = k

        X_c = X_train[:,:,start:end]

        # create thread
        p = multiprocessing.Process(
            target=ga_worker,
            args=(X_c, test_files, win_len, win_overlap, f_s, seizure_time, X_c.shape[2], out_q, start,control_parameters)
        )

        procs.append(p)
        p.start()


    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    print "Procs finished tuning channel SVMs"

    # collect results in a list
    pi_list = []
    fitness_list = []
    for i in range(num_channels):
        pi_list += resultdict[i,0].tolist()
        fitness_list += resultdict[i,1].tolist()

    best_genes_selected_channels = pi_list
    best_fitness_selected_channels = fitness_list



    # Find the parameters for the good channels
    v_s=best_genes_selected_channels[0::4]
    g_s=best_genes_selected_channels[1::4]
    p_s=best_genes_selected_channels[2::4]
    adapt_rates=best_genes_selected_channels[3::4]
    # T_pers = best_genes_selected_channels[4::5]
    num = len(v_s)
    T_pers = [int(180*f_s)]*num


    return SELECTED_CHANNELS, v_s, g_s, p_s, adapt_rates, T_pers,best_fitness_selected_channels



def loocv_testing(all_files, data_filenames, file_type, seizure_times, seizure_print, win_len, win_overlap, num_windows, f_s, save_path,control_parameters,folds):

    file_num = len(all_files)

    fitnesses=[]

    # leave-one-out-cross-validation
    for i, X_test in enumerate(all_files):

        if i>=folds:
            break


        # set up test files, seizure times, etc. for this k-fold
        print 'Cross Validation on k-fold %d of %d ...' %(i+1, file_num)
        cv_test_files = all_files[:i] + all_files[i+1:]
        cv_file_type = file_type[:i] + file_type[i+1:]
        cv_seizure_times = seizure_times[:i] + seizure_times[i+1:]

        # collect interictal files
        print '\tCollecting Interictal File'
        inter_ictal = [cv_test_files[j] for j in range(len(cv_test_files)) if cv_file_type[j] is not "ictal"]
        X_train = collect_windows(inter_ictal, num_windows)

        # single channel parameter_tuning
        print '\tSingle Channel Parameter Tuning'
        good_channel_indices, nu, gamma, C, adapt_rate, T_per, all_fitness = parameter_tuning_1(X_train, cv_test_files, win_len,
                                                                                   win_overlap, f_s, cv_seizure_times,
                                                                                   X_train.shape[2],control_parameters)


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
            # print j
            _, out_frac_total[:,j] = anomaly_detection(new_X_test[:,:,j], svm_list[j], adapt_rate[j])
            decision=np.sign(out_frac_total[:,j]-C[j])

            sensitivity, latency,FP,time,FP_times = get_stats_extended(decision, T_per[j], seizure_times[i], f_s, win_len, win_overlap)


            print '\t\tFor channel ',j
            print '\t\t\tSensitivity %.2f\tLatency %.3f\tFP %.2f\tTime %.2f' %(sensitivity, latency, FP, time)
            if seizure_times[i]== None:
                print "\t\t\t No seizure!"
            else:
                print '\t\t\t The actual seizure happened on ', (seizure_times[i][0]/1000.0,seizure_times[i][1]/1000.0)

            print '\t\t\t Persistence times are ',[T/1000.0 for T in T_per], 'seconds'

            if FP_times!=[]:
                print "\t\tFalse Positives occured on these times ",tuple(FP_times), "seconds"

            else:
                print "No False Positives!"

        #plot graph
        actual_channel= SELECTED_CHANNELS[good_channel_indices[j]]
        plot_outlier(actual_channel, out_frac_total[:, j], C[j], file_path=
        str(i + 1) + "fold " + "test file " + str(i + 1) + "channel" + str(
            actual_channel) + ".png", FP_times=FP_times, seizure_time=seizure_times[i], T_per=T_per[j])

    # print "fitnesses over all folds", fitnesses
    return fitnesses




def analyze_patient(patient_id, data_path, save_path, log_file, parameters, folds, win_len=1.0, win_overlap=0.5, num_windows=1000, f_s=1e3,include_awake=True, include_asleep=False, long_interictal=False):

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
    b, a = signal.butter(filt_order, band_norm, 'bandpass') # design filter

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
                        X_chunk = signal.filtfilt(b,a,X_chunk,axis=0) # filter the data

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
                X = signal.filtfilt(b,a,X,axis=0) # filter the data

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
    fitnesses=loocv_testing(all_files, data_filenames, file_type, seizure_times, seizure_print, win_len, win_overlap, num_windows, f_s, save_path,parameters, folds)
    #update_log(log_file, patient_id, sensitivity, latency, FP, time)

    return fitnesses

def experiment(parameters, folds):

    # parameters -- sampling data
    win_len = 1.0 # in seconds
    win_overlap = 0.5 # in seconds
    num_windows = 1000 # number of windows to sample to build svm model
    f_s = float(1e3) # sampling frequency
    include_awake = True
    include_asleep = False

    # list of patients that will be used
    # patients = ['TS039', 'TS041', 'TA533', 'TA023']
    # patients = ['TA533', 'TA023']

    patients = ['TS041']
    long_interictal = [False]

    # get the paths worked out
    # to_data = '/scratch/smh10/DICE'
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(to_data, 'data', NAME_FOR_SAVING_FOLDER)

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # create results file
    log_file = os.path.join(save_path, 'log_file.txt')
    f = open(log_file, 'w')

    # write the first few lines
    f.write("Results file for "+ METHOD +"\n")
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
        return analyze_patient(patient_id, p_data_path, p_save_path, log_file,parameters, folds, win_len, win_overlap, num_windows, f_s, include_awake, include_asleep, long_interictal[i])


# experiment([.8,.1,.25],6)