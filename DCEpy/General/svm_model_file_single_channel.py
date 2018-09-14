import os
import pickle
import random
import sys

import numpy as np
from scipy import signal
from sklearn import svm

from DCEpy.Features.GardnerStudy.edfread import edfread
from deap import base
from deap import creator
from deap import tools


def collect_windows(inter_ictal, num_windows):

    # data dimension
    p = inter_ictal[0].shape[1]

    # get different sizes of
    k = len(inter_ictal)
    size1 = np.floor(num_windows / k)
    size2 = np.floor(num_windows / k) + 1

    # initialize sampled data
    X = np.ones((num_windows,p))

    # collect sampled data
    for i in range(k):

        X_inter = inter_ictal[i]
        n = X_inter.shape[0]

        if (i <= (k - num_windows % k)):
            ind = np.random.choice(n, size=size1, replace=False)
            X[i:i+size1] = X_inter[ind,:]

        else:
            ind = np.random.choice(n, size=size2, replace=False)
            X[i:i+size2] = X_inter[ind,:]

        if np.any(np.isnan(X)):
                pass

    if np.any(np.isnan(X)):
            print '\tUh-oh, NaN encountered while collecting windows'

    # return the collected data
    return X

"""
energy_features()

INPUT:
    X --> ndarray
        time series data of size n x p

OUTPUT:
    Y --> ndarray
        feature vector of size n x 3 where entries are log of
        (i) Mean Curve Length
        (ii) Mean Energy
        (iii) Mean Absolute Teager Energy
"""
def energy_features(X):

    # get data dimensions
    n,p = X.shape

    # compute energy based statistics
    CL = np.log10(np.mean(np.abs(np.diff(X, axis=0)), axis=0)) # mean curve length
    E = np.log10(np.mean(X ** 2, axis=0)) # mean energy
    tmp = X[1:n-1,:] ** 2 -  X[0:n-2,:] * X[2:n,:] # Teager energy
    TE = np.log10(np.mean(np.abs(tmp), axis=0)) # ABSOLUTE Teager energy

    # get feature vectors together
    Y = np.hstack((CL,E,TE))
    return Y

"""
learn_support()

INPUT:
    X_train --> ndarray
        training feature vectors of size n x p
    nu --> float
        outlier fraction
    gamma --> float
        coefficient for gaussian kernel

OUTPUT:
    clf --> one-class SVM model
"""
def learn_support(X_train, nu=0.1, gamma=1.0):

    if np.any(np.isnan(X_train)):
        print '\tUh-oh, NaN encountered in learn_support()'

    clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
    clf.fit(X_train)
    return clf

"""
anomaly_detection()

INPUT:
    X_test --> ndarray
        testing feature vectors of size n x p
    clf --> one-class SVM model
        model learned from training data
    adapt_rate --> int
        size of window used to estimate outlier fraction
    C --> float (between 0 and 1)
        threshold for outlier fraction

OUTPUT:
    n_seq --> ndarray (size n)
        anomaly sequence returned by one-class SVM
    out_frac --> ndarray (size n)
        estimated outlier fraction
"""
def anomaly_detection(X_test, clf, adapt_rate=40):

    # label returned by one-class SVM
    n,p = X_test.shape
    n_seq = clf.predict(X_test)

    # change output so that +1 is anomalous and 0 is not anomalous
    z = np.copy(n_seq)
    z[np.where(n_seq==-1)] = 1 # anomalous
    z[np.where(n_seq==1)] = 0 # not anomalous

    # small sliding window of size n
    out_frac = np.zeros(n_seq.size)
    for i in np.arange(adapt_rate,n):
        out_frac[i] = np.mean(z[i-adapt_rate:i])

    return n_seq, out_frac

def get_alarm_seq(out_frac, C, window_length, window_overlap):

    # get length of non overlapping portion of windows andd number of time points
    window_nonoverlap = window_length - window_overlap
    n = (out_frac.size + 1) * window_nonoverlap

    # create alarm sequence
    alarm_seq = np.zeros(n, dtype=bool)
    for i in range(out_frac.size):
        alarm_seq[i:i+window_nonoverlap] = (out_frac[i] > C)

    return alarm_seq


def evaluate_single(alarm_seq, T_per, seizure_time, f_s):

    # if inter-ictal file
    if seizure_time is None:

        # get the amount of time passed
        time = float(len(alarm_seq)) / float(f_s * 60. * 60.)
        FP = 0.

        # while an anomaly is detected ahead
        while np.any(alarm_seq):

            # add a false positive
            FP += 1.

            # discard everything in the persistance block
            start = np.where(alarm_seq == True)[0][0]
            if (start + T_per) > alarm_seq.size:
                break
            else:
                alarm_seq = alarm_seq[start+T_per:]

        # compute performance metrics
        # fpr = FP / time
        sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        latency = np.nan # latency is meaningless since there is no seizure

    else: # ictal time

        start = seizure_time[0]
        end = seizure_time[1]
        pre_alert = max(0, start-T_per)

        # compute false positive rate (only on T_per time before the seizure)
        pre_seizure = np.copy(alarm_seq[:pre_alert])
        if pre_seizure > 0:
            time = float(len(pre_seizure)) / float(f_s * 60. * 60.)
            FP = 0.

            # while an anomaly is detected ahead
            while np.any(pre_seizure):

                # add a false positive
                FP += 1.

                # discard everything in the persistance block
                s_start = np.where(pre_seizure == True)[0][0]
                if (s_start + T_per) > pre_seizure.size:
                    break
                else:
                    pre_seizure = pre_seizure[s_start+T_per:]

            # fpr = FP / time
        else:
            FP = np.nan
            time = np.nan

        # compute sensitivity
        # Note: true positive is defined here as when a seizure is detected during seizure time
        #       and false negative is defined here as when a seizure is not detected early

        if not np.any(alarm_seq[pre_alert:end]):
            sensitivity = 0 # seizure not detected
        elif not np.any(alarm_seq[pre_alert:start]):
            sensitivity = 0.5 # seizure detected late
        else:
            sensitivity = 1.0 # seizure detected early

        # compute latency
        if np.any(alarm_seq[pre_alert:end]):
            detect_time = np.where(alarm_seq[pre_alert:end] == True)[0][0] + pre_alert
            latency = float(detect_time - start) / float(f_s) # (in seconds)
        else:
            latency = np.nan
            #print 'Never detected :('

    return sensitivity, latency, FP, time

def evaluate_fit(nu, gamma, C, adapt_rate, T_per, X_train, feat_vec, seizure_times, f_s, window_length, window_overlap):

    # fit SVM model
    clf = learn_support(X_train, nu=nu, gamma=gamma)

    # initialize performance metrics
    file_num = len(feat_vec)
    sensitivity = np.empty(file_num)
    latency = np.empty(file_num)
    time = np.empty(file_num)
    FP = np.empty(file_num)

    # get performance metrics on each seizure
    for i in range(file_num):

        # perform anomaly detection using SVM model
        n_seq, out_frac = anomaly_detection(feat_vec[i], clf, adapt_rate=adapt_rate)
        alarm_seq = get_alarm_seq(out_frac, C, window_length, window_overlap)

        # score performance
        sensitivity[i], latency[i], FP[i], time[i] = evaluate_single(alarm_seq, T_per, seizure_times[i], f_s)

    # get aggregate statistics on performance metrics
    # S = np.nanmedian(sensitivity)
    S = np.median(sensitivity[np.isfinite(sensitivity)]) # stupid work-around
    # FPR = np.nansum(FP) / np.nansum(time)
    FPR = np.sum(FP[np.isfinite(FP)]) / np.sum(time[np.isfinite(time)]) # stupid work-around

    # if any seizure is detected
    if np.any(np.isfinite(latency)):
        # mu = np.nanmedian(latency)
        mu = np.median(latency[np.isfinite(latency)])
        # EDF = float(np.where(latency < 0)[0].size) / float(np.where(latency > -np.inf)[0].size)
        detected = latency[np.isfinite(latency)]
        EDF = float(np.where(detected < 0)[0].size) / float(detected.size)

    else:
        mu = 500.0 # else give terrible values
        EDF = 0.0

    return S, EDF, FPR, mu

def within_constraints(MIN, MAX):
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

def fitness_fn(individual, X_train, feat_vec, seizure_times, f_s, window_length, window_overlap):

    nu = individual[0]
    gamma = individual[1]
    C = individual[2]
    adapt_rate = individual[3]
    T_per = individual[4]

    S, EDF, FPR,  mu = evaluate_fit(nu, gamma, C, adapt_rate, T_per, X_train, feat_vec, seizure_times, f_s, window_length, window_overlap)

    alpha1 = 100*S-10*(1-np.sign(S-0.75))
    alpha2 = 20*EDF
    alpha3 = -10*FPR-20*(1-np.sign(5-FPR))
    alpha4 = max(-mu,30)
    result=alpha1+alpha2+alpha3+alpha4

    return result,


def parameter_tuning(X_train, feat_vec, seizure_times, f_s, window_length, window_overlap):

    #creating types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    t_per_min = 10*f_s
    t_per_max = 200*f_s

    #defining genes
    #ranges are given by Gardner paper
    toolbox.register("attr_v", random.uniform, .02, .2)
    toolbox.register("attr_g", random.uniform, .25, 10)
    toolbox.register("attr_p", random.uniform, .3, 1)
    toolbox.register("attr_N", random.uniform, 10, 100)
    toolbox.register("attr_T", random.uniform, t_per_min, t_per_max)

    #defining an individual as a group of the five genes
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N, toolbox.attr_T), 1)

    #defining the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the fitness function
    toolbox.register("evaluate", lambda x: fitness_fn(x, X_train, feat_vec, seizure_times, f_s, window_length, window_overlap))

    MIN = [0.02, 0.25, 0.3, 10, t_per_min]
    MAX = [.2, 10, 1, 100, t_per_max]
    # register the crossover operator
    # other options are: cxOnePoint, cxUniform (requires an indpb input, probably can just use CXPB)
    # there are others, more particular than these options
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.decorate("mate", within_constraints(MIN,MAX))

    # register a mutation operator with a probability to mutate of 0.05
    # can change: mu, sigma, and indpb
    # there are others, more particular than this
    toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=10, indpb=0.03)
    toolbox.decorate("mutate", within_constraints(MIN,MAX))

    # operator for selecting individuals for breeding the next generation
    # other options are: tournament: randonly picks tournsize out of population, chosses fittest, and has that be
    # a parent. continues until number of parents is equal to size of population.
    # there are others, more particular than this
    toolbox.register("select", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selRoulette)

    #create an initial population of size 20
    pop = toolbox.population(n=20)

    # CXPB  is the probability with which two individuals are crossed
    CXPB = 0.3

    # MUTPB is the probability for mutating an individual
    MUTPB = 0.5

    # NGEN  is the number of generations until final parameters are picked
    NGEN = 40

    print("Start of evolution")

    # find the fitness of every individual in the population
    fitnesses = list(map(toolbox.evaluate, pop))

    #assigning each fitness to the individual it represents
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    #defining variables to keep track of best indivudals throughout species
    best_species_genes = tools.selBest(pop, 1)[0]
    best_species_value = best_species_genes.fitness.values
    best_gen = 0

    next_mean=1
    prev_mean=0

    #start evolution
    for g in range(NGEN):
        if abs( next_mean - prev_mean ) > 0.005 :

            prev_mean=next_mean
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

            #updating best species values
            if max(fitnesses):
                if max(fitnesses) > best_species_value:
                    best_species_genes = tools.selBest(pop, 1)[0]
                    best_species_value = best_species_genes.fitness.values
                    best_gen = g
                best_next_obj = max(fitnesses)

            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            next_mean = sum(fits) / length

    return best_species_genes[0], best_species_genes[1], best_species_genes[2], best_species_genes[3], best_species_genes[4]


def create_model_file(data_path, win_len, win_overlap, f_s, model_file, param_file, num_windows=500, include_awake=True, include_asleep=False):

    # use pickle files
    p_file = os.path.join(data_path, 'patient_pickle.txt')
    with open(p_file,'r') as pickle_file:
        patient_info = pickle.load(pickle_file)

    # add data file names and types
    data_filenames = patient_info['seizure_data_filenames']
    seizure_times = patient_info['seizure_times']
    file_types = ['ictal'] * len(data_filenames)

    if include_awake:
        data_filenames += patient_info['awake_inter_filenames']
        seizure_times += [None] * len(patient_info['awake_inter_filenames'])
        file_types += ['awake'] * len(patient_info['awake_inter_filenames'])

    if include_asleep:
        data_filenames += patient_info['asleep_inter_filenames']
        seizure_times += [None] * len(patient_info['asleep_inter_filenames'])
        file_types += ['sleep'] * len(patient_info['asleep_inter_filenames'])

    # attach data file names to data path
    data_filenames = [os.path.join(data_path,filename) for filename in data_filenames]
    num_files = len(data_filenames)

    # get best channel to train on
    # TODO: (this will change in the future to include all channels)
    good_channels = patient_info['best_channel']


    # TODO: change this to read an edf file, then get energy statistic BEFORE going to next edf file
    #
    # read files and store in an array
    print 'Reading data from edf files to numpy array'
    all_data = []
    num_channels = []
    i = 1
    for seizure_file in data_filenames:
        print '\tReading ' + str(i) + ' of ' + str(num_files)
        i += 1
        X,_,_ = edfread(seizure_file,good_channels=good_channels)
        num_channels.append(X.shape[1])
        all_data.append(X)

    if len(set(num_channels)) == 1:
        num_channels = num_channels[0]
        gt1 = num_channels > 1
        print 'There ' + 'is '*(not gt1) + 'are '*gt1 + str(num_channels) + ' channel' + 's'*gt1
    else:
        print 'Channels: ' + str(num_channels)
        sys.exit('Error: There are different numbers of channels being used for different seizure files...')

    p_feat = 3 # this is the number of energy statistics

    # pre-process data -- filter parameters
    print 'Applying a band-pass filter to the data'
    band = np.array([0.1,100.])
    band_norm = band / (f_s / 2.) # normalize the band
    filt_order = 3

    # band pass filter the data
    b, a = signal.butter(filt_order, band_norm, 'bandpass') # design filter
    for j in range(num_files):
        all_data[j] = signal.filtfilt(b,a,all_data[j],axis=0) # filter the data


    # get features from time series
    num_files = len(all_data)
    feat_vec = []
    print '\tExtracting features from input files...',
    i=1
    for X in all_data:

        # print progress
        print str(i) + ', ',
        i += 1

        # initialize empty feature vector
        n = X.shape[0]
        n_windows = n / (window_length - window_overlap) - 1 # evaluates to floor( n / (L - O ) - 1 since ints
        X_feat = np.zeros((n_windows, p_feat)) # empty feature vector

        k = 0
        # collect features from windows
        for j in range(window_length, n, window_length - window_overlap):
            window = X[(j-window_length):j,:] # select window
            f = energy_features(window) # extract energy statistics
            X_feat[k,:] = f
            k += 1

        # add the new feature vector
        feat_vec.append(X_feat)

    print '' # new line

    # check for NaN
    for X in feat_vec:
        if np.any(np.isnan(X)):
            print '\tUh-oh, NaN encountered while extracting features'

    print '\tCollecting inter-ictal windows'
    inter_ictal = [feat_vec[j] for j in range(len(feat_vec)) if feat_vec[j] is not "ictal"]
    X_train = collect_windows(inter_ictal, num_windows)

    # parameter tuning
    nu, gamma, C, adapt_rate, T_per = parameter_tuning(X_train, feat_vec, seizure_times, f_s, window_length, window_overlap)
    print 'Obtained optimal parameters'

    # run an SVM on the training data
    clf = learn_support(X_train, nu=nu, gamma=gamma)
    num_SV = clf.support_.size

    # create model file
    print 'Writing to model file'
    f = open(model_file, 'w')
    f.write('svm_type one_class\n') # one class SVM
    f.write('kernel_type rbf\n') # kernel type = rbf
    f.write('gamma %.6f\n' %gamma) # gamma
    f.write('nr_class 2\n') # number of classes = 2
    f.write('total_sv %d\n' % num_SV) # total num of support vectors
    f.write('rho %.6f\n' %clf.intercept_[0]) # offset
    f.write('SV\n') # ready for support vectors!

    # write support vectors to model file
    for i in range(num_SV):
        f.write('%.6f ' % clf.dual_coef_[0, i])
        for j in range(p_feat):
            f.write(str(j+1) + ':%.6f ' % clf.support_vectors_[i, j])
        f.write('\n')

    f.close()

    # write other parameters file
    f = open(param_file, 'w')

    # TODO: do not hardcode number of channels
    num_channels = 135
    f.write('adapt_rate: %d\n' %adapt_rate)
    f.write('channel: threshold: weight\n')
    for i in range(num_channels):
        if i == 2:
            weight = 1
        else:
            weight = 0
        f.write("%d: %.4f: %d\n" %(i,C,weight))
    f.close()

    return


if __name__ == "__main__":

    # set parameters about windows
    f_s = 1000
    window_length = 1 * f_s
    window_overlap = int(0.5 * f_s)
    include_awake = True
    include_asleep = False
    num_windows = 500

    # get paths straight
    patientID = 'TS041'
    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(path, 'data', patientID)
    save_path = os.path.join(path, 'data', 'model_files')
    model_file = os.path.join(save_path, patientID + '.model')
    param_file = os.path.join(save_path, patientID + '.params')

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # create model file
    create_model_file(data_path, window_length, window_overlap, f_s, model_file, param_file, num_windows, include_awake, include_asleep)







