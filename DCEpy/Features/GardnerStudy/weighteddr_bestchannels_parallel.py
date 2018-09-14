__author__ = 'Chris'

import numpy as np
#import matplotlib.pyplot as plt
from sklearn import svm
from scipy import signal
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
import matplotlib.pyplot as plt

import array
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import multiprocessing
import math

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


def get_stats_extended(decision, T_per, seizure_time, f_s, win_len, win_overlap):

    # get decision array in time samples
    t_decision = window_to_sample(decision, win_len, win_overlap)

    FP_times=[]
    actual_FP=None

    # if inter-ictal file
    if seizure_time is None:

        # get the amount of time that is passed, false positive
        time = float(t_decision.size) / float(f_s * 60. * 60.)
        FP = 0.

        # while alarms are detected
        while np.any(t_decision > 0):

            # print "T Decision",t_decision

            # record false positive
            FP = FP + 1.0

            # find where it happens
            alarm_ind = np.argmax(t_decision > 0)
            per_ind = alarm_ind + T_per

            # record actual FP time
            if actual_FP == None:
                actual_FP = alarm_ind
            else:
                actual_FP+= per_ind

            FP_times.append(actual_FP)

            # print "T_per ",T_per


            # cut off the amount of persistance
            if per_ind > t_decision.size:
                break
            else:
                per_ind=int(per_ind)
                t_decision = t_decision[per_ind:]

        sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        latency = np.nan # latency is meaningless since there is no seizure

    else:

        # start time, end time, etc
        start = int(f_s * seizure_time[0])
        end = int(f_s* seizure_time[1])
        # pre_alert = max(0, start-T_per)
        pre_alert=0

        # initialize time and FP
        time = float(pre_alert) / float(f_s * 60. * 60.)
        FP = 0.

        # compute false positive rate (only on T_per time before the seizure)
        pre_alert=int(pre_alert)
        pre_seizure = np.copy(t_decision[:pre_alert])

        print "seizure time is not None, while loop starts!"

        while np.any(pre_seizure > 0):
            # print "pre_seizure",pre_seizure

            # record false positive
            FP = FP + 1.0

            # find where it happens
            alarm_ind = np.argmax(pre_seizure > 0)
            per_ind = alarm_ind + T_per
            per_ind=int(per_ind)
            print "in the seizure file, a False positive at window ",alarm_ind
            # print "T_per",T_per
            FP_times.append(alarm_ind)

            # cut off the amount of persistance
            if per_ind > pre_seizure.size:
                break
            else:
                # print "per_ind",per_ind
                pre_seizure = pre_seizure[per_ind:]
                # print "same?",pre_seizure == pre_seizure[per_ind:]
        # print "WHILE loop ends!"

        if not np.any(t_decision[pre_alert:end] > 0):
            sensitivity = 0.0 # seizure not detected
        elif not np.any(t_decision[pre_alert:start] > 0):
            sensitivity = 0.0 # seizure detected late
        else:
            sensitivity = 1.0 # seizure detected early

        # compute latency
        if np.any(t_decision[pre_alert:end] > 0):
            detect_time = np.argmax(t_decision[pre_alert:end] > 0) + pre_alert
            latency = float(detect_time - start) / float(f_s) # (in seconds)
        else:
            latency = np.nan

    # put time and FP as nan if there was no time
    if time <= 0.0:
        time = np.nan
        FP = np.nan

    return sensitivity, latency, FP, time,FP_times



def find_fpr(nu,gamma,C,adapt_rate,T_per, X_train, test_files, win_len, win_overlap, f_s, seizure_time):
    weight = [1]
    sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, weight, nu, gamma,
                                                       T_per[0], adapt_rate, win_len, win_overlap, f_s, seizure_time)

    # use the gardner objective function
    score = score_performance(sensitivity, latency, FP, time)
    FPR=np.nansum(FP) / np.nansum(time)

    print "Score of this good channel is ",score
    print "FPR of this good channel is ",FPR
    return


"""
energy_features()
Compute the energy statistics of a single window of data with many channels

INPUT:
    X --> ndarray
        time series data of size n x p

OUTPUT:
    Y --> ndarray
        feature vector of size 3 x p where entries are log
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
    Y = np.vstack((CL,E,TE))
    return Y



"""
collect_windows()
Initialize a one-class SVM for a single channel

INPUT:
    inter_ictal --> list of ndarray
        ndarrays are feature vectors of size n x p
    num_windows --> int
        number of windows to get (feature vectors)

OUTPUT:
    X --> ndarray
        size num_windows x 3
"""

def collect_windows(inter_ictal, num_windows):

    #  number of channels
    p = inter_ictal[0].shape[2]

    # get different sizes
    k = len(inter_ictal)
    size1 = int(np.floor(num_windows / k))
    size2 = int(np.floor(num_windows / k) + 1)

    # initialize sampled data(3D array)
    X = np.ones((num_windows,3,p))

    # collect sampled data
    for i in range(k):

        X_inter = inter_ictal[i]
        n = X_inter.shape[0]

        if (i <= (k - num_windows % k)):
            ind = np.random.choice(n, size=size1, replace=False)
            X[i:i+size1,:,:] = X_inter[ind,:,:]

        else:
            ind = np.random.choice(n, size=size2, replace=False)
            X[i:i+size2,:,:] = X_inter[ind,:,:]

        if np.any(np.isnan(X)):
                pass

    if np.any(np.isnan(X)):
            print '\tUh-oh, NaN encountered while collecting windows'

    # return the collected data
    return X




"""
init_one_svm()
Initialize a one-class SVM for a single channel

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

def init_one_svm(X_train, nu=0.1, gamma=1.0):

    if np.any(np.isnan(X_train)):
        print '\tUh-oh, NaN encountered in learn_support()'

    clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
    clf.fit(X_train)
    return clf





"""
anomaly_detection()
Run a single channel through one-class SVM. Return
the outlier fraction vector

INPUT:
    X_test --> ndarray
        testing feature vectors of size n x p
    clf --> one-class SVM model

    adapt_rate --> int
        number of past novelty sequence observations
        used in outlier fraction computation

OUTPUT:
    n_seq --> novelty sequence (SVM output)
    out_frac --> outlier fraction (smoothed SVM output)
"""

def anomaly_detection(X_test, clf, adapt_rate=40):

    adapt_rate = int(adapt_rate) # just to be sure...

    # label returned by one-class SVM
    n,p = X_test.shape
    n_seq = clf.predict(X_test)

    # change output so that +1 is anomalous and 0 is not anomalous
    # this is the NOVELTY SEQUENCE
    z = np.copy(n_seq)
    z[np.where(n_seq==-1)] = 1 # anomalous
    z[np.where(n_seq==1)] = 0 # not anomalous

    # small sliding window of size n
    # this is the OUTLIER FRACTION
    out_frac = np.zeros(n_seq.size)
    for i in np.arange(adapt_rate,n):
        out_frac[i] = np.mean(z[i-adapt_rate:i])

    return n_seq, out_frac

"""
decision_rule()
Use thresholds and novelty sequence and weights in
weighted decision rule. Ouputs weighted votes and
decision rule array.

INPUT:
    out_frac --> ndarray
        outlier fraction array of size num_windows x p
    C --> ndarray
        array of channel thresholds size p
    weight --> ndarray
        array of channel weights size p

OUTPUT:
    weighted_vote --> ndarray
        weighted vote array, size num_windows
    decision --> ndarray
        decision array. -1 is not anomalous, 1 is anomalous
"""
def decision_rule(out_frac, C, weight):

    # get shape of data
    n,p = out_frac.shape

    # shifting the outlier fractions to exist [-1,1]
    to_weight = 2.0 * (out_frac - 0.5)

    # threshold for flagging seizures
    weighted_threshold = 2.0 * (np.dot(C,weight) - 0.5)

    # decision rule for a weighted sum of outlier fractions, compared to a best-guess arbitrary threshold
    weighted_vote = np.dot(to_weight, weight)

    decision = np.sign(weighted_vote - weighted_threshold)

    # decision rule for a weighted sum of distance from learned threshold, compared to an arbitrary threshold
    # distance_from_threshold = out_frac - C
    # weighted_vote_distances = np.dot(distance_from_threshold, weight)
    # descision_distance_from_threshold = np.sign(weighted_vote_distances - arbitrary_threshold)

    return weighted_vote, decision

'''
window_to_sample()
There are several arrays that have one value per time window.
This function pads those to have values for each sample.

INPUT:
    win_array --> ndarray
        array with value for each window, size num_windows
    win_len  --> int
        length of window (samples)
    win_overlap --> int
        length of window overlap (samples)
OUTPUT:
    time_array --> ndarray
        array with values for each time sample (padded)
'''
def window_to_sample(win_array, win_len, win_overlap):

    win_array = np.array(win_array) # just to be sure...
    num_windows = win_array.size

    total_len = (num_windows-1)*(win_len - win_overlap) + (win_len - 1)
    time_array = -1 * np.ones(total_len)

    for i in range(num_windows):
        start = i*(win_len - win_overlap) + (win_len -1)
        end = start + (win_len - win_overlap)
        time_array[start:end] = win_array[i]

    return time_array

'''
get_stats()
Given decision array, output performance characteristics
such as sensitivity, fpr, sensitivity, latency.

INPUT:
    decision --> ndarray
        array with decision value for each window, size num_windows
    T_per --> int
        length of persistance time (samples)
    seizure_time --> tuple (if ictal) / None (otherwise)
        tuple that is (start time, end time) of ints (samples)
    f_s --> float
        sampling frequency
    win_len  --> int
        length of window (samples)
    win_overlap --> int
        length of window overlap (samples)
OUTPUT:
    sensitivity --> float
        sensitivity of decision
    latency --> float
        latency of decision (sec)
    FP --> float
        number of false positives
    time --> float
        time where FP could happen
'''

def get_stats(decision, T_per, seizure_time, f_s, win_len, win_overlap):

    # get decision array in time samples
    t_decision = window_to_sample(decision, win_len, win_overlap)

    # if inter-ictal file
    if seizure_time is None:

        # get the amount of time that is passed, false positive
        time = float(t_decision.size) / float(f_s * 60. * 60.)
        FP = 0.

        # while alarms are detected
        # print "seizure time is None, while loop starts!"
        while np.any(t_decision > 0):

            # print "T Decision",t_decision

            # record false positive
            FP = FP + 1.0

            # find where it happens
            alarm_ind = np.argmax(t_decision > 0)
            per_ind = alarm_ind + T_per


            # cut off the amount of persistance
            if per_ind > t_decision.size:
                break
            else:
                per_ind=int(per_ind)
                t_decision = t_decision[per_ind:]

        # print "seizure time is None, while loop ends!"

        sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        latency = np.nan # latency is meaningless since there is no seizure

    else:

        # start time, end time, etc
        start = int(f_s * seizure_time[0])
        end = int(f_s* seizure_time[1])
        pre_alert = max(0, start-T_per)

        # initialize time and FP
        time = float(pre_alert) / float(f_s * 60. * 60.)
        FP = 0.

        # compute false positive rate (only on T_per time before the seizure)
        pre_alert=int(pre_alert)
        pre_seizure = np.copy(t_decision[:pre_alert])

        # print "seizure time is not None, while loop starts!"

        while np.any(pre_seizure > 0):
            # print "pre_seizure",pre_seizure

            # record false positive
            FP = FP + 1.0

            # find where it happens
            alarm_ind = np.argmax(pre_seizure > 0)
            per_ind = alarm_ind + T_per
            per_ind=int(per_ind)
            # print "alarm ind",alarm_ind
            # print "T_per",T_per
            # cut off the amount of persistance
            if per_ind > pre_seizure.size:
                break
            else:
                # print "per_ind",per_ind
                pre_seizure = pre_seizure[per_ind:]
                # print "same?",pre_seizure == pre_seizure[per_ind:]
        # print "WHILE loop ends!"

        if not np.any(t_decision[pre_alert:end] > 0):
            sensitivity = 0.0 # seizure not detected
        elif not np.any(t_decision[pre_alert:start] > 0):
            sensitivity = 0.0 # seizure detected late
        else:
            sensitivity = 1.0 # seizure detected early

        # compute latency
        if np.any(t_decision[pre_alert:end] > 0):
            detect_time = np.argmax(t_decision[pre_alert:end] > 0) + pre_alert
            latency = float(detect_time - start) / float(f_s) # (in seconds)
        else:
            latency = np.nan

    # put time and FP as nan if there was no time
    if time <= 0.0:
        time = np.nan
        FP = np.nan

    return sensitivity, latency, FP, time

'''
performance_stats()
Given parameters, training data, and test data, this function
returns performance characteristics of the many one-class SVM
approach.

INPUT:
    X_train --> ndarray
        array with training data, size n, 3, p
    test_files --> list of ndarray (size test_num)
        list of test data each of size n, 3, p
    C --> ndarray
        array of channel thresholds, size p
    weight --> ndarray
        array of channel weights, size p
    nu --> ndarray
        outlier fraction, size p
    gamma --> ndarray
        coefficient for gaussian kernel, size p
    T_per --> int
        length of persistance time (samples)
    adapt_rate --> int
        number of past novelty sequence observations
        used in outlier fraction computation
    win_len  --> int
        length of window (samples)
    win_overlap --> int
        length of window overlap (samples)
    f_s --> float
        sampling frequency
    seizure_time --> list of tuple (if ictal) / None (otherwise)
        tuples are (start time, end time) of ints (samples)

OUTPUT:
    sensitivity --> ndarray
        sensitivity of decision, size test_num
    latency --> ndarray
        latency of decision (sec), size test_num
    FP --> ndarray
        number of false positives, size test_num
    time --> ndarray
        time where FP could happen, size test_num
'''
def performance_stats(X_train, test_files, C, weight, nu, gamma, T_per, adapt_rate, win_len, win_overlap, f_s, seizure_time):

    # initialize performance metrics
    file_num = len(test_files)
    sensitivity = np.empty(file_num)
    latency = np.empty(file_num)
    time = np.empty(file_num)
    FP = np.empty(file_num)

    # get number of channels
    p = X_train.shape[2]
    svm_list = []

    # train each SVM object
    for i in range(p):
        svm_list.append(init_one_svm(X_train[:,:,i], nu=nu[i], gamma=gamma[i]))

    # for each epoch
    for i, X_test in enumerate(test_files):

        # initialize outlier fraction for all channels
        n = X_test.shape[0]
        out_frac_total = np.empty((n,p))

        # for all channels, get outlier fraction
        for j in range(p):
            _, out_frac_total[:,j] = anomaly_detection(X_test[:,:,j], svm_list[j], adapt_rate[j])

        # decision rule
        weighted_vote, decision = decision_rule(out_frac_total, C, weight)

        # performance characteristics
        sensitivity[i], latency[i], FP[i], time[i] = get_stats(decision, T_per, seizure_time[i], f_s, win_len, win_overlap)

    # return performance characteristics
    return sensitivity, latency, FP, time

'''
score_performance()
This function accepts performance statistics and outputs
the value of the objective function defined by Gardner.

INPUT:
    sensitivity --> ndarray
        sensitivity of decision, size test_num
    latency --> ndarray
        latency of decision (sec), size test_num
    FP --> ndarray
        number of false positives, size test_num
    time --> ndarray
        time where FP could happen, size test_num
OUTPUT:
    score --> float
        the value of the objective function defined by Gardner
'''
def score_performance(sensitivity, latency, FP, time):

    # get aggregate statistics on performance metrics
    S = np.nanmedian(sensitivity)
    FPR = np.nansum(FP) / np.nansum(time)

    # if any seizure is detected
    if np.any(np.isfinite(latency)):
        mu = np.nanmedian(latency)
        # EDF = float(np.where(latency < 0)[0].size) / float(np.where(latency > -np.inf)[0].size)
        detected = latency[np.isfinite(latency)]
        EDF = float(np.where(detected < 0)[0].size) / float(detected.size)

    else:
        mu = 500.0 # else give terrible values
        EDF = 0.0

    # objective function
    desired_latency = -15.0
    allowable_fpr = 6.0
    alpha1 = 100*S-10*(1-np.sign(S-0.75))
    alpha2 = 20*EDF
    alpha3 = -10*FPR-20*(1-np.sign(allowable_fpr-FPR))
    alpha4 = min(30.0*(mu / desired_latency),30)
    # print "alpha 1",alpha1
    # print "alpha 2", alpha2
    # print "alpha 3", alpha3
    # print "alpha 4", alpha4

    score = alpha1+alpha2+alpha3+alpha4
    # score = alpha3 + alpha4
    # print "\t\tScore: %.2f\ta_3: %.2f\ta_4: %.2f"%(score, alpha3, alpha4)
    # print "FPR is ", FPR
    # print "score is", score
    return score

'''
fitness_fun()
This function is meant to be used in the genetic algorithm
for parameter tuning.

INPUT:
    see performance_stats()

OUTPUT:
    score --> float
        the value of the objective function defined by Gardner
'''
# def fitness_fun(individual, X_train, test_files, win_len, win_overlap, f_s, seizure_time):
#
#     # get parameters from individual
#     nu = individual[0::6]
#     gamma = individual[1::6]
#     C = individual[2::6]
#     threshold = C[0]
#     weight = individual[3::6]
#     adapt_rate = individual[4::6]
#     T_per = individual[5::6]
#
#     # get performance characteristics
#     sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, weight, nu, gamma,
#                                                        T_per[0], threshold, adapt_rate[0], win_len, win_overlap, f_s, seizure_time)
#     # use the gardner objective function
#     score = score_performance(sensitivity, latency, FP, time)
#     return score,


def find_fpr(nu,gamma,C,adapt_rate,T_per, X_train, test_files, win_len, win_overlap, f_s, seizure_time):
    weight = [1]
    sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, weight, nu, gamma,
                                                       T_per[0], adapt_rate, win_len, win_overlap, f_s, seizure_time)

    # use the gardner objective function
    score = score_performance(sensitivity, latency, FP, time)
    FPR=np.nansum(FP) / np.nansum(time)

    # print "Score of this good channel is ",score
    # print "FPR of this good channel is ",FPR
    return

def fitness_fun_single_channel(individual, X_train, test_files, win_len, win_overlap, f_s, seizure_time,T_per=180000):


    # get parameters from individual
    nu = [individual[0]]
    gamma = [individual[1]]
    C = [individual[2]]
    adapt_rate = [individual[3]]

    weight = [1]

    # get performance characteristics
    sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, weight, nu, gamma,
                                                       T_per, adapt_rate, win_len, win_overlap, f_s, seizure_time)

    # use the gardner objective function
    score = score_performance(sensitivity, latency, FP, time)
    return score,

def fitness_fun_weights(individual, best_genes_all_channels, X_train, test_files, win_len, win_overlap, f_s, seizure_time,T_per=180000):

    """The function returns the score for any weight assignment for all channels."""
    # get parameters from individual

    weights = individual
    nu = best_genes_all_channels[0::4]
    gamma = best_genes_all_channels[1::4]
    C = best_genes_all_channels[2::4]
    adapt_rate = best_genes_all_channels[3::4]
    # T_per = best_genes_all_channels[4::5]

    # get performance characteristics
    sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, weights, nu, gamma,
                                                       T_per, adapt_rate, win_len, win_overlap, f_s,
                                                       seizure_time)
    # use the gardner objective function
    score = score_performance(sensitivity, latency, FP, time)

    return score,



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
        print '\t\t\t-------Working on generation %d-------' % (g + 1)


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
            #print "\t\t\tMaximum fitness in the current population:  ", max(fits)
            #print "\t\t\tMean fitness in the current generation:  ", next_mean


    return best_species_genes,best_species_value,best_gen

def weights_tuning(best_genes_all_channels, best_fitness_all_channels, X_train, test_files,win_len, win_overlap, f_s, seizure_time, num_channels):

    """
    The funtion optimizes weights for all channels using genetic algorithm after pi is optimized within each channel and returns
    the list of weights.

    """

    # creating types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # defining an individual
    toolbox.register("individual", init_weights, creator.Individual, best_fitness_all_channels)

    # defining the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    MIN = np.tile(0,num_channels)
    toolbox.decorate("population", within_constraints_weights(MIN))

    # register the fitness function
    toolbox.register("evaluate", lambda x: fitness_fun_weights(x, best_genes_all_channels, X_train, test_files, win_len, win_overlap, f_s, seizure_time))

    # register the crossover operator
    # other options are: cxOnePoint, cxUniform (requires an indpb input, probably can just use CXPB)
    # there are others, more particular than these options
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.decorate("mate", within_constraints_weights(MIN))

    # register a mutation operator with a probability to mutate of 0.05
    # can change: mu, sigma, and indpb
    # there are others, more particular than this
    mu = np.tile(0,num_channels)
    sigma = np.tile(.05,num_channels)
    indpb = np.tile([.02], num_channels)
    toolbox.register("mutate", tools.mutGaussian, mu=mu.tolist(), sigma=sigma.tolist(), indpb=indpb.tolist())
    toolbox.decorate("mutate", within_constraints_weights(MIN))

    # operator for selecting individuals for breeding the next generation
    # other options are: tournament: randonly picks tournsize out of population, chosses fittest, and has that be
    # a parent. continues until number of parents is equal to size of population.
    # there are others, more particular than this
    toolbox.register("select", tools.selTournament, tournsize=4)
    # toolbox.register("select", tools.selRoulette)

    # create an initial population of size 20
    pop = toolbox.population(n=50)

    # CXPB  is the probability with which two individuals are crossed
    CXPB = 0.6

    # MUTPB is the probability for mutating an individual
    MUTPB = 0.25

    # NGEN  is the number of generations until final parameters are picked
    NGEN = 10

    # find the fitness of every individual in the population
    fitnesses = list(map(toolbox.evaluate, pop))

    # assigning each fitness to the individual it represents
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # defining variables to keep track of best individuals throughout species
    best_species_genes = tools.selBest(pop, 1)[0]
    # print "initiate indiv", best_species_genes
    best_species_value = best_species_genes.fitness.values
    best_gen = 0

    next_mean = 1
    prev_mean = 0

    # start evolution

    best_species_genes, best_species_value, best_species_gen = evolution(toolbox, pop, NGEN, prev_mean, next_mean,CXPB, MUTPB)
    print "\tAfter weighting all channels, this method has a fitness of ", best_species_value


    return best_species_genes


def ga_worker(X_train, test_files, win_len, win_overlap, f_s, seizure_time, num_channels, out_q, channel_start_ind):

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

    # print "number of channels!",num_channels
    #creating types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    #defining genes
    #ranges are given by Gardner paper

    MIN = [0.02, 0.25, 0.3, 10]
    MAX = [.2, 10, 1, 100]
    toolbox.register("attr_v", random.uniform, MIN[0], MAX[0])
    toolbox.register("attr_g", random.uniform, MIN[1], MAX[1])
    toolbox.register("attr_p", random.uniform, MIN[2], MAX[2])
    toolbox.register("attr_N", random.uniform, MIN[3], MAX[3])

    #defining an individual as a group of the five genes
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N), 1)

    #defining the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.decorate("population", within_constraints_pi(MIN,MAX))


    # register the crossover operator
    # other options are: cxOnePoint, cxUniform (requires an indpb input, probably can just use CXPB)
    # there are others, more particular than these options
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.decorate("mate", within_constraints_pi(MIN,MAX))

    # register a mutation operator with a probability to mutate of 0.05
    # can change: mu, sigma, and indpb
    # there are others, more particular than this
    mu = np.tile([0],4)
    sigma = np.tile([.06, 3.25, .23, 30],1)
    indpb = np.tile([.02],4)
    toolbox.register("mutate", tools.mutGaussian, mu=mu.tolist(), sigma=sigma.tolist(), indpb=indpb.tolist())
    toolbox.decorate("mutate", within_constraints_pi(MIN,MAX))

    # operator for selecting individuals for breeding the next generation
    # other options are: tournament: randonly picks tournsize out of population, chosses fittest, and has that be
    # a parent. continues until number of parents is equal to size of population.
    # there are others, more particular than this
    toolbox.register("select", tools.selTournament, tournsize=2)
    # toolbox.register("select", tools.selRoulette)

    # CXPB  is the probability with which two individuals are crossed
    CXPB = 0.6

    # MUTPB is the probability for mutating an individual
    MUTPB = 0.25

    # NGEN  is the number of generations until final parameters are picked
    NGEN = 5


    # keeps track of the channels that have fitness above 80
    store_ind = channel_start_ind
    outdict = {}

    for channel in range(num_channels):

        # print '\t\tTuning pi for channel %d' % (channel+1)

        # create an initial population of size 20
        pop = toolbox.population(n=20)

        # find X_channel (the channel layer in X_train)
        num_windows=X_train.shape[0]
        para=X_train.shape[1]

        X_channel=np.ones([num_windows,para,1])

        for i in range(num_windows):
            for j in range(para):
                X_channel[i][j][0]=X_train[i][j][channel]


        # register the fitness function
        toolbox.register("evaluate",lambda x: fitness_fun_single_channel(x, X_channel, test_files, win_len, win_overlap, f_s, seizure_time))

        # find the fitness of every individual in the population
        fitnesses = list(map(toolbox.evaluate, pop))

        #assigning each fitness to the individual it represents
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit


        next_mean=1
        prev_mean=0

        #start evolution
        best_species_genes,best_species_value,best_gen=evolution(toolbox, pop, NGEN, prev_mean,next_mean, CXPB, MUTPB)

        print "\t\tThis channel has a fitness of: ", best_species_value[0]

        outdict[store_ind,0] = np.asarray(best_species_genes)
        outdict[store_ind,1] = np.asarray(best_species_value)

        store_ind+=1

    out_q.put(outdict)

    return




def parameter_tuning_1(X_train, test_files, win_len, win_overlap, f_s, seizure_time, num_channels):

    nprocs = 2

    # get data dimensions
    n,p,k = X_train.shape

    # Each process will get 'chunksize' samples and
    # a queue to put his out dict into
    out_q = multiprocessing.Queue()
    channels_per_proc = int(math.ceil(k / float(nprocs)))

    procs = []

    print "Entered parameter_tuning_1, sending data to procs"
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
            args=(X_c, test_files, win_len, win_overlap, f_s, seizure_time, X_c.shape[2], out_q, start)
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


    #set all T_per to the max_T
    T_per_secs=180
    T_pers = [T_per_secs*1000] * num_channels

    # print best_fitness_selected_channels

    #for i in xrange(len(good_channels)):
    #    find_fpr(v_s[i],g_s[i],p_s[i],adapt_rates[i],T_pers[i],X_train[:,:,good_channels],test_files, win_len, win_overlap, f_s, seizure_time)

    print "Tuning weights for all channels"
    weights = weights_tuning(best_genes_selected_channels, best_fitness_selected_channels, X_train,
    test_files, win_len,win_overlap, f_s, seizure_time, num_channels)

    #will return v for all channels, g for all channels, p for all channels, adapt_rate for all channels, and Tper for all channels
    return  v_s, g_s, p_s, adapt_rates,T_pers,weights, sum(best_fitness_selected_channels)



# def parameter_tuning(X_train, test_files, win_len, win_overlap, f_s, seizure_time, num_channels=145):
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
#     MIN = np.tile([0.02, 0.25, 0.3, 0, 10, t_per_min],num_channels)
#     MAX = np.tile([.2, 10, 1, 1, 100, t_per_max],num_channels)
#     toolbox.register("attr_v", random.uniform, MIN[0], MAX[0])
#     toolbox.register("attr_g", random.uniform, MIN[1], MAX[1])
#     toolbox.register("attr_p", random.uniform, MIN[2], MAX[2])
#     toolbox.register("attr_w", random.uniform, MIN[3], MAX[3])
#     toolbox.register("attr_N", random.uniform, MIN[4], MAX[4])
#     toolbox.register("attr_T", random.uniform, MIN[5], MAX[5])
#
#     #defining an individual as a group of the four genes
#     toolbox.register("individual", tools.initCycle, creator.Individual,
#                      (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_w, toolbox.attr_N, toolbox.attr_T), num_channels)
#
#     #defining the population as a list of individuals
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.decorate("population", within_constraints(MIN,MAX))
#
#     # register the fitness function
#     toolbox.register("evaluate", lambda x: fitness_fun(x, X_train, test_files, win_len, win_overlap, f_s, seizure_time))
#
#     # register the crossover operator
#     # other options are: cxOnePoint, cxUniform (requires an indpb input, probably can just use CXPB)
#     # there are others, more particular than these options
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.decorate("mate", within_constraints(MIN,MAX))
#
#     # register a mutation operator with a probability to mutate of 0.05
#     # can change: mu, sigma, and indpb
#     # there are others, more particular than this
#     mu = (MAX+MIN)/2
#     sigma = (MAX-MIN)/3
#     indpb = np.tile([.01],num_channels*6)
#     toolbox.register("mutate", tools.mutGaussian, mu.tolist(), sigma.tolist(), indpb.tolist())
#     toolbox.decorate("mutate", within_constraints(MIN,MAX))
#
#     # operator for selecting individuals for breeding the next generation
#     # other options are: tournament: randonly picks tournsize out of population, chosses fittest, and has that be
#     # a parent. continues until number of parents is equal to size of population.
#     # there are others, more particular than this
#     toolbox.register("select", tools.selTournament, tournsize=3)
#     #toolbox.register("select", tools.selRoulette)
#
#     #create an initial population of size 20
#     pop = toolbox.population(n=20)
#
#     # CXPB  is the probability with which two individuals are crossed
#     CXPB = 0.3
#
#     # MUTPB is the probability for mutating an individual
#     MUTPB = 0.5
#
#     # NGEN  is the number of generations until final parameters are picked
#     NGEN = 40
#
#     # find the fitness of every individual in the population
#     fitnesses = list(map(toolbox.evaluate, pop))
#
#     #assigning each fitness to the individual it represents
#     for ind, fit in zip(pop, fitnesses):
#         ind.fitness.values = fit
#
#     #defining variables to keep track of best indivudals throughout species
#     best_species_genes = tools.selBest(pop, 1)[0]
#     best_species_value = best_species_genes.fitness.values
#     best_gen = 0
#
#     next_mean=1
#     prev_mean=0
#
#     #start evolution
#     for g in range(NGEN):
#         print '\t\tWorking on generation %d' %(g+1)
#         if abs( next_mean - prev_mean ) > 0.5 :
#
#             prev_mean=next_mean
#             # Select the next generation's parents
#             parents = toolbox.select(pop, len(pop))
#
#             # Clone the parents and call them offspring: crossover and mutation will be performed below
#             offspring = list(map(toolbox.clone, parents))
#
#             # Apply crossover to children in offspring with probability CXPB
#             for child1, child2 in zip(offspring[::2], offspring[1::2]):
#                 # cross two individuals with probability CXPB
#                 if random.random() < CXPB:
#                     toolbox.mate(child1, child2)
#                     del child1.fitness.values
#                     del child2.fitness.values
#
#             # Apply mutation to children in offspring with probability MUTPB
#             for mutant in offspring:
#                 if random.random() < MUTPB:
#                     toolbox.mutate(mutant)
#                     del mutant.fitness.values
#
#             # Find the fitnessess for all the children for whom fitness changed
#             invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#             fitnesses = map(toolbox.evaluate, invalid_ind)
#             for ind, fit in zip(invalid_ind, fitnesses):
#                 ind.fitness.values = fit
#
#             # Offspring becomes the new population
#             pop[:] = offspring
#
#             #updating best species values
#             #if max(fitnesses):
#             if max(fitnesses) > best_species_value:
#                 best_species_genes = tools.selBest(pop, 1)[0]
#                 best_species_value = best_species_genes.fitness.values
#                 best_gen = g
#             #best_next_obj = max(fitnesses)
#
#             fits = [ind.fitness.values[0] for ind in pop]
#             length = len(pop)
#             next_mean = sum(fits) / length
#
#
#     best_ind_finalgen = tools.selBest(pop, 1)[0]
#
#     # print("Best individual in final population is %s with fitness value %s" % (best_ind_finalgen, best_ind_finalgen.fitness.values))
#     print("Best individual in species occurred during generation %s with fitness %s" %(best_gen,best_species_value))
#
#     #will return v for all channels, g for all channels, p for all channels, weights for all channels, adapt_rate for all channels, and Tper for all channels
#     return best_species_genes[0::6], best_species_genes[1::6], best_species_genes[2::6], best_species_genes[3::6], best_species_genes[4::6], best_species_genes[5::6]


def create_model_file(model_file, clf, nu, gamma):

    p_feat = 3 # true for energy statistics
    num_SV = clf.support_.size

    # create model file
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
    return

def create_param_file(param_file, C, weight, adapt_rate, num_channels):

    # write other parameters file
    f = open(param_file, 'w')
    f.write('channel_no: %d\n' %num_channels)

    # f.write('adapt_rate: %d\n' % adapt_rate)
    # f.write('adapt_rate: %s\n' %str(adapt_rate))
    f.write('channel: adapt rate: threshold: weight\n')
    for i in range(num_channels):
        print "which channel?",i
        f.write("%d: %.4f: %.4f: %.4f\n" %(i,adapt_rate[i],C[i],weight[i]))
    f.close()
    return

def create_filtered_csv(file_name, X, good_channel_ind):

    # normalize the filtered data
    n,p = X.shape
    X = (X - np.mean(X, axis=0) ) / np.std(X, axis=0)

    # print headers
    f = open(file_name, 'w')
    str = ''
    for j in range(len(good_channel_ind)):
        str += "CH%d, " %j
    f.write(str[:-1] + '\n')

    # print data
    for i in range(n):
        str = ''
        for j in good_channel_ind:
            str += '%.6f,' %(X[i,j])
        f.write(str[:-1] + '\n')
    f.close()
    return

def create_energy_csv(file_name, X):

    n,_ = X.shape

    # print headers
    f = open(file_name, 'w')
    header = 'Curve Length, Average Energy, Teager Energy\n'
    f.write(header)

    # print energy statistics
    for i in range(n):
        str = '%.6f,%.6f,%.6f\n' %(X[i,0], X[i,1], X[i,2])
        f.write(str)
    f.close()
    return

def create_decision_csv(file_name, decision):

    n = decision.size

    # print header
    f = open(file_name, 'w')
    f.write('Decision\n')

    # print decision
    for i in range(n):
        f.write('%.3f\n'%float(decision[i]))
    f.close()
    return

def write_energy_file(file_path, X_feat):

    n,_,p = X_feat.shape
    f = open(file_path, 'w')
    for i in range(n):
        str = ''
        for j in range(p):
            str += "%.6f,%.6f,%.6f," %(X_feat[i,0,j], X_feat[i,1,j], X_feat[i,2,j])
        f.write(str[:-1] + "\n")
    f.close()
    return

def read_energy_file(file_path):

    # get data dimensions
    n = sum(1 for row in csv.reader( open(file_path) ) )
    p = len(next( csv.reader( open(file_path) ) ) ) / 3
    X_feat = np.empty((n,3,p))
    # open up reader
    f = open(file_path, 'rb')
    reader = csv.reader(open(file_path, 'rb'))

    for i,row in enumerate(reader):
        a = np.array([float(b) for b in row])
        a = np.reshape(a, (3,p), 'F')
        X_feat[i,:,:] = a

    return X_feat, p

def update_log(log_file, patient_id, sensitivity, latency, FP, time):

    # print to results file
    f = open(log_file, 'a')
    f.write('\nPatient ' + patient_id + '\n=========================\n')

    # print the results -- aggregates and total
    f.write('Mean Sensitivity: \t%.2f\n' %(np.nanmean(sensitivity)))
    f.write('Mean Latency: \t%.4f\n' %(np.nanmean(latency)))
    f.write('False Positive Rate: \t%.5f (fp/Hr) \n' % (np.nansum(FP) / np.nansum(time)))

    f.write('Sensitivity: ' + str(sensitivity) + '\n')
    f.write('Latency: ' + str(latency) + '\n')
    f.write('False Positives: ' + str(FP) + '\n')
    f.write('FP Time: ' + str(time) + '\n' + '\n')
    f.close()
    return

def update_list(original, update):
    count = 0
    for a in update:
        place = a[0] + count
        original = original[:place] + a[1] + original[place+1:]
        count += len(a[1]) - 1
    return original


def patient_best_channels(patient_id):
    """
    Returns a list of focus channel indices for each patient, selected by Rakesh's method
    :param patient_id:
    :return:
    """

    # based on Rakesh's paper, focus channels for each patient
    map = {"TS041":[1,2,3,14,15],"TS039":[79,80,81,96,97,98]} # needs patient 23

    best_channels = map[patient_id]

    return best_channels



def reduce_channels(all_files_old,chosen_channels):

    """
    reduce channels to only include the "good" ones.

    :param all_files: a list of ndarrays, each of which has shape (number of windows, 3, number of all channels)
    :param chosen_channels:  a list of good channel indices(ints)
    :return: all data with only the "good channels" info selected. A list of arrays, each of which has shape
            (number of windows, 3, number of good channels)

    """

    all_files = []
    for file in all_files_old:
        all_files.append(file[:,:, chosen_channels]) # selects best channel data for every file

    return all_files




def loocv_testing(all_files, data_filenames, file_type, seizure_times, seizure_print, win_len, win_overlap, num_windows, f_s, save_path):

    # num_channels = all_files[0].shape[2]
    file_num = len(all_files)

    # initialize performance metrics
    sensitivity = np.empty(file_num)
    latency = np.empty(file_num)
    time = np.empty(file_num)
    FP = np.empty(file_num)

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

        num_channels = X_train.shape[2]

        # parameter tuning
        print '\tChannel SVM parameter tuning with genetic algorithm'
        nu, gamma, C, adapt_rate, T_per, weights, sum_fitness = parameter_tuning_1(X_train, cv_test_files, win_len,
                                                                                   win_overlap, f_s, cv_seizure_times,
                                                                                   num_channels)

        # train best channel SVMs
        print '\tTraining the SVMs with optimal parameters'
        svm_list = []
        for j in range(num_channels):
            svm_list.append(init_one_svm(X_train[:,:,j], nu=nu[j], gamma=gamma[j]))


        # test result from SVM
        print '\tAnomaly detection'
        n = X_test.shape[0]
        out_frac_total = np.empty((n,num_channels))
        for j in range(num_channels):
            _, out_frac_total[:,j] = anomaly_detection(X_test[:,:,j], svm_list[j], adapt_rate[j])


        # decision rule and performance characteristics
        weighted_vote, decision = decision_rule(out_frac_total, C, weights)
        # sensitivity[i], latency[i], FP[i], time[i] = get_stats(decision, T_per[0], seizure_times[i], f_s, win_len, win_overlap)
        sensitivity[i], latency[i], FP[i], time[i], FP_times = get_stats_extended(decision, T_per[j], seizure_times[i], f_s,
                                                                      win_len, win_overlap)


        print '\tSensitivity %.2f\tLatency %.3f\tFP %.2f\tTime %.2f' %(sensitivity[i], latency[i], FP[i], time[i])

        # plot test result
        plot_outlier(patient_id, out_frac_total[:, j], C[j], file_path=
        str(i + 1) + "fold " + "test file " + str(i + 1) + "channel" + str(
            None) + ".png", FP_times=FP_times, seizure_time=seizure_times[i], T_per=T_per[j])


        # write decision to csv file if it's the desired seizure
        if seizure_print[i]:
            print '\tWriting data to file'
            file_begin = os.path.join(save_path, os.path.splitext(os.path.basename(data_filenames[i]))[0])

            # print weighted votes file
            votes_file = file_begin + '_votes.csv'
            create_decision_csv(votes_file, weighted_vote)

            # print param file
            param_file = file_begin + '.params'
            create_param_file(param_file, C, weights, adapt_rate, num_channels)

            # create model files
            for j in range(num_channels):
                model_file = file_begin + "_%d.model"%(j)
                create_model_file(model_file, svm_list[j], nu[j], gamma[j])

    return sensitivity, latency, FP, time




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


    # selects the best channels for this patient based on Rakesh's paper
    good_channel_indices = patient_best_channels(patient_id)
    all_files = reduce_channels(all_files,good_channel_indices)


    # leave one out cross validation, update log
    sensitivity, latency, FP, time = loocv_testing(all_files, data_filenames, file_type, seizure_times, seizure_print, win_len, win_overlap, num_windows, f_s, save_path)
    update_log(log_file, patient_id, sensitivity, latency, FP, time)

    return

if __name__ == '__main__':

    # parameters -- sampling data
    win_len = 1.0 # in seconds
    win_overlap = 0.5 # in seconds
    num_windows = 1000 # number of windows to sample to build svm model
    f_s = float(1e3) # sampling frequency
    include_awake = True
    include_asleep = False

    # list of patients that will be used
    patients = ['TS041']
    long_interictal = [False]

    # get the paths worked out
    # to_data = '/scratch/smh10/DICE'
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')
    save_path = os.path.join(data_path, 'energy_stats')

    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # create results file
    log_file = os.path.join(save_path, 'log_file.txt')
    f = open(log_file, 'w')

    # write the first few lines
    f.write("Results file for Weighted Decision One-Class SVM Anomaly Detection\n")
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
