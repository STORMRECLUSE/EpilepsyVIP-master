import math
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from DCEpy.Features.AnalyzePatient import analyze_patient_raw
from deap import base
from deap import creator
from deap import tools
from scipy.signal import csd
from scipy.signal import welch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from DCEpy.Features.BurnsStudy import rstat_42
from DCEpy.Features.feature_functions import burns_features


def build_offline_feature_dict(patient_id, data_filenames, all_files, win_len, win_overlap):

    feat_matrices = {}

    for i, filename in enumerate(data_filenames):
        X_feat = burns_features(patient_id, filename)
        feat_matrices[filename] = X_feat.copy()

    return feat_matrices

def hand_label_seizuretimes(data_mat_dict, filetypes, seizuretimes, filenames, win_len, win_overlap):

    label_dict = {}

    for i, filename in enumerate(filenames):
        datamat = data_mat_dict[filename]

        n,p = datamat.shape
        labels = np.zeros(n) #label is 0 if not seizure

        file_seizure_times = seizuretimes[i]

        if filetypes[i] is "ictal":

            seizure_start_time = file_seizure_times[0]
            seizure_end_time = file_seizure_times[1]

            seizure_start_window = int(seizure_start_time / (win_len - win_overlap))
            seizure_end_window = int(seizure_end_time / (win_len - win_overlap))

            #TODO: Figure out what labeling method works best for random forests
            labels[seizure_start_window:seizure_end_window] = np.ones(seizure_end_window-seizure_start_window)

        label_dict[filename] = labels

    return label_dict

def random_forest_feature_selector(feature_matrix, labels, feature_names):

    rf = RandomForestClassifier()
    rf.fit(feature_matrix, labels)
    # print "\t\tFeatures sorted by their score according to random forests & hand labeled data:"
    # print "\t\t", sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)
    return rf.feature_importances_, feature_names

def choose_best_channels(data_dict, label_dict, filenames):

    num_training_files = len(filenames)
    all_dims_to_keep = {}

    all_importances = np.zeros((data_dict[filenames[0]].shape[1],1))

    for i in xrange(num_training_files):

        data = np.vstack([data_dict[key] for key in filenames[:i]+filenames[i+1:]])
        labels = np.hstack([label_dict[key2] for key2 in filenames[:i]+filenames[i+1:]])

        num_feats = data.shape[1]
        feature_names = np.arange(0,num_feats,1)
        #TODO: is there a better feature selector than RF?
        print '\tRandom forests feature selection'
        importances, dimensions = random_forest_feature_selector(data, labels, feature_names)

        for m in dimensions:
            all_importances[m] += importances[m]

    dimensions_to_keep = np.flipud(sorted(range(len(all_importances)),key=lambda x:all_importances[x]))
    dimensions_to_keep = dimensions_to_keep[0:10]

    return dimensions_to_keep

def construct_coherency_matrix(X, f_s, freq_band):

    n,p = X.shape

    # initialize the adjacency matrix
    A = np.zeros((p,p))

    # construct adjacency matrix
    for i in range(p):
        for j in range(i+1,p):
            #TODO: try methods other than coherence to build graph
            fxy, Pxy = csd(X[:,i], X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)
            fxx, Pxx = welch(X[:,i], fs = f_s, nperseg = 1000, noverlap = 500)
            fyy, Pyy = welch(X[:,j], fs = f_s, nperseg = 1000, noverlap = 500)
            Pxy_band = np.mean([Pxy[n] for n in xrange(len(Pxy)) if fxy[n] <= freq_band[1] and fxy[n] >= freq_band[0]])
            Pxx_band = np.mean([Pxx[n] for n in xrange(len(Pxx)) if fxx[n] <= freq_band[1] and fxx[n] >= freq_band[0]])
            Pyy_band = np.mean([Pyy[n] for n in xrange(len(Pyy)) if fyy[n] <= freq_band[1] and fyy[n] >= freq_band[0]])
            c = abs(Pxy_band)**2/(Pxx_band*Pyy_band)
            A[i,j] = c # upper triangular part
            A[j,i] = c # lower triangular part

    # return adjacency matrix
    return A

def build_coherency_array(raw_data, win_len, win_ovlap, f_s, choose_random, num_desired_windows, freq_band):

    data = raw_data.copy()

    n, p = data.shape

    win_len = win_len * f_s
    win_ovlap = win_ovlap * f_s

    num_windows = int( math.floor( float(n) / float(win_len - win_ovlap)) )

    #TODO: could train on a larger data set
    if choose_random and num_desired_windows<num_windows:
        coherency_array = np.zeros((p,p,num_desired_windows))
        window_indices = np.random.choice(num_windows, num_desired_windows, replace=False)
    else:
        coherency_array = np.zeros((p,p,num_windows))
        window_indices = np.arange(num_windows)

    # go through each window
    for spot, index in enumerate(window_indices):
        # print'\t\tCoherency matrix for window', spot
        # isolate time window from seizure data
        start = index*(win_len - win_ovlap)
        end = min(start+win_len, n)
        window_of_data = data[start:end,:] # windowed data
        coherency_array[:,:,spot] = construct_coherency_matrix(window_of_data, f_s, freq_band)

    return coherency_array

def find_normalizing_coherency(matrix):
    mean_mat = np.mean(matrix, axis=2)
    std_mat = np.std(matrix, axis=2)
    return mean_mat, std_mat

def transform_coherency(matrices, mean, std):
    #TODO: try with and without normalizing: does it make a big difference?
    std[std == 0] = 0.001

    if np.ndim(matrices)>2:
        num_matrices = matrices.shape[2]
        for i in xrange(num_matrices):
            matrix = matrices[:,:,i].copy()
            matrix -= mean
            matrix = np.divide(matrix, std)
            matrix = np.divide(np.exp(matrix), 1+ np.exp(matrix))
            matrices[:,:,i] = matrix
    else:
        matrices -= mean
        matrices = np.divide(matrices, std)
        matrices = np.divide(np.exp(matrices), 1+ np.exp(matrices))

    return matrices

def find_evc(matrix):
    if np.ndim(matrix)>2:
        centrality = np.zeros((matrix.shape[2],matrix.shape[1]))
        for i in xrange(matrix.shape[2]):
            # print'\t\tFinding eigenvector centrality for window', i
            sub_matrix = matrix[:,:,i].copy()
            G = nx.Graph(sub_matrix)
            evc = nx.eigenvector_centrality(G, max_iter=500)
            centrality[i,:] = np.asarray(evc.values())
    else:
        G = nx.DiGraph(matrix)
        evc = nx.eigenvector_centrality(G)
        centrality = np.asarray(evc.values())
    return centrality

def train_oneclass_svm(traindat, nu, gamma):
    clf = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    clf.fit(traindat)
    return clf

def create_outlier_frac(testdat, model, adaptrate):

    novelty_seq = model.predict(testdat)
    z = np.copy(novelty_seq)
    z[np.where(novelty_seq==-1)] = 1 # anomalous
    z[np.where(novelty_seq==1)] = 0 # not anomalous
    n,p = testdat.shape
    out_frac = np.zeros(novelty_seq.size)
    for i in np.arange(adaptrate,n):
        out_frac[i] = np.mean(z[i-adaptrate:i])
    return np.asarray(out_frac)

def window_to_sample(array_in_windows, win_len, win_overlap, f_s):

    win_len = int(win_len * f_s)
    win_overlap = int(win_overlap * f_s)

    win_array = np.array(array_in_windows) # just to be sure...
    num_windows = win_array.size

    total_len = (num_windows-1)*(win_len - win_overlap) + (win_len - 1)
    time_array = -1 * np.ones(total_len)

    for i in range(num_windows):
        start = i*(win_len - win_overlap) + (win_len -1)
        end = start + (win_len - win_overlap)
        time_array[start:end] = win_array[i]

    return time_array

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
    min_desired_latency = -15.0
    #TODO: change objective function?
    allowable_fpr = 6.0
    alpha1 = 100*S-10*(1-np.sign(S-0.75))
    alpha2 = 20*EDF
    alpha3 = -10*FPR-20*(1-np.sign(allowable_fpr-FPR))
    alpha4 = np.minimum(30.0*(mu / min_desired_latency),30)
    score = alpha1+alpha2+alpha3+alpha4

    return score

def get_stats(decision, T_per, seizure_time, f_s, win_len, win_overlap):

    # get decision array in time samples
    t_decision = window_to_sample(decision, win_len, win_overlap, f_s)

    # if inter-ictal file
    if seizure_time is None:

        # get length in units of hours of the file
        time = float(t_decision.size) / float(f_s * 60. * 60.)
        FP = 0.

        # while alarms are detected
        # print "seizure time is None, while loop starts!"
        while np.any(t_decision > 0):

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

        sensitivity = np.nan # sensitivity is meaningless since no true positives (no seizure)
        latency = np.nan # latency is meaningless since there is no seizure

    else:

        # start time, end time, etc
        start = int(f_s * seizure_time[0])
        end = int(f_s* seizure_time[1])
        #TODO: find reasonable length of time prior to seizure to call a FP (change pre-alert)
        # pre_alert = max(0, start-T_per)
        pre_alert = 0

        # initialize time and FP
        time = float(pre_alert) / float(f_s * 60. * 60.) #time in which false positives COULD have occured
        FP = 0.

        # compute false positive rate (only on T_per time before the seizure)
        pre_alert=int(pre_alert)
        pre_seizure = np.copy(t_decision[:pre_alert])

        # print "seizure time is not None, while loop starts!"

        #TODO: this loop is being ignored because the preseizure FP period is being ignored; need to fix
        # while np.any(pre_seizure > 0):
        #     # print "pre_seizure",pre_seizure
        #
        #     # record false positive
        #     FP = FP + 1.0
        #
        #     # find where it happens
        #     alarm_ind = np.argmax(pre_seizure > 0)
        #     per_ind = alarm_ind + T_per
        #     per_ind=int(per_ind)
        #     # print "alarm ind",alarm_ind
        #     # print "T_per",T_per
        #     # cut off the amount of persistance
        #     if per_ind > pre_seizure.size:
        #         break
        #     else:
        #         # print "per_ind",per_ind
        #         pre_seizure = pre_seizure[per_ind:]
        #         # print "same?",pre_seizure == pre_seizure[per_ind:]
        # # print "WHILE loop ends!"

        if not np.any(t_decision[pre_alert:end] > 0):
            sensitivity = 0.0 # seizure not detected
        elif not np.any(t_decision[pre_alert:start] > 0):
            sensitivity = 0.5 # seizure detected late
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

def performance_stats(X_train, test_files, C, nu, gamma, T_per, adapt_rate, win_len, win_overlap, f_s, test_seizure_times):

    # initialize performance metrics
    file_num = len(test_files)
    sensitivity = np.empty(file_num)
    latency = np.empty(file_num)
    time = np.empty(file_num)
    FP = np.empty(file_num)

    #TODO: we're testing on our training data during this parameter tuning stage-->overfitting?
    svm_model = train_oneclass_svm(X_train, nu, gamma)

   # for each test file
    for i, X_test in enumerate(test_files):

        outlier_frac = create_outlier_frac(X_test, svm_model, adapt_rate)
        decision = np.sign(outlier_frac - C)

        # performance metrics
        sensitivity[i], latency[i], FP[i], time[i] = get_stats(decision, T_per, test_seizure_times[i], f_s, win_len, win_overlap)

    # return performance characteristics
    return sensitivity, latency, FP, time

def fitness(individual, X_train, test_files, win_len, win_overlap, f_s, seizure_time):

    # get parameters from individual
    nu = individual[0]
    gamma = individual[1]
    C = individual[2]
    adapt_rate = individual[3]
    T_per = individual[4]

    # get performance characteristics
    sensitivity, latency, FP, time = performance_stats(X_train, test_files, C, nu, gamma,
                                                       T_per, adapt_rate, win_len, win_overlap, f_s, seizure_time)

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

def evolution(toolbox, pop, NGEN, prev_mean, next_mean, CXPB, MUTPB):

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

            # Offspring with the best individuals in the past become the new population
            # population = tools.selBest(offsprings + hall_of_fame, len(population))
            #TODO: figure out elitism concept?

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
            print "\t\t\tMaximum fitness in the current population:  ", max(fits)
            print "\t\t\tMean fitness in the current generation:  ", next_mean

    return best_species_genes,best_species_value,best_gen

def train_pi(X_train, test_files, win_len, win_overlap, f_s, test_seizure_times):

    #TODO: metaoptimizer to choose GA parameters
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    #defining genes
    #ranges are chosen; Gardner paper no longer applies
    #TODO: consider ranges more carefully
    t_per_min = 10*f_s
    t_per_max = 240*f_s
    MIN = [0.02, 0.01, 0.3, 10, t_per_min]
    MAX = [.5, 10, 1, 100, t_per_max]
    toolbox.register("attr_v", random.uniform, MIN[0], MAX[0])
    toolbox.register("attr_g", random.uniform, MIN[1], MAX[1])
    toolbox.register("attr_p", random.uniform, MIN[2], MAX[2])
    toolbox.register("attr_N", random.uniform, MIN[3], MAX[3])
    toolbox.register("attr_T", random.uniform, MIN[4], MAX[4])

    #defining an individual as a group of the five genes
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N, toolbox.attr_T), 1)

    #defining the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.decorate("population", within_constraints_pi(MIN,MAX))


    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.decorate("mate", within_constraints_pi(MIN,MAX))

    # register a mutation operator with a probability to mutate of 0.05
    #TODO: choose these mu and sigma better
    mu = np.tile([0],5)
    sigma = np.tile([.06, 3.25, .23, 30, 63.33*f_s],1)
    indpb = np.tile([.2],5)
    toolbox.register("mutate", tools.mutGaussian, mu=mu.tolist(), sigma=sigma.tolist(), indpb=indpb.tolist())
    toolbox.decorate("mutate", within_constraints_pi(MIN,MAX))

    # operator for selecting individuals for breeding the next generation
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selRoulette)

    # CXPB  is the probability with which two individuals are crossed
    CXPB =.6

    # MUTPB is the probability for mutating an individual
    MUTPB = 0.5

    # NGEN  is the number of generations until final parameters are picked
    NGEN = 50

    # initiate lists to keep track of the best genes(best pi) and best fitness scores for everychannel
    best_genes_all_channels =[]
    best_fitness_all_channels=[]

    # create an initial population of size 20
    pop = toolbox.population(n=30)

    # register the fitness function
    toolbox.register("evaluate",lambda x: fitness(x, X_train, test_files, win_len, win_overlap, f_s, test_seizure_times))

    # find the fitness of every individual in the population
    fitnesses = list(map(toolbox.evaluate, pop))

    #assigning each fitness to the individual it represents
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    next_mean=1
    prev_mean=0

    #start evolution
    best_species_genes,best_species_value,best_gen = evolution(toolbox, pop, NGEN, prev_mean,next_mean, CXPB, MUTPB)

    print 'Final fitness score of training is: ', best_species_value

    nu = best_species_genes[0]
    gamma = best_species_genes[1]
    C = best_species_genes[2]
    adapt_rate = best_species_genes[3]
    t_per = best_species_genes[4]

    #TODO: run a test, see if you get the same pi parameters for every fold of the CV given a patient

    return nu, gamma, C, adapt_rate, t_per

def outlier_frac_viz(traindat, testdat, seizure_times, seizurename, win_len, win_overlap):

    # clf = svm.OneClassSVM(kernel='rbf', nu=0.1)
    # clf.fit(traindat)
    # novelty_seq = clf.predict(testdat)
    # z = np.copy(novelty_seq)
    # z[np.where(novelty_seq==-1)] = 1 # anomalous
    # z[np.where(novelty_seq==1)] = 0 # not anomalous
    # n,p = testdat.shape
    #
    # myrange = np.arange(20,70,5)
    #
    # fig, axes = plt.subplots(5,2)
    # row=0
    # col=0
    # print'\tEntering plotting process'
    # for myadapt in myrange:
    #
    #     out_frac = np.zeros(novelty_seq.size)
    #
    #     for iii in np.arange(myadapt,n):
    #         out_frac[iii] = np.mean(z[iii-myadapt:iii])
    #
    #     axes[row, col].plot(out_frac)
    #     axes[row, col].set_ylim(ymin=0, ymax=1)
    #     relevant_times = seizure_times
    #     if relevant_times is not None:
    #         axes[row, col].axvline(x=relevant_times[0]/(win_len-win_overlap), lw=3, color='r')
    #     axes[row, col].set_title('%f'%myadapt)
    #
    #     row+=1
    #     if row>4:
    #         row=0
    #         col=1
    # plt.show()

    myrange = np.arange(0.05,.6,.05)
    fig, axes = plt.subplots(5,2)
    row=0
    col=0
    myadapt = 30

    for mynu in myrange:

        clf = svm.OneClassSVM(kernel='rbf', nu=mynu)
        clf.fit(traindat)
        novelty_seq = clf.predict(testdat)

        z = np.copy(novelty_seq)
        z[np.where(novelty_seq==-1)] = 1 # anomalous
        z[np.where(novelty_seq==1)] = 0 # not anomalous
        n,p = testdat.shape

        out_frac = np.zeros(novelty_seq.size)

        for iii in np.arange(myadapt,n):
            out_frac[iii] = np.mean(z[iii-myadapt:iii])

        axes[row, col].plot(out_frac)
        axes[row, col].set_ylim(ymin=0, ymax=1)
        relevant_times = seizure_times
        if relevant_times is not None:
            axes[row, col].axvline(x=relevant_times[0]/(win_len-win_overlap), lw=3, color='r')
        axes[row, col].set_title('%f'%mynu)

        row+=1
        if row>4:
            row=0
            col=1
    plt.savefig('%s.png'%seizurename, bbox_inches='tight')

def viz_single_outcome(outlier_fraction, thresh, test_times, win_len, win_overlap):
    plt.plot(outlier_fraction)
    if test_times is not None:
        plt.axvline(x=test_times[0]/(win_len-win_overlap), lw=3, color='r')
        plt.axvline(x=test_times[1]/(win_len-win_overlap), lw=3, color='r')
    plt.axhline(y=thresh, lw=2, color='k')
    plt.show()

def parent_function():
    # parameters -- sampling data
    win_len = 1.0  # in seconds
    win_overlap = 0.5  # in seconds
    f_s = float(1e3)  # sampling frequency

    patients = ['TS041']
    long_interictal = [False]
    include_awake = True
    #TODO: try with asleep data
    include_asleep = False

    num_training_windows = 3600
    bands = np.asarray([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])
    rstat_window_len=2500
    rstat_window_interval=1500

    # get the paths worked out
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')

    for k, patient_id in enumerate(patients):

        print "---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)

        # analyze the patient, write to the file
        all_files, data_filenames, file_type, seizure_times, seizure_print = analyze_patient_raw(p_data_path, f_s,
                                                                                             include_awake,
                                                                                             include_asleep,
                                                                                             long_interictal[k])

        file_num = len(data_filenames)
        print'\nReading in stored Burns offline data'
        offline_features_dict = build_offline_feature_dict(patient_id, data_filenames, all_files, win_len, win_overlap)

        irange = [0,1,2,3,4,5]
        # cross-validation
        for i in irange:

            # set up test files, seizure times, etc. for this k-fold
            print '\nCross validations, k-fold %d of %d ...' % (i + 1, file_num)

            testing_file_name = data_filenames[i]
            cv_test_files = all_files[:i] + all_files[i+1:]
            cv_file_names = data_filenames[:i] + data_filenames[i+1:]
            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            print'\nFinidng frequency bands of interest'
            inter_data = np.vstack([cv_test_files[n][0:30*f_s][:] for n in xrange(len(cv_file_names)) if cv_file_type[n] is not 'ictal'])
            ict_data = np.vstack([cv_test_files[n][seizure_times[n][0]*f_s:seizure_times[n][0]*f_s+30*f_s][:] for n in xrange(len(cv_file_names)) if cv_file_type[n] is 'ictal'])
            print '\tEntering rstat code'
            freq_band = rstat_42.calc_rstat(ict_data, inter_data, f_s, bands, rstat_window_len, rstat_window_interval, mode=1)
            print '\tChosen frequency band is ', freq_band
            # freq_band = [0,200]

            print'\nOffline training: compiling offline features and labels'
            offline_training_dict = {}
            for training_file_name in cv_file_names:
                offline_training_dict[training_file_name] = offline_features_dict[training_file_name]
            offline_label_dict = hand_label_seizuretimes(offline_training_dict, cv_file_type, cv_seizure_times, cv_file_names, win_len, win_overlap)

            print'\nOffline training: choosing best channels'
            chosen_channels = choose_best_channels(offline_training_dict, offline_label_dict, cv_file_names)
            print '\tBest channels are: ',chosen_channels

            for local, single_cv_test_file in enumerate(cv_test_files):
                cv_test_files[local] = single_cv_test_file[:, chosen_channels]

            burns_win_len = 3
            burns_win_overlap = 2

            print'\nOffline training: compiling interictal data'
            total_rows = 0
            num_rows = []
            corres_ind = []

            for location in xrange(len(cv_file_names)):
                if cv_file_type[location] is not "ictal":
                    total_rows += cv_test_files[location].shape[0]
                    num_rows += [cv_test_files[location].shape[0]]
                    corres_ind += [location]
                    total_cols = cv_test_files[location].shape[1]
            interictal_training_data = np.zeros((total_rows,total_cols))
            for ind, num_rows_data in enumerate(num_rows):
                print '\tAdding file '
                start_row = sum(num_rows[0:ind])
                end_row = sum(num_rows[0:ind+1])
                interictal_training_data[start_row:end_row,:] = cv_test_files[corres_ind[ind]]
            print'\tBuilding coherency matrices'
            training_coherency_matrices = build_coherency_array(interictal_training_data, burns_win_len, burns_win_overlap, f_s, True, num_training_windows, freq_band)
            print'\tFinding mean and std'
            mean_coherency_matrix, sd_coherency_matrix = find_normalizing_coherency(training_coherency_matrices)
            print'\tTransforming coherency matrices'
            transformed_coherency_matrices = transform_coherency(training_coherency_matrices, mean_coherency_matrix, sd_coherency_matrix)
            print'\tFinding eigenvec centrality'
            training_interictal_evc = find_evc(transformed_coherency_matrices)
            print'\tSaving training data as a mat file'
            # sio.savemat('training%d.mat'%i, {'training_interictal_evc':training_interictal_evc, 'training_coherency':training_coherency_matrices,'chosen_channels':np.array(list(chosen_channels)), 'mean':mean_coherency_matrix, 'std':sd_coherency_matrix})

            cv_evc = []
            for cv, cv_test_file in enumerate(cv_test_files):
                print'\nOffline training: finding evc for all cross validation files'
                print'\tBuilding coherency matrices'
                cv_test_coherency_matrices = build_coherency_array(cv_test_file, burns_win_len, burns_win_overlap, f_s, False, num_training_windows, freq_band)
                print'\tTransforming coherency matrices'
                cv_transformed_test_coherency_matrices = transform_coherency(cv_test_coherency_matrices, mean_coherency_matrix, sd_coherency_matrix)
                print'\tFinding eigenvec centrality'
                cv_evc += [find_evc(cv_transformed_test_coherency_matrices)]
                print'\nFinished finding evc for all cv filess'

            #TODO: alternate parameter tuning methods? adapt rate/C separate from nu gamma?
            nu, gamma, C, adapt_rate, T_per = train_pi(training_interictal_evc, cv_evc, burns_win_len, burns_win_overlap, f_s, cv_seizure_times)

            print'\nOnline testing'
            testing_data = all_files[i]
            testing_data = testing_data[:,chosen_channels]
            print'\tBuilding coherency matrices'
            test_coherency_matrices = build_coherency_array(testing_data, burns_win_len, burns_win_overlap, f_s, False, num_training_windows, freq_band)
            print'\tTransforming coherency matrices'
            transformed_test_coherency_matrices = transform_coherency(test_coherency_matrices, mean_coherency_matrix, sd_coherency_matrix)
            print'\tFinding eigenvec centrality'
            test_evc = find_evc(transformed_test_coherency_matrices)
            print'\nFinished finding online test features'
            # sio.savemat('testing%d.mat'%i, {'test_evc':test_evc, 'test_coherency':test_coherency_matrices})
            # outlier_frac_viz(training_interictal_evc, test_evc, seizure_times[i], data_filenames[i])

            svm_model = train_oneclass_svm(training_interictal_evc, nu, gamma)
            outlier_fraction = create_outlier_frac(test_evc, svm_model, adapt_rate)
            decision = np.sign(outlier_fraction-C)
            S, mu, FP, time = get_stats(decision, T_per, seizure_times[i], f_s, win_len, win_overlap)
            FPR = np.divide(FP,time)
            viz_single_outcome(outlier_fraction, C, seizure_times[i], burns_win_len, burns_win_overlap)
            print 'My parameters chosen were: ', nu, gamma, C, adapt_rate, T_per
            print 'My performance metrics [S, mu, FPR] were: ', S, mu, FPR

if __name__ == '__main__':
    parent_function()
