import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.cluster import vq as clust

# from edfread import edfread
#import feature functions
# from DCEpy.Features.feature_functions import burns_features
# from DCEpy.Features.feature_functions import energy_features
# from DCEpy.Features.feature_functions import spectral_energy_features

data0 = sio.loadmat('testing0.mat')
data1 = sio.loadmat('testing1.mat')
data2 = sio.loadmat('testing2.mat')
data3 = sio.loadmat('testing3.mat')
data4 = sio.loadmat('testing4.mat')
data5 = sio.loadmat('testing5.mat')

all_data = []
all_data += [data0.get('test_evc')]
all_data += [data1.get('test_evc')]
all_data += [data2.get('test_evc')]
all_data += [data3.get('test_evc')]
all_data += [data4.get('test_evc')]
all_data += [data5.get('test_evc')]

file_types = ['ictal','ictal','ictal','awake','awake','awake']
seizure_times = [(3*60 + 9,4*60 + 45), (4*60 + 34,9*60), (4*60 + 37,5*60 + 45),None,None,None]

for cv in xrange(6):

    training_data = all_data[:cv] + all_data[cv+1:]
    training_types = file_types[:cv] + file_types[cv+1:]
    training_times = seizure_times[:cv] +seizure_times[cv+1:]

    testing_data = all_data[cv]
    testing_type = file_types[cv]
    testing_time = seizure_times[cv]

    all_interictal = []
    all_preictal = []

    all_interictal = [training_data[n] for n in xrange(len(training_data)) if training_types[n] is 'awake']
    all_ictal = [training_data[n] for n in xrange(len(training_data)) if training_types[n] is 'ictal']
    ictal_times = [training_times[n] for n in xrange(len(training_times)) if training_types[n] is 'ictal']

    for i in xrange(len(all_ictal)):
        relvtimes = ictal_times[i]
        relvfile = all_ictal[i]
        preictalfile = relvfile[0:relvtimes[0],:]
        all_preictal += [preictalfile]

    interictal_array = np.vstack((all_interictal[n] for n in xrange(len(all_interictal))))
    preictal_array = np.vstack((all_preictal[n] for n in xrange(len(all_preictal))))

    interictal_labels = np.ones(interictal_array.shape[0])
    preictal_labels = np.ones(preictal_array.shape[0])*2

    all_training_data = np.vstack((interictal_array,preictal_array))
    all_training_labels = np.hstack((interictal_labels, preictal_labels))

    print 'starting kmeans'
    interictal_centroids, w_k_r = clust.kmeans(interictal_array, k_or_guess=18, iter=15)
    preictal_centroids, w_k_r = clust.kmeans(preictal_array, k_or_guess=8, iter=15)
    print 'ending kmeans'

    all_centroids = np.vstack((interictal_centroids, preictal_centroids))

    labels = np.empty(testing_data.shape[0])
    for row in xrange(testing_data.shape[0]):
        newpoint = testing_data[row,:]
        distances = np.empty((all_centroids.shape[0]))
        for i in xrange(all_centroids.shape[0]):
            center = all_centroids[i,:]
            distances[i] = np.linalg.norm(center-newpoint)
        labels[row] =  np.argmin(distances)
    print 'done labeling'

#THE WHOLE POINT OF THIS IS THAT YOU HAVE VARIANCE IN YOUR DATA. HOW DO YOU CAPTURE THIS?

    adaptrate = 50
    numpreictal = np.zeros(labels.shape[0]-adaptrate)
    for startinglocation in xrange(labels.shape[0]-adaptrate):
        portion = labels[startinglocation:startinglocation+adaptrate]
        past=100
        for onelabel in xrange(portion.shape[0]):
            new = portion[onelabel]
            compare = portion[onelabel:onelabel+4]
            for help in xrange(3):
                if new != compare[help]:
                    numpreictal[startinglocation]+=1
    plt.plot(numpreictal)
    plt.ylim(ymin=0, ymax=adaptrate)
    if testing_time is not None:
        plt.axvline(x=testing_time[0]-adaptrate, lw=3, color='r')
    plt.show()


    # K = np.arange(1,40,1)
    # n_tests = 10
    # kmeans_iter = 8
    #
    # max_k_inter, gap1 = gap_stat.gap_statistic(interictal_array, K, n_tests, kmeans_iter)
    # max_k_preictal, gap2 = gap_stat.gap_statistic(preictal_array, K, n_tests, kmeans_iter)
    # print 'interictal clusters: ', max_k_inter
    # print 'preictal clusters: ', max_k_preictal
