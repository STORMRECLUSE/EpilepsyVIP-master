# k-means on n-dimensional data

import numpy as np
import matplotlib.pyplot as plt
import scipy
from DCEpy.ML.gap_stat import gap_statistic

# https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/

def test_gap_stat():
    
    # data parameters
    k=4
    N = 400

    # create artificial data
    X = init_board_gauss(N,k)
    plt.scatter(X[:,0], X[:, 1])

    # gap statistic parameters
    K = np.array([range(1,10)])
    rand_test = 1000
    kmeans_iter = 50
    mink, gap = gap_statistic(X, K, rand_tests, kmeans_iter)

    # display results
    gap_fig = plt.figure()
    ax - plt.subplot(111)
    ax.bar(K, gap)
    plt.show()

def test_kmeans():
    k = 4
    N = 400
    X = init_board_gauss(N, k)
    plt.scatter(X[:,0], X[:, 1])
    [centroids, labels] = scipy.cluster.vq.kmeans2(X, k)
    print centroids
    # use distance to color each of the points the right color

    print X
    print labels


def find_cluster(X, centroids):
    """
    Using the centroids from k-means
    take in an array of data, X, (where each row is a data point)
    and make an array of point to cluster number
    """

    labels = []

    for pt in range(X.shape[0]):
        cr_ar = np.asarray(centroids);
        dist_2 = np.sum((cr_ar - X[pt,:])**2, axis=1)
        lbl = np.argmin(dist_2)
        labels.append(lbl)

    return labels


def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        s = np.random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X


