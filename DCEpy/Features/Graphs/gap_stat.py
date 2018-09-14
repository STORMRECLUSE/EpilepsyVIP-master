__author__ = 'Chris'

import numpy as np 
from scipy.cluster import vq as clust

def data_range(X):
    """
    Returns the range of data for use in gap statistic

    Parameters
    ----------
    X: ndarray, shape(n,N)
        data array of n samples, N channels

    Returns
    -------
    Y: ndarray, shape(N,2)
        max and min of given variable
        Y[j,0] = min(X[j,])
        Y[j,1] = max(X[j,])

    """
    Y = np.empty((X.shape[1],2))
    Y[:,0] = X.min(axis=0)
    Y[:,1] = X.max(axis=0)

    return Y

def random_data(Y,n):
    """
    Generates artificial random data for use in gap statistic

    Parameters
    ----------
    Y: ndarray, shape(N,2)
        max and min of given variable
        Y[j,0] = min(X[j,])
        Y[j,1] = max(X[j,])
    n: int
        the desired number of samples

    Returns
    -------
    Z: ndarray, shape(n,N)
        data array of n samples, N channels
    """

    # get number of variables (channels)
    N = Y.shape[0]

    # create uniform random data on [0,1]
    Z = np.random.rand(n,N)

    # transform it to be uniform over data range
    Z = Z * (Y[:,1] - Y[:,0])  + Y[:,0]

    return Z

def gap_statistic(X, K, n_tests, kmeans_inter):
    """
    Determines best k for k-means clustering using gap
    statistic.

    Parameters
    ----------
    X: ndarray, shape(n,N)
        data array of n samples, N channels
    K: ndarray, shape(k_num,1)
        array of possible values for k
        (not necessary, but preferably in increasing order)
    n_test: int
        number of tests to run
    kmeans_inter: int
        number of times to run kmean

    Returns
    -------
    min_k: int
        the k with minimum gap
    gap: ndarray, shape(K)
        the average gap for each k
    """

    # get data range
    n, N = X.shape
    Y = data_range(X)

    # initialize variables
    gap = np.empty(K.shape)
    w_k_rand = np.empty(n_tests)

    for j in range(len(K)):
        print 'Trying out kmeans of k = ', K[j]
        # initialize k
        k = K[j]

        # cluster our data
        centroids, w_k = clust.kmeans(X, k_or_guess=k, iter=kmeans_inter)
        print w_k
        # cluster random data
        for i in range(n_tests):
            # generate uniform data
            Z = random_data(Y,n)

            # cluster random data
            cetroids, w_k_r = clust.kmeans(Z, k_or_guess=k, iter=kmeans_inter)
            w_k_rand[i] = w_k_r

        # expected within cluster distance
        w_k_exp = np.mean(w_k_rand)
        print w_k_exp
        # compute gap
        gap[j] = np.log(w_k_exp) - np.log(w_k)
        print gap[j]

    # find k with minimum gap
    max_k = K[np.argmax(gap)]
    return max_k, gap




