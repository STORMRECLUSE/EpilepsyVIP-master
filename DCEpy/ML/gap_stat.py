__author__ = 'Chris'

import numpy as np
import scipy.cluster.vq as clust
import matplotlib.pyplot as plt

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

def gap_statistic(X, K, n_tests):
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
    n_tests: int
        number of tests to run

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

        # initialize k
        k = K[j]

    	# cluster our data 
    	centroids, w_k = clust.kmeans(X, k)

        # cluster random data
        for i in range(n_tests):

            # generate uniform data
            Z = random_data(Y,n)

    		# cluster random data
            cetroids, w_k_r = clust.kmeans(Z, k)
            w_k_rand[i] = w_k_r

        # expected within cluster distance
        w_k_exp = np.mean(w_k_rand)

        # compute gap
        gap[j] = np.log(w_k_exp) - np.log(w_k)

    # find k with maximum gap
    min_k = K[np.argmax(gap)]
    return min_k, gap

def disp_gap(gap,K):
	"""
	Display the gap from the gap statistic 

	Parameters
	----------
	gap: ndarray, shape(k,2)
	    data array of k means for gaussian distributions
	K: ndarray
		tested k values
	gap: ndarray, shape(K)
	    the average gap for each k

	Returns
	-------
	Plots the gaps
	"""

	# plot gap
	width = 0.8
	ind = np.arange(K.shape[0])
	fig, ax = plt.subplots()
	rects = ax.bar(ind, gap)
	ax.set_ylabel('Gap')
	ax.set_xlabel('k')
	ax.set_title('Gap Statistic')

	# set ticks and labels
	ax.set_xticks(ind + width/2)
	labels = []
	for k in K:
		labels.append(str(k))
	ax.set_xticklabels(labels)

	# show the plot
	plt.show()
	return 

def gap_stat_test(mu, cov, class_size, K):
	"""
	Example code to demonstrate use of gap statistic 
	Samples from provided normal distributions in R^2

	Parameters
	----------
	mu: ndarray, shape(k,2)
	    data array of k means for gaussian distributions
	cov: ndarray, shape(k,2,2)
	    data array of k vcovariance matrices for gaussian distributions
	class_size: ndarray, shape(k)
		data array of class sizes
	K: ndarray
		guesses for k 

	Returns
	-------
	Prints the chosen value of k and values of gaps
	"""

	# check that sizes for k match
	k = mu.shape[0]
	if cov.shape[0] != k or class_size.shape[0] != k:
		raise AttributeError ("k is inconsistent across mu, sigma, class_size")

	# create data
	X = np.empty((0,2))
	for i in range(k):
		A = np.random.multivariate_normal(mu[i], cov[i], class_size[i])
		X = np.vstack((X,A))

	# run gap statistic
	n_tests = 1000
	min_k, gap = gap_statistic(X, K, n_tests)

	# display results
	disp_gap(gap,K)
	
	print("Gap statistic chose k=" + str(min_k))



