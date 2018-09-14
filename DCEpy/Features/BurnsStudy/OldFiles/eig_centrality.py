__author__ = 'Chris'

import numpy as np
from scipy.signal import coherence
# from matplotlib.mlab import coherence

def eig_centrality(X, fs=1000, connections=None, v0=None, tol=1e-3):
    """
    Given a window of data, X, build a graph G with coherency as edges
    and compute eigenvector centrality.

    Parameters
    ----------
    X: ndarray, shape(n,p)
        data array of n samples, p channels
    connections: list
        list of either (i) integers
            the integers are the channels that will become nodes
            and the graph will be complete
        or (ii) length 2 lists of integers
            the integers correspond to directed edges

    Returns
    -------
    eig_vec: ndarray, shape(p)
        eigenvector of adjacency matrix
    """

    # # build coherence graph
    # if connections == None:
    # 	n,p = X.shape
    # 	connections = range(p)
    # weightType = 'coherence'
    # G = build_network(X, connections, weightType)

    # # get the eigenvector centrality
    # eig_dict = nx.eigenvector_centrality(G, weight='coherence')
    # eig_vec = np.zeros(p)
    # for i in range(p):
    #     eig_vec[i] = eig_dict[i]
    # w = [1.0]

    # get the data shape
    n,p = X.shape

    # initialize the adjcency matrix
    A = np.zeros((p,p))

    # construct adjacency matrix
    for i in range(p):
        for j in range(i+1,p):
            f, cxy = coherence(X[:,i], X[:,j], fs=fs)
            # cxy, f = cohere(X[:,i], X[:,j], Fs=fs) # agggg using matplotlib because scipy is being dumb
            c = np.mean(cxy)
            A[i,j] = c # upper triangular part
            A[j,i] = c # lower triangular part
            if np.isnan(c):
                print( '(' + str(i) + ',' + str(j) + ") is nan")


    # EDIT: none of these are working. They return NaN eigenvalues - why??
    # print('Sums are: ', str(np.mean(A,axis=1)))

    # Method 0: Power Iteration
    maxiter = 12
    tol = 1e-5
    if v0 == None:
    	v0 = np.ones(p) / np.sqrt(p)
    for i in range(maxiter):

        # multiply and normalize
        v_new = np.dot(A,v0)
        v_new = v_new / np.linalg.norm(v_new)

        # check for tolerance
        diff = np.linalg.norm(v_new-v0)
        if diff < tol:
            break

        # update iterate
        v0 = v_new

    v = v_new
    w = np.array([np.mean(np.dot(A,v) / v)]) # eigenvalue
    eig_vec = v

    # # Method 1: All eigenvectors of symmetric matrix
    # w,v = np.linalg.eigh(A, UPLO='U') # get the eigenvectors
    # eig_vec = v[:,-1]

    # # Method 2: Top 1 eigenvector of matrix
    # w, v = scipy.sparse.linalg.eigs(A, k=1, v0=v0, tol=tol)
    # eig_vec = v[:,0]x

    # # Method 3: Top 1 eigenvector of a symmetric matrix
    # w, v = scipy.sparse.linalg.eigsh(A, k=1, v0=v0, tol=tol)
    # eig_vec = v[:,0]

    return eig_vec
