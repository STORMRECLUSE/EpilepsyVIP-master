import numpy as np
import networkx as nx
from numpy import linalg as LA
import scipy
from scipy import stats

# network properties:
# https://sph.umd.edu/sites/default/files/files/Rubinov_Sporns_2009.pdf

# four centrality measures:
# https://academic.oup.com/cercor/article/22/8/1862/321860
# degree, eigenvector, page-rank, and subgraph.
# They can be categorized into local (degree), mesoscale (subgraph), and global centralities (eigenvector and page-rank)

# dim reduction
# https://cseweb.ucsd.edu/~saul/papers/smdr_ssl05.pdf

# network data reduction:
# http://proceedings.mlr.press/v32/celik14.pdf




def pagerank_centrality(coherency_matrix):

    # and compute the eigenvector centrality for each window's graph
    sub_matrix = coherency_matrix.copy()
    # cast to a graph type
    G = nx.Graph(sub_matrix)
    try:
        # compute the eigenvector centrality
        evc = nx.pagerank(G, max_iter=500)
        centrality = np.asarray(evc.values())
    except:
        # check to see if convergence criteria failed and if so, return an EVC of all ones
        centrality = np.ones(coherency_matrix.shape[1]) / float(coherency_matrix.shape[1])
        print"Convergence failure in EVC; bad vector returned"
    return centrality


def compute_eigen_centrality(coherency_matrix):
    # and compute the eigenvector centrality for each window's graph
    sub_matrix = coherency_matrix.copy()
    # cast to a graph type
    G = nx.Graph(sub_matrix)
    try:
        # compute the eigenvector centrality
        evc = nx.eigenvector_centrality(G, max_iter=500)
        centrality = np.asarray(evc.values())
    except:
        # check to see if convergence criteria failed and if so, return an EVC of all ones
        centrality = np.ones(coherency_matrix.shape[1]) / float(coherency_matrix.shape[1])
        print"Convergence failure in EVC; bad vector returned"
    return centrality


def compute_katz(coherency_matrix):
    num_chan = 6
    # select attenuation rate
    atten = .1
    # create identity matrix
    ident = np.identity(num_chan)

    w, v = LA.eig(coherency_matrix)
    larg_eig = w.max()
    alpha = (1 / larg_eig) * atten
    A_trans = np.transpose(coherency_matrix)
    cent = LA.inv(ident - alpha * A_trans) - ident
    j = np.matrix('1; 1; 1; 1; 1; 1')
    katz = np.matmul(cent, j)
    centrality_all_files = np.asarray(katz).reshape(-1)
    return centrality_all_files


def eigen(coherency_matrix):
    w, v= np.linalg.eig(coherency_matrix)
    return w


def compute_stats(coherency_matrix):
    mean = np.mean(coherency_matrix)
    abs_mean = np.mean(np.absolute(coherency_matrix))
    std = np.std(coherency_matrix)
    n = np.linalg.norm(coherency_matrix)
    k = scipy.stats.kurtosis(coherency_matrix)
    sk = scipy.stats.skew(coherency_matrix)
    return np.hstack((np.array([mean, abs_mean, std, n]), k, sk))

