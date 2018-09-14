__author__ = 'Sarah'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.sparse.linalg import svds
from numpy.linalg import svd


"""
pca_analysis(X,seizuretimes,k)

Inputs:
    X: matrix of data that pca is to be perfomed on
    seizuretimes: indices that indicate when seizure periods (inter-, pre-, ictal, and post)
    k: number of principal components to be viewed
    file_path_s: storage location for singular values graph
    file_path_PCA: storage location for PCA graphs

Outputs:
    No outputs
    Plot1 is a stem plot of singular values in descending order
    Plot2 is principal components of data plotted against one another
    Data points marked by 'o' indicate seizure (seizure[1]-seizure[2])
    Data points marked by 'x' indicate no seizure
    Black data points occurred in the interictal time window (<seizure[0])
    Colored data points are colored according to the time at which they occurred: pre-ictal data points (seizure[0]
        to seizure[1]) have blue hues, post-ictal data points (>seizure[2]) have red hues
"""



def pca_analysis(X,seizuretimes,k,file_path_s=None,file_path_pca=None):

    #centering data
    n,p = X.shape
    avg_vec = X.mean(0)
    one_vec = np.ones((n,1))
    centered_X = X - one_vec*avg_vec

    #computing k principal components
    U,S,V = svds(centered_X, k)
    princ_comp = U*S

    #computing and plotting all singular values
    full_S = svd(X, compute_uv=0)
    fig1 = plt.figure()
    plt.stem(full_S)
    plt.title('Singular Values')
    plt.ylabel('Magnitude')
    if file_path_s is not None:
        fig1.savefig(file_path_s, bbox_inches='tight')

    #creating vector for labeling seizure points vs nonseizure points
    marking=['x']*n
    marking[seizuretimes[1]:seizuretimes[2]]=['o']*(seizuretimes[2]-seizuretimes[1])

    #creating vector to color data points over time
    colors=['k']*n
    colors[seizuretimes[0]:n] = cm.rainbow(np.linspace(0, 1, n-seizuretimes[0]))

    #plotting principal components
    fig2 = plt.figure(figsize=(9,10))
    for i in range(1,k):
        for j in range(0,n):
            plt.subplot(math.ceil(k/2),2,i)
            plt.scatter(princ_comp[j][k-i], princ_comp[j][k-i-1], color=colors[j], marker=marking[j])
            plt.title('Principal Components')
            plt.xlabel('PC %d' %i)
            plt.ylabel('PC %d' % (i+1))
    if file_path_pca is None:
        plt.show()
    else:
        plt.show()
        fig2.savefig(file_path_pca, bbox_inches='tight')
    return

#creating Gaussian test matrix, X
dim = 6
n_first_points = 250 #before seizuretime[0]
n_middle_points = 150 #seizuretime[0]-seizuretime[1]
n_last_points = 250 #seizuretime[1]-seizuretime[2], seizure happening
n_post_points = 100 #after seizuretime[2]

mean1 = 10*np.ones(dim)
mean2 = -10*np.ones(dim)
cov = np.eye(dim)
X=np.zeros(( n_post_points + n_first_points + n_middle_points + n_last_points,dim))

lam = np.linspace(0,1,n_middle_points)

#create data!
X[0:n_first_points,:] = np.random.multivariate_normal(mean1, cov, size=n_first_points)
X[n_first_points:n_first_points+n_middle_points,:] = np.outer((1-lam), mean1) + np.outer(lam,mean2)
X[n_first_points + n_middle_points:n_first_points + n_middle_points + n_last_points,:] = np.random.multivariate_normal(mean2, cov, size=n_last_points)
X[n_first_points + n_middle_points + n_last_points:,:] = np.random.multivariate_normal(mean1, cov, size=n_post_points)


#testing
k=4
seizuretimes = [250,400,650]
pca_analysis(X,seizuretimes,k,None,None)
