__author__ = 'Chris'

import numpy as np
from DCEpy.ML.gap_stat import gap_stat_test

# give parameters for gaussian 
mu = np.array([ [1,1], [1,-1], [-1,1], [-1,-1] ])
cov = 0.02*np.array( [np.eye(2), np.eye(2), np.eye(2), np.eye(2)] )
class_size = np.array([20,20,20,20])
K = np.array([1,2,4,6,8])

# run gap statistic
gap_stat_test(mu,cov,class_size,K)