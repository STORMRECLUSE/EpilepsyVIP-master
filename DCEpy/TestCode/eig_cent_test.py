# Eigenvector Centrality Test
'''
This test function imports test data,
does not do pre-processing of any sort,
computes eigenvector centrality of graphs at windows 
displays them using PCA 
'''

# TODO:
# 	(1) Add pre-processing
# 	(2) Add R-statistic
#	(3) Add several PC plots with labels


# import packages
import matplotlib.pyplot as plt
import numpy as np
import scipy

from DCEpy.Features.BurnsStudy.eig_centrality import eig_centrality
from DCEpy.Features.GardnerStudy.edfread import edfread

# download a data file
print('Downloading file...')
filename = '/home/chris/Documents/Rice/senior/EpilepsyVIP/data/RMPt2/DA00101U_1-1+.edf' # Chris
seizure_start,seizure_end = 262,330
fs = 1000
bad_channels = ('Events/Markers','EDF Annotations', 'EEG Mark1', 'EEG Mark2')
data,_,labels = edfread(filename, bad_channels=bad_channels)
data_len,nchannels = np.shape(data)
print('shape is ' + str(np.shape(data)))

# window size
window_size = int(5e3)
window_increment = 1250
window_num = len(range(window_size, data_len, window_increment))
eigs = np.empty( (window_num, nchannels) ) # initialize for eigenvectors
col = np.empty((window_num)) # initialize seizure labels

# find eigenvectors
i = 0
v0 = np.ones(nchannels) / np.sqrt(nchannels)
print('Getting Eigenvectors...')
for end_time in range(window_size, data_len, window_increment):

	## compute eigenvector centrality
	window = data[(end_time-window_size):end_time,]
	w, eig_vect = eig_centrality(window, v0=v0)

	# store eigenvectors and update warm start
	eigs[i,:] = eig_vect
	v0 = eig_vect

	# update color -- seizure label
	start_time = end_time-window_size
	if start_time < fs*(seizure_start-5) or start_time > fs*(seizure_end+5):
		col[i] = 0 # interictal
	elif (start_time >= fs*seizure_start-5) and (start_time < fs*seizure_start):
		col[i] = 1 # preictal
	elif start_time >= fs*seizure_start and start_time < fs*seizure_end:
		col[i] = 2 # ictal
	else:
		col[i] = 3 # post ictal

	print('Computed ' + str(i+1) + ' of ' + str(window_num) + ' with col=' + str(col[i]) + ', eig = ' + str(w[0]))
	i = i+1

# perform PCA
print('Computing PCA and plotting...')
U, d, Vh = scipy.linalg.svd(eigs, full_matrices=False)
pcs = ['PC1','PC2','PC3']

for i in range(4):
	plt.subplot(1,4,i+1)
	plt.scatter(U[:,i],U[:,i+1],c=col)
	plt.xlabel(pcs[i])
	plt.ylabel(pcs[i+1])
plt.title('PCA of EigenCentrality')