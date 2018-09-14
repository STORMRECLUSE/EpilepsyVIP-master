# import numpy as np
# import scipy.signal
# import scipy.cluster
# import networkx as nx
# import rstat
#
# def burns_full
# 	"""
# 	"""
#
#
# 	# load data, edf reader
#     #inter_data
#     #ictal_data
#
# 	# pre processing (function below)
#     y_inter = preprocess(inter_data)
#     y_ictal = preprocess(ictal_data)
#
# 	# pick band to use
#     calc_rstat(y_ictal[0], y_inter[0], 1000);
#
# 	evc = []
#
# 	# go through the data pulling out the data
# 	# needed for 3sec graph, right now just doing the first graph
# 	for i in range(0, 1):
# 		# data = pull out specific band and i*1000 - i*1000 + 3000 samples
# 		G = coherence_network(data, band)
# 		current_evc = nx.eigenvector_centrality(G, 100, 1e-06, None, 'weight')
# 		# insert normalization of the vector
# 		evc.append(current_evc)
#
#
#
#
# 	# after creating all, use clustering
# 	# gap statistics to find k
# 	# k-means
# 	# k = gap_stats()
# 	k = 10
#
# 	# also kmeans2?
# 	[codebook, distortion] = scipy.cluster.vq.kmeans(evc, k)
#
# 	# how to use this to classify others?
#
#     # other function to decide which cluster that each one goes in (list?)
#     # take in cluster and the vector to clasify, output the cluster number
#     # create the list based on these
#
#
# def preprocess(data):
#
#     #Filter parameters
#     order = 6
#     fs = 1000 # sample rate, Hz
#     cutoff = 120 # cutoff frequency of the filter, Hz
#
#     y = np.zeros((N,int(data[1].size)))
#
#     for ch in range(0,N):
#
#         # Filter the data
#         y[ch] = butter_lowpass_filter(data[ch], cutoff, fs, order)
#
#         # Apply notch filter around 60 Hz
#         bp_stop_hz = np.array([59.0,61.0])
#         b, a = butter(2,bp_stop_hz/(dsrate/2.0),'bandstop')
#         filtered = np.zeros(y_ds.shape)
#         filtered[ch] = lfilter(b, a, y_ds[ch])
#
#         # Normalization: subtract mean and divide by standard deviation
#         mean = np.zeros((N,))
#         std = np.zeros((N,))
#         normalized = np.zeros((y_ds.shape))
#         mean[ch] = np.mean(filtered[ch])
#         std[ch] = np.std(filtered[ch])
#         normalized[ch] = (filtered[ch] - mean[ch])/std[ch]
#
#     return y
#
#
# # coherence to create graph
# # scipy.signal.coherence
# def coherence_network(data, nodes_used, band):
#     """
#     Creates and returns a NetworkX graph using coherence as edge weights.
#
#     Input:
#         data: an n x s array, n is number of channels
#                               s is the number of samples in the recording
#         nodes_used: a list of integers 1...n that correspond to the electrodes
#                     we want to consider in building our graph
#                     if left empty, all channels will be used
#         band: the specific range we are interested in using (TODO)
#
#     Output:
#         G: a NetworkX graph with channels as nodes and coherence between the
#             channels as edge weights
#
#     """
#
#     n, m = data.shape
#
#     # if empty, use all of the channels
#     if not nodes_used:
#         nodes_used = range(0, n);
#
#     edges = []
#     G=nx.Graph()
#
#     # TODO: Implement using a specific band of the signal
#     for i in nodes_used:
#         G.add_node(i)
#         for j in nodes_used:
#             edges.apped((i, j, {'weight':scipy.signal.coherence(data[i], data[j])}))
#
#     G.add_edges_from(edges)
#
#     return G
#
#
#
#
#
#
#
# # gap stats: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
# # this code is for 2D data, so looking to extend to our case of n dimensional data
# # should not be too hard. also need to increase possible k values and possibly B
# # which is the amount of random data sets to create and use
#
# # might need to alter?
# def Wk(mu, clusters):
#     K = len(mu)
#     return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
#                for i in range(K) for c in clusters[i]])
#
# # create a min vector max vector for each entry
# # since we are normalizing between 0 and 1, we may be able
# # to just use that in every direction, need to read more on
# # whether it is important to use actual min & max or just possible min/max
# def bounding_box(X):
#     xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
#     ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
#     return (xmin,xmax), (ymin,ymax)
#
#  # see above for changes to come
# def gap_statistic(X):
#     (xmin,xmax), (ymin,ymax) = bounding_box(X)
#     # Dispersion for real distribution
#     ks = range(1,10)
#     Wks = zeros(len(ks))
#     Wkbs = zeros(len(ks))
#     sk = zeros(len(ks))
#     for indk, k in enumerate(ks):
#     	# they have a reference for what they use for find_centers, but I think I can use kmeans
#     	# again here, since that is all that is doing, but I need to check on what this needs
#         mu, clusters = find_centers(X,k)
#         Wks[indk] = np.log(Wk(mu, clusters))
#         # Create B reference datasets
#         B = 10
#         BWkbs = zeros(B)
#         for i in range(B):
#             Xb = []
#             for n in range(len(X)):
#                 Xb.append([random.uniform(xmin,xmax),
#                           random.uniform(ymin,ymax)])
#             Xb = np.array(Xb)
#             mu, clusters = find_centers(Xb,k)
#             BWkbs[i] = np.log(Wk(mu, clusters))
#         Wkbs[indk] = sum(BWkbs)/B
#         sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
#     sk = sk*np.sqrt(1+1/B)
#     return(ks, Wks, Wkbs, sk)
#     # I think need to add in an analysis/calculation based on the website
#     # to find when the difference becomes positive and that will give the
#     # actual number of clusters to use
#
#
# #-----------------------------------------------------------------------------
# # ALTERED
#
# # might need to alter?
# def Wk(mu, clusters):
#     K = len(mu)
#     return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
#                for i in range(K) for c in clusters[i]])
#
#
#
#
#
# # create a min vector max vector for each entry
# # since we are normalizing between 0 and 1, we may be able
# # to just use that in every direction, need to read more on
# # whether it is important to use actual min & max or just possible min/max
# def bounding_box_n(X, n):
#
# 	# minimum and maximum value for each entry in the vecotrs in X
# 	min_vec = []
# 	max_vec = []
#
# 	for i in range(0, n):
#     	xmin, xmax = min(X,key=lambda a:a[i])[i], max(X,key=lambda a:a[i])[i]
#     	min_vec.append(xmin)
#     	max_vec.append(xmax)
#     return min_vec, max_vec
#
#  # see above for changes to come
# def gap_statistic_n(X, n):
# 	# could technically get n from X, but for now will just pass in
#    	min_vec, max_vec = bounding_box_n(X)
#     # Dispersion for real distribution
#     ks = range(1,16) # going to start with 16 and see how that does
#     Wks = zeros(len(ks))
#     Wkbs = zeros(len(ks))
#     sk = zeros(len(ks))
#     for indk, k in enumerate(ks):
#     	# they have a reference for what they use for find_centers, but I think I can use kmeans
#     	# again here, since that is all that is doing, but I need to check on if we get "mu" and "clusters" back
#     	# in the format that is needed
#     	# HEREEE
#         mu, clusters = find_centers(X,k)
#         Wks[indk] = np.log(Wk(mu, clusters))
#         # Create B reference datasets
#         B = 10
#         BWkbs = zeros(B)
#         for i in range(B):
#             Xb = []
#             for n in range(len(X)):
#                 Xb.append(rand_vector(min_vec, max_vec, n))
#             Xb = np.array(Xb) # neccessary??
#             mu, clusters = find_centers(Xb,k)
#             BWkbs[i] = np.log(Wk(mu, clusters))
#         Wkbs[indk] = sum(BWkbs)/B
#         sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
#     sk = sk*np.sqrt(1+1/B)
#     return(ks, Wks, Wkbs, sk)
#     # I think need to add in an analysis/calculation based on the website
#     # to find when the difference becomes positive and that will give the
#     # actual number of clusters to use
#
# def rand_vector(min_vec, max_vec, n):
#     # better way to do this with normalizing between the two things
#     rv = []
#
#     for i in range(0, n):
#         rv.append(random.uniform(min_vec[i], max_vec[i]))
#
#     return rv
#
#
#
# # stability analysis or cluster analysis, run same k multiple time and how
# # stable are the clusters