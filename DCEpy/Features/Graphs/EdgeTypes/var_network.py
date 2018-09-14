__author__ = 'vsp'
import networkx as nx
import numpy as np
from statsmodels.tsa.vector_ar import var_model
from .pdc_dtf import DTF, PDC

def var_network(data, connections, weightType, order, n_fft=None):
	"""
	Builds a networkx graph using connectivity measurements
	derived from vector autoregressive models (VAR)

	Parameters
	----------
	data : ndarray, shape (n, N)
		iEEG data with N channels and n samples 
	connections : list 
		list of either (i) integers
			the integers are the channels that will become nodes
			and the graph will be complete
		or (ii) length 2 lists of integers
			the integers correspond to directed edges
	weightType: str
		string to indicate the connectivity measurement used
		see build_network for more details
	order: int
		VAR model order 
	n_fft: int
		length of FFT in PDC, DTF computations

	Returns
	-------
	G : a weighted networkx graph

	"""

	# Notes:
	#	1) Data is centered and scaled --> other transformations?
	# 	2) Do we want to pass parameters for these transformations?

	# check that n_fft is supplied when required
	if weightType == "directed_transfer_function" or weightType == "partial_directed_coherence":
		if n_fft is None:
			raise AttributeError ("n_fft is not supplied")

	# get parameters
	n, N = data.shape # N : number of channels, n: number of observations

	# normalize the channels
	d_mean = np.reshape(np.mean(data,axis=0), (1,N))
	d_std = np.reshape(np.std(data,axis=0), (1,N))
	data = (data - d_mean) / d_std

	# fit MVAR and obtain coefficients
	model = var_model.VAR(data)
	mvar_fit = model.fit(order)
	A = mvar_fit.coefs
	sigma = np.diagonal(mvar_fit.sigma_u_mle)

	# compute connectivity measurement
	if weightType == "directed_transfer_function":
		W, freqs = DTF(A=A, sigma=sigma, n_fft=n_fft)
		keyword = "dtf"
	elif weightType == "partial_directed_coherence":
		W, freqs = PDC(A=A, n_fft=n_fft)
		keyword = "pdc"
	elif weightType == "granger_causality":
		# granger causality code from statsmodels
		keyword = "gc"

	# create directed graph
	G = nx.DiGraph()

	# create an edge list from connections
	if type(connections[0]) is list:
		edges = connections
	elif type(connections[0]) is int:
		edges = []
		for node1 in connections:
			for node2 in connections:
				if node1 != node2:
					edges.append([node1,node2])

	# build the graph with edge weights
	G = nx.DiGraph()
	for edge in edges:
		attr = {keyword: W[edge[0],edge[1]]}
		G.add_edge(edge[0], edge[1], attr)

	return G