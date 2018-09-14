# Chris Harshaw, Megan Kehoe, Emily Meigs, 42 Prieto
# Digital Cure for Epilepsy
# October 2015
#
# network_tools.py
# This file contains functions to build networks from iEEG data and compute network metrics.

import networkx as nx
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math
from scipy import linalg, fftpack
import scipy.io
from autoregressive import MAR_est_LWR

"""
build_network()
	Creates networks from iEEG data.  Individual data channels (or pairs of data channels)
	may be selected to build the network.  The user also selects the method of creating
	weights in the network.
INPUTS:
	data 			==> an n x s numpy array where n is the number of channels in the iEEG
							recording and s is the number of samples in the recording
	connections 	==> a set containing either
							(i) integers from 1 to n, denoting vertices in the graph
							(ii) lists of integers from 1 to n, denoting directed edges in the graph
	weightType 		==> a string dictating how the weights in the network are determined
							see code for options
OUTPUT:
	G 	==> a weighted networkx graph
"""
def build_network(data, connections, weightType):

	if weightType == "phase_lag_index":
		G = pli_network(data, connections) # forty-two
	elif weightType == "partial_directed_coherence":
		G = pdc_network(data,connections) # megan
	elif weightType == "directed_transfer_function":
		G = dtf_network(data, connections) # chris
	elif weightType == "granger_causality":
		G = gc_network(data, connections) # emily
	elif weightType == "directed_information":
		G = di_network(data, connections) # chris

	return G


def cov(X, p):
	""" Vector autocovariance up to order p
	Author: Alexandre Gramfort
	Parameters
	----------
	X : ndarray, shape (N, n)
		The N time series of length n
	Returns
	-------
	R : ndarray, shape (p + 1, N, N)
		The autocovariance up to order p
	"""
	N, n = X.shape
	R = np.zeros((p + 1, N, N))
	for k in range(p + 1):
		R[k] = (1. / float(n - k)) * np.dot(X[:, :n - k], X[:, k:].T)
	return R

def mvar_fit(X, p):
	"""  Fit MVAR model of order p using Yule Walker
	Author: Alexandre Gramfort
	Parameters
	----------
	X : ndarray, shape (N, n)
		The N time series of length n
	n_fft : int
		The length of the FFT
	Returns
	-------
	A : ndarray, shape (p, N, N)
		The AR coefficients where N is the number of signals
		and p the order of the model.
	sigma : array, shape (N,)
		The noise for each time series
	"""
	N, n = X.shape
	gamma = cov(X, p)  # gamma(r,i,j) cov between X_i(0) et X_j(r)
	G = np.zeros((p * N, p * N))
	gamma2 = np.concatenate(gamma, axis=0)
	gamma2[:N, :N] /= 2.

	for i in range(p):
		G[N * i:, N * i:N * (i + 1)] = gamma2[:N * (p - i)]

	G = G + G.T  # big block matrix

	gamma4 = np.concatenate(gamma[1:], axis=0)

	phi = linalg.solve(G, gamma4)  # solve Yule Walker

	tmp = np.dot(gamma4[:N * p].T, phi)
	sigma = gamma[0] - tmp - tmp.T + np.dot(phi.T, np.dot(G, phi))

	phi = np.reshape(phi, (p, N, N))
	for k in range(p):
		phi[k] = phi[k].T

	return phi, sigma

def spectral_density(A, n_fft=None):
	"""Estimate PSD from AR coefficients
	Author: Alexandre Gramfort
	Parameters
	----------
	A : ndarray, shape (p, N, N)
		The AR coefficients where N is the number of signals
		and p the order of the model.
	n_fft : int
		The length of the FFT
	Returns
	-------
	fA : ndarray, shape (n_fft, N, N)
		The estimated spectral density.
	"""
	p, N, N = A.shape
	if n_fft is None:
		n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
	A2 = np.zeros((n_fft, N, N))
	A2[1:p + 1, :, :] = A  # start at 1 !
	fA = fftpack.fft(A2, axis=0)
	freqs = fftpack.fftfreq(n_fft)
	I = np.eye(N)

	for i in range(n_fft):
		fA[i] = linalg.inv(I - fA[i])

	return fA, freqs


def DTF(A, sigma=None, n_fft=None):
	"""Direct Transfer Function (DTF)
	Author: Alexandre Gramfort
	Parameters
	----------
	A : ndarray, shape (p, N, N)
		The AR coefficients where N is the number of signals
		and p the order of the model.
	sigma : array, shape (N, )
		The noise for each time series
	n_fft : int
		The length of the FFT
	Returns
	-------
	D : ndarray, shape (n_fft, N, N)
		The estimated DTF
	"""
	p, N, N = A.shape

	if n_fft is None:
		n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

	H, freqs = spectral_density(A, n_fft)
	D = np.zeros((n_fft, N, N))

	if sigma is None:
		sigma = np.ones(N)

	for i in range(n_fft):
		S = H[i]
		V = (S * sigma[None, :]).dot(S.T.conj())
		V = np.abs(np.diag(V))
		D[i] = np.abs(S * np.sqrt(sigma[None, :])) / np.sqrt(V)[:, None]

	return D, freqs






def pdc_network(data,connections):
	"""
	pdc_network(data,connections)
	Builds a networkx graph using partial directed coherence

	Step 1: Low pass filter (120 Hz)
	Step 2: Down sample (250 Hz)
	Step 3: Notch filter (60 Hz)
	Step 4: Normalization
	Step 5: Calculate PDC
	Step 6: Build graph

	Parameters
	----------
	data : an n x s numpy array where n is the number of channels in the iEEG
		recording and s is the number of samples in the recording
	connections : a set containing integers from 1 to n, denoting vertices in the graph

	Returns
	-------
	G : a weighted networkx graph

	"""

	def butter_lowpass(cutoff, fs, order=5):
		'''
		Generates a butterworth lowpass filter coefficients
		with frequency cutoff, sampling frequency, and of a certain order,
		:param cutoff:
		:param fs:
		:param order:
		:return:
		'''
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = butter(order, normal_cutoff, btype='low', analog=False)
		return b, a

	def butter_lowpass_filter(data, cutoff, fs, order=5):
		'''
		Filters data with a butterworth lowpass filter of given order with given cutoff frequency.
		:param data:
		:param cutoff:
		:param fs:
		:param order:
		:return:
		'''
		b, a = butter_lowpass(cutoff, fs, order=order)
		y = lfilter(b, a, data)
		return y


	def spectral_density(A, n_fft=None):
		"""Estimate PSD from AR coefficients
		Parameters
		----------
		A : ndarray, shape (p, N, N)
			The AR coefficients where N is the number of signals
			and p the order of the model.
		n_fft : int
			The length of the FFT
		Returns
		-------
		fA : ndarray, shape (n_fft, N, N)
			The estimated spectral density.
		"""
		p, N, N = A.shape
		if n_fft is None:
			n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
		A2 = np.zeros((n_fft, N, N))
		A2[1:p + 1, :, :] = A  # start at 1 !
		fA = fftpack.fft(A2, axis=0)
		freqs = fftpack.fftfreq(n_fft)
		I = np.eye(N)

		for i in range(n_fft):
			fA[i] = linalg.inv(I - fA[i])

		return fA, freqs

	def PDC(A, sigma=None, n_fft=None):
		"""Partial directed coherence (PDC)
		Parameters
		----------
		A : ndarray, shape (p, N, N)
			The AR coefficients where N is the number of signals
			and p the order of the model.
		sigma : array, shape (N,)
			The noise for each time series.
		n_fft : int
			The length of the FFT.
		Returns
		-------
		P : ndarray, shape (n_fft, N, N)
			The estimated PDC.
		"""
		p, N, N = A.shape

		if n_fft is None:
			n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

		H, freqs = spectral_density(A, n_fft)
		P = np.zeros((n_fft, N, N))

		if sigma is None:
			sigma = np.ones(N)

		for i in range(n_fft):
			B = H[i]
			B = linalg.inv(B)
			V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
			V = np.diag(V)  # denominator squared
			P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]

		return P, freqs


	def plot_all(freqs, P, name):
		"""Plot grid of subplots
		"""
		m, N, N = P.shape
		pos_freqs = freqs[freqs >= 0]

		f, axes = plt.subplots(N, N)
		for i in range(N):
			for j in range(N):
				axes[i, j].fill_between(pos_freqs, P[freqs >= 0, i, j], 0)
				axes[i, j].set_xlim([0, np.max(pos_freqs)])
				axes[i, j].set_ylim([0, 1])
		plt.suptitle(name)
		plt.tight_layout()


	# =============== Step 1: Low pass filter =============== #

	# Filter parameters
	order = 6
	fs = 1000 # sample rate, Hz
	cutoff = 120 # cutoff frequency of the filter, Hz

	y = np.zeros((N,int(data[1].size)))

	for ch in range(0,N):

		# Filter the data
		y[ch] = butter_lowpass_filter(data[ch], cutoff, fs, order)


		# =============== Step 2: Downsample =============== #

		dsrate = 250 # downsampled sampling rate
		interval = fs/dsrate
		j = 0
		y_ds = np.zeros((N,int(data[1].size/interval)))
		t_ds = y_ds
		for i in range(0,y[ch].size): # downsample low pass filtered iEEG data
			if i % interval == 0:
				y_ds[ch,j] = y[ch,i]
				j += 1

		j = 0
		for i in range(0,t[ch].size): # downsample sample vector
			if i % interval == 0:
				t_ds[ch,j] = t[ch,i]
				j += 1


		# =============== Step 3: Notch Filter =============== #

		# Apply notch filter around 60 Hz
		bp_stop_hz = np.array([59.0,61.0])
		b, a = butter(2,bp_stop_hz/(dsrate/2.0),'bandstop')
		filtered = np.zeros(y_ds.shape)
		filtered[ch] = lfilter(b, a, y_ds[ch])


		# =============== Step 4: Normalization =============== #

		# subtract mean and divide by standard deviation to normalize
		mean = np.zeros((N,))
		std = np.zeros((N,))
		normalized = np.zeros((y_ds.shape))
		mean[ch] = np.mean(filtered[ch])
		std[ch] = np.std(filtered[ch])
		normalized[ch] = (filtered[ch] - mean[ch])/std[ch]


		# =============== Step 5: Calculate PDC =============== #

		plt.close('all')
		# compute PDC
		P, freqs = PDC(A)
		plot_all(freqs, P, 'PDC')


		# =============== Step 6: Build Graph =============== #

		time = 0 # Choose time (sample number) to show graph at

		# Create directed graph
		G = nx.DiGraph()
		for node1 in connections:
			G.add_node(node1) # add a node for each iEEG channel in connections

		i = 0
		for node1 in connections: # set weights of edges as PDC
			j = 0
			for node2 in connections:
				G.add_edge(node1,node2)
				G[node1][node2]['weight'] = P[time,i,j]
				print(node1,end=' '), print(node2, end = ' '), print(P[time,i,j])
				j += 1
			i += 1
		plt.figure() # show graph
		nx.draw(G)
		plt.show()
		return G

	# # Some code taken from https://gist.github.com/agramfort/9875439 (with deletes and edits)
    #
	# """
	# Reference
	# ---------
	# Luiz A. Baccala and Koichi Sameshima. Partial directed coherence:
	# a new concept in neural structure determination.
	# Biological Cybernetics, 84(6):463:474, 2001.
	# """
    #
	# # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
	# #
	# # License: BSD (3-clause)
    #
	# def butter_lowpass(cutoff, fs, order=5):
	# 	nyq = 0.5 * fs
	# 	normal_cutoff = cutoff / nyq
	# 	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	# 	return b, a
    #
	# def butter_lowpass_filter(data, cutoff, fs, order=5):
	# 	b, a = butter_lowpass(cutoff, fs, order=order)
	# 	y = lfilter(b, a, data)
	# 	return y
    #
    #
	# def spectral_density(A, n_fft=None):
	# 	"""Estimate PSD from AR coefficients
	# 	Parameters
	# 	----------
	# 	A : ndarray, shape (p, N, N)
	# 		The AR coefficients where N is the number of signals
	# 		and p the order of the model.
	# 	n_fft : int
	# 		The length of the FFT
	# 	Returns
	# 	-------
	# 	fA : ndarray, shape (n_fft, N, N)
	# 		The estimated spectral density.
	# 	"""
	# 	p, N, N = A.shape
	# 	if n_fft is None:
	# 		n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
	# 	A2 = np.zeros((n_fft, N, N))
	# 	A2[1:p + 1, :, :] = A  # start at 1 !
	# 	fA = fftpack.fft(A2, axis=0)
	# 	freqs = fftpack.fftfreq(n_fft)
	# 	I = np.eye(N)
    #
	# 	for i in range(n_fft):
	# 		fA[i] = linalg.inv(I - fA[i])
    #
	# 	return fA, freqs
    #
	# def PDC(A, sigma=None, n_fft=None):
	# 	"""Partial directed coherence (PDC)
	# 	Parameters
	# 	----------
	# 	A : ndarray, shape (p, N, N)
	# 		The AR coefficients where N is the number of signals
	# 		and p the order of the model.
	# 	sigma : array, shape (N,)
	# 		The noise for each time series.
	# 	n_fft : int
	# 		The length of the FFT.
	# 	Returns
	# 	-------
	# 	P : ndarray, shape (n_fft, N, N)
	# 		The estimated PDC.
	# 	"""
	# 	p, N, N = A.shape
    #
	# 	if n_fft is None:
	# 		n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
    #
	# 	H, freqs = spectral_density(A, n_fft)
	# 	P = np.zeros((n_fft, N, N))
    #
	# 	if sigma is None:
	# 		sigma = np.ones(N)
    #
	# 	for i in range(n_fft):
	# 		B = H[i]
	# 		B = linalg.inv(B)
	# 		V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
	# 		V = np.diag(V)  # denominator squared
	# 		P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]
    #
	# 	return P, freqs
    #
    #
	# def plot_all(freqs, P, name):
	# 	"""Plot grid of subplots
	# 	"""
	# 	m, N, N = P.shape
	# 	pos_freqs = freqs[freqs >= 0]
    #
	# 	f, axes = plt.subplots(N, N)
	# 	for i in range(N):
	# 		for j in range(N):
	# 			axes[i, j].fill_between(pos_freqs, P[freqs >= 0, i, j], 0)
	# 			axes[i, j].set_xlim([0, np.max(pos_freqs)])
	# 			axes[i, j].set_ylim([0, 1])
	# 	plt.suptitle(name)
	# 	plt.tight_layout()
    #
    #
	# # =============== Step 1: Low pass filter =============== #
    #
	# # Filter parameters
	# order = 6
	# fs = 1000 # sample rate, Hz
	# cutoff = 120 # cutoff frequency of the filter, Hz
    #
	# y = np.zeros((N,int(CH1_5[1].size)))
    #
	# for ch in range(0,N):
    #
	# 	# Apply low pass filter to each channel
	# 	y[ch] = butter_lowpass_filter(CH1_5[ch], cutoff, fs, order)
    #
    #
	# 	# =============== Step 2: Downsample =============== #
    #
	# 	dsrate = 250 # downsampled sampling rate
	# 	interval = fs/dsrate
	# 	j = 0
	# 	y_ds = np.zeros((N,int(CH1_5[1].size/interval)))
	# 	t_ds = y_ds
	# 	for i in range(0,y[ch].size): # downsample low pass filtered iEEG data
	# 		if i % interval == 0:
	# 			y_ds[ch,j] = y[ch,i]
	# 			j += 1
    #
	# 	j = 0
	# 	for i in range(0,t[ch].size): # downsample sample vector
	# 		if i % interval == 0:
	# 			t_ds[ch,j] = t[ch,i]
	# 			j += 1
    #
    #
	# 	# =============== Step 3: Notch Filter =============== #
    #
	# 	# Apply notch filter around 60 Hz
	# 	bp_stop_hz = np.array([59.0,61.0])
	# 	b, a = butter(2,bp_stop_hz/(dsrate/2.0),'bandstop')
	# 	filtered = np.zeros(y_ds.shape)
	# 	filtered[ch] = lfilter(b, a, y_ds[ch])
    #
    #
	# 	# =============== Step 4: Normalization =============== #
    #
	# 	# subtract mean and divide by standard deviation to normalize
	# 	mean = np.zeros((N,))
	# 	std = np.zeros((N,))
	# 	normalized = np.zeros((y_ds.shape))
	# 	mean[ch] = np.mean(filtered[ch])
	# 	std[ch] = np.std(filtered[ch])
	# 	normalized[ch] = (filtered[ch] - mean[ch])/std[ch]
    #
    #
	# 	# =============== Step 5: Calculate PDC =============== #
    #
	# 	plt.close('all')
	# 	# compute PDC
	# 	P, freqs = PDC(A)
	# 	plot_all(freqs, P, 'PDC')
    #
    #
	# 	# =============== Step 6: Build Graph =============== #
    #
	# 	time = 0 # Choose time (sample number) to show graph at
    #
	# 	# Create directed graph
	# 	G = nx.DiGraph()
	# 	for node1 in connections:
	# 		G.add_node(node1) # add a node for each iEEG channel in connections
    #
	# 	i = 0
	# 	for node1 in connections: # set weights of edges as PDC
	# 		j = 0
	# 		for node2 in connections:
	# 			G.add_edge(node1,node2)
	# 			G[node1][node2]['weight'] = P[time,i,j]
	# 			print(node1,end=' '), print(node2, end = ' '), print(P[time,i,j])
	# 			j += 1
	# 		i += 1
	# 	plt.figure() # show graph
	# 	nx.draw(G)
	# 	plt.show()
	# 	return G


def dtf_network(data, connections):

	"""
	dtf_network()
		Creates networks from iEEG data using given connections.  Each directed edge has a
		directed transfer function (DTF) vector.
	INPUTS:
		data 			==> an n x s numpy array where n is the number of channels in the iEEG
								recording and s is the number of samples in the recording
		connections 	==> a set containing either
								(i) integers from 1 to n, denoting vertices in the graph
								(ii) lists of integers from 1 to n, denoting directed edges in the graph
	OUTPUT:
		G 	==> a directed networkx graph.  Each directed edge has DTF vector
	"""

	# fit the MVAR and compute DTF coefficients
	# NOTE: p and n_fft are currently NOT inputs to function.  BRING THIS DECISION UP
	A, sigma = mvar_fit(data, p)
	D, freqs = DTF(A, sigma, n_fft)

	# get the edges
	if type(connections[0]) is list:
		edges = connections
	elif type(connections[0]) is int:
		edges = []
		for node1 in connections:
			for node2 in connections:
				if node1 != node2:
					edges.append([edge1,edge2])

	# build the graph with normalized DFT as edge weights
	G = nx.DiGraph()
	for edge in edges:
		dft = D[:, edge[1], edge[0]] / sum(D[:, :, edge[0]], axis=1)
		attr = {"DFT": dft}
		G.add_edge(edge[0], edge[1], attr)

	return G

def gc_network(data, connections):
	return

def di_network(data, connections):
	return


# =============== Import data =============== #

# Load iEEG data
mat = scipy.io.loadmat('C:\\Users\\Megan Kehoe\\Documents\\VIP\\Patient Data\\TA023\\TA023_08aug2009_18_23_06_Seizure.mat')
iEEG = mat['record_RMpt5']
N = 5 # Number of channels selected
CH1_5 = iEEG[0:N-1] # Select first 5 channels
start = 0
stop = 1000
CH1_5 = CH1_5[:,start:stop] # Take 1s window
t = np.zeros((N,int(CH1_5[0].size))) # Create sample number vector for iEEG data
for i in range(0,N):
	t[i,:] = np.arange(0,int(CH1_5[1].size))

# Load AR coefficients calculated for data in Matlab file
mat2 = scipy.io.loadmat('C:\\Users\\Megan Kehoe\\Documents\\VIP\\AR.mat')
A = mat2['Am']
p = 5 # MVAR model order
A = np.reshape(A,(p,N,N))


build_network(CH1_5,['1','2','3','4','5'],"partial_directed_coherence")
