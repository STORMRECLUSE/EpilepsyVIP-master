import scipy.signal
import numpy as np
import math
from DCEpy.Features.Preprocessing.preprocess import lpf, notch, normalize

def calc_rstat(x_ict,x_inter,fs,mode):
	"""
	Choose the frequency band of interest by calculating the band with the maximum R statistic 
	(the ratio of power spectra in ictal state to interictal state). 
	
	Inputs:
	-------
	x_ict = ictal time-series data: nxN array, n = number of samples, N = number of channels
	x_inter = interictal time-series data: nxN array, n = number of samples, N = number of channels
	fs = sampling rate, in Hz: int,float
	mode = whether zero-phase filtering is used in preprocessing: 0 = no, 1 = yes
	
	Output:
	-------
	band = selected frequency band, Hz: 1x2 array
	"""
	band_ch = np.zeros([x_ict.shape[0],2])
	for ch in range(x_ict.shape[0]):
		band_ch[ch] = calc_band(x_ict[ch],x_inter[ch],fs,mode)
	band,count = stats.mode(band_ch)
	return band
	
def calc_band(x_ict,x_inter,fs,mode=0):
	
	seg = 1000
	f_ict = np.zeros([math.ceil(x_ict.size/2500.0),seg/2+1])
	f_inter = np.zeros([math.ceil(x_inter.size/2500.0),seg/2+1])
	Pxx_ict = f_ict
	Pxx_inter = f_inter
	# Preprocessing and compute power spectrum for ictal and interictal data
	for i in range(int(math.ceil(x_inter.size/2500.0))):
		x_inter_seg = x_inter[1500*i:1500*i+2499]
		print x_inter_seg.shape
		x_inter_seg = lpf(x_inter_seg,fs,120.0,6,mode)
		x_inter_seg = notch(x_inter_seg,59.0,61.0,fs,mode)
		x_inter_seg = normalize(x_inter_seg)
		f, Pxx = scipy.signal.welch(x_inter_seg.T, fs, nperseg=seg, noverlap=750, scaling='spectrum')
		f_inter[i,:] = f
		Pxx_inter[i,:] = Pxx
	
	for i in range(int(math.ceil(x_ict.size/2500.0))):
		x_ict_seg = x_ict[1500*i:1500*i+2499]
		x_ict_seg = lpf(x_ict_seg,fs,120.0,6,mode)
		x_ict_seg = notch(x_ict_seg,59.0,61.0,fs,mode)
		x_ict_seg = normalize(x_ict_seg)
		f, Pxx = scipy.signal.welch(x_ict_seg.T, fs, nperseg=seg, noverlap=750, scaling='spectrum')
		f_ict[i,:] = f
		Pxx_ict[i,:] = Pxx
		
	# Frequency bands of interest
	bands = np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])

	# Average the power spectra over the frequency bands of interest
	# and calculate r statistic
	rstat = np.zeros(bands.shape[0])
	for b in range(0,bands.shape[0]):
		b_ict = np.mean(Pxx_ict[:,bands[b,0]:bands[b,1]])
		b_inter = np.mean(Pxx_inter[:,bands[b,0]:bands[b,1]])
		rstat[b] = b_ict/b_inter
	print(rstat)
	
	# Choose band with maximum r statistic
	I = np.argmax(rstat)
	band = bands[I]
	
	f_ict = f_ict.T
	Pxx_ict = Pxx_ict.T 
	f_inter = f_inter.T 
	Pxx_inter = Pxx_inter.T 
	
	return band, f_ict, Pxx_ict, f_inter, Pxx_inter
