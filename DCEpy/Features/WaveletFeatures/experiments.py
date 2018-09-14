# import wave
from DCEpy.General.DataInterfacing.edfread import edfread
# from scipy.io import wavfile
# from DCEpy.Features.Wavelet_Transform import wavelet_decomp
# from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import lyapunov_exponent
import numpy as np
# import scipy.stats
# # from wavelet_features import wavelet_non_window_features
# import matplotlib.pyplot as plt
# from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import signal_entropy
# from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import lyapunov_exponent
# from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import teager_energy
# from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import mcl
#
# ## EEG signal
"""Wavelet Single Features"""
# #-------------------read in patient 41, channel LAH2(index=1)-----------------------
# seizure_file="/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS041/DA00101W_1-1+.edf"
#
# X, _, labels = edfread(seizure_file)    #X is ndarray with shape (number of samples, number of channels), labels are the name for the output channels
# X= np.array(X)
# X_ch1=X[:,100]
# print X_ch1.shape
# coeff1=wavelet_features.discrete_wavelet_decomp(X=X_ch1,f_s=1000,plot=False, seizure_time=(191,405))
# coeff_mat=coeff1
#
#
# # inter_file = "/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS041/CA00100E_1-1_03oct2010_01_03_05_Awake+.edf"
# # X, _, labels = edfread(inter_file)
# # X= np.array(X)
# # X_ch1_awake=X[:,100]
# # coeff2=wavelet_features.discrete_wavelet_decomp(X=X_ch1_awake,f_s=1000,plot=False)
# # coeff_mat=coeff2
# # new_X= np.hstack((X_ch1_awake,X_ch1))
# #
# # new_coeff = np.vstack((coeff2.transpose(),coeff1.transpose())).transpose()
# # coeff_mat=new_coeff
# # print new_coeff.shape
# # from wavelet_features import plot_dwt
# # # plot_dwt(X=new_X,f_s=1000,coeff_matrix=new_coeff,signal_name="unnormalized+41+inter+seizure", seizure_time=(191+1267,405+1267))
# #
# # feature_matrix = wavelet_non_window_features(new_coeff)
# # print feature_matrix.size
# # plot_dwt(X=new_X,f_s=1000,coeff_matrix=feature_matrix,signal_name="unnormalized+41+inter+seizure", seizure_time=(191+1267,405+1267))
#
#
# # ------------------read in patient 23, channel HD1,MST1,MST2---------------------
# seizure_file = "/Users/TianyiZhang/Documents/EpilepsyVIP/data/TA023/DA001020_1-1+.edf"
# X, _, labels = edfread(seizure_file)
# good_channel_names = ["HD1","MST1","MST2"]
# good_channel_idxs = [82, 107, 108]
# X_ch=X[:,82]
#
# start=56*60 + 42
# end=57*60 + 42
# coeff_mat = wavelet_decomp.discrete_wavelet_decomp(X=X_ch, f_s=1000, plot=True, seizure_time=(start, end))
#
#
# # ##linchirp and whale sounds
# #
# # bird_file="sp_lifeclef.wav"
# # fs_1, bird = wavfile.read(bird_file)
# # print "bird",bird
# # # fs_1=44100, human ear range 200~20000
# # # file_time = 41 seconds
#
# # whale_file="sp.NIPS4B.wav"
# # fs_2, whale = wavfile.read(whale_file)
# # # fs_2=44100
# # # file_time = 20 seconds
# # a= np.random.rand(20,40,60)
# # c = signal_entropy(a,axis=0)
#
#
# ## Debugging wavelet features
# # coeff_mat = wavelet_features.discrete_wavelet_decomp(whale,fs_2,plot=True)
# norms = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func= lambda x: np.einsum('ij,ij ->i', x, x),
#                                                 numpy_style=False)
# sums    = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func= np.sum)
# stds    = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func=np.std)
# entropy = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func=signal_entropy)
# ly_exponent = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func=lyapunov_exponent)
# teager = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func=teager_energy)
# mcl = wavelet_decomp.window_features_from_mat(coeff_mat, window_step=1000, window_len=2000, window_func=mcl)
# num_features=6
#
# print "shape of average",norms.shape
#
#
# plt.figure(2)
# plt.subplot(num_features,1,1)
# plt.title('Square Sums (log scale)')
# plt.axvline(x=start-2, color="b")
# plt.axvline(x=end-2, color="b")
# plt.imshow(np.log(norms.T + 1e-6),aspect='auto',interpolation='nearest')
# plt.subplot(num_features,1,2)
# plt.title('Entropy of Frequencies (log scale)')
# plt.imshow(entropy.T,aspect='auto',interpolation='nearest')
# plt.axvline(x=start-2, color="b")
# plt.axvline(x=end-2, color="b")
# plt.subplot(num_features,1,3)
# plt.title('Standard Devs (log scale)')
# plt.imshow(np.log(stds.T + 1e-6),aspect='auto',interpolation='nearest')
# # plt.axvline(x=56*60 + 42-2, color="b")
# # plt.axvline(x=57*60 + 42-2, color="b")
#
# plt.subplot(num_features,1,4)
# plt.title('Lyapunov Exponent (log scale)')
# plt.imshow(ly_exponent.T,aspect='auto',interpolation='nearest')
# plt.axvline(x=start-2, color="b")
# plt.axvline(x=end-2, color="b")
#
# plt.subplot(num_features,1,5)
# plt.title('Teager Energy (log scale)')
# plt.axvline(x=start-2, color="b")
# plt.axvline(x=end-2, color="b")
# plt.imshow(np.log(teager.T+1e-6),aspect='auto',interpolation='nearest')
#
#
# plt.subplot(num_features,1,6)
# plt.title('Mean Curve Length (log scale)')
# plt.axvline(x=start-2, color="b")
# plt.axvline(x=end-2, color="b")
# plt.imshow(mcl.T,aspect='auto',interpolation='nearest')
#



"""Test Channels For Asleep and Awake channels"""

# seizure_file="/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS041/DA00101W_1-1+.edf"
#
# X_seizure, _, labels_seizure = edfread(seizure_file)    #X is ndarray with shape (number of samples, number of channels), labels are the name for the output channels
# print "how many channels for seizure file?",len(labels_seizure)
#
#
# asleep_file= "/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS041/DA00101Q_1-1_02oct2010_03_00_05_Sleep+.edf"
#
# X_asleep, _,labels_asleep = edfread(asleep_file)
#
# print "how many channels for asleep file?",len(labels_asleep)
#
# awake_file= "/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS041/DA00101P_1-1_02oct2010_09_00_38_Awake+.edf"
#
# X_awake, _,labels_awake = edfread(awake_file)
#
# print "how many channels for asleep file?",len(labels_awake)




file= "/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS039/CA00100D_1-1+.edf"
X,_,labels = edfread(file)
expected="RAH3"
print labels.index(expected)


"""Test Filters"""

# # Filter a noisy signal.
# T = 0.05
# nsamples = T * fs
# t = np.linspace(0, T, nsamples, endpoint=False)
# a = 0.02
# f0 = 600.0
# x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
# x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
# x += a * np.cos(2 * np.pi * f0 * t + .11)
# x += 0.03 * np.cos(2 * np.pi * 2000 * t)
# plt.figure(2)
# plt.clf()
# plt.plot(t, x, label='Noisy signal')
# y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
# plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
# plt.xlabel('time (seconds)')
# plt.hlines([-a, a], 0, T, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')
# plt.show()

from scipy.io import wavfile
# from Wavelet_coherence import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from DCEpy.General.DataInterfacing.edfread import edfread

# #bird
# bird_file="sp_lifeclef.wav"
# fs_1, bird = wavfile.read(bird_file)
# print "bird",bird
# # fs_1=44100, human ear range 200~20000
# # file_time = 41 seconds
#
# #seizure
# seizure_file="/Users/TianyiZhang/Documents/EpilepsyVIP/data/TS041/DA00101W_1-1+.edf"
# X, _, labels = edfread(seizure_file)    #X is ndarray with shape (number of samples, number of channels), labels are the name for the output channels
# X= np.array(X)
# X=X[:,100]
#
# 191,405
# x=X[100000:500000]
# fs=1000
#
# # f=plt.figure()
#
# #unfiltered signal time domain
# t=len(x)   #in samples
# x1=preprocessing(x,fs)
# times= np.linspace(0,t-1,t)
#
# plt.subplot(211)
# plt.plot(times, x)
# plt.xlim(0, times[-1])
# plt.title('Filter input - Time Domain')
# # plt.grid(True)
#
#
#
# plt.plot(times, x1)
# # plt.xlim(0, times[-1])
# # plt.title('Filter output - Time Domain')
# plt.grid(True)
#
#
# # unfiltered signal frequency domain
# xfreq = np.fft.fft(x)
# fft_freqs = np.fft.fftfreq(t, d=1./fs)
# plt.subplot(212)
# plt.loglog(fft_freqs[0:t/2], np.abs(xfreq)[0:t/2])
# # plt.title('Filter input - Frequency Domain')
# plt.text(0.03, 0.01, "freqs: "+" Hz")
# # plt.grid(True)
#
#
#
# xfreq = np.fft.fft(x1)
# fft_freqs = np.fft.fftfreq(t, d=1./fs)
# plt.loglog(fft_freqs[0:t/2], np.abs(xfreq)[0:t/2])
# # plt.title('Filter input - Frequency Domain')
# plt.text(0.03, 0.01, "freqs: "+" Hz")
# plt.grid(True)
#
# plt.show()