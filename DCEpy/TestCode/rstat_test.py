import sys
sys.path.append('C:\Users\User\Documents\GitHub\EpilepsyVIP')

import numpy as np
import matplotlib.pyplot as plt
from DCEpy.Features.BurnsStudy.rstat_42 import calc_rstat

"""
Demonstrates use of preprocessing functions and rstat
"""

# Load iEEG data
#mat = scipy.io.loadmat(r'C:\Users\Megan Kehoe\Documents\VIP\Patient Data\TA023\TA023_08aug2009_18_23_06_Seizure.mat')
#iEEG = mat['record_RMpt5']
#ch1 = iEEG[0]

# take random data
ch1 = np.random.rand(300000,1)

bands = np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])

N = 10000
x_inter = ch1[0:N-1] # Take N samples during interictal period
x_ict = ch1[260000:260000+N-1] # Take N samples during ictal period
fs = 1000 # sampling rate in Hz

x_inter = np.reshape(x_inter,(1,x_inter.size)) # Nxn
x_ict = np.reshape(x_ict,(1,x_ict.size)) # Nxn
x_inter = x_inter.T #nxN
x_ict = x_ict.T #nxN

# Calculate R statistic to choose frequency band
# calc_rstat performs necessary preprocessing itself
band, f_ict, Pxx_ict, f_inter, Pxx_inter = calc_rstat(x_ict,x_inter,fs)


# Plot power spectrum for first time window
plt.subplot()
plt.semilogy(f_ict, Pxx_ict)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [mV**2]')
plt.title('Ictal Power Spectrum')

plt.figure()
plt.semilogy(f_ict, Pxx_inter)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [mV**2]')
plt.title('Preictal Power Spectrum')




plt.show()
