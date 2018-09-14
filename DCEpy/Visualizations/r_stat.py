import matplotlib.pyplot as plt
import numpy as np

from DCEpy.Features.BurnsStudy.rstat_42 import calc_rstat

"""
Demonstrates use of preprocessing functions and rstat
"""

# Load iEEG data
#mat = scipy.io.loadmat(r'C:\Users\Megan Kehoe\Documents\VIP\Patient Data\TA023\TA023_08aug2009_18_23_06_Seizure.mat')
#iEEG = mat['record_RMpt5']
#ch1 = iEEG[0]

# take random data
bands = np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])

ch1 = np.random.rand(300000,1)



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
band, f_ict, Pxx_ict, f_inter, Pxx_inter = calc_rstat(x_ict,x_inter,fs,bands)



f = f_ict
# Plot power spectrum for first time window
plt.subplot(131)
plt.semilogy(f, Pxx_ict)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [mV**2]')
plt.title('Ictal Power Spectrum')

plt.subplot(132)
plt.semilogy(f, Pxx_inter)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectrum [mV**2]')
plt.title('Preictal Power Spectrum')

plt.subplot(133)
r_stat = Pxx_inter/Pxx_ict
plt.loglog(f,r_stat)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Frequency Ratio [mV**2]')
plt.title('R-satistic plot')

colors = 'kbgrm'
for color_index,freq_band in enumerate(bands):
    color = colors[color_index%len(colors)]
    plt.vlines(freq_band,ymin=min(r_stat),
               ymax = max(r_stat),colors=color,linestyles='dashed')





plt.show()
