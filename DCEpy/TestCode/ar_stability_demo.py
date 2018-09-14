import matplotlib.pyplot as plt

from DCEpy.Features.ARModels.ar_stability import ar_stability_window
from DCEpy.Features.GardnerStudy.edfread import edfread

__author__ = 'vsp'

filenames = ('/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101U_1-1+.edf',
             '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101V_1-1+.edf',
             '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101W_1-1+.edf',
             '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101P_1-1_02oct2010_09_00_38_Awake+.edf')
good_channels = ['LAH1','LAH2','LPH6','LPH7','LPH9','LPH10','LPH11','LPH12']
seizure_starts = 262,107,191,0
seizure_ends   = 330,287,405,0

for plot_no,(filename, seizure_start, seizure_end) in \
        enumerate(zip(filenames,seizure_starts,seizure_ends)):
    data,_,labels = edfread(filename,good_channels=good_channels)
    # data = normalize(np.random.rand(3e4,5))
    #data, A = artif_VAR_data(N=5, n=2000, p=4, burn=50, A_type="tridiag")
    eig_seq,tim = ar_stability_window(data, order=15, n_eigs = 8, w_len=5000, w_gap = 1000)
    plt.subplot(1,len(filenames),plot_no+1)
    plt.plot(tim/1000,eig_seq)
    plt.xlabel('Time(s)')
    plt.ylabel('Top Eigenvalues')
    plt.vlines((seizure_start,seizure_end),0.9,1.04,'g',linestyles='dashed')

plt.show()
a = 1