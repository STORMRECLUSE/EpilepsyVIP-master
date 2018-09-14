from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from DCEpy.Features.GardnerStudy.edfread import edfread

filenames = ('/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101U_1-1+.edf',
             '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101V_1-1+.edf',
             '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101W_1-1+.edf',
             '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101P_1-1_02oct2010_09_00_38_Awake+.edf'
             )
good_channels = ['LAH1','LAH2','LPH6','LPH7',
                 #'LPH9','LPH10','LPH11','LPH12'
                 ]
seizure_starts = 262,\
                  107,191,0
seizure_ends   = 330,\
                  287,405,0



window_len = 10000
window_step = 1000


# choose the function that calculates a feature based on data.
# input args to function as keyword dict
from DCEpy.Features.ARModels.prediction_err import mvar_pred_err
inter_fxn = mvar_pred_err
extra_params = {'order':7,'pred_window_size':4000}
feat_name = 'Prediction_val' #don't forget to give a good ylabel!
n_feats = len(good_channels)*2

for plot_no,(filename, seizure_start, seizure_end) in \
        enumerate(zip(filenames,seizure_starts,seizure_ends)):
    data,_,labels = edfread(filename,good_channels=good_channels)

    ## Insert preprocessing steps


    tim = np.arange(window_len,step = window_step,stop = np.size(data,axis=0))
    feat_plot = np.zeros((np.size(tim,0),n_feats))
    for index,end_tim in enumerate(tim):
        feat_plot[index,:] = inter_fxn(data[end_tim-window_len:end_tim],**extra_params)


    plt.subplot(1,len(filenames),plot_no+1)
    plt.plot(tim/1000,feat_plot)
    plt.xlabel('Time(s)')
    plt.ylabel(feat_name)

    range_range = np.max(feat_plot)-np.min(feat_plot)
    plt.vlines((seizure_start,seizure_end),np.min(feat_plot) -.3*range_range ,
               np.max(feat_plot + .3*range_range),'g',linestyles='dashed')

plt.show()
a = 1