from __future__ import print_function

# Emily
# sys.path.append('C:\Users\User\Documents\GitHub\EpilepsyVIP')

from DCEpy.Features.Graphs.build_network import build_network
from DCEpy.Features.Graphs.threshold import graph_thresh
from DCEpy.Features.GardnerStudy.edfread import edfread
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# filenames = ('/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101U_1-1+.edf',
#              '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101V_1-1+.edf',
#              '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101W_1-1+.edf',
#              '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101P_1-1_02oct2010_09_00_38_Awake+.edf')
# good_channels = ['LAH1','LAH2','LPH6','LPH7','LPH9','LPH10','LPH11','LPH12']
# seizure_starts = 262,107,191,0
# seizure_ends   = 330,287,405,0
#
# for plot_no,(filename, seizure_start, seizure_end) in \
#         enumerate(zip(filenames,seizure_starts,seizure_ends)):
#     data,_,labels = edfread(filename,good_channels=good_channels)

weight_types = ['pli', 'imag_coherency', 'coherence']
thresh_param_types = [{'thresh_val':.002},{'thresh_val':0},{'thresh_val':0}]


# filename = '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs/DA00101U_1-1+.edf' # 42
# filename = 'C:\Users\User\Documents\EpilepsySeniorDesign\Burns\DA00101U_1-1+.edf' # Emily
filename = '/home/chris/Documents/Rice/senior/EpilepsyVIP/data/RMPt2/DA00101U_1-1+.edf' # Chris

good_channels = ['LAH1','LAH2','LPH6','LPH7','LPH9','LPH10','LPH11','LPH12']
seizure_start,seizure_end = 262,330
#falsify data. Get actual data later
data,_,labels = edfread(filename,good_channels=good_channels)

#data,annotations,labels  = edfread(filename)
connections = list(range(8))

data_len,nchannels = np.shape(data)

window_size = int(5e3)
window_increment = 1250


#main code loop
## preprocessing

processed_data = data
print('Processed Data has size ' + str(processed_data.shape))


line_handles = []
for weight,thresh_params in zip(weight_types,thresh_param_types):
    cc_vect,t_vect = [],[]
    print('Weight Type: ' + weight)

    for end_time in range(window_size,data_len,window_increment):

        ## build network
        G = build_network(processed_data[(end_time-window_size):end_time][:],connections,weight)

        ## Threshold
        graph_thresh(G,weight,**thresh_params)

        ## Compute CC, store in vector
        avg_coeff = nx.average_clustering(G,weight=weight)
        cc_vect.append(avg_coeff)
        t_vect.append(end_time/1000)


    t_arr,cc_arr = np.array(t_vect),np.array(cc_vect)
    line_handles.append(plt.plot(t_arr,cc_arr))

plt.legend(weight_types)
plt.vlines((seizure_start,seizure_end),-.3,.9,'g',linestyles='dashed')
plt.xlabel('Time (s)')
plt.ylabel('CC of Networks')
plt.show()
a = 1