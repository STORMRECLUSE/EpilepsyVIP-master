from __future__ import print_function

import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import hsv2rgb
import numpy as np

from DCEpy.General.DataInterfacing.edfread import edfread
import SOMPY

#This is not on pip! Download the working copy from https://github.com/sevamoo/SOMPY
#into the site-packages folder where your Python is installed.

## features imported


#TODO: Save figures generated from viz

label_hues = {'Interictal Asleep':0, 'Interictal Awake':.15, 'Preictal':.35,
                       'Ictal':.6,'Postictal':.8}
label_indices = {'Interictal Asleep':0, 'Interictal Awake': 1, 'Preictal':2,
                       'Ictal':3,'Postictal':4}
label_hue_arr = np.zeros(len(label_indices))

for label,index in label_indices.items():
    label_hue_arr[index] = label_hues[label]

def prepare_and_store_data(patient_paths, window_size,window_step,feature_func = None, read_chunktime=300, info_pickle_name ='patient_pickle.txt',
                           feature_type='Single',feature_pickle_name = 'correlation.txt',channel_name=None, fs = 1000.,**feature_params):
    '''
    A generator that, on a given patient, computes the necessary features across all of the windows commanded.
    :param patient_paths: The paths to the folders containing patient data. A list.
    :param window_size: The window size, in seconds, associated with the method
    :param window_step: The window step, in seconds, of the feature.
    :param feature_func: The function, that given a window of data, will compute the necessary feature.
    :param read_chunktime: For long files, data is chunked up into segments no longer than read_chunktime seconds.
    :param info_pickle_name: The name of the file where information about patient data is stored. This is necessary
    :param feature_type: A string, "Single" or "Multiple" which describes the type of feature given.
                         Is it computed on a single or multiple channel(s) at a time?
    :param feature_pickle_name: The file where the computed features will be stored upon computing.
    :param channel_name: The names of the channels used to compute the features.
    :param fs: Sampling frequency of the files.
    :param **feature_params:
    :return: A list of dictionaries containing the features computed on each window.
    '''
    def compute_features_from_file(seiz_file, window_size,window_step,fs, chunk_time,channel_name = None,
                                   feature_type ='Single', feature_func = None, **feature_params):
        '''
        the subfunction that interfaces with each file. Returns a list of dicts ("records") that include window start time
        :param seiz_file: the file containing iEEG data
        :param patient_info: the patient info, taken from the pickle file
        :param window_size: the size of the window, in samples.
        :param window_step: The step between each sliding window, in samples.
        :param chunk_time:
        :param channel_name:
        :param feature_type:
        :param fs:
        :param feature_params:
        :return rec_list:
        '''
        def chunk_file_iterator(chunk_samples,fs,seiz_file, good_channels = None):

            """
            iterates over chunks of data. Useful for long (longer than 5 minutes) edf files, saves memory.
            """

            n = 0
            try:
                while True:
                    data,_,labels = edfread(seiz_file,
                                            rec_times=(n*chunk_samples/float(fs),(n+1)*chunk_samples/float(fs)),
                                            good_channels=good_channels)
                    #data_is_full = np.size(data,0)>=chunk_samples
                    start_time = n*chunk_samples/float(fs)
                    yield data, start_time,n, labels
                    n+=1
            except ValueError:
                pass

        if feature_func is None:
            raise ValueError('No feature function was input!!')

        if feature_type =='Single':
            channel_name = [channel_name]

        num_windows = np.round((chunk_time*fs - window_size)/float(window_step)+1)
        actual_chunksamples = window_step*(num_windows-1) + window_size
        rec_list = []
        for data_chunk,chunk_start,chunk_num, labels in chunk_file_iterator(actual_chunksamples,fs,seiz_file, good_channels=channel_name):
            for start_sample in range(0,np.size(data_chunk,0),window_step):
                rec = {}
                window = data_chunk[start_sample:start_sample+window_size,:]
                window_end_time = (actual_chunksamples*chunk_num +
                                   start_sample + window_size)/float(fs)
                if feature_type =='Single':
                    rec['feats'] = feature_func(window[:,[labels.index(channel_name[0])]],**feature_params)


                elif feature_type =='Multiple':
                    rec['feats'] = feature_func(window,**feature_params)

                rec['wind_end']= window_end_time
                rec['preict_time'] = np.nan
                rec_list.append(rec)
        return rec_list

    def process_seiz_file(seiz_file,patient_info,index):
        prelim_feats = compute_features_from_file(seiz_file, window_size,window_step,fs, read_chunktime,
                                                  feature_func=feature_func, feature_type=feature_type,channel_name=channel_name,**feature_params)
        seiz_start,seiz_end = patient_info['seizure_times'][index]
        for rec in prelim_feats:
            if rec['wind_end'] < seiz_start:
                rec['label'] = "Preictal"
                rec['preict_time'] = seiz_start- rec['wind_end']
            elif seiz_start<=rec['wind_end']<=seiz_end:
                rec['label'] = 'Ictal'
            else:
                rec['label'] = 'Postictal'

        return prelim_feats

    def process_inter_files(patient_path,patient_info):

        awake_files = patient_info['awake_inter_filenames']
        inter_feats = []
        for inter_file in awake_files:
            inter_file = os.path.join(patient_path,inter_file)
            prelim_feats = compute_features_from_file(inter_file,window_size,window_step,fs, read_chunktime,
                                                  feature_func=feature_func, feature_type=feature_type,
                                                      channel_name=channel_name,**feature_params)
            for rec in prelim_feats:
                rec['label'] = 'Interictal Awake'
            inter_feats.extend(prelim_feats)

        asleep_files = patient_info['asleep_inter_filenames']
        for inter_file in asleep_files:
            inter_file = os.path.join(patient_path,inter_file)
            prelim_feats = compute_features_from_file(inter_file,window_size,window_step,fs, read_chunktime,
                                                  feature_func=feature_func, feature_type=feature_type,
                                                      channel_name=channel_name,**feature_params)
            for rec in prelim_feats:
                rec['label'] = 'Interictal Asleep'
            inter_feats.extend(prelim_feats)
        return inter_feats


    for patient_path in patient_paths:
        file_save_path= os.path.join(patient_path,'Results')
        info_pickle_path = os.path.join(patient_path, info_pickle_name)
        feature_pickle_path = os.path.join(patient_path,feature_pickle_name)
        #make sure that the pickle file was made
        if os.path.isfile(feature_pickle_path):
            with open(feature_pickle_path,'rb') as feature_pickle:
                yield pickle.load(feature_pickle)

        else:# generate patient data, store into pickle files
            with open(info_pickle_path) as info_pickle:
               patient_info = pickle.load(info_pickle)

            patient_data = []
            for ind,data_file in enumerate(patient_info['seizure_data_filenames']):
                data_file = os.path.join(patient_path,data_file)
                patient_data.extend(process_seiz_file(data_file,patient_info,ind))

            patient_data.extend(process_inter_files(patient_path,patient_info))

            #save the file out
            with open(feature_pickle_path,'wb') as pickle_out:
                pickle.dump(patient_data,pickle_out)
            yield patient_data






def make_SOM(features, SOM_dims = (30,30),**SOM_params):
    #if SOM_params is None:
    #    SOM_params = {}
    SOM = SOMPY.SOMFactory.build(features, SOM_dims, **SOM_params)
    SOM.train(verbose='info')
    return SOM

def visualize_SOM(SOM,dataset, labels,preict_times, extra_info= None):
    '''
    Visualize a self-organized map
    :param SOM:
    :param dataset:
    :param labels:
    :return:
    '''

    def compute_class_densities():
        best_units = SOM._bmu[0]
        prototype_density = np.zeros(np.append(SOM.codebook.nnodes,5))
        avg_preictal_times = np.zeros(SOM.codebook.nnodes)
        for label,label_ind in label_indices.items():
            for prototype in range(SOM.codebook.nnodes):
                prototype_density[prototype,label_ind] = np.sum(
                        np.logical_and(labels==label_ind , best_units==prototype))
                if label=='Preictal':
                    avg_preictal_times[prototype] =np.mean(
                        preict_times[np.logical_and(labels==label_ind,best_units==prototype)])



        return prototype_density,avg_preictal_times

    def mod_u_matrix(prototype_density,avg_preictal_times):

        raw_dens = np.sum(prototype_density,axis=1)
        norm_times = avg_preictal_times/np.nanmax(avg_preictal_times)

        u_matrix = SOMPY.visualization.umatrix.UMatrixView(*SOM.codebook.mapsize,
                                                                          title='SOM Visualization', text_size=10).build_u_matrix(SOM)

        prep_H = np.argmax(prototype_density,axis=1)
        prep_H = label_hue_arr[prep_H]
        prep_H[prep_H==label_hue_arr[label_indices['Preictal']]] -= \
            .15*norm_times[prep_H==label_hue_arr[label_indices['Preictal']]]
        H = np.reshape(prep_H,SOM.codebook.mapsize)

        S_prep = 0.75* np.ones_like(u_matrix.flatten())
        S_prep[raw_dens==0] = 0
        S = np.reshape(S_prep,SOM.codebook.mapsize)


        V = 0.85*(1-u_matrix/np.max(u_matrix))



        image_mod_mat = hsv2rgb(
            np.stack((H,S,V),axis=2))
        plt.subplot(1,2,1)
        plt.imshow(image_mod_mat)
        plt.title('Prototype classes and Neighborhood Distances')

        proj = SOM.project_data(SOM.data_raw)
        coord = SOM.bmu_ind_to_xy(proj)

        mn = np.min(u_matrix.flatten())
        #mx  =  np.max(u_matrix.flatten())
        std  = np.std(u_matrix.flatten())
        md =np.median(u_matrix.flatten())
        mx = md + 0.5*std
        plt.contour(u_matrix, np.linspace(mn, mx, 15), linewidths=0.7, cmap=plt.cm.get_cmap('Blues'))
        plt.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray', marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
        plt.axis('off')

        preictal_hues = np.linspace(label_hues['Preictal'],label_hues['Preictal']-.15,100)
        prelim_colormap = np.squeeze(hsv2rgb(np.dstack((preictal_hues,
                                                        np.full_like(preictal_hues,.75),
                                                        np.full_like(preictal_hues,.85)))))
        cmap = mpl.colors.ListedColormap(prelim_colormap)
        ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.75)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                       norm=mpl.colors.Normalize(vmin=np.nanmin(avg_preictal_times), vmax=np.nanmax(avg_preictal_times)))


        plt.subplot(1,2,2)
        plt.imshow(np.log(np.reshape(raw_dens,SOM.codebook.mapsize) + .05),cmap='gray')
        plt.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray', marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
        plt.title('Prototype Density Map')
        plt.axis('off')

    def bar_graph_plot(prototype_density,avg_preictal_times):
        hsv_colors =np.dstack((label_hue_arr,np.full_like(label_hue_arr,.75),np.full_like(label_hue_arr,.85)))
        avg_preictal_times =np.reshape(avg_preictal_times/ np.nanmax(avg_preictal_times),SOM.codebook.mapsize)


        xmax,ymax = SOM.codebook.mapsize
        f,ax_arr = plt.subplots(xmax,ymax,num=2)
        prototype_density = np.reshape(prototype_density,np.append(SOM.codebook.mapsize,len(label_hue_arr)))
        #bar_data = np.random.randint(1,100,(xmax,ymax,4))
        for xind in range(xmax):
            for yind in range(ymax):
                bar_colors = hsv_colors.copy()

                #plt.subplot(xmax,ymax,yind+1 +xind*ymax)
                #plt.axis('off')
                if prototype_density[xind,yind,label_indices['Preictal']]:
                    bar_colors[0,label_indices['Preictal'],0] -= .15*avg_preictal_times[xind,yind]
                #plt.bar(np.arange(len(label_hue_arr)),prototype_density[xind,yind,:],color=np.squeeze(hsv2rgb(bar_colors)))
                ax_arr[xind,yind].axis("off")
                ax_arr[xind,yind].bar(np.arange(len(label_hue_arr)),prototype_density[xind,yind,:],color=np.squeeze(hsv2rgb(bar_colors)))
        #plt.show()

        pass

    prototype_density,avg_preictal_times = compute_class_densities()
    plt.figure(1)
    mod_u_matrix(prototype_density,avg_preictal_times)
    #plt.colorbar()
    bar_graph_plot(prototype_density,avg_preictal_times)

    SOMPY.visualization.mapview.View2DPacked(SOM.codebook.mapsize[0],
                                                            SOM.codebook.mapsize[1],'Woop').show(SOM)

    #SOM.

    #a=1
    #weights = SOM.K

def cleanup_data_for_viz(dataset):
    features,labels,preictal_times, = [],[],[]
    for rec in dataset:
        features.append(rec['feats'])
        labels.append(label_indices[rec['label']])
        preictal_times.append(rec['preict_time'])
    return np.array(features),np.array(labels),np.array(preictal_times)


##execute main code
def viz_all_patients(patients,feature_func,channel_name,feature_type="Single",**func_params):

    data_base_path = os.path.abspath(os.path.join(os.getcwd(),'../../data'))
    patient_paths = [os.path.join(data_base_path,patient) for patient in patients]

    ## visualization of features, i changed window_size to 1000 from 2000
    for patient_data in prepare_and_store_data(patient_paths, window_size=15000, window_step=00, feature_func=feature_func,
                           read_chunktime=300, feature_type=feature_type, channel_name=channel_name, fs=1000,
                                               feature_pickle_name = 'feature_pickle.txt', **func_params):
        features,labels,preictal_times = cleanup_data_for_viz(patient_data)
        SOM = make_SOM(features, SOM_dims=(25,20),training='seq')
        visualize_SOM(SOM,features,labels,preictal_times)
        plt.show()

#
if __name__  == "__main__":
    pass

    ## feature func imported
    # from DCEpy.Features.ARModels.single_ar import ar_features
    # from DCEpy.Features.Bivariates.cross_correlation import cross_correlate
    # from DCEpy.Features.Bivariates.cross_correlation import compress
    #
    # viz_all_patients(['TS041'],lambda x : compress(cross_correlate(x)),['LAH2', 'LAH3', 'LAH4', 'LPH1', 'LPH2'],"Multiple")
    # from DCEpy.Features.Wavelet_Transform.wavelet_features import


# Fasai code:
# get your features - number of windows x number of features, in ndarray 'features'
# SOM = make_SOM(features,SOM_dims = (25,20), training = 'seq')
# fake_labels = np.ones(np.size(features,0))
# visualize_SOM(SOM,features,fake_labels,fake_labels)
# plt.show()

