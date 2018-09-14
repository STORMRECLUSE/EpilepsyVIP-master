from __future__ import print_function

import json
import os
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import hsv2rgb

from DCEpy.General.DataInterfacing.edfread import edfread
# from DCEpy.TestCode import SOMPY
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
                           feature_type='Single',feature_pickle_name = 'ar_pickle.txt',channel_name=None, fs = 1000.,**feature_params):
    '''

    :param patient_paths:
    :param feature:
    :param info_pickle_name:
    :param feature_pickle_name:
    :return:
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
        def chunk_file_iterator(chunk_samples,fs,seiz_file):

            n = 0
            try:
                while True:
                    data,_,labels = edfread(seiz_file,
                                            rec_times=(n*chunk_samples/float(fs),(n+1)*chunk_samples/float(fs)))
                    #data_is_full = np.size(data,0)>=chunk_samples
                    start_time = n*chunk_samples/float(fs)
                    yield data, start_time,n, labels
                    n+=1
            except ValueError:
                pass

        if feature_func is None:
            raise ValueError('No feature function was ineput!!')
        num_windows = np.round((chunk_time*fs - window_size)/float(window_step)+1)
        actual_chunksamples = window_step*(num_windows-1) + window_size
        rec_list = []
        for data_chunk,chunk_start,chunk_num, labels in chunk_file_iterator(actual_chunksamples,fs,seiz_file):
            for start_sample in range(0,np.size(data_chunk,0),window_step):
                rec = {}
                window = data_chunk[start_sample:start_sample+window_size,:]
                window_end_time = (actual_chunksamples*chunk_num +
                                   start_sample + window_size)/float(fs)
                if feature_type =='Single':
                    rec['feats'] = feature_func(window[:,[labels.index(channel_name)]],**feature_params)
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


def cleanup_data_for_viz(dataset):
    features,labels,preictal_times, = [],[],[]
    for rec in dataset:
        features.append(rec['feats'])
        labels.append(label_indices[rec['label']])
        preictal_times.append(rec['preict_time'])
    return np.array(features),np.array(labels),np.array(preictal_times)


##execute main code
def viz_all_patients():
    patients = ['TS039']

    ## features imported

    from DCEpy.Features.ARModels.single_ar import ar_features

    data_base_path = os.path.abspath(os.path.join(os.getcwd(),'../../data'))
    patient_paths = [os.path.join(data_base_path,patient) for patient in patients]
    for patient_data in prepare_and_store_data(patient_paths,window_size=2000,window_step=500,feature_func=ar_features,
                                               read_chunktime=300,feature_type='Single',channel_name='LAH1',fs = 1000,
                                               order=30):
        features,labels,preictal_times = cleanup_data_for_viz(patient_data)
        SOM = make_SOM(features, SOM_dims=(25,20),training='seq')
        visualize_SOM(SOM,features,labels,preictal_times)
        plt.show()


if __name__  == "__main__":

    fs=1000
    win_len = 1*fs
    win_overlap = int(.5*fs)

    include_awake = True
    include_asleep = True

    patients = ['TS041']
    long_interictal = [False]

    # get the paths worked out
    # to_data = '/scratch/smh10/DICE'
    to_data = os.path.dirname(os.path.dirname(os.getcwd()))
    data_path = os.path.join(to_data, 'data')

    for p, patientID in enumerate(patients):

        # specify data paths
        if not os.path.isdir(data_path):
            sys.exit('Error: Specified data path does not exist')

        p_file = os.path.join(data_path, patientID, 'patient_pickle.txt')

        with open(p_file,'r') as pickle_file:
            print("Open Pickle: {}".format(p_file)+"...\n")
            patient_info = pickle.load(pickle_file)

        # add data file names
        data_filenames = patient_info['seizure_data_filenames']
        seizure_times = patient_info['seizure_times']
        file_type = ['ictal'] * len(data_filenames)
        seizure_print = [True] * len(data_filenames)      # mark whether is seizure

        if include_awake:
            data_filenames += patient_info['awake_inter_filenames']
            seizure_times += [None] * len(patient_info['awake_inter_filenames'])
            file_type += ['awake'] * len(patient_info['awake_inter_filenames'])
            seizure_print += [False] * len(patient_info['awake_inter_filenames'])

        if include_asleep:
            data_filenames += patient_info['asleep_inter_filenames']
            seizure_times += [None] * len(patient_info['asleep_inter_filenames'])
            file_type += ['sleep'] * len(patient_info['asleep_inter_filenames'])
            seizure_print += [False] * len(patient_info['asleep_inter_filenames'])

        min_channels = np.inf

        labels = []
        preictal_times = []

        for i, seizure_file in enumerate(data_filenames):

            seizureID = seizure_file[:-4]
            # update paths specific to each patient

            data_path = os.path.join(to_data,'data',patientID,seizureID) + 'eigenvec_centrality.json'

            print("Reading in file ",i)
            with open(data_path) as data_file:
                raw_data = json.load(data_file)

            num_windows = len(raw_data)
            num_channels = len(raw_data["0"])

            data = np.empty([num_windows,num_channels])
            for k in xrange(num_windows):
                for j in xrange(num_channels):
                    data[k,j]=raw_data["%d"%k]["%d"%j]
            win_array = np.array(data) # just to be sure...

            num_windows = win_array.shape[0]
            total_len = (num_windows-1)*(win_len - win_overlap) + (win_len)

            t_data = np.empty([total_len-win_len+win_overlap, num_channels])

            for j in xrange (data.shape[1]):

                time_array = -1 * np.ones(total_len-win_len+win_overlap)

                for k in range(num_windows):
                    start = (k)*(win_len - win_overlap)
                    end = start + (win_len - win_overlap)
                    time_array[start:end] = win_array[k,j]

                t_data[:,j] = time_array

            t_data = t_data[::(win_len - win_overlap), :]
            unique_data_persec = fs / (win_len - win_overlap)

            if i == 0:
                features = t_data[:, :]
            else:
                if num_channels < min_channels:
                    min_channels = num_channels
                    features = features[:, :min_channels]
                if num_channels > min_channels:
                    t_data = t_data[:, :min_channels]
                    num_channels = min_channels
                features = np.vstack((features, t_data))

            if (seizure_times[i] is not None):
                seiz_times = [time * unique_data_persec for time in seizure_times[i]]

            if (file_type[i] is 'ictal'):
                labels.extend([2]*(seiz_times[0]))
                labels.extend([3]*(seiz_times[1]-seiz_times[0]))
                labels.extend([4]*(t_data.shape[0]-seiz_times[1]))
                for num_preictal_datapoints in xrange(seiz_times[0]):
                    preictal_times.extend([(seiz_times[0] - num_preictal_datapoints) / float(unique_data_persec)])
                preictal_times.extend([np.nan]*(t_data.shape[0]-seiz_times[0]))

            if (file_type[i] is 'awake'):
                labels.extend([1] * t_data.shape[0])
                preictal_times.extend([np.nan]*(t_data.shape[0]))

            if (file_type[i] is 'sleep'):
                labels.extend([0] * t_data.shape[0])
                preictal_times.extend([np.nan]*(t_data.shape[0]))

            print(features.shape)
            print(len(labels))
            print(len(preictal_times))

        features = np.array(features)
        labels = np.array(labels)
        preictal_times = np.array(preictal_times)

        SOM = make_SOM(features, SOM_dims=(25,20),training='seq')
        visualize_SOM(SOM,features,labels,preictal_times)
        plt.show()


