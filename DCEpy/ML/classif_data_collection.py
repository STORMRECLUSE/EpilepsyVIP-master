'''
This file takes in data files, as well as seizure start and end times, and

There is some example code at the end.
'''
from __future__ import print_function

import os

import numpy as np

from DCEpy.General.DataInterfacing.edfread import edfread
from DCEpy.Features.Preprocessing.preprocess import RstatPreprocessor,notch,normalize


def interval_inclusion_index(win_start,win_end,start_times,end_times):
    for (lab_index,start),end in zip(enumerate(start_times),end_times):
        if win_start>=start and win_end<= end:
            return lab_index
    else:
        return None

def classif_data_collect(seiz_filenames, seizure_times,inter_filenames, window_len,
                         preictal_time, postictal_time, n_windows,
                         sliding_window = False, window_overlap = .8,
                         fs = 1000., good_channels=None,bad_channels=None,
                         rstat_bands = ((1,4),(5,8),(9,13),(14,25),(25,90),(100,200)),
                         rstat_win_len=20000, notch_filt=True,
                         norm_whole_file = True, norm_window = False):
    '''

    :param seiz_filenames:       A list of the .edf filenames which contain epileptic data
    :param seizure_times:   A list corresponding to the ictal filenames of:
                                tuples containing seizure start and end times (in seconds),
                                    if the recording contains of seizure

                            For instance, if my filename list looks like
                            ['seizure_a.edf','seizure_b.edf']

                            Then my seizure_times list might be:
                            [(123,135),(60,300)]
    :param inter_filenames: A list of .edf filenames of interictal data
    :param good_channels:   A list of the channel names that should be observed for the
    :param window_len:      The length of the windows that the program takes,
                                in number of samples
    :param preictal_time:   The amount of time, in seconds, before a seizure, for which
                                a window will still be considered as preictal for training purposes

    :param postictal_time:The amount of time, in seconds, after a seizure
                                that the period is considered postictal
    :param n_windows:       The number of windows of each class we choose to extract
    :param sliding_window: Do you want the data to be a sliding window?
    :param window_overlap: The maximum allowable overlap (percentage) between two windows.
    :param fs: the sampling frequency of the sample
    :param bad_channels: if you want to read in all channels, except for a select few
    :param rstat_bands: An iterable of pairs of two elements,
                        representing the start and stop of the bands of interest. Can be NoneType if no rstat is desired
    :param rstat_win_len: (int) The length of the random windows
                                extracted in computing the r-statistic
    :param notch_filt: (bool) A boolean value determining whether a notch filter is to be applied to the data
    :param norm_whole_file: (bool) A boolean value that normalizes the whole file, not just the window.
    :param norm_window: (bool) A boolean value that chooses to normalize each window, not the whole file.

    :return:
    '''



    def select_windows_from_interval(int_start,int_end,max_iter=10):
        '''
        :param int_start:
        :param int_end:
        :param max_iter:
        :return: window_ends, a numpy array with indices at which the sample ends
        '''
        window_ends = np.sort(np.random.randint(int_start+window_len,int_end,n_windows))
        if n_windows==1:
            return window_ends
        for i in range(max_iter):
            diffs = np.diff(window_ends)
            if all(diffs>=(window_len*window_overlap)):
                return window_ends
            #TODO: refine window logic
            window_ends = np.sort(np.random.randint(int_start+window_len,int_end,n_windows))


        else:
            #manually select window ends
            window_ends = int_end - window_overlap * window_len * np.arange(n_windows)
            return window_ends
    def generate_windows_from_seizure(sliding_window = False):
        '''
        Given seizure data, select random windows
        from the preictal,ictal,and postictal phases
        :return:
        '''
        preictal_samples,postictal_samples =  int(preictal_time*fs),int(postictal_time*fs)
        seiz_start,seiz_end = seize_times

        ictal_samples = int((seiz_end-seiz_start)*fs)
        max_iter = 4

        if not sliding_window:
            if not window_overlap:
                if (n_windows*window_len) > \
                        (min(preictal_samples, postictal_samples,ictal_samples)):
                    raise ValueError('Nonoverlapping windows not possible to produce')
            else:
                if (n_windows*window_len - (n_windows-1)*window_len*window_overlap)>\
                    min(preictal_samples,postictal_samples,ictal_samples):
                    raise ValueError('Not possible to produce overlapping windows with certain max_proportion')

        seiz_start = int(seiz_start*fs)
        seiz_end = int(seiz_end*fs)

        interval_starts = (seiz_start-preictal_samples,seiz_start,seiz_end)
        interval_ends = (seiz_start,seiz_end,np.size(seiz_data,axis=0))
        labels = ['Preictal','Ictal','Postictal']

        if sliding_window:
            end_times = range(window_len-1,np.size(seiz_data,0),int(window_len*(1-window_overlap)))

            for end in end_times:
                start = end-window_len+1
                lab_index = interval_inclusion_index(start,end,interval_starts,interval_ends)
                if lab_index is None:
                    label = 'None of the Above'
                else:
                    label = labels[lab_index]
                window = preprocess(seiz_data[start:end+1,:],normaliz=norm_window,notch_filt = False)
                data_container.append(
                        {'window':window,'label':label,'fold_lab':'S{}'.format(seiz_count),'time':end/fs}
                )
        else:

            for interval_start,interval_end,label in zip(interval_starts,interval_ends,labels):
                window_ends = select_windows_from_interval(interval_start,interval_end,max_iter)
                for end in window_ends:
                    window = preprocess(seiz_data[end-window_len+1:end+1,:],normaliz=norm_window,notch_filt = False)
                    data_container.append(
                        {'window':window,'label':label,'fold_lab':'S{}'.format(seiz_count),'time':end/fs}
                         )
        return



    def generate_windows_from_nonseizure(sliding_window = False):
        label = 'Interictal'
        if sliding_window:
            end_times = range(window_len,np.size(seiz_data,0),int(window_len*(1-window_overlap)))
            for end in end_times:
                window = preprocess(seiz_data[end-window_len+1:end+1,:],normaliz=norm_window,notch_filt = False)
                data_container.append(
                        {'window':window,'label':label,
                         'fold_lab':'NS{}'.format(non_seiz_count),'time':end/fs}
                )

        else:
            max_iter = 4 # maximum times to pick a random list for nonoverlapping windows
            end_times = select_windows_from_interval(0,np.size(seiz_data,axis=0),max_iter)
            for end in end_times:
                window = preprocess(seiz_data[end-window_len+1:end+1,:],normaliz=norm_window,notch_filt = False)
                data_container.append(
                    {'window':window,'label':label,
                     'fold_lab':'NS{}'.format(non_seiz_count),'time':end/fs}
                     )
        return

    def preprocess(seiz_data,normaliz = False,notch_filt=True,rstat_band=False,causal = True):
        if normaliz:
            seiz_data = normalize(seiz_data)

        if notch_filt:
            seiz_data = notch(seiz_data,56.,64.,fs,mode=not causal)

        if rstat_band:
            seiz_data = rstat_processor.optimal_bandpass(seiz_data,mode=not causal)
        return seiz_data


    ###start main code of the function
    rstat_filt = False
    if rstat_bands is not None:

        rstat_processor = RstatPreprocessor(inter_filenames[0],seiz_filenames[0],
                                            seizure_times=seizure_times[0],
                                            fs=1000.)
        rstat_processor.prepare_rstat(rstat_bands,good_channels=good_channels, bad_channels=bad_channels, window_len=20000, mode=0)

        rstat_filt=True




    data_container = []

    if not window_overlap:
        window_overlap = 0

    for seiz_count,(seizure_file,seize_times) in enumerate(zip(seiz_filenames,seizure_times)):
        seiz_data,_,_ = edfread(seizure_file,good_channels=good_channels, bad_channels=bad_channels)

        seiz_data = preprocess(seiz_data,normaliz=norm_whole_file,notch_filt=notch_filt,rstat_band=rstat_filt)
        generate_windows_from_seizure(sliding_window)


    for non_seiz_count,nonseizure_file in enumerate(inter_filenames):
        seiz_data,_,_ = edfread(nonseizure_file,good_channels=good_channels, bad_channels=bad_channels)

        seiz_data = preprocess(seiz_data,normaliz=norm_whole_file,notch_filt=notch_filt,rstat_band=rstat_filt)
        generate_windows_from_nonseizure(sliding_window)
    return {'data':data_container,'seize_times':seizure_times}



if __name__ =='__main__':
    from DCEpy.ML.ML_pipeline_driver import EigCentralityTester
    # Example code to see how this works. Data taken from patient TS0041
    file_dir = '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs'

    filenames = ('DA00101U_1-1+.edf',
                 # 'DA00101V_1-1+.edf',
                 # 'DA00101W_1-1+.edf',
                 # 'CA00100E_1-1_03oct2010_01_03_05_Awake+.edf',
                 # 'CA00100F_1-1_03oct2010_13_05_13_Awake+.edf',
                 # 'DA00101P_1-1_02oct2010_09_00_38_Awake+.edf'
                 )
    filenames = [os.path.join(file_dir,filename) for filename in filenames]

    good_channels = [
                    'LAH1','LAH2','LPH6','LPH7',
                    'LPH9','LPH10','LPH11','LPH12'
                    ]
    # seizure_starts = 262,\
    #               107,191,0
    # seizure_ends   = 330,\
    #                 287,405,0
    rstat_bands = np.array([ [1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])

    seizure_times = (262,330),\
                    #(107,287),(191,405),None,None,None

    data_container = classif_data_collect(filenames,seizure_times,
                                          good_channels=good_channels,window_len=5000,sliding_window=True,
                                          preictal_time=30,postictal_time=40,n_windows=2,
                                          window_overlap=.75,rstat_bands= None,norm_window=True,norm_whole_file= False)


    #ar_driver = ARCoeffsTester(data_container,
    #                        order=5,nonlinear_feats=['lyapunov_exponent'],nonlinear_params=[{}])

    ar_driver = EigCentralityTester()
    ar_driver.set_data(data_container)
    ar_driver._write_to_files_and_call_fxn('lolll.txt','696969696420.csv',{})
    X,Y,T = ar_driver.get_feature_list()
    b =1