#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A pipeline for evaluating performance on algorithms.
Uses cross validation (across files) to more accurately assess algorithm performance.
'''
from __future__ import division
from __future__ import print_function

import os
import pickle
import copy
import logging
from collections import defaultdict, namedtuple

import numpy as np

from DCEpy.General.DataInterfacing.edfread import edfread
from DCEpy.Features.Preprocessing.preprocess import box_filter
from sklearn.cross_validation import PredefinedSplit



def dict_like_product(param_grid):
    '''
    Get a cartesian product of every dict inside the structure.
    This is to imitate sklearn's implementation.
    :param param_grid:
    :return:
    '''
    if isinstance(param_grid,dict):
        for element in dict_product(param_grid):
            yield element
    else:
        for dict_with_lists in param_grid:
            for element in dict_product(dict_with_lists):
                yield element

def dict_product(dict_with_lists):
    for x in cartesian_product(*dict_with_lists.itervalues()):
        yield (dict(zip(dict_with_lists, x)))


def cartesian_product(L=None,*lists):
    '''
    Returns the cartesian product of all lists in  L
    :param L:
    :param lists:
    :return:
    '''
    if L is None:
        return
    if not lists:
        for x in L:
            yield (x,)
    else:
        for x in L:
            for y in cartesian_product(lists[0],*lists[1:]):
                yield (x,)+y


def prepare_and_store_data(patient_paths,preprocessing_routine, window_size,window_step,
                           preprocessing_routine_params = None, feature_func = None, read_chunktime=300, info_pickle_name ='patient_pickle.txt',
                           feature_type='Single',feature_pickle_name = 'energy_pickle(winlen=2000).txt',channel_name=None, fs = 1000.,**feature_params):
    '''
    A generator that, on a given patient, computes the necessary features across all of the windows commanded.
    :param patient_paths: The paths to the folders containing patient data. A list.
    :param preprocessing_routine: A function executed on each "chunk" of data that returns preprocessed data.
            Mandatory input: fs, sampling frequency.
    :param preprocessing_routine_params: A dictionary containing all keyword arguments, if any,
            that the preprocessing routine needs in order to run. Default: no parameters needed
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
    These dictionaries contain the following information:
    'Label': a naïve layer

    '''
    def compute_features_from_file(seiz_file,  window_size,window_step,fs, chunk_time,channel_name = None,
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
        def chunk_file_iterator(chunk_samples,seiz_file, good_channels = None):

            """iterates over chunks of data. Useful for long (longer than 5 minutes) edf files, saves memory."""

            n = 0
            try:
                while True:
                    data,_,labels = edfread(seiz_file,
                                            rec_times=(n*chunk_samples/float(fs),(n+1)*chunk_samples/float(fs)),
                                            good_channels=good_channels)
                    if preprocessing_routine is not None:
                        data = preprocessing_routine(data,fs = fs,**preprocessing_routine_params)
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
        for data_chunk,chunk_start,chunk_num, labels in chunk_file_iterator(actual_chunksamples,
                                                                            seiz_file, good_channels=channel_name):
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
                                                  feature_func=feature_func, feature_type=feature_type,
                                                  channel_name=channel_name,**feature_params)
        seiz_start,seiz_end = patient_info['seizure_times'][index]
        patient_data['seizure_times'][seiz_file] = (seiz_start,seiz_end)
        for rec in prelim_feats:
            rec['file_type'] = "Seizure"
            rec['file_index'] = index
            rec['filename'] = seiz_file
            if rec['wind_end'] < seiz_start:
                rec['label'] = "Preictal"
                #preictal time in seconds
                rec['preict_time'] = seiz_start- rec['wind_end']
            elif seiz_start<=rec['wind_end']<=seiz_end:
                rec['label'] = 'Ictal'
            else:
                rec['label'] = 'Postictal'

        return prelim_feats

    def process_inter_files(patient_path,patient_info):

        awake_files = patient_info['awake_inter_filenames']
        inter_feats = []
        for index,inter_file in enumerate(awake_files):
            inter_file = os.path.join(patient_path,inter_file)
            prelim_feats = compute_features_from_file(inter_file,window_size,window_step,fs, read_chunktime,
                                                  feature_func=feature_func, feature_type=feature_type,
                                                      channel_name=channel_name,**feature_params)
            for rec in prelim_feats:
                rec['file_type'] = "Interictal Awake"
                rec['file_index'] = index
                rec['label'] = 'Interictal Awake'
                rec['filename'] = inter_file
            inter_feats.extend(prelim_feats)

        asleep_files = patient_info['asleep_inter_filenames']
        for index,inter_file in enumerate(asleep_files):
            inter_file = os.path.join(patient_path,inter_file)
            prelim_feats = compute_features_from_file(inter_file,window_size,window_step,fs, read_chunktime,
                                                  feature_func=feature_func, feature_type=feature_type,
                                                      channel_name=channel_name,**feature_params)
            for rec in prelim_feats:
                rec['file_type'] = "Interictal Asleep"
                rec['file_index'] = index
                rec['filename'] = inter_file
                rec['label'] = 'Interictal Asleep'
            inter_feats.extend(prelim_feats)
        return inter_feats

    if preprocessing_routine is not None:
        if preprocessing_routine_params is None:
            preprocessing_routine_params = {}

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

            patient_data = {'windows':[],'seizure_times':{},'info':patient_info}
            for ind,data_file in enumerate(patient_info['seizure_data_filenames']):
                data_file = os.path.join(patient_path,data_file)
                patient_data['windows'].extend(process_seiz_file(data_file,patient_info,ind,))

            patient_data['windows'].extend(process_inter_files(patient_path,patient_info))

            #save the file out
            with open(feature_pickle_path,'wb') as pickle_out:
                pickle.dump(patient_data,pickle_out)
            yield patient_data


def prepare_train_test_viz_data(patient_data,preictal_time, train_only_interictal = True,
                                total_labels = ('Interictal Asleep','Interictal Awake','Preictal'),
                                training_labels = ('Interictal Asleep','Interictal Awake')):
    '''

    :param patient_data: The patient data collected from prepare_and_store_data()
    :param preictal_time: The time, in seconds, before a seizure, that is considered preictal.
    :param train_only_interictal: make train data only interictal?
    :param total_labels:
    :param training_labels:
    :return:
    '''


    # first separate data into folds (leave one out of each type given)
    # then only include preictal from seizure files in train/test data, but keep it for visualization.
    # ONLY INTERICTAL IN TRAIN DATA. FOLDS FROM REMOVING ONE INTERICTAL FILE. VIZ FROM BEST FOLD??
    # Structure ideas-
    # -List/dict of folds (dict)
    fold_dict = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))
    viz_dict = defaultdict(lambda:defaultdict(lambda: defaultdict(list)))

    total_labels,training_labels = set(total_labels),set(training_labels)
    viz_labels = {'Interictal Asleep','Interictal Awake','Seizure'}

    #create dictionary structures storing data
    for rec in patient_data['windows']:
        if rec['label'] in ['Interictal Asleep','Interictal Awake']:
            fold_dict[rec['label']][rec['file_index']]['data'].append(rec['feats'])
            fold_dict[rec['label']][rec['file_index']]['window_times'].append(rec['wind_end'])

        else:
            if rec['label']== 'Preictal':
                if rec['preict_time'] <= preictal_time:
                    fold_dict[rec['label']][rec['file_index']]['data'].append(rec['feats'])
                    fold_dict[rec['label']][rec['file_index']]['window_times'].append(rec['wind_end'])

        viz_dict[rec['file_type']][rec['file_index']]['data'].append(rec['feats'])
        viz_dict[rec['file_type']][rec['file_index']]['window_times'].append(rec['wind_end'])
        if not viz_dict[rec['file_type']][rec['file_index']]['filename']:
                viz_dict[rec['file_type']][rec['file_index']]['filename'] = rec['filename']

                #TODO: Change method of acquiring interictal data (i.e. allow data before a time to be interictal)


    if not train_only_interictal:
        training_labels.update('Preictal')

    max_common_train_index = 0
    empty_classes = []

    label_len = {}
    for label in fold_dict:
        if label in total_labels:
            label_len[label] = len(fold_dict[label])
            for fold in fold_dict[label]:
                if fold_dict[label][fold]:
                    fold_dict[label][fold]['data'] =         np.asarray(fold_dict[label][fold]['data'])
                    fold_dict[label][fold]['window_times'] = np.asarray(fold_dict[label][fold]['window_times'])

    for label in viz_dict:
        for fold in viz_dict[label]:
            viz_dict[label][fold]['data'] =         np.asarray( viz_dict[label][fold]['data'])
            viz_dict[label][fold]['window_times'] = np.asarray(viz_dict[label][fold]['window_times'])


    val_only_labels = total_labels.copy()
    val_only_labels.difference_update(training_labels)

    max_common_train_index = min((label_len[label] for label in fold_dict if label in training_labels))
    min_common_val_index = min((label_len[label] for label in fold_dict if label in val_only_labels)) \
        if val_only_labels else 0
    empty_classes = set(training_labels) - set(label_len)


    #TODO: Separate long files only if necessary (implement insufficient_data function)
    if ('Interictal Asleep' in empty_classes and 'Interictal Awake' in empty_classes):
        raise ValueError('No interictal data found to train or test on!')
    elif max_common_train_index < 3:

        raise ValueError('Not enough interictal data to train and test on. Need at least 3 distinct files.')
    elif min_common_val_index < 2 and val_only_labels:
        raise ValueError('Not enough data to train and test on.')




    return {'fold_data':fold_dict,'viz_data':viz_dict,'len_labels':label_len,
            'num_val_folds':min_common_val_index,'num_train_folds':max_common_train_index}



default_label_mapping = {'Interictal Asleep': 'Interictal', 'Interictal Awake': 'Interictal'}

label_viz_mapping = {'Interictal': 0, 'Preictal':1}



def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    return l


class BasicGridCrossValidator(object):
    def __init__(self, estimator, grid_params, score_func, out_dir ,  supervised=True, fit_params = None,
                 total_labels = ('Interictal Asleep','Interictal Awake','Preictal'),
                 training_labels = ('Interictal Asleep','Interictal Awake'),
                 label_map = default_label_mapping,label_to_viz = label_viz_mapping, **score_func_params):
        '''

        :param estimator: The classifier/anomaly detector, assumed to be sklearn-like
        :param grid_params: The parameters to tune over
        :param score_func:
        :param out_dir:
        :param fit_params:
        :param total_labels:
        :param training_labels:
        :param label_map:
        :param score_func_params:
        :return:
        '''
        self.total_labels = total_labels
        self.training_labels = training_labels

        #TODO: Check to see if logic has changed for assignment of following variables
        self.estimator = estimator
        self.grid_params = grid_params
        self.score_func = score_func
        self.score_func_params = score_func_params
        self.fit_params = fit_params if fit_params is not None else {}
        self.label_map = label_map
        self.label_to_viz = label_to_viz
        self.supervised = supervised
        self.out_dir = out_dir


        pass
    def cross_validate(self, patient_data,preictal_time = 300,train_only_interictal = True):

        def collect_test_data():
            testing_data = {
                'old_labels':[], 'new_labels':[], 'window_times':[], 'data':[]}
            for label in test_file_indices:
                #add test fold based on indices
                testing_data['data'].append(folds_dict[label][num_index]['data'])

                new_label = self.label_map[label] if label in self.label_map else label
                testing_data['old_labels'].extend([label]*np.size(folds_dict[label][num_index]['data'],0))
                testing_data['new_labels'].extend([new_label]*np.size(folds_dict[label][num_index]['data'],0))
                testing_data['window_times'].append(folds_dict[label][num_index]['window_times'])
            #clean up data
            testing_data['data'] = np.vstack(testing_data['data'])
            testing_data['window_times']= np.hstack(testing_data['window_times'])
            return testing_data

        def collect_validation_train_data():
            validation_data = {'old_labels':[],'new_labels':[],'window_times':[],'data':[]}
            training_data = {'old_labels':[], 'new_labels':[],'window_times':[],'data':[]}

            for label in folds_dict:
                if label in self.total_labels:
                    for fold in folds_dict[label]:
                        new_label = self.label_map[label] if label in self.label_map else label
                        # if we don't need some data for training, add it
                        if label not in self.training_labels:
                            # add labels
                            validation_data['old_labels'].extend([label]*np.size(folds_dict[label][fold]['data'],0))
                            validation_data['new_labels'].extend([new_label]*np.size(folds_dict[label][fold]['data'],0))
                            # add window times
                            validation_data['window_times'].append(folds_dict[label][fold]['window_times'])
                            # add data
                            validation_data['data'].append(folds_dict[label][fold]['data'])
                            pass

                        else:
                            if label ==validation_label and fold==validation_index:
                                # add labels, window end times and data to validation
                                validation_data['old_labels'].extend([label]*np.size(folds_dict[label][fold]['data'],0))
                                validation_data['new_labels'].extend([new_label]*np.size(folds_dict[label][fold]['data'],0))
                                validation_data['window_times'].append(folds_dict[label][fold]['window_times'])
                                validation_data['data'].append(folds_dict[label][fold]['data'])
                            else:
                                # add labels, window end times and data to training
                                training_data['old_labels'].extend([label]*np.size(folds_dict[label][fold]['data'],0))
                                training_data['new_labels'].extend([new_label]*np.size(folds_dict[label][fold]['data'],0))
                                training_data['window_times'].append(folds_dict[label][fold]['window_times'])
                                training_data['data'].append(folds_dict[label][fold]['data'])
            training_data['data'],validation_data['data'] = np.vstack(training_data['data']), np.vstack(validation_data['data'])

            training_data['window_times'],validation_data['window_times'] = \
                np.hstack(training_data['window_times']), np.hstack(validation_data['window_times'])

            return training_data,validation_data


        data_dict = prepare_train_test_viz_data(patient_data,preictal_time=preictal_time,
                                                train_only_interictal = train_only_interictal,
                                    total_labels=self.total_labels,training_labels = self.training_labels
                                    )
        folds_dict,viz_dict = data_dict['fold_data'],data_dict['viz_data']
        self.folds_dict, self.viz_dict = folds_dict,viz_dict
        labels_len = data_dict['len_labels']
        test_num = min(data_dict['num_val_folds'],data_dict['num_train_folds'])
        test_file_indices = { label :np.random.choice(labels_len[label],test_num,replace=False)
                              for label in labels_len}
        self.test_file_indices, self.test_num = test_file_indices,test_num

        all_indices = {label:set(range(labels_len[label])) for label in labels_len}

        GridPoint = namedtuple('GridPoint',['mean_score','fold_scores','point_params'])



        for num_index in range(test_num):
            self.num_index = num_index
            print('----- Begin Test {} of {} ----\n\n'.format(num_index+1,test_num))
            #for each label, get the length of the number of files in that category and get the labels you can choose from
            test_data = collect_test_data()
            nontest_indices =copy.deepcopy(all_indices)
            test_results = []
            for label in labels_len:
                nontest_indices[label].remove(test_file_indices[label][num_index])

            num_folds = 0
            for label in self.training_labels:
                num_folds += len(nontest_indices[label])
            #construct a grid for each parameter value
            max_score = -np.Inf
            validation_grid = []
            for grid_params in dict_like_product(self.grid_params):
                #perform leave-one-out cross-validation on the files
                print('Validating parameters: {}'.format(grid_params))

                all_scores = []
                curr_fold = 1
                for validation_label in self.training_labels:
                    for validation_index in nontest_indices[validation_label]:
                        train_data,validation_data = collect_validation_train_data()
                        score = self._train_and_validate_on_fold(train_data,validation_data,grid_params)
                        all_scores.append(score)
                        print('Evaluated Fold {} of {}\n'.format(curr_fold,num_folds))
                        curr_fold +=1

                this_score = np.mean(all_scores)

                if this_score>max_score:
                    best_point = GridPoint(mean_score = np.mean(all_scores), fold_scores = all_scores,
                                                 point_params = grid_params)
                    best_index = len(validation_grid)
                validation_grid.append(GridPoint(mean_score = np.mean(all_scores), fold_scores = all_scores,
                                                 point_params = grid_params))
            self.visualize_validations(validation_grid,validation_grid, best_point.point_params)
            print('--- Done Parameter Tuning {} of {} Testing Data --'.format(num_index,test_num))
            self._test_one_time(test_data,best_point.point_params)






            #average, or somehow weight results across different sets to get overall performance.
    def _train_and_validate_on_fold(self, train_data, validate_data,estimator_params):
        self.estimator.set_params(**estimator_params)
        if self.supervised:
            self.estimator.fit(train_data['data'],train_data['new_labels'],**self.fit_params)
        else:
            self.estimator.fit(train_data['data'],**self.fit_params)
        score = self.score_func(validate_data,estimator,**self.score_func_params)
        return score

    def visualize_validations(self,validation_grid,best_params):
        '''
        Visualizes and logs the performance of the grid on the best algorithm, and some other random parameters.
        :param grid_performance:
        :param best_index:
        :return:
        '''

        def visualize_validation(point_params,best_val):
            title_str = str(point_params)
            self.estimator.set_params(**point_params)
            self.logg

        pass

    def visualize_test(self,estimator,best_params):
        estimator.set_params(**best_params)
        for label in self.test_file_indices:
            viz_predictions = estimator.predict(self.viz_dict[label][self.test_file_indices[label][self.test_num]]['data'])
            if viz_predictions.dtype.name.startswith('string'):
                viz_predictions = np.array((self.label_to_viz[pred] for pred in viz_predictions))





    def _test_one_time(self, test_data,estimator_params):
        estimator.set_params(**estimator_params)
        test_predictions = estimator.predict(test_data['data'])

        #get the data for information
        for label in self.test_file_indices:
            estimator.predict(self.viz_dict[label][self.test_file_indices[label][self.num_index]])
        pass

    def cross_validate_with_cross_validator(self,patient_data,cross_validator, preictal_time = 300,predef_fold_split = None,
                                            train_only_interictal = True,**cross_validator_params):
        def collect_test_data():
            testing_data = {
                'new_labels':[], 'window_times':[], 'data':[]}
            for label in test_file_indices:
                #add test fold based on indices
                testing_data['data'].append(folds_dict[label][num_index]['data'])
                testing_data['new_labels'].extend([label]*np.size(folds_dict[label][num_index]['data'],0))
                testing_data['window_times'].append(folds_dict[label][num_index]['window_times'])
            #clean up data
            testing_data['data'] = np.vstack(testing_data['data'])
            testing_data['window_times']= np.hstack(testing_data['window_times'])
            return testing_data

        def collect_remaining_data():
            remain_data = []
            labels = []
            for label in nontest_indices:
                for fold_index in nontest_indices[label]:
                    remain_data.append(folds_dict[label][fold_index]['data'])
                    labels.append(folds_dict[label][fold_index]['new_labels'])

            return np.vstack(remain_data),np.hstack(labels)

        def train_and_test():

            validator = cross_validator(**cross_validator_params)
            validator.fit(remaining_data,remaining_labels)

            #validator.grid_params

            #log training results
            #test on best results
            validator.predict(test_data)
            #score test,

            return validator


        def visualize_data():
            return

        data_dict = prepare_train_test_viz_data(patient_data,preictal_time=preictal_time,
                                                train_only_interictal = train_only_interictal,
                                    total_labels=self.total_labels,training_labels = self.training_labels
                                    )
        folds_dict,viz_dict = data_dict['fold_data'],data_dict['viz_data']
        self.folds_dict, self.viz_dict = folds_dict,viz_dict
        labels_len = data_dict['len_labels']
        test_num = min(data_dict['num_val_folds'],data_dict['num_train_folds'])
        test_file_indices = { label :np.random.choice(labels_len[label],test_num,replace=False)
                              for label in labels_len}
        if self.fit_params:
            cross_validator_params['fit_params'] = self.fit_params
        all_indices = {label:set(range(labels_len[label])) for label in labels_len}
        for num_index in range(test_num):
            #for each label, get the length of the number of files in that category and get the labels you can choose from
            test_data = collect_test_data()
            nontest_indices =copy.deepcopy(all_indices)
            estimator = self.estimator()
            for label in labels_len:
                nontest_indices[label].remove(test_file_indices[label][num_index])

            if train_only_interictal  and 'cv' not in cross_validator_params:
                raise ValueError('Sklearn currently does not easily support cross validation for anomaly detection')

            remaining_data, remaining_labels = collect_remaining_data()
            train_and_test()

def AutoEncoder_balanced_percentile_score(testing_data, estimator, beta=1,positive_classes = ('Preictal',)):
    percentiles = estimator.percentiles_estimator(testing_data['data'])
    p_vals = 1-percentiles
    pos_count, pos_val = 0,0.
    neg_val = 0.

    for number, label in zip(p_vals,testing_data['new_labels']):
        if label in positive_classes:
            pos_count +=1
            pos_val += number
        else:
            neg_val +=number

    return -pos_val/pos_count + beta* neg_val/(len(p_vals)-pos_count)

    pass

def AutoEncoder_balanced_percentile_score2(testing_data,estimator,beta=1, positive_classes = ('Preictal',),
                                           max_val = 100.):
    percentiles = estimator.percentile_estimator(testing_data['data'])
    one_over_p = 1/(1-percentiles)
    # noinspection PyUnresolvedReferences
    one_over_p[np.isnan(one_over_p)] = max_val
    one_over_p[one_over_p>max_val] = max_val
    pos_count, pos_val = 0,0.
    neg_val = 0.

    for number, label in zip(one_over_p,testing_data['new_labels']):
        if label in positive_classes:
            pos_count +=1
            pos_val += number
        else:
            neg_val +=number

    return pos_val/pos_count - beta* neg_val/(len(one_over_p)-pos_count)



def classification_accuracy(testing_data,estimator, positive_classes = ('Preictal',)):
    pred_labels = estimator.predict(testing_data['data'])
    TP, FP, TN, FN = get_binary_decision_stats(pred_labels,testing_data['new_labels'],positive_classes=positive_classes)
    return (TP + TN)/(TP + FP + TN + FN)

def balanced_classif_accuracy(testing_data,estimator,beta = 1,positive_classes = ('Preictal',)):
    pred_labels = estimator.predict(testing_data['data'])
    TP, FP, TN, FN = get_binary_decision_stats(pred_labels,testing_data['new_labels'],positive_classes=positive_classes)
    return TP*(TN + FP) + beta*TN*(TP + FN)


def f_beta_score(testing_data,estimator,beta =1, positive_classes = ('Preictal',)):
    pred_labels = estimator.predict(testing_data['data'])
    TP, FP, TN, FN = get_binary_decision_stats(pred_labels,testing_data['new_labels'],positive_classes=positive_classes)
    return (1 + beta**2)* TP/((1 + beta**2)* TP + beta**2*FN + FP)


def get_binary_decision_stats(predicted_labels,actual_labels,positive_classes=('Preictal',)):
    TP, FP, TN, FN = 0,0,0,0
    for predicted_label,actual_label in zip(predicted_labels,actual_labels):
        if predicted_label in positive_classes:
            if actual_label in positive_classes:
                TP +=1
            else:
                FP +=1
        else:
            if actual_label in positive_classes:
                FN +=1
            else:
                TN +=1

    return TP,FP,TN,FN

def gardner_decision_rule(outlier_sequence,window_times,adapt_rate,holdoff_time,decision_threshold=0.5):
    '''
    Computes gardner-like decisions on a sequence of outliers
    :param outlier_sequence: numpy array of numbers
    :param adapt_rate: the length of the box filter, in windows
    :param holdoff_time: (persistence time) time for which after a positive decision is reached, the decision is held.
    :param decision_threshold:
    :return: TP, FP, negative_duration, latency
    '''
    outlier_fraction = box_filter(outlier_sequence,adapt_rate)
    gardner_decisions = threshold_decision_rule(outlier_fraction,window_times,holdoff_time,decision_threshold)
    return gardner_decisions
    pass

def threshold_decision_rule(outlier_fraction,window_times,holdoff_time,decision_threshold):
    '''
    Raise a flag for the function given if it rises above a certain threshold.
    Hold off further evaluation if it reaches this for a defined time.

    Returns the times at which a positive is flagged.
    :param outlier_fraction: a vector of the outlier fraction
    :param window_times: a vector of the times at which the window was collected
    :param holdoff_time:
    :param decision_threshold: for what value do tou decide?
    :return:
    '''
    last_time = -np.inf
    pos_times = []
    for fraction,window_time in outlier_fraction,window_times:
        if fraction > decision_threshold and window_time > holdoff_time + last_time:
            last_time = window_time
            pos_times.append(last_time)
    return np.array(pos_times)

def score_holdoff_d_rule(pos_times,holdoff_time,file_duration,min_false_pos_time = None,seizure_file = True, seiz_times = None):
    '''
    score a decision rule with a holdoff period (like Gardner's method)
    :param pos_times:
    :param holdoff_time:
    :param seizure_file: Does a seizure happen DURING the pos_times given?
    :param min_false_pos_time: What is the minimum amount of time we must have in
                               order to have a false positive right before a seizure?
                               Default holdoff_time. (Will not allow FPR to reach more than highest possible)
    :param seiz_start: At what time, in same units as other times, does the seizure start?
    :return:
    '''


    if seizure_file:
        if seiz_times is None:
            raise ValueError('No seizure start time given even though file declared a seizure')

        start_seiz, end_seiz = seiz_times


    if min_false_pos_time is None:
        min_false_pos_time = holdoff_time/2

    TP, FP = 0,0

    if seizure_file:

        if start_seiz < holdoff_time + min_false_pos_time:
            count_no_FPs = True
            negative_duration = 0
        else:
            count_no_FPs = False
            negative_duration = start_seiz-holdoff_time

        for pos_time in pos_times:
            #check to see that the time before the seizure is long enough to allow false positives
            if count_no_FPs:
                if pos_time < start_seiz-holdoff_time:
                    continue
            if pos_time < start_seiz - holdoff_time:
                FP +=1
            elif pos_time < start_seiz:
                TP +=1
                latency = pos_time-start_seiz
                return TP,FP,negative_duration, latency
            elif pos_time<end_seiz:
                latency = pos_time-start_seiz
            else:
                return TP,FP,negative_duration,np.nan

    else:
        TP = 0
        FP = len(pos_times)
        negative_duration = file_duration
        return TP, FP, negative_duration, np.nan


def default_estimator_score(testing_data,estimator,**score_func_params):
    return estimator.score(testing_data['data'], testing_data['lab'])





from DCEpy.Features.Preprocessing.preprocess import easy_seizure_preprocessing as simple_preprocessor

def cv_all_patients(patients,estimator,estimator_params,score_func,score_func_params,
                    feature_func,preprocessing_routine,channel_name,
                    fit_params = None, feature_type="Single",
                    window_size = 4000,window_step =2000,
                    total_labels =('Interictal Asleep','Interictal Awake','Preictal'),
                    training_labels = ('Interictal Asleep','Interictal Awake'),**func_params):
    fit_params = fit_params if fit_params is not None else {}
    data_base_path = os.path.abspath(os.path.join(os.getcwd(),'../../data'))
    patient_paths = [os.path.join(data_base_path,patient) for patient in patients]
    patient_index = 0
    for patient_data in prepare_and_store_data(patient_paths,preprocessing_routine, window_size=window_size,
                            window_step=window_step, feature_func=feature_func,read_chunktime=300,
                            feature_type=feature_type, channel_name=channel_name, fs=1000,
                            feature_pickle_name = 'feature_pickle.txt', **func_params):


        cross_validator = BasicGridCrossValidator(estimator,estimator_params,score_func,
                                                  patient_paths[patient_index],supervised=False,
                                                  fit_params = fit_params,total_labels=total_labels,
                                              training_labels=training_labels,**score_func_params)
        cross_validator.cross_validate(patient_data)
        patient_index +=1


if __name__ =='__main__':
    #feature func imported
    from DCEpy.Features.ARModels.single_ar import ar_features
    # from DCEpy.Features.Bivariates.cross_correlation import cross_correlate
    # from DCEpy.Features.Bivariates.cross_correlation import compress
    from sklearn.svm import OneClassSVM
    from DCEpy.ML.anomaly_detect import AutoEncoder

    score_func = lambda:None
    estimator = AutoEncoder()

    #the parameters we want to evaluate our estimator with. Dict of lists
    grid_params = [
    {'hidden_layers':[(3,),(4,),(5,),(6,),(7,)],'learn_rate':[0.03],'err_conv':[0.1],
     'batchsize':[5],'momentum':[.4],'normalizer':['variance'],'max_steps':[75000]}
]
    # If applicable, the parameters we want to feed into the .fit() method of out estimator
    fit_params = {'learn_step':500,'learn_hist':True}

    cv_all_patients(['TS041'],estimator,grid_params,AutoEncoder_balanced_percentile_score2,{'beta':1},ar_features,simple_preprocessor,'LAH2',feature_type="Single",
                    fit_params = fit_params ,order=9)

    # visualization of features, i changed window_size to 1000 from 2000