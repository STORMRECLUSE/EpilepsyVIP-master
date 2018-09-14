from __future__ import division
from __future__ import print_function

import os
import pickle
import copy
import logging
from collections import defaultdict, namedtuple

import numpy as np

from sklearn.cross_validation import PredefinedSplit
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


from DCEpy.General.DataInterfacing.edfread import edfread
from DCEpy.Features.Preprocessing.preprocess import box_filter
from DCEpy.ML.anomaly_pipeline import prepare_and_store_data, prepare_train_test_viz_data
from DCEpy.Features.Preprocessing.preprocess import easy_seizure_preprocessing as simple_preprocessor


def f_beta_score(testing_data,estimator,beta =1, positive_classes = ('Preictal',)):
    pred_labels = estimator.predict(testing_data['data'])
    TP, FP, TN, FN = get_binary_decision_stats(pred_labels,testing_data['new_labels'],positive_classes=positive_classes)
    return (1 + beta**2)* TP/((1 + beta**2)* TP + beta**2*FN + FP)

def sklearn_f_beta_score(estimator, test_data,actual_labels, beta = 1,positive_classes = ('Preictal',)):
    pred_labels = estimator.predict(test_data)
    TP, FP, TN, FN = get_binary_decision_stats(pred_labels, actual_labels,
                                               positive_classes=positive_classes)
    return (1 + beta ** 2) * TP / ((1 + beta ** 2) * TP + beta ** 2 * FN + FP)

def percent_correct(estimator, test_data, actual_labels, positive_classes = ('Preictal',)):
    pred_labels = estimator.predict(test_data)
    TP, FP, TN, FN = get_binary_decision_stats(pred_labels, actual_labels,
                                               positive_classes=positive_classes)
    return  (TP+TN)/(TP + FP + TN + FN)


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



def perform_svm_grid_search(patient_data, classifier, svm_grid, decision_rule_grid, svm_score_func, decision_score_func,
                            estimator_fit_params = None, preictal_time = 210):


    # Separate data into folds
    estimator_fit_params = estimator_fit_params if estimator_fit_params is not None else {}
    # decision_score_func_params = decision_score_func_params if decision_score_func_params is not None else {}

    sorted_container = prepare_train_test_viz_data(patient_data,preictal_time=preictal_time, train_only_interictal=False,
                                                   )
    # consolidate folds data into array, give indices to separate folds for validation
    fold_data = sorted_container['fold_data']
    total_data = []
    data_labels = []
    fold_numbers = []
    running_fold_number =0
    for data_type in fold_data:

        for fold_number in fold_data[data_type]:
            total_data.extend(fold_data[data_type][fold_number]['data'])
            data_labels.extend([data_type]*np.size(fold_data[data_type][fold_number]['data'],0))
            fold_numbers.extend([running_fold_number]*np.size(fold_data[data_type][fold_number]['data'],0))
            running_fold_number +=1

    total_data = np.array(total_data)
    # Use custom fold iterator (Predefined Split), along files.
    cv = PredefinedSplit(fold_numbers)




    # Perform Cross Validation on data using GridSearchCV
    cross_validator = GridSearchCV(classifier,svm_grid,scoring=svm_score_func,fit_params=estimator_fit_params,cv = cv,verbose=1)
    # TODO: with best parameters, optimize other stuff
    best_estimator = cross_validator.fit(total_data,data_labels)

    a = 1

def cv_all_patients(patients,classifier, svm_grid,decision_rule_grid, svm_score_func, decision_score_func, feature_func,
                    preprocessor = simple_preprocessor, estimator_fit_params = None,feature_type = 'Single',
                    window_size=3000, window_step=1000,
                   fs = 1000, read_chunktime = 300, info_pickle_name ='patient_pickle.txt', channel_name = None,
                   feature_pickle_name='feature_pickle.txt', **feature_params):

    # fit_params = fit_params if fit_params is not None else {}
    data_base_path = os.path.abspath(os.path.join(os.getcwd(),'../../../data'))
    patient_paths = [os.path.join(data_base_path,patient) for patient in patients]
    patient_index = 0

    for patient_data in prepare_and_store_data(patient_paths, preprocessor,window_size,window_step,preprocessing_routine_params=None,
                           feature_func = feature_func,read_chunktime = read_chunktime,info_pickle_name = info_pickle_name,
                           feature_type= feature_type, channel_name = channel_name,
                           feature_pickle_name = feature_pickle_name, fs = fs,**feature_params):
        perform_svm_grid_search(patient_data,classifier, svm_grid, decision_rule_grid, svm_score_func,decision_score_func,
                                estimator_fit_params = estimator_fit_params)



if __name__ == '__main__':
    # feature func imported
    from DCEpy.Features.ARModels.single_ar import ar_features
    # from DCEpy.Features.Bivariates.cross_correlation import cross_correlate
    # from DCEpy.Features.Bivariates.cross_correlation import compress
    from sklearn.svm import OneClassSVM, SVC
    from DCEpy.ML.anomaly_detect import AutoEncoder


    # the parameters we want to evaluate our estimator with. Dict of lists
    svm_grid_params = [
        {'C': [1,2,3], }
    ]

    decision_rule_grid = [
        {'adapt_rate':[20,30]}
    ]
    # If applicable, the parameters we want to feed into the .fit() method of out estimator
    estimator_fit_params = {}
    estimator = SVC()

    #svm_score_func =lambda estimator,X,Y: sklearn_f_beta_score(estimator,X,Y,beta = 1,positive_classes=('Preictal',))

    svm_score_func = percent_correct
    decision_score_func = lambda:None

    cv_all_patients(['TS041'],estimator, svm_grid_params, decision_rule_grid,svm_score_func, decision_score_func, ar_features,
                    preprocessor=simple_preprocessor, estimator_fit_params=estimator_fit_params,
                    feature_type="Single", window_size=3000, window_step=1000,fs = 1000,read_chunktime=300,
                    info_pickle_name='patient_pickle.txt',channel_name= 'LAH2',feature_pickle_name='feature_pickle.txt',
                    order=9)

