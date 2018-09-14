"""

nu-svm parameter tuning with gridsearchCV.

Note: The burns_parent function reads in Burns features, optimizes nu,gamma based on Burns features.

"""

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import NuSVC
from DCEpy.Features.BurnsStudy.BurnsPipeline import analyze_patient_raw
from sklearn.metrics import classification_report
import numpy as np
import scipy.io as sio
import os
from DCEpy.Features.Classifiers.classifier_toolbox import label_classes
from classifier_toolbox import viz_labels
from scipy.ndimage.filters import gaussian_filter
from classifier_toolbox import mat_corrcoeff, hinge_loss,ham_loss,jss,f_beta, chs


def nusvm_gridsearch(train_data,train_labels):

    """
    performs grid search with cross validation for scoring on the training set.
    :param train_data: data matrix with shape (number of samples, number of features)
    :param train_labels: label array with shape (number of samples,)
    :return: the best classifier
    """

    print "Started Parameter Tuning..."

    # define the grid
    # Options: different ranges, more parameters
    nu_range = [.001, .002, .004, .005, .008, .01, .015, .02, .04, .05, .08, .1,.15,.2]
    gamma_range = [.001, .002, .003, .004, .005, .006, .008, .01, .05, .1, .2, .5,1,2,4,8]



    grid = [{"nu": nu_range, "gamma": gamma_range}]


    # define the scoring function(for model evaluation)
    # Options see: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = "f1"

    #initiate nu-svc object
    svr = NuSVC(class_weight="balanced")

    # initiate grid search object
    print "\tSearching for the best nu-svm model..."
    clf = GridSearchCV(svr,grid,scoring=scoring)
    clf.fit(train_data,train_labels)
    print "\tThe best parameters are: ", clf.best_params_
    print "\tThis set of parameters receive a score of: ", clf.best_score_

    return clf.best_estimator_





def burns_parent(win_len_secs=3.0, win_overlap_secs=2.0,f_s=float(1e3)):


    # define parameters
    # parameters -- sampling data
    win_len_secs = 3.0  # in seconds
    win_overlap_secs = 2.0  # in seconds
    f_s = float(1e3)  # sampling frequency
    patients = ['TS039']

    long_interictal = [False]
    include_awake = True
    include_asleep = False

    # get the paths worked out
    to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_path = os.path.join(to_data, 'data')


    for patient_index, patient_id in enumerate(patients):
        print "---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths specific to each patient
        p_data_path = os.path.join(data_path, patient_id)
        # analyze the patient, write to the file
        all_files, data_filenames, file_type, seizure_times, seizure_print = analyze_patient_raw(p_data_path, f_s,
                                                                                                 include_awake,
                                                                                                 include_asleep,
                                                                                                 long_interictal[
                                                                                                     patient_index])
        file_num = len(data_filenames)

        # cross-validation
        for i in [0, 1, 2, 3, 4, 5]:

            # set up test files, seizure times, etc. for this k-fold
            print '\nCross validations, k-fold %d of %d ...' % (i + 1, file_num)
            testing_file_name = data_filenames[i]
            cv_file_names = data_filenames[:i] + data_filenames[i + 1:]
            # cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            # training set for Burns
            stored_features_path = '/Users/TianyiZhang/Documents/EpilepsyVIP/DCEpy/Features/BurnsStudy'
            train_name = 'evc_training{}_{}.mat'
            test_name = 'evc_testing{}_{}.mat'
            train_path = os.path.join(stored_features_path,train_name)
            test_path = os.path.join(stored_features_path,test_name)


            load_train_data = sio.loadmat(train_path.format(patient_id[-2:], i))
            training_evc_cv_files = load_train_data.get('data')[0]
            training_evc_cv_files = np.ndarray.tolist(training_evc_cv_files)
            training_labels = [label_classes(data=training_evc_cv_files[j],data_seizure_time=cv_seizure_times[j], win_len_seconds=win_len_secs,win_overlap_seconds=win_overlap_secs) for j in range(len(cv_file_names))]

            training_data = np.vstack(training_evc_cv_files)
            training_labels = np.hstack(training_labels)


            # parameter tuning
            best_clf = nusvm_gridsearch(train_data=training_data,train_labels=training_labels)


            # load test data
            load_test_data = sio.loadmat(test_path.format(patient_id[-2:], i))
            testing_evc_cv_files = load_test_data.get("data")
            testing_data = np.vstack(testing_evc_cv_files)
            test_seizure_time = seizure_times[i]
            actual_labels = label_classes(data=testing_data,data_seizure_time=test_seizure_time,win_len_seconds=win_len_secs,win_overlap_seconds=win_overlap_secs)
            predicted_labels = best_clf.predict(testing_evc_cv_files)


            # post processing

            # report+visualize(save) results
            print classification_report(actual_labels,predicted_labels)

            viz_labels(labels=[actual_labels,predicted_labels],testing_file_name=testing_file_name,test_data_seizure_time=test_seizure_time,f_s=f_s,win_len_secs=win_len_secs,win_overlap_secs=win_overlap_secs)



burns_parent()

#
# def parent(feat_func,win_len_secs=2.0, win_overlap_secs=1.0,f_s=float(1e3),burns=True):
#
#
#     # define parameters
#     # parameters -- sampling data
#     win_len_secs = 2.0  # in seconds
#     win_overlap_secs = 1.0  # in seconds
#     f_s = float(1e3)  # sampling frequency
#     win_len = int(win_len_secs * f_s)
#     win_overlap = int(win_overlap_secs * f_s)
#
#     patients = ['TS039']
#     chosen_channels = [81]  # varies for each patient
#
#     long_interictal = [False]
#     # include_awake = True
#     include_awake = False
#     # include_asleep = False
#     include_asleep = True
#
#     # get raw data
#
#     # reduce channels
#
#     # get features
#
#     # label data
#
#     # grid search for nu-svm
#
#
#     # test on test file
#     # best_clf = nusvm_gridsearch(train_data= train_features,train_labels=train_labels)
#
#     # predicted_labels = best_clf(test_features)
#
#     pass

