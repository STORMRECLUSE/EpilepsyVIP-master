
from DCEpy.Features.BurnsStudy.BurnsPipeline import analyze_patient_raw
from sklearn.metrics import classification_report
import numpy as np
import scipy.io as sio
import os
from DCEpy.Features.Classifiers.classifier_toolbox import label_classes
from DCEpy.Features.Classifiers.classifier_toolbox import viz_labels
from DCEpy.Features.Classifiers.csvm_gridsearch import svm_gridsearch
from sklearn.decomposition import PCA

def take_frames(feature_mat,frame_size=5):

    """
    Computes patterns from windowed features.

    :param feature_mat: feature matrix with shape (number of windows, number of features)
    :param frame_mat:  frame matrix with (number of frames, number of features*frame_size)
    :param frame_size: int
    :return:

    """

    n,p = feature_mat.shape # n - number of windows, p - number of features
    num_frames = n//frame_size
    frame_mat = np.ones((num_frames,p*frame_size))
    frame_idx = 0

    for win_end in range(frame_size-1,n,frame_size):
        feature= np.hstack(feature_mat[win_end+1-frame_size:win_end+1])
        frame_mat[frame_idx] = feature
        frame_idx+=1

    return frame_mat


def label_frames(frame_mat,seizure_time,win_len_seconds=2.0,win_overlap_seconds=1.0,frame_size=5,test=False):
    """

    :param feat_mat: feature matrix with shape (number of windows, number of features)
    :param frame_mat: pattern matrix with shape (number of frames, number of features*frame_size), as returned from take_frames
    :param seizure_time: tuple or None
    :param win_len_seconds:
    :param win_overlap_seconds:
    :param frame_size: int
    :return:
    """

    seizure_window = None

    if seizure_time != None:
        seizure_start_time, seizure_end_time = seizure_time
        seizure_start_window = int((seizure_start_time - win_len_seconds) / (win_len_seconds - win_overlap_seconds) + 1)
        seizure_end_window = int((seizure_end_time - win_len_seconds) / (win_len_seconds - win_overlap_seconds) + 1)
        seizure_window = (seizure_start_window,seizure_end_window)

    # label the frames with label_classes by seeing window as "time" and seeing frame as "window"
    frame_labels = label_classes(frame_mat, seizure_window, win_len_seconds=frame_size, win_overlap_seconds=0.0)

    if test:
        return frame_labels,seizure_window

    return frame_labels



def burns_parent(win_len_secs=3.0, win_overlap_secs=2.0,f_s=float(1e3)):


    # define parameters
    # parameters -- sampling data
    win_len_secs = 3.0  # in seconds
    win_overlap_secs = 2.0  # in seconds
    f_s = float(1e3)  # sampling frequency
    frame_size = 5
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
            #???? not sure if each file feature data in training evc cv files is a np matrix

            framed_training = [take_frames(feat_mat) for feat_mat in training_evc_cv_files]
            framed_training_labels = [label_frames(frame_mat=framed_training[j],seizure_time=cv_seizure_times[j],win_len_seconds=win_len_secs,win_overlap_seconds=win_overlap_secs) for j in range(len(cv_file_names))]
            framed_training = np.vstack(framed_training)
            framed_training_labels = np.hstack(framed_training_labels)

            # parameter tuning
            best_clf = svm_gridsearch(framed_training,framed_training_labels)

            # load test data
            load_test_data = sio.loadmat(test_path.format(patient_id[-2:], i))
            testing_evc_cv_files = load_test_data.get("data")
            testing_data = np.vstack(testing_evc_cv_files)
            framed_testing = take_frames(testing_data)
            test_seizure_time = seizure_times[i]
            actual_framed_labels,test_seizure_windows = label_frames(frame_mat=framed_testing,seizure_time=test_seizure_time,win_len_seconds=win_len_secs,win_overlap_seconds=win_overlap_secs,test=True)
            predicted_framed_labels = best_clf.predict(framed_testing)


            # post processing

            # report+visualize(save) results
            print classification_report(actual_framed_labels,predicted_framed_labels)

            viz_labels(labels=[actual_framed_labels,predicted_framed_labels],testing_file_name=testing_file_name,test_data_seizure_time=test_seizure_windows,f_s=f_s,win_len_secs=frame_size,win_overlap_secs=0)



# burns feature
burns_parent()