import os

import numpy as np
import scipy.io as sio

from DCEpy.Features.BurnsStudy.spatiotemp_pipeline import take_frames
from DCEpy.Features.Classifiers.classifier_toolbox import label_classes
from DCEpy.Features.Classifiers.classifier_toolbox import viz_labels

# get some burns matrix
i=3

stored_features_path = '/Users/TianyiZhang/Documents/EpilepsyVIP/DCEpy/Features/BurnsStudy'
test_name = 'evc_testing{}_{}.mat'
test_path = os.path.join(stored_features_path, test_name)
patient_id = 'TS039'
load_test_data = sio.loadmat(test_path.format(patient_id[-2:], i))
testing_evc_cv_files = load_test_data.get("data")
testing_data = np.vstack(testing_evc_cv_files)

# frame

seizure_time = None
mat = take_frames(testing_data)
print mat.shape
win_len_seconds=3.0
win_overlap_seconds = 2.0


seizure_start_window = None
seizure_end_window = None
if seizure_time!=None:
    seizure_start_time, seizure_end_time = seizure_time
    seizure_start_window = int((seizure_start_time - win_len_seconds) / (win_len_seconds - win_overlap_seconds) + 1)
    seizure_end_window = int((seizure_end_time - win_len_seconds) / (win_len_seconds - win_overlap_seconds) + 1)

frame_labels = label_classes(mat,None,win_len_seconds=5.0,win_overlap_seconds=0.0)

viz_labels([frame_labels],"Hi")