__author__ = 'Chris'

import os
import time

from DCEpy.Features.GardnerStudy.features.Old_files import analyze_patient

# parameters -- sampling data
window_length = 1.0 # in seconds
window_overlap = 0.5 # in seconds
num_windows = 800 # number of windows to sample to build svm model
f_s = float(1e3) # sampling frequency
include_awake = True
include_asleep = False
p_feat = 3 # only applicable for energy statistics
feature = 'energy_stat'

# list of patients that will be used
patients = ['TS039', 'TS041']

# get the paths worked out
to_data = os.path.dirname( os.path.dirname( os.path.dirname( os.getcwd()) ) )
data_path = os.path.join(to_data, 'data')
save_path = os.path.join(to_data, 'data', 'single_channel_gardner_results_short_overlap')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# create results file
results_file = os.path.join(save_path, 'gardner_results.txt')
f = open(results_file, 'w')

# write the first few lines
print >>f, "Results file for One-Class SVM Anomaly Detection\n"
print >>f, 'Ran on %s\n' %time.strftime("%c")

# write the parameters
print >>f, 'Parameters used for this test\n================================\n'
print >>f, 'Feature used is %s' %feature
print >>f, 'Window Length \t%.3f\nWindow Overlap \t%.3f\nNumber of training windows \t%d' %(window_length, window_overlap, num_windows)
print >>f, 'Sampling Frequency \t%.3f' % f_s
print >>f, 'Awake Times are ' + (not include_awake)*'NOT ' + ' included in training'
print >>f, 'Asleep Times are ' + (not include_asleep)*'NOT ' + ' included in training\n\n'

# write the patients
print >>f, 'Patients are ' + " ".join(patients) + "\n\n"

for patient_id in patients:

    # update paths specific to each patient
    p_data_path = os.path.join(data_path, patient_id)
    p_save_path = os.path.join(save_path, patient_id)

    # analyze the patient, write to the file
    analyze_patient(p_data_path, feature, p_save_path, patient_id, f, p_feat, window_length, window_overlap, num_windows, f_s, include_awake, include_asleep)


