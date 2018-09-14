__author__ = 'Chris'

import csv
import os
import sklearn

import numpy as np
from DCEpy.Features.GardnerStudy.gardner_many_svm_functions import *


def read_csv(file_name):
    reader = csv.reader(open(file_name, 'r'))
    header = reader.next()[:-1]
    X = []
    for row in reader:
        if len(header) > 1:
            X.append([float(a) for a in row[:-1]])
        else:
            X.append(float(row[0]))
    return np.array(X), header

# get mean, std of filtered data
to_data = os.path.dirname( os.path.dirname( os.getcwd()) )
data_path = os.path.join(to_data, 'data', 'Erik')
filt_file = os.path.join(data_path, 'filt_erik.csv')
X_filt, channels = read_csv(filt_file)
channel_num = len(channels)
mean_vec = np.mean(X_filt, axis=0)
std_vec = np.std(X_filt, axis=0)

# write mean, std to a csv file
file_name = os.path.join(data_path, 'normalization.csv')
f = open(file_name, 'w')
str1 = ''
str2 = ''
str3 = ''
for i in range(channel_num):
    str1 += channels[i] + ","
    str2 += '%.3f,' %mean_vec[i]
    str3 += '%.3f,' %std_vec[i]
str1 = str1[:-1] + "\n"
str2 = str2[:-1] + "\n"
str3 = str3[:-1] + "\n"
f.write(str1 + str2 + str3)
f.close()

# get the energy statistics
energy_file = os.path.join(data_path, 'energy_erik.csv')
X_feat, header = read_csv(energy_file)

# put all parameters there
nu = 0.01
kernel = 'rbf'
gamma = 1/float(3)
C = 0.9* np.ones(channel_num)
weight = 1/float(channel_num) * np.ones(channel_num)
adapt_rate = 30
param_file = os.path.join(data_path, 'erik_parameters.params')
model_file = os.path.join(data_path, 'energy_erik.model')

# create SVM model
clf = sklearn.svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
X_feat = np.nan_to_num(X_feat)
clf.fit(X_feat)

# create model and param files
create_model_file(model_file, clf, nu, gamma)
create_param_file(param_file, C, weight, adapt_rate, channel_num)

