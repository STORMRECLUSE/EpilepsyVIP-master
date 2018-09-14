from __future__ import print_function
__author__ = 'Chris'

import numpy as np
import  DCEpy.ML.mcd_alg as mcd_alg
import os

def anom_test(X_inter, X_seizure, X_labels, feature_str, save_path, h=0.85):
    """
    This function uses robust methods to fit interictal data to a normal distribution.
    Then computes mahalanobis distances (equivalently, p-values) for seizure windows.
    Finally, plots the p-value.

    Parameters
    ----------
    X_inter: ndarray, shape(n,p)
        interictal data array of n observations, p features
        used to build model of 'normal'
    X_seizure: list of ndarray,
        each ndarray in list corresponds to seizure data in
        a single seizure file
    X_labels : list of ndarray
        each ndarray in list contains strings that label the time windows
    feature_str: string
        a string that is the feature we are testing.
        Spaces are okay
    save_path: string
        file path to where we should save stuff

    Returns
    -------
    m_dist: list of ndarray, shape(L)
        mahalanobhis distances of seizure windows
    """
    # number of seizures
    K = len(X_seizure)

    # title and save strings
    title_str = feature_str
    save_str = feature_str.replace(" ", "_") + "_s"

    # build gaussian model
    mcd = mcd_alg.build_model(X_inter, h)
    m_dist = []

    # for each seizure file
    for k in range(K):

        # get seizure data with labels
        X = X_seizure[k]
        labels = X_labels[k]
        file_path = os.path.join(save_path, save_str + str(k) + '.png')

        # get anomaly score for sliding time window (mahalanobis distance)
        score = mcd_alg.anomaly_score(X, mcd)
        m_dist.append(score)

        # visualize and save the anomaly score
        mcd_alg.plot_score(score, labels, file_path, title_str + ' - Seizure ' + str(k), threshold=np.inf, plot_log=True)

    return m_dist