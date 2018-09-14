from __future__ import print_function
import numpy as np


def decision_rule(labels,pos_classes,n_windows,flag_tol):
        '''
        Implements a primitive decision rule
        from classifications -
        how can we decide if a patient is having a seizure?

        From labels, determine if a patient is having
        a seizure in the following way:

        If, out of the last n_windows windows,
        (as a proportion of the total) more than flag_tol are labels from pos_classes, give  true as the value
        :param labels: a list or numpy array of all of the labels
        :param pos_classes: the class outputs considered to be a positive
        :param n_windows:
        :param flag_tol:
        :return:
        '''
        decision = np.zeros_like(labels,dtype=bool)
        intermed_val = np.zeros_like(labels,dtype=np.float64)
        for ind in range(len(labels)):
            start_ind = max(0,ind-n_windows+1)
            window_len = ind+1-start_ind
            intermed_val[ind] = float(np.sum((label in pos_classes for label in labels[start_ind:ind+1])))/float(window_len)
            decision[ind] = float(intermed_val[ind])>=flag_tol
        return decision,intermed_val