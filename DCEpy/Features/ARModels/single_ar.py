'''
Fits an autoregressive model to the

Relevant functions:
'''
from __future__ import print_function
import numpy as np
from statsmodels.tsa import ar_model
from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import nonlinear_func_dict

def ar_features(data,order=30):
    '''
    Fit an order n AR model to the input single channel of data.
    :param data: preprocessed data
    :param order:
    :return:
    '''
    model_ar = ar_model.AR(data)
    model_results =model_ar.fit(maxlag=order)
    return model_results.params



def ar_features_nonlinear(data,order,nonlinear_feats,nonlinear_params):
    '''
    Compute ar features, as well as extra nonlinear
    characteristics of the data. Returns the desired features all together.

    As of November 12, 2015, the only nonlinear feature
    that has been implemented is the lyapunov exponent.


    Because it has no extra parameters to compute, its
    associated nonlinear_params entry is an empty dictionaty.

    To call this function, then, with the lyapunov exponent as an extra feature,
    call it like so:

    ar_features_nonlinear(data,order=n,['lyapunov_exponent'],[{}])
    Note that we specify it with a string. The string you use
    to specify a nonlinear feature is in the nonlinear_grab_bag.py file in
    NonlinearLib


    :param data:
    :param order: the order of the AR model associated with
    :param nonlinear_feats: a list of strings
    :param nonlinear_params: a list
    :return:
    '''
    ar_coeffs = ar_features(data,order)
    extra_feature_list = np.array([])
    for extra_feature,feat_params in zip(nonlinear_feats,nonlinear_params):
        feature_func = nonlinear_func_dict[extra_feature]
        new_feat = feature_func(data,**feat_params)
        extra_feature_list = np.append(
                                extra_feature_list,new_feat
                                )

    return np.append(ar_coeffs,extra_feature_list)
