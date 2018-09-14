"""This file performs feature computation and runs vizualizations on different features."""

from __future__ import print_function
import aggregate_viz

from DCEpy.Features.ScatCo.scat_coeffs import scat_coeffs
from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import lyapunov_exponent


## import all the feature functions here
from DCEpy.Features.ARModels.single_ar import ar_features
# from feature_functions import energy_features


def feature_viz_experiment(patients_lst,feature_func, channel_name,feature_type="Single",order=None):
    """
    runs feature visualization.

    :param patients_lst:
    :param feature_func:
    :param channel_name:
    :return:
    """

    aggregate_viz.viz_all_patients(patients_lst,feature_func,channel_name,order=order)



#--------------- Feature Visualizations------------------

## Autoregression Features
# feature_viz_experiment(['TS039'],ar_features,'LAH1', "Single",order=30)


## Gardner's Energy Features
# feature_viz_experiment(['TS039'],energy_features,"LAH1")

