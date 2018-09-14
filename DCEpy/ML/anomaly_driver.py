'''
This file collects features for patient TS0041, tests ML schemes on these features, and then
'''
from __future__ import print_function
from DCEpy.Features.ARModels.ar_order import ar_order_sel,var_order_sel
from DCEpy.Features.ARModels.ar_stability import ar_stability
from DCEpy.Features.ARModels.prediction_err import mvar_pred_err,ar_pred_err
from DCEpy.Features.ARModels.single_ar import ar_features_nonlinear,ar_features
from DCEpy.Features.BurnsStudy.eig_centrality import eig_centrality


from DCEpy.ML.classif_data_collection import classif_data_collect
from DCEpy.ML.anomaly_methods import anom_test

import numpy as np
import os

def dict_product(dict_with_lists):
    return (dict(zip(dict_with_lists, x))
            for x in cartesian_product(*dict_with_lists.itervalues()))

def cartesian_product(L=None,*lists):
    '''
    Returns the cartesian product of all lists
    :param L:
    :param lists:
    :return:
    '''
    if L is None:
        return
    if not lists:
        for x in L:
            yield (x,)
    else:
        for x in L:
            for y in cartesian_product(lists[0],*lists[1:]):
                yield (x,)+y



class IncompleteClassError(Exception):
    def __init__(self,val):
        self.val = val
    def __str__(self):
        return 'Incomplete class. No function for: {}'.format(self.val)



###
class FeatureDriver(object):
    feature = 'NoFeature'
    feature_func = None
    def __init__(self,data_container ,**params):
        '''

        :param data_container:
        :param summary_fo:
        :param longfile_fo:
        :param params:
        :return:
        '''
        self.data_container = data_container
        self.params = params
    def get_feature_list(self):
        '''

        :return: X,Y, and T, where

                    X is the feature matrix: nxp
                        where n is no observations
                              p is features/observation
                    Y is the label matrix: size: n
                        each label is a string out of:
                            'Preictal','Ictal','Postictal','Interictal'
                    T is the time chunk vector
                        an array of ints specifying the time chunk
                        a window is a member of
        '''
        self.feature_vect = []
        self.label_vect = []
        self.time_chunk_vect = []
        for window,label,seiz_str in self.data_container:
            time_chunk = int(seiz_str[-1])
            self.obtain_feature(window,label, time_chunk)

        X,Y,T = map(np.asarray,(self.feature_vect,self.label_vect,
                           self.time_chunk_vect))
        return X,Y,T



    def obtain_feature(self,window,label,time_chunk):
        raise IncompleteClassError('Calling Features')

    def call_func(self,data,label,time_chunk):
        channel_feats = self.feature_func(data,**self.params)
        self.feature_vect.append(channel_feats)
        self.label_vect.append(label)
        self.time_chunk_vect.append(time_chunk)


class SingleFeatureDriver(FeatureDriver):
    def obtain_feature(self,window,label,time_chunk):
        '''
        From a window, get all features and return their observations
        :param window:
        :param label:
        :param time_chunk:
        :return:
        '''
        window = window.T
        for channel in window:
            try:
                self.call_func(channel,label,time_chunk)
            except TypeError as e:
                raise IncompleteClassError('No Function selected')

class MultFeatureDriver(FeatureDriver):
    def obtain_feature(self,window,label,time_chunk):
        try:
            self.call_func(window,label,time_chunk)
        except TypeError as e:
            raise IncompleteClassError('No Function selected')

class ARStability(MultFeatureDriver):
    feature = 'VAR eigenstability'
    feature_func = staticmethod(ar_stability)

class VARPredictErr(MultFeatureDriver):
    feature = 'VAR prediction error'
    feature_func = staticmethod(mvar_pred_err)

class EigCentrality(MultFeatureDriver):
    feature = 'Eigenvector Centrality'
    feature_func = staticmethod(eig_centrality)

class ARPredictErr(SingleFeatureDriver):
    feature = 'AR prediction error'
    feature_func = staticmethod(ar_pred_err)

class ARNonlinear(SingleFeatureDriver):
    feature = 'AR Coefficients + Nonlinear Features'
    feature_func = staticmethod(ar_features_nonlinear)

class ARCoeffs(SingleFeatureDriver):
    feature = 'AR Coefficients without nonlinear features'
    feature_func = staticmethod(ar_features)


class FeatureTester(object):
    window_len_class=5000
    preictal_time_class=30
    postictal_time_class=50
    n_windows_class=6
    window_overlap_class=.5
    variable_params_class = {}
    params_class = {}
    classification_names_class = ['ridge','svm_lin','svm_poly','svm_rad',
                                    'nb','lda','qda','rf']

    FeatureClass_class = FeatureDriver

    def __init__(self,window_len=None,
                 preictal_time=None,postictal_time=None,n_windows=None,
                 window_overlap=None,
                 variable_params=None,
                 FeatureClass=None,
                 classification_names = None,**params):
        self.window_len=window_len \
            if window_len is not None else self.window_len_class
        self.preictal_time=preictal_time  \
            if preictal_time is not None else self.preictal_time_class
        self.postictal_time=postictal_time \
            if postictal_time is not None else self.postictal_time_class
        self.n_windows=n_windows \
            if n_windows is not None else self.n_windows_class
        self.window_overlap=window_overlap \
            if window_overlap is not None else self.window_overlap_class
        self.variable_params = variable_params \
            if variable_params is not None else self.variable_params_class
        self.params = params \
            if params else self.params_class
        self.FeatureClass = FeatureClass\
            if FeatureClass is not None else self.FeatureClass_class
        self.classification_names = classification_names\
            if classification_names is not None else self.classification_names_class

    def collect_data(self,filenames,seizure_times,good_channels=None,bad_channels = None ):
        self.data_container = \
            classif_data_collect(filenames,seizure_times,
                                 self.window_len,self.preictal_time,self.postictal_time,
                                 self.n_windows,self.window_overlap,
                                 good_channels=good_channels, bad_channels=bad_channels)

    def set_data(self,data_container):
        self.data_container = data_container

    def write_to_files(self,summary_obj,longfile_obj):
        '''
         Write out to files given input features.
        :param summary_obj: file
        The file containing the summary
        :param longfile_obj: file
        The file containing all information output
        :return:
        '''
        if self.variable_params:
            for variable_param_dict in dict_product(self.variable_params):
                self._write_to_files_and_call_fxn(summary_obj,
                                                  longfile_obj,
                                                  variable_param_dict)
        else:
            variable_param_dict = {}
            self._write_to_files_and_call_fxn(summary_obj,
                                              longfile_obj,
                                              variable_param_dict)


    def _write_to_files_and_call_fxn(self,summary_obj,longfile_obj, variable_param_dict):
        '''
        Write out to files given input features.
        :param summary_obj: file
        The file containing the summary
        :param longfile_obj: file
        The file containing all information output
        :return:
        '''

        full_params = self.params.copy()
        full_params.update(variable_param_dict)
        feat_implement = self.FeatureClass(
                            self.data_container,**full_params)
        feat_str = feat_implement.feature
        X,Y,T = feat_implement.get_feature_list()

        # write out feature type and param type to both files
        self._write_to_file(summary_obj,feat_str,variable_param_dict)
        self._write_to_file(longfile_obj,feat_str,variable_param_dict)
        anom_test(X,Y,T,summary_obj,longfile_obj, self.classification_names)

        return



    @staticmethod
    def _write_to_file(fileobj,feat_str,params_used):
        '''
        The output that will be written to each file on every iteration
        :param fileobj:
        :param feat_str: str a string describing the feature
        :param params_used: the varied parameters used this time around
        :return:
        '''
        fileobj.write('******************************************\n')
        fileobj.write('Feature: {}  \nVaried Param Run: {}\n'.format(feat_str,params_used))


class ARStabilityTester(FeatureTester):
    FeatureClass_class = ARStability
    params_class = {'order':30} #VAR value
    variable_params_class = {'n_eigs':[5,6,7,8,9,10]}

class ARNonlinearTester(FeatureTester):
    FeatureClass_class = ARNonlinear
    params_class = {'order':30, #single AR value
                    'nonlinear_feats':['lyapunov_exponent'],
                    'nonlinear_params':[{}]}

class EigCentralityTester(FeatureTester):
    FeatureClass_class = EigCentrality
    params_class = {} # there are no parameters to feed in

class ARCoeffsTester(FeatureTester):
    FeatureClass_class = ARCoeffs
    params_class = {'order':30} #single AR value

class ARPredictErrTester(FeatureTester):
    FeatureClass_class = ARPredictErr
    params_class = {'order':30, #single AR value
                    }
    variable_params_class = {'pred_window_size':
                             [500,750,1000,1250,1500]}

class VARPredictErrTester(ARPredictErrTester):
    FeatureClass_class = VARPredictErr
    params_class = {'order':30} #MVAR value

if __name__ == '__main__':
    # Example code to see how this works. Data taken from patient TS0041

    summary_file = 'ML_summary_Centrality.txt'
    longform_file = 'ML_log_Centrality.txt'

    # file_dir = '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs' # 42
    file_dir = '/home/chris/Documents/Rice/senior/EpilepsyVIP/data/RMPt2' # Chris

    data_filenames = ('DA00101U_1-1+.edf',
                 'DA00101V_1-1+.edf',
                 'DA00101W_1-1+.edf',
                 'CA00100E_1-1_03oct2010_01_03_05_Awake+.edf',
                 'CA00100F_1-1_03oct2010_13_05_13_Awake+.edf',
                 'DA00101P_1-1_02oct2010_09_00_38_Awake+.edf'
                 ) # 42
    # data_filenames = ('DA00101U_1-1+.edf',
    #              'DA00101V_1-1+.edf',
    #              'DA00101W_1-1+.edf',
    #              'CA00100E_1-1+.edf',
    #              'CA00100F_1-1+.edf',
    #              'DA00101P_1-1+.edf'
    #              ) # Chris

    seizure_times = (262,330),(107,287),(191,405),None,None,None

    data_filenames = [os.path.join(file_dir,filename) for filename in data_filenames]

    good_channels = [
                    'LAH1','LAH2','LPH6','LPH7',
                    'LPH9','LPH10','LPH11','LPH12'
                    ]

    rstat_bands= np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])


    data_container1 = classif_data_collect(data_filenames,seizure_times,window_len=5000,
                         preictal_time=30,postictal_time=50,n_windows=6,window_overlap=.5, good_channels=good_channels,
                                          rstat_bands=None)

    data_container2 = classif_data_collect(data_filenames,seizure_times,window_len=5000,
                         preictal_time=30,postictal_time=50,n_windows=6,window_overlap=.5,
                                           bad_channels=('Events/Markers','EDF Annotations'),
                                          rstat_bands=rstat_bands)
    print('Data read in')

    for ind,(_,label,_) in enumerate(data_container1):
        if label == 'Interictal':
            first_interictal_ind= ind
            break


    interictal_test_window,_,_ = data_container1[first_interictal_ind]

    #mvar_order = var_order_sel(interictal_test_window,maxorder=40)
    #ar_order = ar_order_sel(interictal_test_window,maxorder=300)
    # print('Ideal MVAR order: {} \t '
    #       #'Ideal AR order{}'
    #       ''.format(mvar_order,#ar_order
    #                 ))



    testclass_list = [ARCoeffsTester,ARNonlinearTester,
                    ARPredictErrTester,
                    ARStabilityTester,VARPredictErrTester,
                    EigCentralityTester]
    with open(summary_file,'w') as summary_obj,\
        open(longform_file,'w') as longform_obj:
        for testclass in testclass_list:
            print('Feature: {}'.format(testclass.FeatureClass_class.feature))
            # if our feature is AR, get the optimal order to run it
            # with and test the feature with it
            if testclass in [ARStabilityTester, #mvar
                             VARPredictErrTester]:
                feat_tester = testclass()
            elif testclass in [ARCoeffsTester,ARNonlinearTester,
                             ARPredictErrTester]:#ar TODO: fix ar_order
                feat_tester = testclass()

            else:   #not AR
                feat_tester = testclass()

            if testclass is EigCentralityTester:
                feat_tester.set_data(data_container2)
            else:
                feat_tester.set_data(data_container1)
            feat_tester.write_to_files(summary_obj,longform_obj)


