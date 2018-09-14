'''
This file collects features for patient TS0041, tests ML schemes on these features, and then
'''
from __future__ import print_function
from DCEpy.Features.ARModels.ar_order import ar_order_sel,var_order_sel
from DCEpy.Features.ARModels.ar_stability import ar_stability
from DCEpy.Features.ARModels.prediction_err import mvar_pred_err, ar_pred_err
from DCEpy.Features.ARModels.single_ar import ar_features_nonlinear, ar_features
from DCEpy.Features.BurnsStudy.eig_centrality import eig_centrality

# from DCEpy.Features.ScatCo.scat_coeffs import flat_scat_coeffs
# import matlab.engine

from DCEpy.ML.anomaly_methods import anom_test

from DCEpy.ML.classif_data_collection import classif_data_collect
from DCEpy.ML.classifiers import ML_test

from DCEpy.ML.DecisionRules.classification_decision_rule import decision_rule
from DCEpy.ML.DecisionRules.testing import update_decision_stats,check_for_fp,diagnose_seiz_file

import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt, ticker as ticker
import os
import pickle

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
    class_scale = 'linear'
    def __init__(self,data_container, scale = None ,**params):
        '''

        :param data_container:
        :param summary_fo:
        :param longfile_fo:
        :param params:
        :return:
        '''
        self.data_container = data_container
        self.params = params
        self.scale = scale if scale else self.class_scale
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
        self.window_end_vect = []
        self.filetype_vect = []
        self.channel_vect = []

        for window_container in self.data_container:
            seiz_str = window_container['fold_lab']
            window_end = window_container['time']
            time_chunk = int(seiz_str[-1])
            self.obtain_feature(window_container,time_chunk,window_end)

        X,Y,T,end_times,filetypes,channels = map(np.asarray,(self.feature_vect,self.label_vect,
                           self.time_chunk_vect,self.window_end_vect,self.filetype_vect,
                                                    self.channel_vect))
        return X,Y,T,end_times,filetypes,channels



    def obtain_feature(self,window_container,time_chunk,window_end):
        raise IncompleteClassError('Calling Features')

    def call_func(self,data,window_container,label,time_chunk,window_end):
        channel_feats = self.feature_func(data,**self.params)
        if self.scale == 'logarithmic':
            channel_feats = np.log(np.array(channel_feats) + 1e-7)
        self.feature_vect.append(channel_feats)
        self.label_vect.append(label)
        self.time_chunk_vect.append(time_chunk)
        self.window_end_vect.append(window_end)
        self.filetype_vect.append(window_container['fold_lab'][0])


class SingleFeatureDriver(FeatureDriver):
    def obtain_feature(self,window_container,time_chunk,window_end):
        '''
        From a window, get all features and return their observations
        :param window:
        :param label:
        :param time_chunk:
        :return:
        '''
        window,label = window_container['window'],\
                       window_container['label'],
        window = window.T
        for channel_no,channel in enumerate(window):
            try:
                self.call_func(channel,window_container,label,time_chunk,window_end)
                self.channel_vect.append(channel_no)
            except TypeError as e:
                raise IncompleteClassError('No Function selected')

class MultFeatureDriver(FeatureDriver):
    def obtain_feature(self,window_container,time_chunk,window_end):
        window,label = window_container['window'],\
                       window_container['label']
        try:
            self.channel_vect.append(0)
            self.call_func(window,window_container,label,time_chunk,window_end)
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

# class ScatCoeffs(SingleFeatureDriver):
#     feature = 'Scattering Coefficients'
#     feature_func = staticmethod(flat_scat_coeffs)


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

    decision_rule_params_class = {'pos_classes':('Preictal','Ictal'),'n_windows':5,'flag_tol':.35,'holdoff_per':180. }


    FeatureClass_class = FeatureDriver

    def __init__(self, save_path, window_len=None,
                 preictal_time=None, postictal_time=None, n_windows=None,
                 window_overlap=None,
                 variable_params=None,
                 FeatureClass=None,
                 classification_names = None,
                 testmethod ='classifier',
                 sliding_window = False,
                 decision_rule_params = None,
                 **params):
        self.save_path = save_path
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

        self.decision_rule_params = decision_rule_params\
            if decision_rule_params is not None else self.decision_rule_params_class

        self.testing_method,self.sliding_window = testmethod, sliding_window

    def collect_data(self,filenames,seizure_times,good_channels=None,bad_channels = None ):
        data_container = \
            classif_data_collect(filenames,seizure_times,
                                 self.window_len,self.preictal_time,self.postictal_time,
                                 self.n_windows,self.window_overlap,
                                 good_channels=good_channels, bad_channels=bad_channels)

        self.data_container = data_container['data']
        self.seize_times = data_container['seize_times']

    def set_data(self,data_container):
        self.data_container = data_container['data']
        self.seize_times = data_container['seize_times']

    def write_to_files(self,summary_obj,longfile_obj,decision_obj):
        '''
         Write out to files given input features.
        :param summary_obj: file
        The file containing the summary
        :param longfile_obj: file
        The file containing all information output
        :return:
        '''
        self.decision_obj = decision_obj
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
        self.feat_str = feat_implement.feature
        X,Y,T,end_times,filetypes,channel_nos= feat_implement.get_feature_list()

        if self.testing_method =='classifier':
            # write out feature type and param type to both files
            self._write_to_file(summary_obj,self.feat_str,variable_param_dict)
            self._write_to_file(longfile_obj,self.feat_str,variable_param_dict)
            self.ml_info = ML_test(X,Y,T,summary_obj,longfile_obj, self.classification_names)
            self._visualize_and_save_seiz(X,Y, T, end_times, filetypes, channel_nos, self.feat_str, variable_param_dict)
        elif self.testing_method == 'anomaly':
            X_inter,X_seize,Y =  self._parse_feat_list_for_anomaly(X,Y,T)

            anom_test(X_inter,X_seize,Y,self.feat_str,self.save_path)



        else:
            raise IncompleteClassError(
                    'No testing method called \'{}\' found'.format(self.testing_method))


        return



    def _visualize_and_save_seiz(self, feats, labels, folds, end_times, filetypes,
                                 channel_nos, feat_str, variable_params):
        decision_rule_params = self.decision_rule_params.copy()
        holdoff_per = decision_rule_params.pop('holdoff_per')
        decision_stats = {}

        for classifier_key,seiz_info in self.ml_info.items():
            decision_stats[classifier_key] = {}
            for fold, model in enumerate(seiz_info['models']):



                fold_features = feats[(folds == fold)]
                fold_filetypes = filetypes[(folds == fold)]
                fold_endtimes = end_times[(folds==fold) ]
                fold_channels = channel_nos[(folds == fold)]

                pred_labels = model.predict(fold_features)
                pred_classes,plot_pred_labels = np.unique(pred_labels,return_inverse=True)

                #used for snazzy y-axis labels
                locator = ticker.FixedLocator(np.arange(len(pred_classes)))
                formatter = ticker.FixedFormatter(pred_classes)


                seiz_preds = plot_pred_labels[fold_filetypes=='S']
                #seiz_non_plot_preds = pred_labels[fold_filetypes=='S']
                seiz_channels = channel_nos[fold_filetypes=='S']
                seiz_endtimes  = fold_endtimes[fold_filetypes=='S']


                num_seiz_chans = np.max(seiz_channels) + 1
                seiz_preds,_ = mode(np.reshape(seiz_preds,(-1,num_seiz_chans)),axis=1)
                seiz_preds = seiz_preds.flatten()
                seiz_non_plot_preds = pred_classes[seiz_preds]
                seiz_endtimes = np.reshape(seiz_endtimes,(-1,num_seiz_chans))[:,0]


                fig = plt.figure(0,figsize=(10,6))
                fig.clf()


                #plot the predictions
                ax = fig.add_subplot(2,1,1)
                plt.plot(seiz_endtimes,seiz_preds)
                adj_str = 'Adjusted Params {}, '.format(list(variable_params.items())) if variable_params else ''
                seiz_str = ('Feature: {},' + adj_str+'Classifier: {}, Seizure {}').format(
                                                           feat_str,classifier_key,fold+1)
                plt.title(seiz_str)
                plt.ylabel('Classification')

                #snazzy labels continued
                ax.yaxis.set_major_locator(locator)
                ax.yaxis.set_major_formatter(formatter)

                #plot seizure lines
                plt.vlines(self.seize_times[fold],min(seiz_preds),max(seiz_preds),'g',linestyles='dashed')

                #compute decision rules for seizure, plot them
                fig.add_subplot(2,1,2)
                seiz_decision,prop_windows = decision_rule(seiz_non_plot_preds,**decision_rule_params)

                seiz_stats = diagnose_seiz_file(
                    seiz_endtimes,seiz_decision,holdoff_per,self.seize_times[fold]
                                                  )
                update_decision_stats(decision_stats[classifier_key],seiz_stats)

                plt.plot(seiz_endtimes,prop_windows,'g')
                plt.hlines(self.decision_rule_params['flag_tol'],min(seiz_endtimes),max(seiz_endtimes),'r',linestyles='dashed')
                plt.title('Post-processing outputs')
                plt.xlabel('Time (s)')
                plt.ylabel('Fraction deemed ictal')

                plt.vlines(self.seize_times[fold],0,1,'g',linestyles='dashed')


                plt.savefig(os.path.join(self.save_path,seiz_str+ '.png'))

                #compute the predictions/decisions for the interictal period

                inter_preds = plot_pred_labels[fold_filetypes=='N']
                #inter_non_plot_preds = pred_labels[fold_filetypes=='N']
                inter_channels = channel_nos[fold_filetypes=='N']
                inter_endtimes  = fold_endtimes[fold_filetypes=='N']

                num_inter_chans = np.max(inter_channels) + 1

                inter_preds,_ = mode(np.reshape(inter_preds,(-1,num_inter_chans)),axis=1)
                inter_preds = inter_preds.flatten()
                inter_non_plot_preds = pred_classes[inter_preds]
                inter_endtimes = np.reshape(inter_endtimes,(-1,num_inter_chans))[:,0]

                #plot the interictal decision rules and classes over time
                fig = plt.figure(2,figsize=(10,6))
                fig.clf()

                ax = fig.add_subplot(2,1,1)
                plt.plot(inter_endtimes,inter_preds)

                inter_str = ('Feature: {},' + adj_str+'Classifier: {}, Interictal {}').format(
                                                           feat_str,classifier_key,fold+1)
                plt.title(inter_str)
                ax.yaxis.set_major_locator(locator)
                ax.yaxis.set_major_formatter(formatter)
                plt.ylabel('Classification')


                fig.add_subplot(2,1,2)

                #compute the decisions, statistics
                inter_decision,prop_windows = decision_rule(inter_non_plot_preds,**decision_rule_params)
                inter_stats = check_for_fp(inter_endtimes,inter_decision,holdoff_per)
                update_decision_stats(decision_stats[classifier_key],inter_stats)


                plt.plot(inter_endtimes,prop_windows,'g')
                plt.hlines(self.decision_rule_params['flag_tol'],min(inter_endtimes),max(inter_endtimes),'r',linestyles='dashed')
                plt.title('Post-processing outputs')
                plt.xlabel('Time (s)')
                plt.ylabel('Fraction deemed ictal')



                plt.savefig(os.path.join(self.save_path,inter_str+ '.png'))

        self._summarize_decision_results(self.decision_obj, decision_stats,variable_params)


    def _summarize_decision_results(self,decision_obj,decision_results,variable_params):
        self._write_to_file(decision_obj,self.feat_str,variable_params)
        for classif_key,result_set in decision_results.items():


            decision_obj.write('=========================================\n Classifier: {}\n'.format(classif_key))

            FP = sum(result_set['false_pos'])
            TP = sum(result_set['found_seiz'])
            sens = float(TP)/float(len(result_set['found_seiz'])) if TP else 0
            decision_obj.write('Sensitivity: {}%\n'.format(sens*100))

            FPH = FP/float(sum(result_set['inter_time']))*3600.
            decision_obj.write('FPH: {}, Ind. False pos.: {}\n'.format(FPH,result_set['false_pos'])
                               )
            mean_latency = np.nanmean(result_set['seiz_latency'])
            decision_obj.write('mean_latency: {}, All_latency: {}\n\n'.format(mean_latency,result_set['seiz_latency']))




    @staticmethod
    def _parse_feat_list_for_anomaly(X,Y,T):
        num_folds = np.max(T)
        X_inter =[]
        X_seize, labels = [[] for fold in range(num_folds)],[[] for fold in range(num_folds)]
        for feat,label,fold_num in zip(X,Y,T):

            if label =='Interictal':
                X_inter.append(feat)
            else:
                X_seize[fold_num-1].append(feat)
                labels[fold_num-1].append(label)
        X_seize = [np.asarray(mat) for mat in X_seize]
        X_inter = np.asarray(X_inter)
        return X_inter,X_seize,labels


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
                    'scale':'logarithmic',
                    'pred_window_size': 1000}

class VARPredictErrTester(ARPredictErrTester):
    FeatureClass_class = VARPredictErr
    params_class = {'order':30,'scale':'logarithmic'} #MVAR value
#
# class ScatCoeffsTester(FeatureTester):
#     FeatureClass_class = ScatCoeffs
#     params_class = {'mat_eng':matlab.engine.start_matlab()}
#     variable_params_class = {'T':[128,512],'Q':[8,4,2]}


def run_pipeline_on_patient(patient_data_path,testclass_list,summary_filename,longform_filename,decision_filename,
                            pickle_name = 'patient_pickle.txt',preictal_time = 30,postictal_time = 50,
                            norm_window = False, testmethod = 'classifier'):
    '''
    This function will take in the
    :param patient_data_path:
    The folder where the patient data (and pickle file) is. The results will be
    stored in a "results" folder in
    :param testclass_list:
    A list of the testing classes to be used in
    :param summary_filename:
    :param longform_filename:
    :return:
    '''
    rstat_bands= np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])
    file_save_path = os.path.join(patient_data_path,'Results')
    pickle_path = os.path.join(patient_data_path,pickle_name)
    with open(pickle_path,'r') as pickle_file:
        patient_info = pickle.load(pickle_file)

    seiz_filenames = [os.path.join(patient_data_path,filename) for
                      filename in patient_info['seizure_data_filenames']]

    inter_filenames = [os.path.join(patient_data_path,filename)
                       for filename in patient_info['awake_inter_filenames']]

    seizure_times = patient_info['seizure_times']

    try:
        bad_channels = patient_info['bad_channels']
    except KeyError:
        bad_channels = ('Events/Markers','EDF Annotations')


    try:
        good_channels = patient_info['good_channels']
    except KeyError:
        good_channels = None

    if not DEBUG:
        # data container for all information except for eigenvector centrality
        data_container1 = classif_data_collect(seiz_filenames,seizure_times,inter_filenames,window_len=5000,
                        preictal_time=preictal_time,postictal_time=postictal_time,n_windows=6,
                                           window_overlap=.5, good_channels=good_channels, bad_channels=bad_channels,
                                          rstat_bands=None,sliding_window=True)

        #eigenvector centrality needs rstat bands
        data_container2 = classif_data_collect(seiz_filenames,seizure_times,inter_filenames,window_len=5000,
                         preictal_time=preictal_time,postictal_time=postictal_time,n_windows=6,
                                           window_overlap=.5, good_channels = good_channels,
                                           bad_channels=('Events/Markers','EDF Annotations'),
                                           rstat_bands=rstat_bands,sliding_window=True)

    else:
        with open('data1_pickle.txt') as data1,open('data2_pickle.txt') as data2:
            data_container1 = pickle.load(data1)
            #data_container2 = pickle.load(data1)

    summary_file = os.path.join(file_save_path,summary_filename)
    longform_file = os.path.join(file_save_path,longform_filename)
    decision_file = os.path.join(file_save_path,decision_filename)
    with open(summary_file,'w') as summary_obj,\
        open(longform_file,'w') as longform_obj,\
        open(decision_file,'w') as decision_obj:
        for testclass in testclass_list:
            print('Feature: {}'.format(testclass.FeatureClass_class.feature))
            # if our feature is AR, get the optimal order to run it
            # with and test the feature with it

            feat_tester = testclass(file_save_path, testmethod = testmethod)

            if testclass is EigCentralityTester:
                feat_tester.set_data(data_container2)
            else:
                feat_tester.set_data(data_container1)
            feat_tester.write_to_files(summary_obj,longform_obj,decision_obj)







    #TODO: implement final decision rule, get metrics there


if __name__ == '__main__':
    # # Example code to see how this works. Data taken from patient TS0041
    #
    # summary_file = 'ML_ summary_Centrality.txt'
    # longform_file = 'ML_log_Centrality.txt'
    # python setup.py config --compiler=intelem --fcompiler=intelem build_clib --compiler=intelem --fcompiler=intelem build_ext --compiler=intelem --fcompiler=intelem install --user --record files.txt

    # file_dir = '/Users/vsp/Google Drive/MATLAB/Scattering Coeffs' # 42
    # #file_dir = '/home/chris/Documents/Rice/senior/EpilepsyVIP/data/RMPt2' # Chris
    #
    # data_filenames = ('DA00101U_1-1+.edf',
    #               'DA00101V_1-1+.edf',
    #              # 'DA00101W_1-1+.edf',
    #                'CA00100E_1-1_03oct2010_01_03_05_Awake+.edf',
    #               'CA00100F_1-1_03oct2010_13_05_13_Awake+.edf',
    #              # 'DA00101P_1-1_02oct2010_09_00_38_Awake+.edf'
    #               ) # 42
    # # data_filenames = ('DA00101U_1-1+.edf',
    # #              'DA00101V_1-1+.edf',
    # #              'DA00101W_1-1+.edf',
    # #              'CA00100E_1-1+.edf',
    # #              'CA00100F_1-1+.edf',
    # #              'DA00101P_1-1+.edf'
    # #              ) # Chris
    #
    # seizure_times = ((262,330),
    #                  (107,287),
    #                  #(191,405),
    #                  None,None,
    #                 #None
    #                 )
    # data_filenames = [os.path.join(file_dir,filename) for filename in data_filenames]
    #
    # good_channels = [
    #                 'LAH1','LAH2','LPH6','LPH7',
    #                 'LPH9','LPH10','LPH11','LPH12'
    #                 ]
    #
    # rstat_bands= np.array([[1,4],[5,8],[9,13],[14,25],[25,90],[100,200]])
    #
    #
    # data_container1 = classif_data_collect(data_filenames,seizure_times,window_len=5000,
    #                      preictal_time=30,postictal_time=50,n_windows=6,window_overlap=.5, good_channels=good_channels,
    #                                       rstat_bands=None,sliding_window=True)
    # #COMMENTED OUT FOR DEBUGGING
    # #data_container2 = classif_data_collect(data_filenames,seizure_times,window_len=5000,
    # #                      preictal_time=30,postictal_time=50,n_windows=6,window_overlap=.5,
    # #                                        bad_channels=('Events/Markers','EDF Annotations'),
    # #                                        rstat_bands=rstat_bands,sliding_window=True)
    # print('Data read in')
    #
    #
    #
    # #This block of code was used to determine the ideal order for the data.
    # # for ind,window_cont in enumerate(data_container1):
    # #     if window_cont['label'] == 'Interictal':
    # #         first_interictal_ind= ind
    # #         break
    # #
    # #
    # # interictal_test_window,_,_ = data_container1[first_interictal_ind]
    # #
    # #mvar_order = var_order_sel(interictal_test_window,maxorder=40)
    # #ar_order = ar_order_sel(interictal_test_window,maxorder=300)
    # # print('Ideal MVAR order: {} \t '
    # #       #'Ideal AR order{}'
    # #       ''.format(mvar_order,#ar_order
    # #                 ))
    #
    #
    # testmethod= 'classifier'#anomaly
    #
    # testclass_list = [ARCoeffsTester,ARNonlinearTester,
    #                 #ARPredictErrTester,
    #                 #ARStabilityTester,VARPredictErrTester,
    #                 #EigCentralityTester
    #                 ]
    # with open(summary_file,'w') as summary_obj,\
    #     open(longform_file,'w') as longform_obj:
    #     for testclass in testclass_list:
    #         print('Feature: {}'.format(testclass.FeatureClass_class.feature))
    #         # if our feature is AR, get the optimal order to run it
    #         # with and test the feature with it
    #         if testclass in [ARStabilityTester, #mvar
    #                          VARPredictErrTester]:
    #             feat_tester = testclass(testing_method = testmethod)
    #
    #         elif testclass in [ARCoeffsTester,ARNonlinearTester,
    #                          ARPredictErrTester]:#ar TODO: fix ar_order
    #             feat_tester = testclass(testing_method = testmethod)
    #
    #         else:   #not AR
    #             feat_tester = testclass()
    #
    #         if testclass is EigCentralityTester:
    #             feat_tester.set_data(data_container2)
    #         else:
    #             feat_tester.set_data(data_container1)
    #         feat_tester.write_to_files(summary_obj,longform_obj)
    #

    # Example code to see how this works. Data taken from patient TS0041
    data_folder = '../../data/'
    data_path = os.path.join(os.getcwd(),data_folder)
    patient_folders = os.listdir(data_path)

    testmethod= 'classifier'#'anomaly'

    testclass_list = [VARPredictErrTester,ARStabilityTester,
                      ARCoeffsTester,ARPredictErrTester,ARNonlinearTester,

                      EigCentralityTester,
                    #ScatCoeffsTester
                    ]
    scanned = False
    DEBUG = False
    patients = [
        'TS041'
        #'TS039',
        ]
    for patient in patient_folders:
        scanned = True
        if patient not in patients:
            continue
        patient_dir = os.path.join(data_path,patient)
        #TODO: handle data_container parameter changes in saves
        summary_filename = 'ML_summary_Centrality.txt'
        longform_filename = 'ML_log_Centrality.txt'
        decision_filename = 'Decision_Results.txt'
        print('Begin {}'.format(patient))
        run_pipeline_on_patient(patient_dir,testclass_list,summary_filename,longform_filename,decision_filename,
                                testmethod=testmethod)


    else:
        if not scanned:
            raise(ValueError('No patients were scanned.'))


