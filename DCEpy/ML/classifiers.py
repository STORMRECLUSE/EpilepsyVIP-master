from __future__ import print_function
__author__ = 'Chris'

import numpy as np
from sklearn import linear_model, svm, naive_bayes, discriminant_analysis, ensemble, metrics
from DCEpy.Features.Preprocessing.preprocess import normalize


def ML_test(X, Y, T, f_sum, f_log, class_keys):
    """
    Given feature data matrix X with categorical response vector Y,
    and folds T, this function computes the effectiveness of given
    machine learning classification algorithms and records results
    in provided summmary and log files.

    Parameters
    ----------
    X: ndarray, shape(n,p)
        data array of n observations, p features
    Y: ndarray, shape(n)
        array of n class labels
    T: ndarray, shape(n)
        fold assignments
    class_keys: list of strings
        classification algorithms to test
    f_sum: file object
        summary file (shorter)
    f_log: file object
        log file (longer)

    Returns
    -------
    ml_info: dict of dicts
        Info about different machine learning classification methods
        keys are classification methods inner dict keys are
            'conf_mat' --> list of confusion matrices (ndarrays)
            'avg_mis' --> average misclassification rate (float)
            'avg_pre_tp' --> average preictal true positive rate (float)
            'avg_pre_fp' --> average preictal false positive rate (float)
    """

    # TO DO:
    # (1) Add writing to log and summary files
    # (2) Write test function using zip code digit data

    # check that classification keys are valid
    valid_classes = ['ridge', 'svm_lin', 'svm_poly', 'svm_rad', 'nb', 'lda', 'qda', 'rf']
    for key in class_keys:
        if key not in valid_classes:
            raise AttributeError (key + ' is not recognized')

    # get data dimensions
    n, p  = X.shape
    k = np.max(T) + 1

    # get label stuff
    set_labels = set(Y)
    set_labels.remove('None of the Above')
    mat_labels = list(set_labels)
    mat_labels.sort()
    preict = mat_labels.index('Preictal')
    nc = len(mat_labels)
    log_col_space = max(5, max([len(s) for s in mat_labels])+3) # spacings for columns
    sum_col_space = max(5, max([len(s) for s in class_keys])+3)
    sum_row_label = ['Avg MissClass', 'Avg Pre TP', 'Avg Pre FP']
    sum_row_space = max([len(s) for s in sum_row_label])+3

    # create machine learning info dictionary
    ml_info = {}
    for key in class_keys:
        ml_info[key] = {'conf_mat':[], 'avg_mis': np.nan, 'avg_pre_tp': np.nan, 'avg_pre_fp': np.nan,
                        'preds':[],'models':[]}

    # for each fold
    for i in range(k):

        # get fold indices
        test_ind, = np.where((T==i) & (Y!='None of the Above'))
        train_ind, = np.where((T!=i) & (Y!='None of the Above'))

        # create folds
        X_train = X[train_ind,:]
        Y_train = Y[train_ind]
        X_test = X[test_ind,:]
        Y_test = Y[test_ind]

        # run classification tests
        preds,c_mat,models = class_tests(X_train, Y_train, X_test, Y_test, class_keys,mat_labels)

        # store results
        for key in class_keys:
            ml_info[key]['conf_mat'].append(c_mat[key])
            #store the predictions in the information
            ml_info[key]['preds'].append(preds[key])

            ml_info[key]['models'].append(models[key])
    # for each classification method
    for key in class_keys:

        # initialize averages
        avg_mis = 0
        avg_ptp = 0
        avg_pfp = 0

        # for each fold
        for i in range(k):

            # get confusion matrix
            c_mat = ml_info[key]['conf_mat'][i]

            # print confusion matrix to log file
            print(key + ' -- ' + str(i+1) + '\n' + '='*log_col_space*(len(mat_labels)+1) + '\n', file=f_log)
            print(''.join(' '*log_col_space + ('{0:<'+str(log_col_space)+'}').format(s) for s in mat_labels), file=f_log)
            for row_label, row in zip(mat_labels, c_mat):
                print('%s [%s]' % (('{0:<'+str(log_col_space)+'}').format(row_label), ''.join(('{0:<'+str(log_col_space)+'}').format('%2.1f' % i) for i in row)), file=f_log)
            print("\n", file=f_log)

            # update averages
            total = np.sum(np.reshape(c_mat, np.prod(c_mat.shape)))
            preictal = np.sum(c_mat,axis=1)[preict]
            not_preict = total - preictal
            correct_preict = c_mat[preict,preict]
            incorrect_preict = np.sum(c_mat,axis=0)[preict] - correct_preict

            avg_mis = avg_mis + 1.0 - np.trace(c_mat) / total
            avg_ptp = avg_ptp + correct_preict / preictal
            avg_pfp = avg_pfp + incorrect_preict / not_preict

        # store averages
        ml_info[key]['avg_mis'] = avg_mis / float(k)
        ml_info[key]['avg_pre_tp'] = avg_ptp / float(k)
        ml_info[key]['avg_pre_fp'] = avg_pfp / float(k)

    # print table to summary file

    # header
    print(' '*sum_row_space + ''.join(('{0:<'+str(sum_col_space)+'}').format(s) for s in class_keys), file=f_sum)

    # print data to summary file
    print_data = []
    print_data.append([ml_info[key]['avg_mis'] for key in class_keys])
    print_data.append([ml_info[key]['avg_pre_tp'] for key in class_keys])
    print_data.append([ml_info[key]['avg_pre_fp'] for key in class_keys])
    for i in range(len(print_data)):
        print('%s [%s]' % ( ('{0:<'+str(sum_row_space)+'}').format(sum_row_label[i]),
            ''.join(('{0:<'+str(sum_col_space)+'}').format('%2.1f' % (100*d)) for d in print_data[i])),
              file=f_sum)
    print("\n", file=f_sum)

    return ml_info


def class_tests(X_train, Y_train, X_test, Y_test, class_keys,mat_labels):
    """
    Returns the range of data for use in gap statistic

    Parameters
    ----------
    X_train: ndarray, shape(n_train,p_train)
        data array of n observations, p features
    Y_train: ndarray, shape(n_train)
        data array of n class labels
    X_test: ndarray, shape(n_test,p_test)
        testing data of n_test observations, p_test features
    Y_test: ndarray, shape(n_test)
        data array of n class labels

    Returns
    -------
    conf_mat: dict
        confusion matrices for different learning methods

    preds: dict
        predicted class labels for different learning methods

    """

    # get data dimensions
    n_train, p_train = X_train.shape
    n_test, p_test = X_test.shape

    # normalize data
    mean = np.mean(X_train,axis=0)
    sd = np.std(X_train,axis=0)
    sd[np.where(sd==0)] = 1 # handling divide by zero errors
    X_train_norm = (X_train - mean) / sd
    X_test_norm = (X_test - mean) / sd

    conf_mat = {}
    preds = {}
    models = {}


    # (1) Ridge Regression
    if 'ridge' in class_keys:
        alphas = np.logspace(start=1e-5, stop=1e2, num=30)
        if n_train >= 10:
            cv = 10
        else:
            cv = None
        ridge_model = linear_model.RidgeClassifierCV(alphas=alphas, cv=cv)
        ridge_model.fit(X_train, Y_train)
        preds["ridge"] = ridge_model.predict(X_test)
        models["ridge"] = ridge_model
        conf_mat["ridge"] = metrics.confusion_matrix(Y_test, preds["ridge"],labels = mat_labels)

    # (2) SVM -- linear kernel
    if 'svm_lin' in class_keys:
        lin_svm = svm.SVC(kernel="linear")
        lin_svm.fit(X_train_norm,Y_train)
        preds["svm_lin"] = lin_svm.predict(X_test_norm)
        models["svm_lin"] = lin_svm
        conf_mat["svm_lin"] = metrics.confusion_matrix(Y_test, preds["svm_lin"],labels=mat_labels)

    # (3) SVM -- polynomial kernel
    if 'svm_poly' in class_keys:
        poly_svm = svm.SVC(kernel="poly")
        poly_svm.fit(X_train_norm,Y_train)
        preds["svm_poly"] = poly_svm.predict(X_test_norm)
        models ["svm_poly"] = poly_svm
        conf_mat["svm_poly"] = metrics.confusion_matrix(Y_test, preds["svm_poly"],labels=mat_labels)

    # (4) SVM -- gaussian kernel
    if 'svm_rad' in class_keys:
        rad_svm = svm.SVC(kernel="rbf")
        rad_svm.fit(X_train_norm,Y_train)
        preds["svm_rad"] = rad_svm.predict(X_test_norm)
        models["svm_rad"] = rad_svm
        conf_mat["svm_rad"] = metrics.confusion_matrix(Y_test, preds["svm_rad"],labels=mat_labels)

    # (5) Naive Bayes
    if 'nb' in class_keys:
        nb_model = naive_bayes.GaussianNB()
        nb_model.fit(X_train, Y_train)
        preds["nb"] = nb_model.predict(X_test)
        models["nb"] = nb_model
        conf_mat["nb"] = metrics.confusion_matrix(Y_test, preds["nb"],labels=mat_labels)

    # (6) LDA
    if 'lda' in class_keys:
        lda_model = discriminant_analysis.LinearDiscriminantAnalysis()
        lda_model.fit(X_train, Y_train)
        preds["lda"] = lda_model.predict(X_test)
        models["lda"] = lda_model
        conf_mat["lda"] = metrics.confusion_matrix(Y_test, preds["lda"],labels=mat_labels)

    # (7) QDA
    if 'qda' in class_keys:
        qda_model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        qda_model.fit(X_train, Y_train)
        preds["qda"] = qda_model.predict(X_test)
        models["qda"] = qda_model
        conf_mat["qda"] = metrics.confusion_matrix(Y_test, preds["qda"],labels=mat_labels)

    # (8) Random forests
    if 'rf' in class_keys:
        rf_model = ensemble.RandomForestClassifier()
        rf_model.fit(X_train,Y_train)
        preds["rf"] = rf_model.predict(X_test)
        models["rf"] = rf_model
        conf_mat["rf"] = metrics.confusion_matrix(Y_test, preds["rf"],labels=mat_labels)

    return preds, conf_mat, models
