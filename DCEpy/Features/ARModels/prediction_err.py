from __future__ import print_function
import numpy as np
from statsmodels.tsa.vector_ar import var_model
from statsmodels.tsa import ar_model


def mvar_pred_err(data,pred_window_size =1000 , order=30):
    '''
    Fits a multivatiate autoregressive model to data
    :param data:
    :param pred_window_size:
    :param order:
    :return:
    '''
    model_var = var_model.VAR(data[:-pred_window_size,:])
    model_results = model_var.fit(maxlags=order)
    pred_data = np.asarray(model_results.forecast(
        data[-pred_window_size-order:-pred_window_size],pred_window_size))


    fun_energy = function_energy(data[-pred_window_size:,:])
    err_energy = function_energy(data[-pred_window_size:,:]-pred_data)

    return np.append(fun_energy,err_energy)

def ar_pred_err(data,pred_window_size=1000,order=30):
    '''

    :param data:
    :param pred_window_size:
    :param order:
    :return:
    '''
    model_ar = ar_model.AR(data[:-pred_window_size])
    model_results = model_ar.fit(maxlag=order)
    pred_data = model_ar.predict(model_results.params,
                                 start=np.size(data,axis=0)-pred_window_size-1,
                                 end = np.size(data,axis=0)-1)
    fun_energy = function_energy(data[-pred_window_size-1:])
    err_energy = function_energy(data[-pred_window_size-1:]-pred_data)

    return fun_energy,err_energy


def function_energy(data):
    '''
    compute the naive L2 norm of a function.
    :param fxn: numpy array
    :return:
    '''
    return np.sum(np.power(data,2),axis=0)
