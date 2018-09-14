from __future__ import print_function

import numpy as np
from scipy import linalg
from statsmodels.tsa.vector_ar import var_model

def ar_stability(data, order, n_eigs):
    '''
    Builds an vector autoregressive (VAR) model of iEEG data,
    constructs a state matrix, and returns first n_eigs eigenvalues
    of this matrix.
    :param data:
    :param order:
    :param n_eigs:
    :return:
    '''
    n, N = data.shape # N : number of channels, n: number of observations
    model = var_model.VAR(data)
    mvar_fit = model.fit(order)
    A = mvar_fit.coefs

    # build the state matrix
    top = np.concatenate(A,axis=1)
    bottom = np.eye(N*(order-1), N*order)
    state_mat = np.concatenate(np.array([top,bottom]), axis=0)

    # compute the top n_eigs largest eigenvalues
    eigs = linalg.eig(state_mat,right=False, overwrite_a=True)
    abs_eigs = np.abs(eigs)
    abs_eigs =np.sort(abs_eigs)
    return abs_eigs[-n_eigs:]

def ar_stability_window(data, order, n_eigs, w_len, w_gap):
    """
    Builds an vector autoregressive (VAR) model of iEEG data, 
    constructs a state matrix, and returns first n_eigs eigenvalues
    of this matrix. 

    Parameters
    ----------
    data : ndarray, shape (n, N)
        iEEG data with N channels and n samples 
    order: int
        VAR model order 
    n_eigs: int
        number of eigenvalues to return 
    w_len: int 
        length of the time window (in indices)
    w_gap: int 
        length between time windows (in indices)

    Returns
    -------
    eig_seq: ndarray, shape(w, n_eigs)
        sequence of top eigenvalues of the state matrix

    """
    n, N = data.shape # N : number of channels, n: number of observations
    
    # get beginnings of windows and initialize eigenvalue sequence
    w_end = np.arange(start=w_len, stop=n, step=w_gap)
    eig_seq = np.empty((len(w_end),n_eigs), dtype=float)

    for index,i in enumerate(w_end):

        # get window and fit AR model
        window = data[i-w_len:i,]
        eig_seq[index,:] = ar_stability(window,order,n_eigs)

    return eig_seq,w_end