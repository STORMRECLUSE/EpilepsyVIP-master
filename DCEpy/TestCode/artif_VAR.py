__author__ = 'Chris'
import numpy as np

def artif_VAR_data(N, n, p, burn, A_type):
    """
    Creates artificial data from one of two VAR models

    Parameters
    ----------
    N: int
        inumber of variables in VAR model
    n: int
        number of samples / observations
    p: int
        order of the VAR model
    burn: int
        number of initial data points to throw away
    A_type: str 
        "rand" for random VAR model (not helpful so far)
        "tridiag" for a tridiagonal model 

    Returns
    -------
    X: ndarray, shape(n,N)
        artifical VAR time series
    A: ndarray, shape(p,N,N)
        VAR coefficients 
    """
    if A_type == "rand":
        A = np.zeros((p, N, N))
        for i in range(p):
            A[i] = (2**-i) * ( np.random.rand(N,N) - 0.5 * np.ones((N, N)) )
        sigma = 0.1*np.ones(N)
    if A_type == "tridiag":
        A = np.zeros((p, N, N))
        for i in range(p):
            A[i] = (2**-i) * ( np.diag(np.ones(N),k=0) - 0.5*np.diag(np.ones(N-(p+1)),k=(1+p)) - 0.5*np.diag(np.ones(N-(p+1)),k=-(1+p)) )
        sigma = 0.1*np.ones(N)  

    # the artifical data X --> start from a random vector and push it through
    X = np.zeros((N,n+burn))
    noise = np.empty((N,1))
    X[:,0] = np.random.rand(N) - 0.5
    for i in range(1,n+burn):
        if i < p:
            A_cat = np.concatenate(A[0:i],axis=1)
            X_cat = X[:,0:i].T.reshape((i*N,1))
        else:
            A_cat = np.concatenate(A, axis=1)
            X_cat = X[:,(i-p):i].T.reshape((p*N,1))

        for j in range(N):
            noise[j] = np.random.normal(0, sigma[j])

        res = np.dot(A_cat, X_cat) + noise
        X[:,i] = res.reshape(N)

    # throw away the burn entries
    X = X[:,burn:]

    return X.T, A

