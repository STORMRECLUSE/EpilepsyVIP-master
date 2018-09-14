__author__ = 'Chris'

import numpy as np
import pdc_dtf as mv
from statsmodels.tsa.vector_ar import var_model

# create artificial data
N = 5
n = 60
p = 4
burn = 20

# the VAR coefficients are uniform random between -2^(-k) and 2^(k) for
# each order 1,2,...p ==> This should be a fine model, nothing crazy (i.e. unstability) here
A = np.zeros((p, N, N))
for i in range(p):
    A[i] = (2**-i) * ( np.random.rand(N,N) - 0.5 * np.ones((N, N)) )
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

# centered with unit variance
X_mean = np.reshape(np.mean(X,axis=1), (N,1))
X_std = np.reshape(np.std(X,axis=1), (N,1))
X = (X - X_mean) / X_std

# print stuff to make sure I'm not going crazy
print "Channel Means: " + str(np.mean(X, axis=1))
print "Channel Standard Deviation: " + str(np.std(X, axis=1))
print "Data Dimension: " + str(X.shape)

# try pdc_dtf.py code
A_bar, sigma_bar = mv.mvar_fit(X, p)
est_total = sum(np.array([np.linalg.norm(A_bar[i],ord='fro') for i in range(p)]))
true_total = sum(np.array([np.linalg.norm(A[i],ord='fro') for i in range(p)]))
for i in range(p):
    est_power = np.linalg.norm(A_bar[i],ord='fro') / est_total
    true_power = np.linalg.norm(A[i],ord='fro') / true_total
    print "For i=" + str(i) +", the estimated regression coefficients (with norm " + str(est_power) + " | " + str(true_power) + ") : "
    print str(A_bar[i])
print "Noise variances: "
print str(sigma_bar)

# # create known VAR process with parameters A and sigma -- useful later?
# known = var_model.VARProcess(coefs=A, intercept=np.zeros(N), sigma_u=sigma)
# known.plotsim()
# print "Simulation successful"

# create the VAR with data X
model = var_model.VAR(X.T)
print "Model Successfully created"

# fit the model -- trying a method
results = model.fit(ic='aic', maxlags=10, trend="nc", verbose=True)
results.summary()


# for i in range(p):
#     print "For i=" + str(i) +", the difference of estimated and true regression coefficients: "
#     print str(A[i]-A_bar[i])
# print "Noise variances: "
# print str(sigma_bar)