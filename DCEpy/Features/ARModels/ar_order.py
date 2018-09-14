
import numpy as np
from scipy.stats import mode
from statsmodels.tsa.vector_ar import var_model
from statsmodels.tsa import ar_model

def var_order_sel(data, maxorder=5000):
    model = var_model.VAR(data)
    order = model.select_order(maxorder)
    return order['aic']

def ar_order_sel(data, maxorder=5000):
    n,p = data.shape
    orders = np.empty(p)

    # determine best order for each channel
    for i in range(p):
        model = ar_model.AR(data[:,i])
        orders[i] = model.select_order(maxorder,ic='aic')
    order_mode,_ = mode(orders)
    # return the maximum order (?)
    return order_mode