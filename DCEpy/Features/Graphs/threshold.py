__author__ = 'vsp'


def graph_thresh(G,conn_method = None, **thresh_params):
    '''
    Given a graph G, use a thresholding method to eliminate edges that do not satisfy the threshholding criteria.

    For instance, to threshold phase lag index graph G with a threshold of .3, type:
    graph_thresh(G,'pli',thresh_val= .3)

    :param G: The graph with weighted edges to threshold
    :param conn_method: A string describing the connection method
    :param thresh_params: This means extra keyword arguments.
    Check the thresholding function associated with your method to see what these must be.
    If your function uses default_threshold, the ONLY keyword argument is thresh_val.
    :return: Returns a thresholded graph
    '''

    if not conn_method:
        conn_method = 'weight'
        try:
            thresh_val = thresh_params['thresh_val']
        except KeyError:
            raise ValueError('No connection parameter specified and no threshold value given. \nRemember to '
                             'set keyword argument "thresh_val" to the desired threshold value.')


    thresh_func = thresh_func_dict[conn_method]

    for graph_edge in G.edges(data=True):
        # check to see that the thresholding function "does not like" that edge
        try:
            if not thresh_func(graph_edge[2][conn_method],**thresh_params):
                G.remove_edge(*graph_edge[:2])
        except KeyError:
            raise AttributeError('The graph edge does not have a weight associated with the connection_method!')

def bad_vect_thresh(vect_val,thresh_val):
    '''
    Performs a bad thresholding measure on a vector.
    :param vect_val: a numpy vector (or list of numbers)
    :param thresh_val: the value to be compared against
    :return:
    '''
    return abs(vect_val[0]) >= abs(thresh_val)

def default_threshold(weight_val, thresh_val):
    '''
    Performs a basic single-valued threshold on two values.
    Returns True (keep edge) if the weighted value is greater than the threshold value
    :param weight_val:
    :param thresh_val:
    :return:
    '''
    return weight_val>=thresh_val

def abs_threshold(weight_val, thresh_val):
    '''
    Performs a basic single-valued threshold on two values.
    Returns True (keep edge) if the weighted value is greater in magnitude than the threshold value
    :param weight_val:
    :param thresh_val:
    :return:
    '''
    return abs(weight_val)>=abs(thresh_val)



thresh_func_dict = {'weight':default_threshold, #when called without connection type
                    'pli':abs_threshold, #phase lag index
                    'imag_coherency': default_threshold, #imaginary coherency
                    'coherence': default_threshold, # coherence
                    '':bad_vect_thresh,
                    '':bad_vect_thresh,}