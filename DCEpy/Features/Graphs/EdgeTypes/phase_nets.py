import networkx as nx
import numpy as np
import scipy.fftpack as sig_proc

def phase_conn_network(data,connections,phase_type):
    '''
    Creates a graph based on some phase-related measure of connectivity between signals.
    :param data: Either a numpy array or a dict of numpy arrays
    :param connections:
    :param phase_type
    :return:
    '''

    data = np.transpose(data)

    if phase_type=='pli':
        edge_func = phase_lag_index
    elif phase_type == 'imag_coherency':
        edge_func = imag_coherency
    else:
        raise(ValueError('"{}" is not a supported phase-related connectivity measure.'.format(phase_type)))


    def compute_from_vertices(params):

        G.add_nodes_from(connections)

        for i in connections:
            for j in connections:
                if (not G.has_edge(i,j)) and (i!=j):
                    weight_dict[phase_type] = edge_func(data[i],data[j])
                    G.add_edge(i,j, **weight_dict)
        return G


    def compute_from_edges(params):


        for edge in connections:
            i, j = edge
            weight_dict[phase_type] = edge_func(data[i],data[j])
            G.add_edge(i,j, **weight_dict)
        return G

    params = {}

    #check one connection, to see if it is an edge or a vertex
    test_conn = next(iter(connections))

    weight_dict = {phase_type:0}
    compute_all_edges = True
    try:
        if np.floor(test_conn/1) != test_conn:
            raise ValueError('Noninteger element found in "connections"')
    except TypeError:
        if len(test_conn)!=2:
            raise ValueError("Elements of 'connections' must be lists of two integers")
        else:
            compute_all_edges= False
    G = nx.Graph()
    if compute_all_edges:
        compute_from_vertices(params)
    else:
        compute_from_edges(params)

    return G

def phase_lag_index(x,y):
    '''
    compute the phase lag index of two signals
    :param x:
    :param y:
    :return:
    '''
    N = len(x)
    if N!=len(y):
        raise ValueError('Signals must be the same length')
    #window the signals for less high-freq interference
    window = np.hamming(N)
    x_hat= sig_proc.hilbert(x*window)
    y_hat = sig_proc.hilbert(np.conjugate(y*window))

    x_phase = np.arctan2(x_hat,x)
    y_phase = np.arctan2(y_hat,y)

    phase_diff = np.mod(x_phase-y_phase,2*np.pi)
    phase_diff[phase_diff>np.pi] = phase_diff[phase_diff>np.pi]-2*np.pi


    almost_index = np.sign(phase_diff)
    almost_index[almost_index==0] = 1
    index = np.abs(np.mean(almost_index))

    return index


def weighted_phase_lag_index(x,y):
    pass

def imag_coherency(x,y):
    '''
    Compute the imaginary coherence of two signals.
    :param x: numpy array
    :param y: numpy array
    :return: numpy float
    '''
    N = len(x)
    if N!=len(y):
        raise ValueError('Signals must be the same length')
    x,y = np.array(x),np.array(y)
    window = np.hamming(N)
    x_hat = sig_proc.hilbert(x*window)
    y_hat = sig_proc.hilbert(y*window)

    x_phase = np.arctan2(x_hat,x)
    y_phase = np.arctan2(y_hat,y)

    phase_diff = x_phase-y_phase

    c = np.mean(x*y*np.sin(phase_diff))/(
        np.sqrt(np.mean(np.power(x,2))*
                np.mean(np.power(y,2)))
    )
    return c

