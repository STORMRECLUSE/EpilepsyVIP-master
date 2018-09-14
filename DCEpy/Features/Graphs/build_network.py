from DCEpy.Features.Graphs.EdgeTypes.phase_nets import phase_conn_network
from .EdgeTypes.var_network import var_network
from .EdgeTypes.coherence_network import coherence_network

def build_network(data, connections, weightType, **params):
    '''
    Builds a networkx graph using specified connectivity measure

    Parameters
    ----------
    data : ndarray, shape (n, N)
        iEEG data with N channels and n samples  
    connections: list 
        list of either (i) integers
            the integers are the channels that will become nodes
            and the graph will be complete
        or (ii) length 2 lists of integers
            the integers correspond to directed edges
    weightType: str
        the connectivity measure to be used for graph weights.
    params: dict 
        dictionary to provide specific paramters to connectivity 
        computations
    '''


    if weightType == "pli":
        G = phase_conn_network(data, connections,'pli') # forty-two
    elif weightType == "partial_directed_coherence":
        G = var_network(data, connections, weightType, **params)
    elif weightType == "directed_transfer_function":
        G = var_network(data, connections, weightType, **params)
    elif weightType == "granger_causality":
        G = var_network(data, connections, weightType, **params)
    elif weightType == "directed_information":
        pass
        #G = di_network(data, connections, **params) #TODO: directed information
    elif weightType == "imag_coherency":
        G = phase_conn_network(data, connections, 'imag_coherency')
    elif weightType == "coherence":
        G = coherence_network(data, connections)

    return G


