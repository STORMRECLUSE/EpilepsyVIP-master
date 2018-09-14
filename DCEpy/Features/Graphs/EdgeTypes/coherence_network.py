import networkx as nx
import matplotlib.mlab as mplm
import numpy as np
from scipy.signal import coherence

def coherence_network(data, connections, fs=1000):
    """
    Creates and returns a NetworkX graph using coherence as edge weights.

    Input:
        data: an s x n array, s is the number of samples in the recording
                              n is number of channels
                              
        connections: a list of integers 1...n that correspond to the electrodes
                    we want to consider in building our graph
                    if left empty, all channels will be used

    Output:
        G: a NetworkX graph with channels as nodes and coherence between the
            channels as edge weights

    """
    s, n = data.shape

    # create edge list from connections
    if connections == []:
        edges = range(n)
    elif type(connections[0]) is list:
        edges = connections
    elif type(connections[0]) is int:
        edges = []
        for node1 in connections:
            for node2 in connections:
                if node1 != node2:
                    edges.append([node1, node2])

    # build coherence graph
    G = nx.Graph()
    for edge in edges:
        # cxy, f = mplm.cohere(data[:, edge[0]], data[:, edge[1]])
        f, cxy = coherence(data[:,edge[0]], data[:,edge[1]], fs=fs)
        c = np.mean(cxy) 
        G.add_edge(edge[0], edge[1], attr_dict={'coherence': c}) 

    return G