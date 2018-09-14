__author__ = 'Chris'

import networkx as nx
import numpy as np

def large_graph_metric(G, weight=None, thresh=None):

    # threshold graph edges based on weight
    if thresh is None:
        H = G
    else:
        if weight is None:
            raise ValueError('No weight is given for thresholding')

        rm_edges = []
        for edge in G.edges_iter(data=True):
            if edge[2]['weight'] < thresh:
                rm_edges.append([edge[0],edge[1]])
        H = G.copy()
        H.remove_edges_from(rm_edges)

    # create feature vector
    x = np.ones(13)

    # clustering coefficient
    x[0] = nx.average_clustering(H)

    # node connectivity
    x[1] = nx.node_connectivity(H)

    # average betweeness centrality
    # bc = nx.betweenness_centrality(G, weight=weight)
    bc = nx.betweenness_centrality(G)
    x[2] = float(sum(bc[node] for node in bc)) / len(bc)

    # average closeness centrality
    # cc = nx.closeness_centrality(G, weight=weight)
    cc = nx.closeness_centrality(G)
    x[3] = float(sum(cc[node] for node in cc)) / len(cc)

    # estrada index
    x[4] = nx.estrada_index(H)

    # average degree
    x[5] = float(sum(len(H[u]) for u in H)) / len(H)

    # standard deviation degree
    # x[6] = float(sum((x[4] - len(H[u]))**2)) / len(H)

    # degree assortativity coefficient
    # x[7] = nx.degree_assortativity_coefficient(G, weight=weight)
    x[7] = nx.degree_assortativity_coefficient(G)

    # transitivity
    x[8] = nx.transitivity(H)

    # diameter
    x[9] = nx.diameter(H)

    # eccentricity
    # x[10] = nx.eccentricity(H)

    # periphery
    # x[11] = nx.periphery(H)

    # radius
    x[12] = nx.radius(H)

    return x