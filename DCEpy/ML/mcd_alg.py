__author__ = 'Chris'

import numpy as np
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt

def build_model(X_inter, h):

    mcd = MinCovDet(support_fraction=h).fit(X_inter)
    return mcd

def anomaly_score(X, mcd):

    score = mcd.mahalanobis(X)
    return score

def plot_score(score, labels, file_path, title_str, threshold=np.inf, plot_log=True):

    fig = plt.figure(0,figsize=(10,6))
    plt.clf()
    plt.cla()
    ax = fig.add_subplot(1,1,1)


    if plot_log:
        plot_dist = np.log10(score)
        ylabel = "Mahalanobis Distance (log)"
    else:
        plot_dist = score
        ylabel = "Mahalanobis Distance"

    # plot the mahalanobis distance
    N = len(labels)
    time = np.arange(N) # time vector
    ax.plot(time, plot_dist, c='blue') # plot the distance
    plt.xlabel('Time Window Indices', fontsize='x-large')
    plt.ylabel(ylabel, fontsize='x-large')
    plt.title(title_str, fontsize = 'xx-large')

    # plot threshold if it exists
    if threshold < np.inf:
        if plot_log:
            plot_thresh = np.log10(threshold)
        else:
            plot_thresh = threshold
        ax.plot(time, plot_thresh, 'r--', linewidth=2.5)
        ax.text(N-1, plot_thresh[0], 'Anomaly Threshold', horizontalalignment='right', size='large')

    # plot the lines denoting seizure times
    y_pos = (ax.get_ylim()[0] + np.min(plot_dist) ) / 2.0

    pre_ictal_time = next((i for i, v in enumerate(labels) if v == 'Preictal'), -1)
    ax.axvline(x=pre_ictal_time, linestyle='-.', c="green", linewidth=4.0)
    ax.text(pre_ictal_time, y_pos, 'Pre-ictal', horizontalalignment='left', size='large')

    ictal_time = next((i for i, v in enumerate(labels) if v == 'Ictal'), -1)
    ax.axvline(x=ictal_time, linestyle='-.', c="green", linewidth=4.0)
    ax.text(pre_ictal_time, y_pos, 'Seizure Onset', horizontalalignment='left', size='large')

    post_ictal_time = next((i for i, v in enumerate(labels) if v == 'Postictal'), -1)
    ax.axvline(x=post_ictal_time, linestyle='-.', c="green", linewidth=4.0)
    ax.text(pre_ictal_time, y_pos, 'Post-Ictal', horizontalalignment='left', size='large')

    # save the figure
    fig.savefig(file_path, bbox_inches='tight')

    return



