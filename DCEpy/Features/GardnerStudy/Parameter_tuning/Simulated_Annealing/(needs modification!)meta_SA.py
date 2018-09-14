"""

Meta-heuristics for Genetic Algorithm in Gardners multichannel+weighted dr pipeline

This file tunes the GA control parameters: [CXPB, INDPB, MTPB], for each channel.

"""


import meta_SA_helper as ptest
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import svm
import scipy
import os, pickle, sys, time, csv
from DCEpy.General.DataInterfacing.edfread import edfread
# from edfread import edfread
import array
import random
import multiprocessing
import math


def score_parameters(parameters, folds):
    """"
    The function takes in:
    1. parameters: a list of [CXPB,INDPB,MTPB] used in genetic algorithm
    2, the number of folds desired to be run on

    The function runs the desired number of folds of experiment on a selected channel and returns the
    negative value of the sum of fitnesses.

    """

    # pass in the parameters to parameters test

    fitnesses= ptest.experiment(parameters,folds)

    return -sum(fitnesses)



def tune_pbs():


    # Initialize random set of [CXPB,INDPB,MTPB]

    MIN = [0.5, .1, .1]
    MAX = [.95, .25, .25]
    CXPB = random.uniform(MIN[0], MAX[0])
    INDPB = random.uniform(MIN[1], MAX[1])
    MTPB = random.uniform(MIN[2], MAX[2])
    pb_init=[CXPB,INDPB,MTPB]
    bounds = zip(MIN, MAX)

    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

    print "start optimization!"
    results = scipy.optimize.basinhopping(
        func=lambda x: score_parameters(x,6),
        x0=pb_init,
        minimizer_kwargs=minimizer_kwargs,
        disp=False
        )

    parameters=results.x
    print "CXPB: %f, INDPB : %f, MTPB: %f." %(parameters[0],parameters[1],parameters[2])




tune_pbs()