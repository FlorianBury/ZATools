import glob
import os
import re
import sys
import math
import socket
import json
import warnings

import array
import numpy as np
from operator import itemgetter

from scipy.stats import chisquare

import matplotlib.pyplot as plt                                                                  

###############################################################################
# PlotComparison #
###############################################################################
def PlotComparison(inter_dict,name):
    """
    Takes a serie of N dict histograms defined as dictionaries themselves : 
    Inputs :
        - inter_dict : dict
            Contains the interpolation dict
                key -> name for the title
                value ->  dict
                    key -> (mA,mH) mass configuration
                    value -> 6 bin contents of the rho distribution 
        - name : str
            Name of the output directory
    Outputs :
        None
    Plots :
        Comparison of the three distribution for each mass point 
    """

    # Create directory #
    path = os.path.join(os.getcwd(),name)
    if not os.path.isdir(path):
        os.makedirs(path)

    # Binning parameters #
    N = len(inter_dict.keys())
    bin_dict = {} # Will contain the bins for each hist
    i = 0
    for key in inter_dict.keys():
        if i==0:
            bin_dict[key] =  np.linspace(0,2.5,6)
        else:
            bin_dict[key] = np.linspace(0,2.5,6)+0.5*i/(N+1)
        i += 1

    # Plot section #
    print ('[INFO] Starting interpolate plot section')
    for masspoint in list(inter_dict.values())[0].keys(): # Loop over mass points of first dict
        print ('Masspoint : m_H = %0.f, m_A = %0.f'%(masspoint[1],masspoint[0]))
        color=iter(plt.cm.rainbow(np.linspace(0.3,1,N)))
        fig = plt.figure()
        for name, hist_dict in inter_dict.items(): # Loop over the histograms dict
            # Plot the bars #
            c=next(color)
            p = plt.bar(bin_dict[name],
                    hist_dict[masspoint],
                    align='edge',
                    width=0.5/(N+1),
                    color=c,
                    linewidth=2,
                    label=name)

        # Optional parameters #
        plt.legend(loc='upper right')
        plt.xlabel(r'$\rho$')
        plt.ylabel('Arb. units')
        plt.title(r'Mass point $m_H=$%d GeV, $m_A$=%d GeV'%(masspoint[1],masspoint[0]))

        # Estethic : distinguishable groups of bins #
        #for i in range(0,6):
        #    if i%2==1: 
        #        p_avg[i].set_hatch('/')
        #        p_tri[i].set_hatch('/')
        #        p_dnn[i].set_hatch('/')
        #        p_real[i].set_hatch('/')
        #    else:
        #        p_avg[i].set_hatch('\\')
        #        p_tri[i].set_hatch('\\')
        #        p_dnn[i].set_hatch('\\')
        #        p_real[i].set_hatch('\\')

        # Save #
        fig.savefig(os.path.join(path,'m_H_%d_m_A_%d.png'%(masspoint[1],masspoint[0])))



