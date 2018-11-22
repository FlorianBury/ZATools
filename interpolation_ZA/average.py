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
# InterpolateAverage #
###############################################################################
# Define euclidean distance #
def distance(a,b):
    """ Distance between two tuples in 2D """
    assert len(a)==2
    assert len(b)==2
    return math.sqrt((float(a[0])-float(b[0]))**2+(float(a[1])-float(b[1]))**2)
    

def InterpolateAverage(neighbours,eval_grid,n_neigh):
    """
    Interpolate the rho distributions over a grid of tuples (m_H,m_A)
    Interpolation is made by averaging the n_neigh closest neighbours weighted by the distance
    Inputs :
        - neighbours : dict 
            points where rho distribution is know
                -> key = ('mH','mA') tuple
                -> value = np.array of six bins
        - eval_grid : list of list (nx2 elements)
            contains the points on which interpolation is to be done
        - n_neigh : int
            number of neighbours to be used for the averaging 
    Outputs :
        - grid = dict 
            interpolated points
                -> key = ('mH','mA') tuple
                -> value = np.array of six bins
    """
    grid = {} # To be returned
    dist_dict = {} # keep memory of the distances 
    for val in eval_grid: # Loop over the points to interpolate
        hist_arr = np.zeros(6) # will be the hist array for the grid element
        for key in neighbours: # Loop over neighbours to find the closests
            dist_dict[key] = distance(val,key)
        sort_dist = sorted(dist_dict.items(), key=itemgetter(1)) # sorts dist_dict : tuple ((mH,mA),distance)
        if sort_dist[0][1] == 0:
            n_neigh += 1 # if first is 0, add one because we will not take it into account

        # Get total distance of n_neigh closest neighbours #
        total_dist = 0
        for e in sort_dist[:n_neigh]: # Loop over the n_neigh closest neighbours
            total_dist += e[1]
            # don't need to remove point where dist=0 because won't impact the sum 

        for e in sort_dist[:n_neigh]: # Loop over the n_neigh closest neighbours
            if e[1] == 0: 
                continue
                # Remove if dist is zero : we don't want to interpolate from the same point
            arr = neighbours[e[0]]*e[1]/total_dist # Gets hist array (=value of dict) corresponding to a close neighbour (weighted)
            hist_arr = np.add(hist_arr,arr)

        grid [tuple(val)] = hist_arr
    
    return grid 

###############################################################################
# EvaluateAverage #
###############################################################################
def EvaluateAverage(hist_dict,max_n):
    """
    Peforms the average interpolation for know points as a cross-check
    Tests different number or neighbours (from 1 to max_n)
    Finds the best case (minimizing chi2) and produces the comparison plots
    Inputs :
        - hist_dict : dict 
            points where rho distribution is know
                -> key = ('mH','mA') tuple
                -> value = np.array of six bins
        - max_n : int 
            maximum number of neighbours to be checked
    Outputs :
        - output_dict : dict
            Result of the interpolation for each mass point 
                -> key = ('mH','mA') tuple
                -> value = np.array of six bins

    """
    # Turn keys from dict into list of list #
    eval_list = []
    for key in hist_dict.keys():
        eval_list.append(list(key))

    # Scan among all the possible number of neighbours #
    chi2_list = []
    for n in range(1,max_n+1):
        chi2_sum = 0.
        # Interpolate #
        eval_avg = InterpolateAverage(hist_dict,eval_list,n)      
        # Evaluate chi2 for each hist #
        for key in eval_avg.keys():
            chi2,p = chisquare(f_obs=eval_avg[key],f_exp=hist_dict[key])
            chi2_sum += chi2     
        # Keeps in memory #
        chi2_list.append(chi2_sum)

    # Prints results #
    for idx,val in enumerate(chi2_list):                                                   
         print ('Average evaluation with %d neighbours :  chi2 sum = %0.5f'%(idx+1,val))

    # Find best model #
    min_index, min_value = min(enumerate(chi2_list), key=itemgetter(1))
    best_n =  min_index +1
    print ('Best number of neighbours -> ',best_n)

    # Get the hist output #
    #output_dict = InterpolateAverage(hist_dict,eval_list,best_n)
    output_dict = InterpolateAverage(hist_dict,eval_list,3) #TODO, use best_n

    return output_dict

###############################################################################
# PlotComparison #
###############################################################################
def PlotComparison(hist_real,hist_avg,hist_DNN,name):
    """
    Takes 3 series of histograms defined in dictionaries : the real one and the two interpolations. Plot the comparisons 
    Inputs :
        - hist_real : dict
            Contains the real rho distribution 
                key -> (mH,mH) mass configuration
                value -> 6 bin contents of the rho distribution 
        - hist_avg : dict
            Contains the interpolation from the average method
                key -> (mH,mH) mass configuration
                value -> 6 bin contents of the rho distribution 
        - hist_real : dict
            Contains the interpolation from the DNN method
                key -> (mH,mH) mass configuration
                value -> 6 bin contents of the rho distribution 
        - name : str
            Name of the output directory
    Outputs :
        None
    Plots :
        Comparison of the three distribution for each mass point 
    """

    # Create directory #
    path = os.path.join(os.getcwd(),name+'/verification')
    if not os.path.isdir(path):
        os.makedirs(path)

    # Plot section #
    for key in hist_real.keys(): # Loop over mass points 
        # Binning parameters #
        n_bin = hist_real[key].shape[0]
        bins_real = np.linspace(0,2.5,6)
        bins_avg = bins_real+0.5*2/3
        bins_DNN = bins_real+0.5/3

        # Plot the bars #
        fig = plt.figure()
        p_avg = plt.bar(bins_avg,
                hist_avg[key],
                align='edge',
                width=0.5/3,
                color='y',
                #edgecolor=c,
                linewidth=2,
                label='Averaged distribution')
        p_dnn = plt.bar(bins_DNN,
                hist_DNN[key],
                align='edge',
                width=0.5/3,
                color='b',
                #edgecolor=c,
                linewidth=2,
                label='DNN distribution')
        p_real = plt.bar(bins_real,
                hist_real[key],
                align='edge',
                width=0.5/3,
                color='g',
                #edgecolor=c,
                linewidth=2,
                label='True distribution')

        # Optional parameters #
        plt.legend(loc='upper right')
        plt.xlabel(r'$\rho$')
        plt.ylabel('Arb. units')
        plt.title(r'Mass point $m_H=$%d GeV, $m_A$=%d GeV'%(key[0],key[1]))

        # Estethic : distinguishable groups of bins #
        for i in range(0,6):
            if i%2==1: 
                p_avg[i].set_hatch('/')
                p_dnn[i].set_hatch('/')
                p_real[i].set_hatch('/')
            else:
                p_avg[i].set_hatch('\\')
                p_dnn[i].set_hatch('\\')
                p_real[i].set_hatch('\\')

        # Save #
        fig.savefig(os.path.join(path,'m_H_%d_m_A_%d.png'%(key[0],key[1])))



