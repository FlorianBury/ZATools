import glob
import os
import re
import sys
import math
import json
import warnings
import pprint

import array
import numpy as np
from operator import itemgetter

from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error 

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
    Interpolate the rho distributions over a grid of tuples (m_A,m_H)
    Interpolation is made by averaging the n_neigh closest neighbours weighted by the distance
    Inputs :
        - neighbours : dict 
            points where rho distribution is know
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
        - eval_grid : list of list (nx2 elements)
            contains the points on which interpolation is to be done
        - n_neigh : int
            number of neighbours to be used for the averaging 
    Outputs :
        - grid = dict 
            interpolated points
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
    """
    grid = {} # To be returned
    dist_dict = {} # keep memory of the distances 
    for point in eval_grid: # Loop over the points to interpolate
        hist_arr = np.zeros(6) # will be the hist array for the grid element
        for neighbour_point in neighbours: # Loop over neighbours to find the closests
            dist_dict[neighbour_point] = distance(point,neighbour_point)
        sort_dist = sorted(dist_dict.items(), key=itemgetter(1)) # sorts dist_dict : tuple ((mA,mH),distance)

        # Get total distance of n_neigh closest neighbours #
        total_dist = 0
        for close_neighbour in sort_dist[:n_neigh]: # Loop over the n_neigh closest neighbours
            # close_neighbour[0] -> tuple (mA,mH)
            # close_neighbour[1] -> distance
            if close_neighbour[1] == 0:
                print ('Distance is 0 -> Same point found')
            arr = neighbours[close_neighbour[0]]*close_neighbour[1]  # Gets hist array (=value of dict) corresponding to a close neighbour (weighted)
            total_dist += close_neighbour[1]
            hist_arr = np.add(hist_arr,arr)
        
        hist_arr /= total_dist # weighted
        grid [tuple(point)] = hist_arr
    
    return grid 

###############################################################################
# EvaluateAverage #
###############################################################################
def EvaluateAverage(train_dict,test_dict,max_n,scan=False):
    """
    Performs the average interpolation for know points as a cross-check
    Tests different number or neighbours (from 1 to max_n)
    Finds the best case (minimizing chi2) and produces the comparison plots
    Inputs :
        - hist_dict : dict 
            points where rho distribution is know
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
        - max_n : int 
            maximum number of neighbours to be checked
    Outputs :
        - output_dict : dict
            Result of the interpolation for each mass point 
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins

    """
    # Turn keys from dict into list of list #
    eval_list = []
    for key in test_dict.keys():
        eval_list.append(list(key))

    # Scan among all the possible number of neighbours #
    if scan:
        chi2_list = []
        mse_list = []
        for n in range(1,max_n+1):
            chi2_sum = 0.
            mse_sum = 0.
            # Interpolate #
            eval_avg = InterpolateAverage(train_dict,eval_list,n)      
            # Evaluate chi2 for each hist #
            for key in eval_avg.keys():
                #chi2,p = chisquare(f_obs=eval_avg[key],f_exp=hist_dict[key])
                mse  = mean_squared_error(y_true=np.transpose(test_dict[key]),y_pred=np.transpose(eval_avg[key]))
                #chi2_sum += chi2     
                mse_sum += mse
            # Keeps in memory #
            chi2_list.append(chi2_sum)
            mse_list.append(mse_sum)


        # Prints results #
        for idx,val in enumerate(mse_list,1):                                                   
             #print ('Average evaluation with %d neighbours :  chi2 sum = %0.5f'%(idx+1,val))
             print ('Average evaluation with %d neighbours :  MSE sum = %0.5f'%(idx,val))

        # Find best model #
        min_index, min_value = min(enumerate(mse_list), key=itemgetter(1))
        best_n =  min_index +1
        print ('Best number of neighbours -> ',best_n)

    # Get the hist output #
    if scan:
        output_dict = InterpolateAverage(train_dict,eval_list,best_n)
    else:
        output_dict = InterpolateAverage(train_dict,eval_list,max_n) 

    return output_dict

