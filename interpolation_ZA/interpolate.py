#!/usr/bin/env python 

import glob
import os
import re
import math
import socket
import json
import sys

import array
import numpy as np
import argparse 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

# Private modules #
from useful_functions import *
from get_hist_dict import *
from NeuralNet import * 


def get_options():
    """                                         
    Parse and return the arguments provided by the user.
    """
    parser = argparse.ArgumentParser(description='Compare names of input and output files for matches and potentiel failed files')
    parser.add_argument('-a','--average', action='store', required=False, type=int, default=0,
        help='If averaged interpolation is required -> input number of neighbours')
    parser.add_argument('-s','--scan', action='store', required=False, type=str, default='',
        help='If scan for hyperparameters to be performed [edit NeuralNet.py] and given name')
    parser.add_argument('-r','--reporting', action='store', required=False, type=str, default='',
        help='If reporting is necessary for analyzing scan given the csv file given')

    return parser.parse_args()

def main():
    
    # Get options from user #
    opt = get_options()

    # Input path #
    print ('[INFO] Starting histograms input')

    #path_to_files = '/nfs/scratch/fynu/fbury/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/test_for_signal/slurm/output/'
    path_to_files = '/nfs/scratch/fynu/fbury/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/test_full_for_signal/slurm/output/'

    # Loop to get histograms #
    #name_dict = GetHistDict_previous()
    name_dict = GetHistDict_new()
    hist_dict = LoopOverHists(path_to_files,name_dict,verbose=False,return_numpy=True)
    #TH1_dict = LoopOverHists(path_to_files,name_dict,verbose=True,return_numpy=False)
    print ('... Done')

    # Normalize numpy histograms #
    hist_dict = NormalizeHist(hist_dict)

    # Get grid on which evaluate the network for the interpolation #
    print ('[INFO] Extracting output grid')
    eval_grid = EvaluationGrid() 
    print ('... Done')

    # Interpolate with average #
    if opt.average!=0:
        print ('[INFO] Using the average interpolation')
        inter_avg = InterpolateAverage(hist_dict,eval_grid,opt.average)
        print ('... Done')

    # Interpolate with DNN #
    print ('[INFO] Using the DNN interpolation')
    x_DNN = np.empty((0,2)) # Inputs (mH,mA) of the DNN
    y_DNN = np.empty((0,6)) # Outputs (6 bins of rho dist) of the DNN
    # Need to prepare inputs and outputs #
    for k,v in hist_dict.items():
        x_DNN = np.append(x_DNN,np.asarray(k).reshape(-1,2),axis=0)
        y_DNN = np.append(y_DNN,[v],axis=0)

    # Splitting and preprocessing
    x_train,x_test,y_train,y_test = train_test_split(x_DNN,y_DNN,test_size=0.4) # MUST ADD WEIGHTS

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Make HyperScan #
    if opt.scan!='':
        print ("Total training size : ",x_train.shape[0],'/',x_DNN.shape[0])
        h = HyperScan(x_train,y_train,name=opt.scan)
    print ('... Done')
    
    # Reporting #
    if opt.reporting!='': 
        print ("Total testing size : ",x_test.shape[0],'/',x_DNN.shape[0])
        HyperReport(opt.reporting,x_test,y_test)
    



if __name__ == "__main__":                                     
    main()

