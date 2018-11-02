#!/usr/bin/env python 

import glob
import os
import re
import math
import socket
import json

import array
import numpy as np
import argparse 
#from ROOT import TFile, TTree, TH1, TCanvas

# Private modules #
from useful_functions import *
#from NeuralNet import * 


def get_options():
    """                                         
    Parse and return the arguments provided by the user.
    """
    parser = argparse.ArgumentParser(description='Compare names of input and output files for matches and potentiel failed files')
    parser.add_argument('-a','--average', action='store', required=False, type=int, default=0,
                        help='If averaged interpolation is required -> input number of neighbours')
    #parser.add_argument('--part_name', action='store', required=False, default="", type=str,
    #                    help='Common part of the file names [DEFAULT = FALSE]')
    #parser.add_argument('--not_in_name', action='store', required=False, default=None, type=str,
    #                    help='Excluded part of name (use to exclude files) [DEFAULT = FALSE]')
    #parser.add_argument('--verbose', action='store', required=False, default=False, type=bool,
    #                    help='Verbosity : wether to look at sizes and number of events inside each files [DEFAULT = FALSE]')

    return parser.parse_args()

def main():
    
    # Get options from user #
    opt = get_options()
    print (opt)

    # Get histograms #
    print ('[INFO] Starting histograms input')
    path_to_files = '/nfs/scratch/fynu/fbury/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/test_for_signal/slurm/output/'
    name_dict = GetHistDict()
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
    print (x_DNN.shape)

    for k,v in hist_dict.items():
        print (np.asarray(k).reshape(1,2).shape)
        np.append(x_DNN,np.asarray(k).reshape(-1,2),axis=0)
        np.append(y_DNN,[v],axis=0)



    print ('... Done')

    print (x_DNN)
    print (y_DNN)

if __name__ == "__main__":                                     
    main()

