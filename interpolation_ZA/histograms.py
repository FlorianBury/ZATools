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

from ROOT import TChain, TFile, TTree, TH1F, TH1

###############################################################################
# LoopOverHists #
###############################################################################
def LoopOverHists(input_dir,hist_dict,verbose=False, return_numpy=False):
    """
        Loop over the hist from a ROOT file 
        Input :
            - input_dir = directory containing the ROOT files
            - hist_dict = dict containing the names of the file and the corresponding hist to extract
            - verbose : wether to print all info
            - return_numpy :    False -> returns TH
                                True -> returns numpy array
        Output :
            - TH1_dict = dict containing the TH1/numpy and their names
    """
    TH1_dict = {}

    if not os.path.isdir(input_dir):
        sys.exit("[INFO] LoopOverHists : Not a directory")

    if verbose:
        print ("[INFO] Accessing directory : ",input_dir)

    # Loop over root files #
    for name in glob.glob(input_dir+"*.root"):
        filename = name.replace(input_dir,'')

        if verbose:
            print ("[INFO] Accessing file : ",filename)

        # Look for a match between filename and hist name according to hist_dict #
        if filename in hist_dict:
            hist_name = hist_dict[filename]
            if verbose:
                print ('[INFO] Found match with %s'%(hist_name))
        else: 
            if verbose:
                print ("[WARNING] Could not match %s file with hist name"%(filename))

        # Extract TH1 and store in dict #
        try:
            f = TFile.Open(name)
            TH1.AddDirectory(0) # Required, "Otherwise the file owns and deletes the histogram." (cfr Root forum)
            h = f.Get(hist_name)
            mHmA = (float(re.findall(r'\d+', filename)[2]),float(re.findall(r'\d+', filename)[3])) # record mH, mA as tuple t be used as a key in the dict
            #mHmA = (float(re.findall(r'\d+', filename)[2])+.01*float(re.findall(r'\d+', filename)[3]),float(re.findall(r'\d+', filename)[4])+.01*float(re.findall(r'\d+', filename)[5])) # record mH, mA as tuple t be used as a key in the dict
            TH1_dict[mHmA] = h.Clone()
            f.Close()
            if verbose:
                print ("\t-> [INFO] Extracted hist")
        except:
            print ("\t-> [WARNING] Could not extract hist")
            
    if not return_numpy:
        return TH1_dict
    else:
        numpy_dict = {}
        for key,val in TH1_dict.items():
            arr = np.zeros(6)
            for i in range(0,6):
                arr[i] = val.GetBinContent(i+1)
            numpy_dict[key] = arr
        return numpy_dict
            
        
    # Loop over hist inside file #
    #hnames = []
    #for key in f.GetListOfKeys():
    #    h = key.ReadObj()
    #    if h.ClassName() == 'TH1F' or h.ClassName() == 'TH2F':
    #        hnames.append(h.GetName())
   
