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
from root_numpy import tree2array, rec2array, hist2array

import matplotlib.pyplot as plt                                                                  
###############################################################################
# Tree2Numpy#
###############################################################################

def Tree2Numpy(input_file, variables, weight, cut=None, reweight_to_cross_section=False, n=None):
    """
    Convert a ROOT TTree to a numpy array.
    Inputs :
        input_file = name of the ROOT file
        variables = list of names of the variables to be extracted
        weight = dict of weights to be used
        cut = optionnal cut on data 
        reweight_to_cross_section = wether to reweight according to the cross-section
        n = number of events to be printed for tests
    Outputs :
        dataset = array containing events in rows with variable in columns
        weight = array of weights following dataset
    """

    file_handle = TFile.Open(input_file)

    tree = file_handle.Get('t')

    cross_section = 1
    relative_weight = 1
    if reweight_to_cross_section:
        cross_section = file_handle.Get('cross_section').GetVal()
        relative_weight = cross_section / file_handle.Get("event_weight_sum").GetVal()

    if isinstance(weight, dict):
        # Keys are regular expression and values are actual weights. Find the key matching
        # the input filename
        found = False
        weight_expr = None
        if '__base__' in weight:
            weight_expr = weight['__base__']

        for k, v in weight.items():
            if k == '__base__':
                continue

            groups = re.search(k, input_file)
            if not groups:
                continue
            else:
                if found:
                    raise Exception("The input file is matched by more than one weight regular expression. %r" % input_file)

                found = True
                weight_expr = join_expression(weight_expr, v)

        if not weight_expr:
            raise Exception("Not weight expression found for input file %r" % weight_expr)

        weight = weight_expr

    # Read the tree and convert it to a numpy structured array
    a = tree2array(tree, branches=variables + [weight], selection=cut)

    # Rename the last column to 'weight'
    a.dtype.names = variables + ['weight']

    dataset = a[variables]
    weights = a['weight'] * relative_weight

    # Convert to plain numpy arrays
    dataset = rec2array(dataset)

    if n:
        print("Reading only {} from input tree".format(n))
        dataset = dataset[:n]
        weights = weights[:n]

    return dataset, weights

###############################################################################
# LoopOverTrees #
###############################################################################

def LoopOverTrees(input_dir, variables, weight, part_name=None, cut=None, reweight_to_cross_section=False, n=None, verbose=False):
    """
    Loop over ROOT trees inside input_dir and process them using Tree2Numpy.
    Inputs :
        input_dir = directory of the ROOT files
        part_name = optionnal string included in the name of the file (used to differentiate different ROOT files inside directory)
        variables = list of names of the variables to be extracted
        weight = dict of weights to be used
        cut = optionnal cut on data 
        reweight_to_cross_section = wether to reweight according to the cross-section
        n = number of events to be printed for tests
        verbose : wether to print all info
    Outputs :
        dataset = array containing events in rows with variable in columns
        weight = array of weights following dataset
    """

    if not os.path.isdir(input_dir):
        sys.exit("[INFO] LoopOverTrees : Not a directory")

    if verbose:
        print ("[INFO] Accessing directory : ",input_dir)

    for name in glob.glob(input_dir+"*.root"):
        filename = name.replace(input_dir,'')

        if verbose:
            print ("[INFO] Accessing file : ",filename)
            
        if part_name is not None:
            if re.search(part_name,filename):
                if verbose:
                    print ('[INFO] Found match')
            else:
                if verbose:
                    print ('[INFO] Could not find match')
                continue 
        
        dataset,weights = Tree2Numpy(name,variables,weight,cut,reweight_to_cross_section,n) 
    
        return dataset,weights

###############################################################################
# LoopOverTree #
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
        for k,v in TH1_dict.items():
            numpy_dict[k] = hist2array(v)
        return numpy_dict
            
        
    # Loop over hist inside file #
    #hnames = []
    #for key in f.GetListOfKeys():
    #    h = key.ReadObj()
    #    if h.ClassName() == 'TH1F' or h.ClassName() == 'TH2F':
    #        hnames.append(h.GetName())
    
###############################################################################
# NormalizeHist #
###############################################################################
def NormalizeHist(dico):
    """
    Normalize histograms to unit area 
    Input :
        - Hist with bin contents
    Output :
        - Hist with bin contents normalized
    """
    new_dico = dict()
    for key,val in dico.items(): # For each hist, normalize
        sumval = np.sum(val)
        for i,v in enumerate(val):
            val[i] /= sumval
        new_dico [key] = val
    return new_dico

###############################################################################
# AddHist #
###############################################################################
def AddHist(dict1,dict2):
    """
    Add two dict of histogram bin contents
    Input :
        - dict1 : dict
            contains mass point (tuple) as key and bin content (numpy array [1,6]) as values
        - dict2 : dict
            contains mass point (tuple) as key and bin content (numpy array [1,6]) as values
    Output :
        - dict_add : dict
            = "dict1+dict2" in the values
    """
    dict_add = {}
    for key,val in dict1.items(): 
        dict_add[key] = np.add(dict1[key],dict2[key])

    return dict_add

###############################################################################
# CheckHist #
###############################################################################
def CheckHist(dico,verbose=False):
    """
    Checks that the histogram has decreasing bins (because signal si concentrated at center of ellipse 
    Input :
        - dico : dict
            contains mass point (tuple) as key and bin content (numpy array [1,6]) as values
    Output : Prints cases to investigate
    """
    for key,val in dico.items():
        diff = np.diff(val) # Get diff a[n+1]-a[n]
        if diff[0]>0:
            if verbose:
                warnings.warn("First bin looks weird") 
                print (val)
                # Signal not exactly at center, possible but have to check
        check = diff[1:]>0
        if  np.any(check):
            warnings.warn("At least one bin is not decreasing compared to previous one, might need to check that")
            print ('Config : ',key)
            print ('Bins : ',val)
            # Bin content should decrease but might be a little variation, to be checked

###############################################################################
# EvaluationGrid #
###############################################################################

def EvaluationGrid(path_to_json='/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/points_0.500000_0.500000.json'):
    with open(path_to_json) as f:
        data = json.load(f)
    return data


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



