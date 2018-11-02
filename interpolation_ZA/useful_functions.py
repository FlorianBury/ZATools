import glob
import os
import re
import math
import socket
import json

import array
import numpy as np

from ROOT import TChain, TFile, TTree, TH1F, TH1
from root_numpy import tree2array, rec2array, hist2array


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
            mHmA = (re.findall(r'\d+', filename)[2],re.findall(r'\d+', filename)[3]) # record mH, mA as tuple t be used as a key in the dict
            TH1_dict[mHmA] = h.Clone()
            if verbose:
                print ("\t-> [INFO] Extracted hist")
        except:
            print ("\t-> [WARNING] Could not extract hist")
    f.Close()
            
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
    Ouput :
        - Hist with bin contents normalized
    Ouput :
    """
    new_dico = dict()
    for key,val in dico.items(): # For each hist, normalize
        sumval = np.sum(val)
        for i,v in enumerate(val):
            val[i] /= sumval
        new_dico [key] = val
    return new_dico
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
def InterpolateAverage(neighbours,eval_grid,n_neigh):
    """
    Interpolate the rho distributions over a grid of tuples (m_H,m_A)
    Interpolation is made by averaging the n_neigh closest neighbours
    Inputs :
        - neighbours = points where rho distribution is know
                Dict    -> key = ('mH','mA') tuple
                        -> value = np.array of six bins
    Outputs :
        - grid = interpolated points
                Dict    -> key = ('mH','mA') tuple
                        -> value = np.array of six bins
    """

###############################################################################
# GetHistDict #
###############################################################################
def GetHistDict():
    """ 
        Get Matching between hist number and filename
        Returns 
            - dico 
                key = filename (.root)
                value = hist name (TH1)
    """
    dico = {}
    dico ['HToZATo2L2B_MH-1000_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_0.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-1000_MA-500_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_1.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-1000_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_2.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-200_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_3.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-200_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_4.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-250_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_5.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-250_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_6.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-300_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_7.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-300_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_8.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-300_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_9.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-500_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_10.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-500_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_11.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-500_MA-300_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_12.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-500_MA-400_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_13.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-500_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_14.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-650_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_15.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-800_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_16.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-800_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_17.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-800_MA-400_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_18.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-800_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_19.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-800_MA-700_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_20.root'] = 'rho_steps_histo_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    
    return dico


