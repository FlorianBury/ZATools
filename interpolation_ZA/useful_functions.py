import glob
import os
import re
import math
import socket
import json

import array
import numpy as np

from ROOT import TChain, TFile, TTree
#from root_numpy import tree2array, rec2array


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
# EvaluationGrid #
###############################################################################

def EvaluationGrid(path_to_json='/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/points_0.500000_0.500000.json'):
    with open(path_to_json) as f:
        data = json.load(f)
    return data
