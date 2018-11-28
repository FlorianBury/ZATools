#!/usr/bin/env python

import os
import sys
import warnings

import argparse
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from scipy.stats import chisquare

import matplotlib.pyplot as plt


def get_options():
    """
    Parse and return the arguments provided by the user.
    """
    parser = argparse.ArgumentParser(description='Compare names of input and output files for matches and potentiel failed files')
    parser.add_argument('-a','--average', action='store', required=False, type=int, default=0,
        help='If averaged interpolation is required -> input number of neighbours')
    parser.add_argument('-t','--triangle', action='store_true', required=False, default=False,
        help='If triangle interpolation is required')
    parser.add_argument('-s','--scan', action='store', required=False, type=str, default='',
        help='If scan for hyperparameters to be performed [edit NeuralNet.py] and given name')
    parser.add_argument('-r','--reporting', action='store', required=False, type=str, default='',
        help='If reporting is necessary for analyzing scan given the csv file given')
    parser.add_argument('-e','--evaluate', action='store', required=False, type=int, default=0,
        help='Wether to evaluate the models with cross validation : provide number of folds.\nCAUTION : requires the option --scan' )
    parser.add_argument('-d','--deploy', action='store', required=False, type=str, default='',
        help='Wether to deploy the model, provide the name\n.CAUTION : requires the option --scan.\nWARNING : if option --evaluate, will select the best model according to cross validation,\nif not, takes the one with lowest val_loss')
    parser.add_argument('-i','--interpolate', action='store', required=False, type=str, default='',
        help='Interpolation with the Neural Network. Must provide the name of the .zip package to apply the interpolation')
    parser.add_argument('-c','--compare', action='store_true', required=False, default=False,
        help='Whether to compare the interpolation between the different techniques. WARNING : at least -i option must have been used ')
    parser.add_argument('-v','--verification', action='store', required=False, default='', type=str,
        help='Wether to apply the verification cross check with average and DNN techniques, aka apply regression on know points. For average, best number of neighbours is found based on chi2 minimization, for the DNN the path to a Talos zip file must be provided (without .zip)')

    opt = parser.parse_args()

    # Option checks #
    if opt.evaluate!=0 and opt.scan=='':
        warnings.warn('--evaluate options requires --scan option')
        sys.exit(1)
    if opt.deploy!='' and opt.scan=='':
        warnings.warn('--deploy options requires --scan option')
        sys.exit(1)

    if opt.compare and opt.interpolate=='':
        warnings.warn('--compare must be used with at least --interpolate')
        sys.exit(1)

    return opt

def main():
    #############################################################################################
    # Preparation #
    #############################################################################################
    # Get options from user #
    opt = get_options()

    # Private modules #
    from histograms import LoopOverHists, NormalizeHist, EvaluationGrid, AddHist, CheckHist
    from average import InterpolateAverage, EvaluateAverage
    from triangles_interpolation import InterpolateTriangles, EvaluateTriangles
    from comparison import PlotComparison
    from get_link_dict import GetHistDictOld, GetHistDictNew
    from NeuralNet import HyperScan, HyperReport, HyperEvaluate, HyperDeploy, HyperVerif, HyperRestore
    # Needed because PyROOT messes with argparse

    #############################################################################################
    # Data Input and preprocessing #
    #############################################################################################
    # Input path #
    print ('[INFO] Starting histograms input')

    path_to_files = '/nfs/scratch/fynu/fbury/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/test_for_signal/slurm/output/'
    #path_to_files = '/nfs/scratch/fynu/fbury/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/test_full_for_signal/slurm/output/' # TODO change way to extract mH, mA in LoopOverHists

    # Loop to get histograms #
    print ('-'*80)
    name_dict_MuMu = GetHistDictOld('MuMu')
    name_dict_ElEl = GetHistDictOld('ElEl')
    print ('\n[INFO] MuMu Histograms\n')
    hist_dict_MuMu = LoopOverHists(path_to_files,name_dict_MuMu,verbose=False,return_numpy=True)
    print ('-'*80)
    print ('\n[INFO] ElEl Histograms\n')
    hist_dict_ElEl = LoopOverHists(path_to_files,name_dict_ElEl,verbose=False,return_numpy=True)
    #TH1_dict = LoopOverHists(path_to_files,name_dict,verbose=True,return_numpy=False)
    print ('-'*80)
    print ('... Done')

    print ('-'*80)
    # Checks shape of distributions #
    print ('[INFO] Checking MuMu sample')
    CheckHist(hist_dict_MuMu,verbose=False)
    print ('[INFO] Checking ElEl sample')
    CheckHist(hist_dict_ElEl,verbose=False)
    print ('-'*80)

    # Treat numpy arrays : normalization and addition #
    hist_dict = AddHist(hist_dict_MuMu,hist_dict_ElEl)
    hist_dict = NormalizeHist(hist_dict)

    # Get grid on which evaluate the network for the interpolation #
    print ('[INFO] Extracting output grid')
    eval_grid = EvaluationGrid()
    print ('... Done')

    #############################################################################################
    # Interpolation #
    #############################################################################################
    # Interpolate with average #
    if opt.average!=0:
        print ('[INFO] Using the average interpolation')
        inter_avg = InterpolateAverage(hist_dict,eval_grid,opt.average)
        print ('... Done')

    # Interpolate with triangle #
    if opt.triangle:
        print ('[INFO] Using the triangle interpolation')
        inter_tri = InterpolateTriangles(hist_dict,eval_grid)
        print ('... Done')


    # Interpolate with DNN #
    x_DNN = np.empty((0,2)) # Inputs (mH,mA) of the DNN
    y_DNN = np.empty((0,6)) # Outputs (6 bins of rho dist) of the DNN
    # Need to prepare inputs and outputs #
    for k,v in hist_dict.items():
        x_DNN = np.append(x_DNN,np.asarray(k).reshape(-1,2),axis=0)
        y_DNN = np.append(y_DNN,[v],axis=0)

    # Splitting and preprocessing
    x_train,x_test,y_train,y_test = train_test_split(x_DNN,y_DNN,test_size=0.3) # TODO change when enough data

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test) #TODO : change when enough data

    # Make HyperScan #
    if opt.scan != '':
        print ('[INFO] Starting Hyperscan')
        print ("Total training size : ",x_train.shape[0],'/',x_DNN.shape[0])
        h = HyperScan(x_train,y_train,name=opt.scan)
        print ('... Done')

    # Make HyperEvaluate #
    best_model = -1
    if opt.evaluate != 0:
        print ('[INFO] Starting HyperEvaluate')
        print ("Total validation size : ",x_test.shape[0],'/',x_DNN.shape[0])
        if (x_test.shape[0]==0):
            print ("[WARNING] No data to apply cross validation, not applied")
            return # Avoids being stuck with no data in evaliation 
        best_model = HyperEvaluate(h,x_test,y_test,opt.evaluate)
        print ('... Done')

    # Make HyperDeploy #
    if opt.deploy!='':
        print ('[INFO] Starting HyperDeploy')
        HyperDeploy(h,opt.deploy,best_model)
        print ('... Done')

    # Make HyperReport #
    if opt.reporting!='':
        print ('[INFO] Analyzing with Report')
        HyperReport(opt.reporting)
        print ('... Done')

    # Make HyperRestore #
    if opt.interpolate!='':
        print ('[INFO] Starting interpolation with model')
        inputs = np.asarray(eval_grid)
        inter_DNN = HyperRestore(inputs,scaler,opt.interpolate+'.zip')
        print ('... Done')

    # Comparison between different techniques #
    if opt.compare:
        inter_dict = {} # Will contain the dict of the different interpolation outputs (which are dict also)
        try :
            inter_dict['Average'] = inter_avg
            print ('Average interpolation detected')
        except:
            print ('Average interpolation not detected')
        try :
            inter_dict['Delaunay Triangles'] = inter_tri
            print ('Triangle interpolation detected')
        except:
            print ('Triangle interpolation not detected')
        try :
            inter_dict['DNN'] = inter_DNN
            print ('DNN interpolation detected')
        except:
            print ('DNN interpolation not detected')

        PlotComparison(inter_dict,opt.interpolate+'/interpolation/')

    #############################################################################################
    # Verification #
    #############################################################################################
    if opt.verification!='':
        print ('[INFO] Cross check verification with average')
        check_avg = EvaluateAverage(hist_dict, 20)
        print ('... Done')

        print ('[INFO] Cross check verification with triangles')
        check_tri = EvaluateTriangles(hist_dict)
        print ('... Done')

        print ('[INFO] Cross check verification with Neural Network')
        check_DNN = HyperVerif(hist_dict,scaler=scaler,path=opt.verification+'.zip')
        print ('... Done')

        check_dict = {'True':hist_dict,'Average':check_avg,'Delaunay Triangle':check_tri,'DNN':check_DNN}

        print ('[INFO] Comparison plots')
        PlotComparison(check_dict,opt.verification+'/verification/')
        print ('... Done')


if __name__ == "__main__":
    main()
