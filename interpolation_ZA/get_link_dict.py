import math
import glob
import os
import re
import json
import sys
import copy
import pprint

#################################################################################################
# GetHistDictOld #
#################################################################################################
def GetHistDictOld(part_type):
    """ 
        Get Matching between hist number and filename
        Inputs :
            - part_type : str (MuMu or ElEl)
                Type of decay from the Z boson
        Returns 
            - dico 
                key = filename (.root)
                value = hist name (TH1)
    """
    if part_type != 'MuMu' and part_type != 'ElEl':
        print("The particle type is not correct in GetHistDictOld")        
        sys.exit(1)
    dico = {}
    dico ['HToZATo2L2B_MH-1000_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_0.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_18'
    dico ['HToZATo2L2B_MH-1000_MA-500_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_1.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_17'
    dico ['HToZATo2L2B_MH-1000_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_2.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_19'
    dico ['HToZATo2L2B_MH-200_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_3.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_1'
    dico ['HToZATo2L2B_MH-200_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_4.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_6'
    dico ['HToZATo2L2B_MH-250_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_5.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_20'
    dico ['HToZATo2L2B_MH-250_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_6.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_15'
    dico ['HToZATo2L2B_MH-300_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_7.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_2'
    dico ['HToZATo2L2B_MH-300_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_8.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_3'
    dico ['HToZATo2L2B_MH-300_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_9.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_0'
    dico ['HToZATo2L2B_MH-500_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_10.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_7'
    dico ['HToZATo2L2B_MH-500_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_11.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_8'
    dico ['HToZATo2L2B_MH-500_MA-300_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_12.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_9'
    dico ['HToZATo2L2B_MH-500_MA-400_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_13.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_10'
    dico ['HToZATo2L2B_MH-500_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_14.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_5'
    dico ['HToZATo2L2B_MH-650_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_15.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_4'
    dico ['HToZATo2L2B_MH-800_MA-100_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_16.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_11'
    dico ['HToZATo2L2B_MH-800_MA-200_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_17.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_12'
    dico ['HToZATo2L2B_MH-800_MA-400_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_18.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_13'
    dico ['HToZATo2L2B_MH-800_MA-50_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_19.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_16'
    dico ['HToZATo2L2B_MH-800_MA-700_13TeV-madgraph_Summer16MiniAODv2_v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a_histos_20.root'] = 'rho_steps_histo_'+part_type+'_hZA_lljj_deepCSV_btagM_mll_and_met_cut_14'
    
    return dico

#################################################################################################
# GetHistDictNew#
#################################################################################################
def GetHistDict(path_files,path_json,cat):
    dico = {}
    with open(path_json.format(cat)) as data_file:    
        config = json.load(data_file) # List of list of ellipse config
    
    basic_histo = 'rho_steps_histo_{}_hZA_lljj_deepCSV_btagM_mll_and_met_cut_{}'

    for idx, line in enumerate(config):
        # Value of dict -> Name of histo from jason file #
        histo = copy.copy(basic_histo.format(cat,idx))

        # key of dict -> Must find corresponding root file #
        for f in glob.glob(path_files+'/*.root'):
            p = re.compile(r'\d+[p]\d+')
            mH = float(p.findall(f)[0].replace('p','.'))
            mA = float(p.findall(f)[1].replace('p','.'))
            if mH == line[-1] and mA == line[-2]:
                dico[f.replace(path_files,'')] = histo
                break # Found, let's not waste any more time
                
    return dico
        
