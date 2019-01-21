#! /bin/env python

import re
import math
import sys, os, json
import copy
import glob
import argparse

import ROOT
  
def get_options():
    parser = argparse.ArgumentParser(description='Computes the ellipse parameters with or without centroid fit')
    parser.add_argument('-fit','--fit', action='store_true', required=False, default=False,
        help='If option used, the script will try to find the pol2 fit coefficients and use them to fix the centroid in the fit') 
    parser.add_argument('-window','--window', action='store_true', required=False, default=False,
        help='If option used, the script will restrict the 2D fit to a window around the centroid (max peak or from the fit)')

    opt = parser.parse_args()                                                                                                                                                                           

    return opt  

def get_ratio(histo):
    in_ell = histo.GetBinContent(2)
    out_ell = histo.GetBinContent(1)
    return in_ell/(in_ell+out_ell)
    


def main():
    # Preparation # 
    opt = get_options()

    if opt.window and not opt.fit:
        print ('[ERROR] You need the fit to use the window')
        sys.exit(1)

    # Path to files #
    path_json ='fullEllipseParam_{}.json'
    path_histo = '../../factories_ZA/countEvents_base'
    out_json = 'ratioParam_{}.json'

    if opt.fit:
        path_json = path_json.replace('Param','ParamFit')
        out_json = out_json.replace('Param','ParamFit')
        path_histo = path_histo.replace('base','fit')
        if opt.window:
            path_json = path_json.replace('Param','ParamWindow')
            out_json = out_json.replace('Param','ParamWindow')
            path_histo = path_histo.replace('fit','fit_window')

    path_histo = os.path.join(path_histo,'slurm','output')

    # Loop over categories #
    for cat in ['MuMu','ElEl']:
        list_out = []
        with open(path_json.format(cat),'r') as f:
            data = json.load(f)
            print 'Looking at %s'%(path_json.format(cat))

        for idx,line in enumerate(data):
            # Loop over the ellipse conf in json file #
            mH = line[-1]
            mA = line[-2]

            # Find corresponding root file #
            mH_string = '%0.fp%02.f'%(math.modf(mH)[1],math.modf(mH)[0]*100)
            mA_string = '%0.fp%02.f'%(math.modf(mA)[1],math.modf(mA)[0]*100)
            
            file_histo = ''
            for rf in glob.glob(path_histo+'/*.root'):
                if rf.find(mH_string)!=-1 and rf.find(mA_string)!=-1:
                    file_histo = rf
                    break

            if file_histo == '':
                sys.exit('Wait, something is wrong ... I could not locate the histogram')

            # From this root file, get the correct histo (aka the one with index idx, because ordered according to json file #
            tfile = ROOT.TFile.Open(file_histo)
            for key in tfile.GetListOfKeys():
                h = key.ReadObj()
                if h.ClassName() == 'TH1F' and h.GetName().find(cat)!=-1:
                    num_histo = int(re.findall(r'\d+',h.GetName())[0])
                    if idx == num_histo: # Found the correct histogram
                        #c1 = ROOT.TCanvas()
                        #h.Draw()
                        ratio = get_ratio(h)
                        #c1.Close()
                        break

            list_out.append([mA,mH,ratio])

                        
        with open(out_json.format(cat), 'w') as fp:
            json.dump(list_out, fp)
            print 'Dumped results at %s'%(out_json.format(cat))



if __name__ == "__main__":                                                                                                                                                                              
    main()  
    
   
