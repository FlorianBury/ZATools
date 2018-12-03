#! /bin/env python

import sys, os, json
import copy
import numpy as np

import ROOT

from computeEllipseParameters import getMassAndWidth

# Plots the m_lljj and m_jj distributions for each mass point with the associated Gaussian fit

def main():

    path = "/home/ucl/cp3/fbury/cp3_llbb/ZATools/factories_ZA/test_for_signal/slurm/output/"

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    
    first_plot = True

    for inputfile in os.listdir(path):
        if inputfile.startswith("HToZA") and inputfile.endswith(".root"):
            c1 = ROOT.TCanvas( 'c1', 'dist', 200, 10, 1200, 700 )

            if first_plot: # Starts the canvas pdf for the first plot
                c1.Print('dist.pdf[')
                first_plot = False

            #Get the simulated masses: MA and MH
            print ('-'*80)
            splitPath = inputfile.split('/')
            filename = splitPath[-1]
            print filename
            splitFilename = filename.replace('_', '-').split('-')
            MH = int(splitFilename[2])
            MA = int(splitFilename[4])
            print "MH: ", MH
            print "MA: ", MA

            print str(path+inputfile)
            inputs = ROOT.TFile(path+inputfile,"READ")

            # Get the histograms #

            hist_MuMu_m_jj = inputs.Get("jj_M_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut")
            hist_ElEl_m_jj = inputs.Get("jj_M_ElEl_hZA_lljj_deepCSV_btagM_mll_and_met_cut")
            hist_MuMu_m_lljj = inputs.Get("lljj_M_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut")
            hist_ElEl_m_lljj = inputs.Get("lljj_M_ElEl_hZA_lljj_deepCSV_btagM_mll_and_met_cut")

            # Esthetic choices #

            hist_MuMu_m_jj.SetMinimum(0)
            hist_MuMu_m_lljj.SetMinimum(0)
            hist_ElEl_m_jj.SetMinimum(0)
            hist_ElEl_m_lljj.SetMinimum(0)

            hist_MuMu_m_jj.GetXaxis().SetRangeUser(0,MA+200)
            hist_MuMu_m_lljj.GetXaxis().SetRangeUser(0,MH+300)
            hist_ElEl_m_jj.GetXaxis().SetRangeUser(0,MA+200)
            hist_ElEl_m_lljj.GetXaxis().SetRangeUser(0,MH+300)

            hist_MuMu_m_jj.SetLineColor(ROOT.kGreen+2)
            hist_MuMu_m_lljj.SetLineColor(ROOT.kGreen+2)
            hist_ElEl_m_jj.SetLineColor(ROOT.kRed+2)
            hist_ElEl_m_lljj.SetLineColor(ROOT.kRed+2)

            hist_MuMu_m_jj.SetLineWidth(2)
            hist_MuMu_m_lljj.SetLineWidth(2)
            hist_ElEl_m_jj.SetLineWidth(2)
            hist_ElEl_m_lljj.SetLineWidth(2)

            # Generate legend #

            legend_jj = ROOT.TLegend(0.7,0.80,0.9,0.9)
            legend_jj.SetHeader("Legend")
            legend_jj.AddEntry(hist_ElEl_m_jj,"ElEl")
            legend_jj.AddEntry(hist_MuMu_m_jj,"MuMu")


            legend_lljj = ROOT.TLegend(0.7,0.80,0.9,0.9)
            legend_lljj.SetHeader("Legend")
            legend_lljj.AddEntry(hist_ElEl_m_lljj,"ElEl")
            legend_lljj.AddEntry(hist_MuMu_m_lljj,"MuMu")

            # Recover the Fit #

            fit_MuMu_jj = getMassAndWidth(hist_MuMu_m_jj,MA)
            fit_MuMu_lljj = getMassAndWidth(hist_MuMu_m_lljj,MH)
            fit_ElEl_jj = getMassAndWidth(hist_ElEl_m_jj,MA)
            fit_ElEl_lljj = getMassAndWidth(hist_ElEl_m_lljj,MH)
            # fit contains (m_reco, sigma, pvalue, fit_hist) 

            fit_MuMu_jj[3].SetLineColor(ROOT.kGreen+2)
            fit_MuMu_lljj[3].SetLineColor(ROOT.kGreen+2)
            fit_ElEl_jj[3].SetLineColor(ROOT.kRed+2)
            fit_ElEl_lljj[3].SetLineColor(ROOT.kRed+2)

            c1.Clear() # clear the fits from canvas

            # Generates Pads and title #

            pad1 = ROOT.TPad( 'pad1', 'm_llbb', 0.03, 0.10, 0.50, 0.85)
            pad2 = ROOT.TPad( 'pad2', 'm_bb', 0.53, 0.10, 0.98, 0.85)
            pad1.Draw()
            pad2.Draw()
            ROOT.SetOwnership(c1, False) # otherwise pyroot crashes, needed for the garbage collector
            ROOT.SetOwnership(pad1, False)
            ROOT.SetOwnership(pad2, False)
            title = ROOT.TPaveText( .3, 0.9, .7, .99 )
            title.SetFillColor(0)
            title.AddText('M_{H} = %0.f GeV, M_{A} = %0.f GeV)'%(MH,MA))
            title.Draw()

            # m_jj #
            pad1.cd()
            hist_MuMu_m_jj.Draw()
            hist_MuMu_m_jj.SetTitle('M_{jj}')
            hist_ElEl_m_jj.Draw("same")
            fit_MuMu_jj[3].Draw("same")
            fit_ElEl_jj[3].Draw("same")
            legend_jj.Draw()
            

            # m_lljj #
            pad2.cd()
            hist_MuMu_m_lljj.Draw()
            hist_MuMu_m_lljj.SetTitle('M_{lljj}')
            hist_ElEl_m_lljj.Draw("same")
            fit_MuMu_lljj[3].Draw("same")
            fit_ElEl_lljj[3].Draw("same")
            legend_lljj.Draw()

            c1.Update()
            #raw_input('Press key')
            c1.Print('dist.pdf')
                
            c1.Clear()
            #c1.Close()
            
    c1.Print('dist.pdf]')
        
    
if __name__ == "__main__":
    main()  



