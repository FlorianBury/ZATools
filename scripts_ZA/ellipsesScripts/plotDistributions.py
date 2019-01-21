#! /bin/env python

import math
import time
import sys, os, json
import copy
import argparse
import random

import ROOT

from computeEllipseParameters import getMassAndWidth
from cutWindow import massWindow
import tdrstyle


def getBinsAndNumber(histo):
    N = 0
    all_entries = []
    for x in range(1,histo.GetNbinsX() + 1):            # Loop over bins in x axis
        for y in range(1,histo.GetNbinsY() + 1):        # Loop over bins in y axis
            x_low = histo.GetXaxis().GetBinLowEdge(x)
            x_high = histo.GetXaxis().GetBinUpEdge(x)
            y_low = histo.GetXaxis().GetBinLowEdge(y)
            y_high = histo.GetXaxis().GetBinUpEdge(y)
            z_val = histo.GetBinContent(x,y)
            # Generate four tuples of corners of the bin + z_val #
            entry = [[(x_low,y_low),(x_low,y_high),(x_high,y_high),(x_high,y_low)],z_val]
            all_entries.append(entry)
            N += histo.GetBinContent(x,y)*((x_high-x_low)*(y_high-y_low))
    return N,all_entries

def binArea(corners):
    # corner = [(LxLy),(LxUy),(UxUy),(UxLy)], U= upper, L= lower
    Ux = corners[2][0]
    Lx = corners[0][0]
    Uy = corners[1][1]
    Ly = corners[0][1]
    Area = (Ux-Lx)*(Uy-Ly)
    return Area
    
def subDivideBin(corners,z):
    # corner = [(LxLy),(LxUy),(UxUy),(UxLy)], U= upper, L= lower
    Ux = corners[2][0]
    Lx = corners[0][0]
    Uy = corners[1][1]
    Ly = corners[0][1]

    a = (Lx,Uy)
    b = ((Ux-Lx)/2+Lx,Uy)
    c = (Ux,Uy)
    d = (Lx,(Uy-Ly)/2+Ly)
    e = ((Ux-Lx)/2+Lx,(Uy-Ly)/2+Ly)
    f = (Ux,(Uy-Ly)/2+Ly)
    g = (Lx,Ly)
    h = ((Ux-Lx)/2+Lx,Ly)
    i = (Ux,Ly)

    corner1 = [[d,a,b,e],z]
    corner2 = [[e,b,c,f],z]
    corner3 = [[g,d,e,h],z]
    corner4 = [[h,e,f,i],z]
    
    new_corners = [corner1,corner2,corner3,corner4] 
    return new_corners

    # LxUy a---b---c  UxUy
    #      |-1-|-2-|  
    #      d---e---f  
    #      |-3-|-4-|  
    # LxLy g---h---i  UxLy

def evaluateSum(array_bin,center,path_json):
    tot = 0
    first = True
    instance = massWindow(path_json)
    for corners,z in array_bin:
        for corner in corners:
            if first:
                print ('Current bin size : %0.5f\tNumber of bins : %d'%(binArea(corners),len(array_bin)))
                first = False
            In = instance.isInWindow(center,1,corner)
            if In:
                tot += binArea(corners)*z
                break
    return tot
 

def countInEllipse(array_bin,center,path_json):
    previous_tot = 0
    corners = array_bin
    while True:
        new_tot = evaluateSum(corners,center,path_json)
        if new_tot != 0:
            print ('Previous sum : %0.2E\tCurrent result : %0.2E \tDifference : %0.2f%%'%(previous_tot,new_tot,math.fabs(previous_tot-new_tot)*100/new_tot))
            if math.fabs(previous_tot-new_tot)/new_tot<0.25:
                break
        else:
            print ('Previous sum : %0.2E\tCurrent result : %0.2E'%(previous_tot,new_tot))
        print ('Too many variations in the sum, increased number of bins')
        previous_tot = copy.copy(new_tot)
        new_corners = []
        for corner,z in corners:
            new_corners += subDivideBin(corner,z)
        corners = copy.deepcopy(new_corners)
        if len(corners)>1500000:
            break

    return new_tot

def MC_countInEllipse(center,path_json,histo,n_points=100000):
    instance = massWindow(path_json)
    xmax = histo.GetXaxis().GetBinUpEdge(histo.GetNbinsX())
    ymax = histo.GetYaxis().GetBinUpEdge(histo.GetNbinsY())
    sum_tot = 0
    sum_in = 0
    for i in range(0,n_points):
        # Process print #
        #time.sleep(0.1) 
        sys.stdout.write('Current Process : %0.2f%%\r'%(i*100/n_points))
        sys.stdout.flush()
        # Generate a point #
        x = random.uniform(0,xmax)
        y = random.uniform(x,ymax) # We only want the upper triangle

        # Check if inside ellipse #
        in_ellipse = instance.isInWindow(center,1,(x,y))

        # Get bin content #
        binx = histo.GetXaxis().FindBin(x)
        biny = histo.GetXaxis().FindBin(y)
        z = histo.GetBinContent(binx,biny)

        if in_ellipse:
            sum_in += z
        sum_tot += z

    integral = histo.Integral(0,histo.GetNbinsX(),0,histo.GetNbinsY()) 

    print (('%s'%(path_json)).ljust(40,'.')+' Finished MC : in = %0.2E / tot = %0.2E \tRatio = %0.5f%%\tIntegral = %0.2E'%(sum_in,sum_tot,sum_in*100/sum_tot,integral))
    if sum_in*100/sum_tot<10:  
        print ('[WARNING] Something wrong')

    return sum_in/sum_tot

def get_ratio(MA,MH,path_json):
    with open (path_json,'r') as f:
        data = json.load(f)
        ratio = -1
        for line in data:
            if line[0]==MA and line[1]==MH:
                ratio = line[2]
    if ratio == -1:
        print '[ERROR] Could not match config in %s'%path_json
    return ratio
    
    

def getConfig(ellipse,MH,MA): # get ellipse params corresponding to MH and MA
    for line in ellipse:
        if MH == line[-1] and MA == line[-2]:
            params = line
            break
    # params = [mbb,mllbb,a,b,theta,MA,MH]
    return params

def generateEllipse(ellipse,MH,MA,rho=1): # Generates the ellipse given MH and MA
    params = getConfig(ellipse,MH,MA)
    t = params[4] * 57.29 # radians -> Degrees
    ell = ROOT.TEllipse(params[0],params[1],rho*math.sqrt(params[2]),rho*math.sqrt(params[3]),0,360,t)
    ell.SetFillStyle(0)
    ell.SetLineWidth(2)
    return ell

   
def get_options():
    parser = argparse.ArgumentParser(description='Computes the ellipse parameters with or without centroid fit')
    parser.add_argument('-fit','--fit', action='store_true', required=False, default=False,
        help='If option used, the script will try to find the pol2 fit coefficients and use them to fix the centroid in the fit') 
    parser.add_argument('-window','--window', action='store_true', required=False, default=False,
        help='If option used, the script will restrict the 2D fit to a window around the centroid (max peak or from the fit)')
    parser.add_argument('-count','--count', action='store_true', required=False, default=False,
        help='Wether to compute the number of events in each ellipse (takes some time)')

    opt = parser.parse_args()                                                                                                                                                                           

    return opt  

# Plots the m_lljj and m_jj distributions for each mass point with the associated Gaussian fit

def main():
    # Preparation # 
    opt = get_options()

    if opt.window and not opt.fit:
        print ('[ERROR] You need the fit to use the window')
        sys.exit(1)

    tdrstyle.setTDRStyle()

    #path = "/home/ucl/cp3/fbury/cp3_llbb/ZATools/factories_ZA/test_for_signal/slurm/output/"
    path = "/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/new2018prod_massHistosForEllParametrization/slurm/output/"

    ROOT.gROOT.SetBatch(True)
    
    # generate pdf #
    TH1_name = 'dist.pdf'
    TH2_name = 'ellipse.pdf'
    if opt.fit:
        TH1_name = TH1_name.replace('dist','dist_centroid_fit')
        TH2_name = TH2_name.replace('ellipse','ellipse_centroid_fit')
    if opt.window:
        TH2_name = TH2_name.replace('ellipse','ellipse_window')

    # Extract ellipse config from json #
    with open('fullEllipseParam_ElEl.json', "r") as read_file:
        ellipse_ElEl = json.load(read_file)
    if opt.fit:
        with open('fullEllipseParamFit_ElEl.json', "r") as read_file:
            ellipse_fit_ElEl = json.load(read_file)
        if opt.window:
            with open('fullEllipseParamWindowFit_ElEl.json', "r") as read_file:
                ellipse_fit_window_ElEl = json.load(read_file)
    with open('fullEllipseParam_MuMu.json', "r") as read_file:
        ellipse_MuMu = json.load(read_file)
    if opt.fit:
        with open('fullEllipseParamFit_MuMu.json', "r") as read_file:
            ellipse_fit_MuMu = json.load(read_file)
        if opt.window:
            with open('fullEllipseParamWindowFit_MuMu.json', "r") as read_file:
                ellipse_fit_window_MuMu = json.load(read_file)
        
    c1 = ROOT.TCanvas( 'c1', 'dist', 200, 10, 1200, 700 )
    c2 = ROOT.TCanvas( 'c2', 'ellipse', 200, 10, 1200, 700 )
    c1.Print(TH1_name+'[')
    c2.Print(TH2_name+'[')

    # Loop over files #
    for inputfile in os.listdir(path):
        if inputfile.startswith("HToZA") and inputfile.endswith(".root"):

            #Get the simulated masses: MA and MH
            print ('-'*80)
            splitPath = inputfile.split('/')
            filename = splitPath[-1]
            print (filename)
            splitFilename = filename.replace('_', '-').split('-')
            #MH = int(splitFilename[2])
            #MA = int(splitFilename[4])
            MH = float(splitFilename[1].replace('p','.'))                                                                                                                                           
            MA = float(splitFilename[2].replace('p','.'))
            print ("MH: ", MH)
            print ("MA: ", MA)
            print (str(path+inputfile))
            inputs = ROOT.TFile(path+inputfile,"READ")
                
            ##########################################################################################
            # TH1 distributions -> m_llbb and m_bb #
            ##########################################################################################

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

            legend_jj = ROOT.TLegend(0.6,0.75,0.9,0.85)
            legend_jj.SetHeader("Legend")
            legend_jj.AddEntry(hist_ElEl_m_jj,"H #rightarrow ZA #rightarrow e^{+} e^{-} #bar{b} b")
            legend_jj.AddEntry(hist_MuMu_m_jj,"H #rightarrow ZA #rightarrow #mu^{+} #mu^{-} #bar{b} b")


            legend_lljj = ROOT.TLegend(0.6,0.75,0.9,0.85)
            legend_lljj.SetHeader("Legend")
            legend_lljj.AddEntry(hist_ElEl_m_lljj,"H #rightarrow ZA #rightarrow e^{+} e^{-} #bar{b} b")
            legend_lljj.AddEntry(hist_MuMu_m_lljj,"H #rightarrow ZA #rightarrow #mu^{+} #mu^{-} #bar{b} b")

            # Recover the Fit #

            fit_MuMu_jj = getMassAndWidth(hist_MuMu_m_jj,MA,cat='mA_MuMu',centroid='mbb',use_fit=opt.fit)
            fit_MuMu_lljj = getMassAndWidth(hist_MuMu_m_lljj,MH,cat='mH_MuMu',centroid='mllbb',use_fit=opt.fit)
            fit_ElEl_jj = getMassAndWidth(hist_ElEl_m_jj,MA,cat='mA_ElEl',centroid='mbb',use_fit=opt.fit)
            fit_ElEl_lljj = getMassAndWidth(hist_ElEl_m_lljj,MH,cat='mH_ElEl',centroid='mllbb',use_fit=opt.fit)
            # fit contains (m_reco, sigma, pvalue, fit_hist) 

            fit_MuMu_jj[3].SetLineColor(ROOT.kGreen+2)
            fit_MuMu_lljj[3].SetLineColor(ROOT.kGreen+2)
            fit_ElEl_jj[3].SetLineColor(ROOT.kRed+2)
            fit_ElEl_lljj[3].SetLineColor(ROOT.kRed+2)

            c1.Clear() # clear the fits from canvas

            # Generates Pads and title #
            pad11 = ROOT.TPad( 'pad1', 'm_llbb', 0.03, 0.10, 0.50, 0.85)
            pad12 = ROOT.TPad( 'pad2', 'm_bb', 0.53, 0.10, 0.98, 0.85)
            pad11.Draw()
            pad12.Draw()
            ROOT.SetOwnership(c1, False) # otherwise pyroot crashes, needed for the garbage collector
            ROOT.SetOwnership(pad11, False)
            ROOT.SetOwnership(pad12, False)
            title = ROOT.TPaveText( .3, 0.88, .7, .99 )
            title.SetFillColor(0)
            title.SetBorderSize(1)
            title.AddText('M_{H} = %0.2f GeV, M_{A} = %0.2f GeV)'%(MH,MA))
            title.Draw()

            # m_jj #
            pad11.cd()
            hist_MuMu_m_jj.Draw()
            hist_MuMu_m_jj.SetTitle('M_{jj};M_{jj} [GeV]')
            hist_ElEl_m_jj.Draw("same")
            fit_MuMu_jj[3].Draw("same")
            fit_ElEl_jj[3].Draw("same")
            legend_jj.Draw()
            

            # m_lljj #
            pad12.cd()
            hist_MuMu_m_lljj.Draw()
            hist_MuMu_m_lljj.SetTitle('M_{lljj};M_{lljj} [GeV]')
            hist_ElEl_m_lljj.Draw("same")
            fit_MuMu_lljj[3].Draw("same")
            fit_ElEl_lljj[3].Draw("same")
            legend_lljj.Draw()

            # Update and save Canvas #
            c1.Update()
            c1.Print(TH1_name)
            c1.Clear()

            ##########################################################################################
            # TH2 distributions -> Ellipses #
            ##########################################################################################
            c2.cd()
            # Get 2D hist #
            mass2D_ElEl = inputs.Get("Mjj_vs_Mlljj_ElEl_hZA_lljj_deepCSV_btagM_mll_and_met_cut")
            mass2D_MuMu = inputs.Get("Mjj_vs_Mlljj_MuMu_hZA_lljj_deepCSV_btagM_mll_and_met_cut")
               
            #N_ElEl, bins_ElEl = getBinsAndNumber(mass2D_ElEl)
            #N_MuMu, bins_MuMu = getBinsAndNumber(mass2D_MuMu)


            # Get numbers in Ellipses #
            if opt.count:
                ratio_ElEl = get_ratio(MA,MH,'ratioParam_ElEl.json')
                ratio_fit_ElEl = get_ratio(MA,MH,'ratioParamFit_ElEl.json')
                ratio_fit_window_ElEl = get_ratio(MA,MH,'ratioParamWindowFit_ElEl.json')
                ratio_MuMu = get_ratio(MA,MH,'ratioParam_MuMu.json')
                ratio_fit_MuMu = get_ratio(MA,MH,'ratioParamFit_MuMu.json')
                ratio_fit_window_MuMu = get_ratio(MA,MH,'ratioParamWindowFit_MuMu.json')
                                

            # pseParam_MuMu.jsonRange #
            mass2D_ElEl.GetXaxis().SetRangeUser(0,MH*2)
            mass2D_MuMu.GetXaxis().SetRangeUser(0,MH*2)
            mass2D_ElEl.GetYaxis().SetRangeUser(0,MH*2)
            mass2D_MuMu.GetYaxis().SetRangeUser(0,MH*2)


            # Generate Legend #
            legend_ElEl = ROOT.TLegend(0.65,0.38,0.9,0.85)
            legend_ElEl.SetHeader("Legend")
            legend_ElEl.SetTextFont(42)
            legend_ElEl.SetTextSize(0.023)

            legend_MuMu = ROOT.TLegend(0.65,0.38,0.9,0.85)
            legend_MuMu.SetHeader("Legend")
            legend_MuMu.SetTextFont(42)
            legend_MuMu.SetTextSize(0.023)
 
            # Generates Pads and title #
            c2.Clear()
            pad21 = ROOT.TPad( 'pad1', 'MuMu', 0.03, 0.10, 0.50, 0.85)
            pad22 = ROOT.TPad( 'pad2', 'ElEl', 0.53, 0.10, 0.98, 0.85)
            pad21.Draw()
            pad22.Draw()
            ROOT.SetOwnership(c2, False) # otherwise pyroot crashes, needed for the garbage collector
            ROOT.SetOwnership(pad21, False)
            ROOT.SetOwnership(pad22, False)
            title = ROOT.TPaveText( .3, 0.88, .7, .99 )
            title.SetFillColor(0)
            title.SetBorderSize(1)
            title.AddText('M_{H} = %0.2f GeV, M_{A} = %0.2f GeV)'%(MH,MA))
            title.Draw()

            # Draw MuMu on Canvas #
            pad21.cd()
            mass2D_MuMu.Draw('COLZ')
            mass2D_MuMu.SetTitle('H #rightarrow ZA #rightarrow #mu^{+} #mu^{-} #bar{b} b;M_{bb} ; M_{llbb}')
            #mass2D_MuMu.GetXaxis().SetLabelOffset(1.4)
            #mass2D_MuMu.GetYaxis().SetLabelOffset(1.4)

            TEllipse_MuMu = [0,0,0]
            TEllipse_fit_MuMu = [0,0,0]
            TEllipse_fit_window_MuMu = [0,0,0]
            for i in range(1,4):
                TEllipse_MuMu[i-1] = generateEllipse(ellipse_MuMu,MH,MA,rho=i)
                TEllipse_MuMu[i-1].SetLineColor(ROOT.kGreen-4)
                TEllipse_MuMu[i-1].SetLineStyle(1+(i-1)*2)
                TEllipse_MuMu[i-1].Draw("same")
                legend_MuMu.AddEntry(0,"","")
                legend_MuMu.AddEntry(0,"#rho = %d"%i,'')
                if i==1 and opt.count:
                    legend_MuMu.AddEntry(0,"","")
                    legend_MuMu.AddEntry(TEllipse_MuMu[i-1],"#splitline{Ellipse}{Ratio = %0.2f%%}"%(ratio_MuMu*100),"l")
                    legend_MuMu.AddEntry(0,"","")
                else:   
                    legend_MuMu.AddEntry(TEllipse_MuMu[i-1],"Ellipse","l")

                if opt.fit:
                    TEllipse_fit_MuMu[i-1] = generateEllipse(ellipse_fit_MuMu,MH,MA,rho=i)
                    TEllipse_fit_MuMu[i-1].SetLineColor(ROOT.kRed+1)
                    TEllipse_fit_MuMu[i-1].SetLineStyle(1+(i-1)*2)
                    TEllipse_fit_MuMu[i-1].Draw("same")
                    if i==1 and opt.count:
                        legend_MuMu.AddEntry(0,"","")
                        legend_MuMu.AddEntry(TEllipse_fit_MuMu[i-1],"#splitline{With centroid fit}{Ratio = %0.2f%%}"%(ratio_fit_MuMu*100),"l")
                        legend_MuMu.AddEntry(0,"","")
                    else:
                        legend_MuMu.AddEntry(TEllipse_fit_MuMu[i-1],"With centroid fit","l")
                    if opt.window:
                        TEllipse_fit_window_MuMu[i-1] = generateEllipse(ellipse_fit_window_MuMu,MH,MA,rho=i)
                        TEllipse_fit_window_MuMu[i-1].SetLineColor(ROOT.kOrange)
                        TEllipse_fit_window_MuMu[i-1].SetLineStyle(1+(i-1)*2)
                        TEllipse_fit_window_MuMu[i-1].Draw("same")
                        if i==1 and opt.count:
                            legend_MuMu.AddEntry(0,"","")
                            legend_MuMu.AddEntry(TEllipse_fit_window_MuMu[i-1],"#splitline{With 2D Window}{Ratio = %0.2f%%}"%(ratio_fit_window_MuMu*100),"l")
                            legend_MuMu.AddEntry(0,"","")
                        else:
                            legend_MuMu.AddEntry(TEllipse_fit_window_MuMu[i-1],"With 2D Window","l")
            legend_MuMu.Draw("same")

            # Draw ElEl on Canvas #
            pad22.cd()
            mass2D_ElEl.Draw('COLZ')
            mass2D_ElEl.SetTitle('H #rightarrow ZA #rightarrow e^{+} e^{-} #bar{b} b;M_{bb} ; M_{llbb}')

            TEllipse_ElEl = [0,0,0]
            TEllipse_fit_ElEl = [0,0,0]
            TEllipse_fit_window_ElEl = [0,0,0]
            for i in range(1,4):
                TEllipse_ElEl[i-1] = generateEllipse(ellipse_ElEl,MH,MA,rho=i)
                TEllipse_ElEl[i-1].SetLineColor(ROOT.kGreen-4)
                TEllipse_ElEl[i-1].SetLineStyle(1+(i-1)*2)
                TEllipse_ElEl[i-1].Draw("same")
                legend_ElEl.AddEntry(0,"","")
                legend_ElEl.AddEntry(0,"#rho = %d"%i,'')
                if i==1 and opt.count:
                    legend_ElEl.AddEntry(0,"","")
                    legend_ElEl.AddEntry(TEllipse_ElEl[i-1],"#splitline{Ellipse}{Ratio = %0.2f%%}"%(ratio_ElEl*100),"l")
                    legend_ElEl.AddEntry(0,"","")
                else:   
                    legend_ElEl.AddEntry(TEllipse_ElEl[i-1],"Ellipse","l")

                if opt.fit:
                    TEllipse_fit_ElEl[i-1] = generateEllipse(ellipse_fit_ElEl,MH,MA,rho=i)
                    TEllipse_fit_ElEl[i-1].SetLineColor(ROOT.kRed+1)
                    TEllipse_fit_ElEl[i-1].SetLineStyle(1+(i-1)*2)
                    TEllipse_fit_ElEl[i-1].Draw("same")
                    if i==1 and opt.count:
                        legend_ElEl.AddEntry(0,"","")
                        legend_ElEl.AddEntry(TEllipse_fit_ElEl[i-1],"#splitline{With centroid fit}{Ratio = %0.2f%%}"%(ratio_fit_ElEl*100),"l")
                        legend_ElEl.AddEntry(0,"","")
                    else:
                        legend_ElEl.AddEntry(TEllipse_fit_ElEl[i-1],"With centroid fit","l")
                    if opt.window:
                        TEllipse_fit_window_ElEl[i-1] = generateEllipse(ellipse_fit_window_ElEl,MH,MA,rho=i)
                        TEllipse_fit_window_ElEl[i-1].SetLineColor(ROOT.kOrange)
                        TEllipse_fit_window_ElEl[i-1].SetLineStyle(1+(i-1)*2)
                        TEllipse_fit_window_ElEl[i-1].Draw("same")
                        if i==1 and opt.count:
                            legend_ElEl.AddEntry(0,"","")
                            legend_ElEl.AddEntry(TEllipse_fit_window_ElEl[i-1],"#splitline{With 2D Window}{Ratio = %0.2f%%}"%(ratio_fit_window_ElEl*100),"l")
                            legend_ElEl.AddEntry(0,"","")
                        else:
                            legend_ElEl.AddEntry(TEllipse_fit_window_ElEl[i-1],"With 2D Window","l")
            legend_ElEl.Draw("same")


            
            # Update and save Canvas #
            c2.Update()
            c2.Print(TH2_name)
            c2.Clear()


            
    c1.Print(TH1_name+']')
    c2.Print(TH2_name+']')
        
    
if __name__ == "__main__":
    main()  



