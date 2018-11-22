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

from ROOT import TGraph2D, TH2F, TCanvas, gStyle, gPad, gROOT
   
###############################################################################
# InterpolateTriangles #
###############################################################################
def InterpolateTriangles(hist_dict,eval_grid):
    """
    Performs the Delaunay triangle interpolation withg TGraph2D
    Inputs :
        - neighbours : hist_dict 
            points where rho distribution is know
                -> key = ('mH','mA') tuple
                -> value = np.array of six bins
        - eval_grid : list of list (nx2 elements)
            contains the points on which interpolation is to be done
    Outputs :
        - grid = dict 
            interpolated points
                -> key = ('mH','mA') tuple
                -> value = np.array of six bins
    """
    grid = {}

    # Separates each bin into lists of [x=mH,y=mA,z=bin_value] #
    bin1 = [(key[1],key[0],val[0]) for key,val in hist_dict.items()]
    bin2 = [(key[1],key[0],val[1]) for key,val in hist_dict.items()]
    bin3 = [(key[1],key[0],val[2]) for key,val in hist_dict.items()]
    bin4 = [(key[1],key[0],val[3]) for key,val in hist_dict.items()]
    bin5 = [(key[1],key[0],val[4]) for key,val in hist_dict.items()]
    bin6 = [(key[1],key[0],val[5]) for key,val in hist_dict.items()]
    # bin_ -> bin_[i][0] = x_i (mA)
    #      -> bin_[i][1] = y_i (mH)
    #      -> bin_[i][2] = z_1 (bin content) 
    InterpolateBin(bin1,eval_grid,'Bin1')
    InterpolateBin(bin2,eval_grid,'Bin2')
    InterpolateBin(bin3,eval_grid,'Bin3')
    InterpolateBin(bin4,eval_grid,'Bin4')
    InterpolateBin(bin5,eval_grid,'Bin5')
    InterpolateBin(bin6,eval_grid,'Bin6')
    

###############################################################################
# BuildGraph #
###############################################################################
def BuildGraph(coord):
    """
    Takes list of coordinates (x,y,z) and returns the TGraph2D
    Inputs :
        - coord : list
            list of (x,y,z) 
    Outputs :
        - graph : TGraph2D
            built from the coordinates 
    """
    graph = TGraph2D()    
    for i in range(0,len(coord)):
        graph.SetPoint(i,coord[i][0],coord[i][1],coord[i][2])

    return graph

###############################################################################
# InterpolateBin #
###############################################################################
def InterpolateBin(bin_coord,eval_list,name):
    """
    From a bin coordinates ((x,y,z) , makes the interpolation on (x,y) 
    Inputs :
        - bin_coord : list
            list of ((x,y,z) to learn interpolation 
        - eval_list : list
            list of (mH,mA)=(y,x) that will be interpolated using a TGraph2D
        - name : str
            name of the bin for display purposes
    Outputs :
        - output : list
            list of ((x,y,z) of interpolated z from (x,y)
    """ 
    graph = BuildGraph(bin_coord)

    PlotGraph(graph,eval_list,name)
    return

    z = []
    for x,y in bin_coord:
        #z.append(graph.Interpolate(x,y))
        z.append(graph.Interpolate(x[0],x[1]))
        print (x,graph.Interpolate(x[0]+10.2,x[1]+20.5),y)
        
    output = [(i[0],i[1],o) for i in eval_list for o in z]      
    print (output)
    #return output

###############################################################################
# PlotGraph #
###############################################################################
def PlotGraph(graph,eval_list,name):
    """
    Plots the graph interpolated on a grid for check
    Inputs :
        - graph : TGraph2D
            graph that has learned the interpolation 
        - eval_list : list
            list of (mA,mH)=(x,y) that will be interpolated using a TGraph2D
        - name : str
            name of the bin for display purposes
    Outputs : None
    Plots : plot the graph of (mA,mH,bin) from a grid
    """
    gStyle.SetOptStat("") #no stat frame
    gROOT.SetBatch(True) # No display
    
    # Generate grid #
    grid = [(x,y) for x in range(0,1000,10) for y in range(0,1000,10) if y>=x]

    # Generates TH2F #
    c1 = TCanvas( 'c1', 'MassPlane', 200, 10, 1200, 700 ) 
    graph_TH2 = TH2F('Interpolated graph','Contour Plot;M_{A} [GeV];M_{H} [GeV]',100,0,1000,100,0,1000)
    graph_points = TH2F('','',100,0,1000,100,0,1000)

    # Interpolate and fill graph #
    max_z = 0
    for x,y in grid:
        z = graph.Interpolate(x,y) # x = mA, y = mH
        graph_TH2.Fill(x,y,z)
        if max_z<z:
            max_z=z
    for x,y in eval_list:
        graph_points.Fill(x,y)

    # Plot graph #
    path = os.path.join(os.getcwd(),'Triangles/')
    if not os.path.exists(path):
        os.makedirs(path)

    graph_TH2.Draw('CONTZ')
    graph_points.Draw('p same')
    graph_points.SetMarkerStyle(5);
    graph_TH2.SetTitle('Contour plot : '+name+';M_{bb} [GeV];M_{llbb} [GeV]')
    graph_TH2.GetZaxis().SetTitle('DNN Output')
    graph_TH2.GetZaxis().SetRangeUser(0,max_z)
    gPad.SetRightMargin(0.15)
    gPad.SetLeftMargin(0.15)
    c1.Print(path+name+'.pdf')
    

    


