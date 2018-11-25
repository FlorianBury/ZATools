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
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
        - eval_grid : list of list (nx2 elements)
            contains the points on which interpolation is to be done
    Outputs :
        - grid = dict 
            interpolated points
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
    """
    # Separates each bin into lists of [x=mH,y=mA,z=bin_value] #
    bin1 = [(key[0],key[1],val[0]) for key,val in hist_dict.items()]
    bin2 = [(key[0],key[1],val[1]) for key,val in hist_dict.items()]
    bin3 = [(key[0],key[1],val[2]) for key,val in hist_dict.items()]
    bin4 = [(key[0],key[1],val[3]) for key,val in hist_dict.items()]
    bin5 = [(key[0],key[1],val[4]) for key,val in hist_dict.items()]
    bin6 = [(key[0],key[1],val[5]) for key,val in hist_dict.items()]
    # bin_ -> bin_[i][0] = x_i (mA)
    #      -> bin_[i][1] = y_i (mH)
    #      -> bin_[i][2] = z_1 (bin content) 
    # Interpolation #
    print ('='*80)
    print ('Starting interpolation on bin 1')
    out1 = InterpolateBin(bin1,eval_grid,'Bin1')
    print ('-'*80)
    print ('Starting interpolation on bin 2')
    out2 = InterpolateBin(bin2,eval_grid,'Bin2')
    print ('-'*80)
    print ('Starting interpolation on bin 3')
    out3 = InterpolateBin(bin3,eval_grid,'Bin3')
    print ('-'*80)
    print ('Starting interpolation on bin 4')
    out4 = InterpolateBin(bin4,eval_grid,'Bin4')
    print ('-'*80)
    print ('Starting interpolation on bin 5')
    out5 = InterpolateBin(bin5,eval_grid,'Bin5')
    print ('-'*80)
    print ('Starting interpolation on bin 6')
    out6 = InterpolateBin(bin6,eval_grid,'Bin6')
    print ('-'*80)

    
    # Concatenation and dict #
    print ('Concatenation of the outputs')
    grid = {}
    for i in range(0,len(out1)):
        if out1[i][0]!=out2[i][0] or out1[i][0]!=out3[i][0] or out1[i][0]!=out4[i][0] or out1[i][0]!=out5[i][0] or out1[i][0]!=out6[i][0]:
            print ('[WARNING] m_A are not compatible !')
        if out1[i][1]!=out2[i][1] or out1[i][1]!=out3[i][1] or out1[i][1]!=out4[i][1] or out1[i][1]!=out5[i][1] or out1[i][1]!=out6[i][1]:
            print ('[WARNING] m_H are not compatible !')
        arr = np.array([out1[i][2],out2[i][2],out3[i][2],out4[i][2],out5[i][2],out6[i][2]])
        grid[(out1[i][0],out1[i][1])] = arr

    return grid

    #out = [(x1,y1,z1,z2,z3,z4,z5,z6) for x1,y1,z1 in out1 
    #                                 for x2,y2,z2 in out2
    #                                 for x3,y3,z3 in out3
    #                                 for x4,y4,z4 in out4
    #                                 for x5,y5,z5 in out5
    #                                 for x6,y6,z6 in out6]
                                     #if x1==x2 and x2==x3 and x3==x4 and x4==x5 and x5==x6
                                     #if y1==y2 and y2==y3 and y3==y4 and y4==y5 and y5==y6
                                     #if z1==z2 and z2==z3 and z3==z4 and z4==z5 and z5==z6]
    #grid = dict(((o[0],o[1]),(o[2],o[3],o[4],o[5])) for o in out)
                            
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
    graph.SetNpx(500)
    graph.SetNpy(500)

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
            list of (mA,mH)=(x,y) that will be interpolated using a TGraph2D
        - name : str
            name of the bin for display purposes
    Outputs :
        - output : list
            list of ((x,y,z) of interpolated z from (x,y)
    """ 
    graph = BuildGraph(bin_coord)

    PlotGraph(graph,eval_list,name)

    z = []
    for x,y in eval_list:
        z.append(graph.Interpolate(x,y))

    output = [(a[0][0],a[0][1],a[1]) for a in zip(eval_list,z)]      
    return output

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
    

###############################################################################
# EvaluateTriangles #
###############################################################################
def EvaluateTriangles(hist_dict):
    # Turn keys from dict into list of list #
    eval_list = []
    for key in hist_dict.keys():
        eval_list.append(list((key[0],key[1])))
    
    out_dict = InterpolateTriangles(hist_dict,eval_list)
    return out_dict


