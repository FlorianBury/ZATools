import glob
import os
import re
import sys
import math
import json
import pprint

import array
import numpy as np
from operator import itemgetter

from scipy.stats import chisquare

import matplotlib.pyplot as plt                                                                  

###############################################################################
# PlotRhoComparison #
###############################################################################
def PlotRhoComparison(inter_dict,path):
    """
    Takes a serie of N dict histograms defined as dictionaries themselves : 
    Inputs :
        - inter_dict : dict
            Contains the interpolation dict
                key -> name for the title
                value ->  dict
                    key -> (mA,mH) mass configuration
                    value -> 6 bin contents of the rho distribution 
        - name : str
            Name of the output directory
    Outputs :
        None
    Plots :
        Comparison of the three distribution for each mass point 
    """

    # Binning parameters #
    N = len(inter_dict.keys())
    bin_dict = {} # Will contain the bins for each hist
    i = 0
    for key in inter_dict.keys():
        if i==0:
            bin_dict[key] =  np.linspace(0,2.5,6)
        else:
            bin_dict[key] = np.linspace(0,2.5,6)+0.5*i/(N+1)
        i += 1

    # Plot section #
    print ('[INFO] Starting interpolate plot section')
    for masspoint in list(inter_dict.values())[0].keys(): # Loop over mass points of first dict
        print ('Masspoint : m_H = %0.2f, m_A = %0.2f'%(masspoint[1],masspoint[0]))
        color=iter(plt.cm.rainbow(np.linspace(0.3,1,N)))
        fig = plt.figure()
        for name, hist_dict in inter_dict.items(): # Loop over the histograms dict
            # Plot the bars #
            c=next(color)
            p = plt.bar(bin_dict[name],
                    hist_dict[masspoint],
                    align='edge',
                    width=0.5/(N+1),
                    color=c,
                    linewidth=2,
                    label=name)

        # Optional parameters #
        plt.legend(loc='upper right')
        plt.xlabel(r'$\rho$')
        plt.ylabel('Arb. units')
        plt.title(r'Mass point $m_H=$%d GeV, $m_A$=%d GeV'%(masspoint[1],masspoint[0]))

        # Estethic : distinguishable groups of bins #
        #for i in range(0,6):
        #    if i%2==1: 
        #        p_avg[i].set_hatch('/')
        #        p_tri[i].set_hatch('/')
        #        p_dnn[i].set_hatch('/')
        #        p_real[i].set_hatch('/')
        #    else:
        #        p_avg[i].set_hatch('\\')
        #        p_tri[i].set_hatch('\\')
        #        p_dnn[i].set_hatch('\\')
        #        p_real[i].set_hatch('\\')

        # Save #
        fig.savefig(os.path.join(path,'m_H_%d_m_A_%d.png'%(masspoint[1],masspoint[0])))


###############################################################################
# Plot2DComparison #
###############################################################################
def Plot2DComparison(inter_dict,path):
    ########################### Comparison per method #########################
    # Get common z_values max # 
    z_all_max = np.zeros(6)
    for hist_dict in inter_dict.values(): 
        zmax = np.asarray(list(hist_dict.values())).max(axis=0)
        z_all_max = np.maximum(z_all_max,zmax)
        
    # Loop over methods #
    for name, hist_dict in inter_dict.items(): 
        # Get x and y from keys #
        x = np.asarray(list(hist_dict.keys()))[:,0]
        y = np.asarray(list(hist_dict.keys()))[:,1]
        # Get z bins array from values #
        z = np.asarray(list(hist_dict.values()))
        
        # Generate subplots #
        fig,ax = plt.subplots(2,3,figsize=(16,9))
        fig.subplots_adjust(right=0.85, wspace = 0.3, hspace=0.3, left=0.05, bottom=0.1)
        # Loop over subplots #
        for i in range(0,6): # Loop over bins 
            ix = int(i/3)
            iy = i%3
            im = ax[ix,iy].hexbin(x,y,z[:,i],gridsize=100,vmin=0, vmax=z_all_max.max())
            ax[ix,iy].plot([0, 1000], [0, 1000], ls="--", c=".3")
            ax[ix,iy].set_xlabel('$m_{jj}$')
            ax[ix,iy].set_ylabel('$m_{lljj}$')
            ax[ix,iy].set_title('Bin %d'%i)

        # Add colorbar #
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7]) 
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle('Interpolation with %s'%name, fontsize=16)
        fig.savefig(os.path.join(path,name+'.png'))                                                                                                                                   
        plt.close()
    

    ########################### Comparison bin per bin  #########################
    for b in range(0,6): # Loop over bins 
        # Generate subplots #
        fig,ax = plt.subplots(1,len(list(inter_dict.keys())),figsize=(19,6))
        fig.subplots_adjust(right=0.85, wspace = 0.3, hspace=0.3, left=0.05, bottom=0.1)

        # Make the plot of the given bin i with the different methods #
        i = 0
        for name, hist_dict in inter_dict.items(): # Loop over methods
            # Get x and y from keys #
            x = np.asarray(list(hist_dict.keys()))[:,0]
            y = np.asarray(list(hist_dict.keys()))[:,1]
            # Get corresponding bin array from values #
            z = np.asarray(list(hist_dict.values()))[:,b]
            # Plot on subplot #
            im = ax[i].hexbin(x,y,z,gridsize=100,vmin=0, vmax=z_all_max[b]) 
            ax[i].plot([0, 1000], [0, 1000], ls="--", c=".3")
            ax[i].set_xlabel('$m_{jj}$')
            ax[i].set_ylabel('$m_{lljj}$')
            ax[i].set_title('Method : %s'%name)

            i += 1

        # Add colorbar #
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7]) 
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle('Interpolation of bin %d'%b, fontsize=16)
        fig.savefig(os.path.join(path,'bin%d.png'%b))                                                                                                                                   
        plt.close()
 

    




