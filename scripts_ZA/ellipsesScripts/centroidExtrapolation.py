#! /bin/env python

import sys, os, json
import copy
import itertools
import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from plotDistributions import get_options

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Interpolation of the centroids

#NOTA BENE: MH,MA       = SIMULATED MASSES
#           mllbb, mbb  = RECONSTRUCTED MASSES

def main():
    opt = get_options()

    if not opt.fit and not opt.window:
        save_pol = True
    else:
        save_pol = False

    path_ElEl = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/fullEllipseParam_ElEl.json"
    path_MuMu = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/fullEllipseParam_MuMu.json"

    if opt.fit:                                                                                                                                                                             
        path_ElEl = path_ElEl.replace('ellipseParam','ellipseParamFit')                                                                                                                 
        path_MuMu = path_MuMu.replace('ellipseParam','ellipseParamFit')                                                                                                                 
        if opt.window:                                                                                                                                                                          
            path_ElEl = path_ElEl.replace('ellipseParam','ellipseParamWindow')                                                                                                              
            path_MuMu = path_MuMu.replace('ellipseParam','ellipseParamWindow')                                                                                                              

    with open(path_ElEl,'r') as f:
        ElEl = json.load(f)
    with open(path_MuMu,'r') as f:
        MuMu = json.load(f)

    # 0 -> mbb
    # 1 -> mllbb
    # 5 -> mA
    # 6 -> mH

    m_H_llbb_ElEl = np.zeros((0,2))
    m_A_bb_ElEl = np.zeros((0,2))
    m_H_llbb_MuMu = np.zeros((0,2))
    m_A_bb_MuMu = np.zeros((0,2))

    major_ElEl = np.zeros((0,3)) # [mA,mH,a]
    minor_ElEl = np.zeros((0,3)) # [mA,mH,b]
    theta_ElEl = np.zeros((0,3)) # [mA,mH,theta]
    major_MuMu = np.zeros((0,3)) # [mA,mH,a]
    minor_MuMu = np.zeros((0,3)) # [mA,mH,b]
    theta_MuMu = np.zeros((0,3)) # [mA,mH,theta]


    for m in ElEl:
        arr1 = np.array([m[5],m[0]]).reshape(1,2) #mA,mbb
        arr2 = np.array([m[6],m[1]]).reshape(1,2) #mH,mllbb
        arr3 = np.array([m[5],m[6],m[2]]).reshape(1,3) #mA,mH,a
        arr4 = np.array([m[5],m[6],m[3]]).reshape(1,3) #mA,mH,b
        arr5 = np.array([m[5],m[6],m[4]]).reshape(1,3) #mA,mH,theta

        m_A_bb_ElEl = np.append(m_A_bb_ElEl,arr1,axis=0)
        m_H_llbb_ElEl = np.append(m_H_llbb_ElEl,arr2,axis=0)
        major_ElEl = np.append(major_ElEl,arr3,axis=0)
        minor_ElEl = np.append(minor_ElEl,arr4,axis=0)
        theta_ElEl = np.append(theta_ElEl,arr5,axis=0)

    for m in MuMu:
        arr1 = np.array([m[5],m[0]]).reshape(1,2) #mA,mbb
        arr2 = np.array([m[6],m[1]]).reshape(1,2) #mH,mllbb
        arr3 = np.array([m[5],m[6],m[2]]).reshape(1,3) #mA,mH,a
        arr4 = np.array([m[5],m[6],m[3]]).reshape(1,3) #mA,mH,b
        arr5 = np.array([m[5],m[6],m[4]]).reshape(1,3) #mA,mH,theta

        m_A_bb_MuMu = np.append(m_A_bb_MuMu,arr1,axis=0)
        m_H_llbb_MuMu = np.append(m_H_llbb_MuMu,arr2,axis=0)
        major_MuMu = np.append(major_MuMu,arr3,axis=0)
        minor_MuMu = np.append(minor_MuMu,arr4,axis=0)
        theta_MuMu = np.append(theta_MuMu,arr5,axis=0)

    # m_.... contains [mb,mllbb,mA,mH]
    
    # Sort arrays #
    m_H_llbb_ElEl = m_H_llbb_ElEl[np.lexsort((-m_H_llbb_ElEl[:,1],m_H_llbb_ElEl[:,0]))] # ascending in x, descending in y
    m_A_bb_ElEl = m_A_bb_ElEl[np.lexsort((-m_A_bb_ElEl[:,1],m_A_bb_ElEl[:,0]))]
    m_H_llbb_MuMu = m_H_llbb_MuMu[np.lexsort((-m_H_llbb_MuMu[:,1],m_H_llbb_MuMu[:,0]))]
    m_A_bb_MuMu = m_A_bb_MuMu[np.lexsort((-m_A_bb_MuMu[:,1],m_A_bb_MuMu[:,0]))]

    major_ElEl = major_ElEl[np.lexsort((-major_ElEl[:,0],major_ElEl[:,1]))]
    minor_ElEl = minor_ElEl[np.lexsort((-minor_ElEl[:,0],minor_ElEl[:,1]))]
    theta_ElEl = theta_ElEl[np.lexsort((-theta_ElEl[:,0],theta_ElEl[:,1]))]
    major_MuMu = major_MuMu[np.lexsort((-major_MuMu[:,0],major_MuMu[:,1]))]
    minor_MuMu = minor_MuMu[np.lexsort((-minor_MuMu[:,0],minor_MuMu[:,1]))]
    theta_MuMu = theta_MuMu[np.lexsort((-theta_MuMu[:,0],theta_MuMu[:,1]))]

    # Remove outliers #

    m_H_llbb_ElEl_mask  = RemovePoints(m_H_llbb_ElEl)
    m_A_bb_ElEl_mask  = RemovePoints(m_A_bb_ElEl)
    m_H_llbb_MuMu_mask  = RemovePoints(m_H_llbb_MuMu)
    m_A_bb_MuMu_mask  = RemovePoints(m_A_bb_MuMu)


    m_H_llbb_ElEl_in = m_H_llbb_ElEl[m_H_llbb_ElEl_mask==True]
    m_H_llbb_ElEl_out = m_H_llbb_ElEl[m_H_llbb_ElEl_mask==False]
    m_A_bb_ElEl_in = m_A_bb_ElEl[m_A_bb_ElEl_mask==True]
    m_A_bb_ElEl_out = m_A_bb_ElEl[m_A_bb_ElEl_mask==False]
    m_H_llbb_MuMu_in = m_H_llbb_MuMu[m_H_llbb_MuMu_mask==True]
    m_H_llbb_MuMu_out = m_H_llbb_MuMu[m_H_llbb_MuMu_mask==False]
    m_A_bb_MuMu_in = m_A_bb_MuMu[m_A_bb_MuMu_mask==True]
    m_A_bb_MuMu_out = m_A_bb_MuMu[m_A_bb_MuMu_mask==False]

    major_ElEl_in = major_ElEl[m_H_llbb_ElEl_mask==True]
    major_ElEl_out = major_ElEl[m_H_llbb_ElEl_mask==False]
    minor_ElEl_in = minor_ElEl[m_H_llbb_ElEl_mask==True]
    minor_ElEl_out = minor_ElEl[m_H_llbb_ElEl_mask==False]
    theta_ElEl_in = theta_ElEl[m_H_llbb_ElEl_mask==True]
    theta_ElEl_out = theta_ElEl[m_H_llbb_ElEl_mask==False]
    major_MuMu_in = major_MuMu[m_H_llbb_MuMu_mask==True]
    major_MuMu_out = major_MuMu[m_H_llbb_MuMu_mask==False]
    minor_MuMu_in = minor_MuMu[m_H_llbb_MuMu_mask==True]
    minor_MuMu_out = minor_MuMu[m_H_llbb_MuMu_mask==False]
    theta_MuMu_in = theta_MuMu[m_H_llbb_MuMu_mask==True]
    theta_MuMu_out = theta_MuMu[m_H_llbb_MuMu_mask==False]

    # Extrapolation #

    m_H_llbb_ElEl_ex = []
    m_A_bb_ElEl_ex = []
    m_H_llbb_MuMu_ex = []
    m_A_bb_MuMu_ex = []
    n_poly = 3
    for i in range(0,n_poly+1):
        m_H_llbb_ElEl_ex.append(Extrapolation(m_H_llbb_ElEl_in,n=i,xmax=1100,label='_mH_ElEl',save_pol=save_pol))
        m_A_bb_ElEl_ex.append(Extrapolation(m_A_bb_ElEl_in,n=i,xmax=1000,label='_mA_ElEl',save_pol=save_pol))
        m_H_llbb_MuMu_ex.append(Extrapolation(m_H_llbb_MuMu_in,n=i,xmax=1100,label='_mH_MuMu',save_pol=save_pol))
        m_A_bb_MuMu_ex.append(Extrapolation(m_A_bb_MuMu_in,n=i,xmax=1000,label='_mA_MuMu',save_pol=save_pol))

    # m_bb plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121) # ElEl
    ax2 = plt.subplot(122) # MuMu

    ax1.scatter(m_A_bb_ElEl_in[:,0],m_A_bb_ElEl_in[:,1],alpha=1,marker='1',s=150,color='g',label='Points kept')
    ax1.scatter(m_A_bb_ElEl_out[:,0],m_A_bb_ElEl_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly+1)))
    for i in range(0,n_poly+1):
        if i==0:
            ax1.plot(m_A_bb_ElEl_ex[i][:,0],m_A_bb_ElEl_ex[i][:,1],color=color.next(),label='Custom fit')
        else:
            ax1.plot(m_A_bb_ElEl_ex[i][:,0],m_A_bb_ElEl_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i))
    ax1.plot([0, 1000], [0, 1000], ls="--", c=".3",label='Theory')
    ax1.legend(loc='upper left')
    ax1.set_ylim((0,1000))
    ax1.set_xlim((0,1000))
    ax1.set_xlabel('$M_{A}$')
    ax1.set_ylabel('Centroid $M_{bb}$')
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.scatter(m_A_bb_MuMu_in[:,0],m_A_bb_MuMu_in[:,1],alpha=1,marker='1',s=150,color='g',label='Points kept')
    ax2.scatter(m_A_bb_MuMu_out[:,0],m_A_bb_MuMu_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly+1)))
    for i in range(0,n_poly+1):
        if i==0:
            ax2.plot(m_A_bb_MuMu_ex[i][:,0],m_A_bb_MuMu_ex[i][:,1],color=color.next(),label='Custom fit')
        else:
            ax2.plot(m_A_bb_MuMu_ex[i][:,0],m_A_bb_MuMu_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i))
    ax2.plot([0, 1000], [0, 1000], ls="--", c=".3",label='Theory')
    ax2.legend(loc='upper left')
    ax2.set_ylim((0,1000))
    ax2.set_xlim((0,1000))
    ax2.set_xlabel('$M_{A}$')
    ax2.set_ylabel('Centroid $M_{bb}$')
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    plt.savefig('m_bb.png')

    # m_llbb plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121) # ElEl
    ax2 = plt.subplot(122) # MuMu

    ax1.scatter(m_H_llbb_ElEl_in[:,0],m_H_llbb_ElEl_in[:,1],alpha=0.5,marker='1',s=150,color='g',label='Points kept')
    ax1.scatter(m_H_llbb_ElEl_out[:,0],m_H_llbb_ElEl_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly+1)))
    for i in range(0,n_poly+1):
        if i==0:
            ax1.plot(m_H_llbb_ElEl_ex[i][:,0],m_H_llbb_ElEl_ex[i][:,1],color=color.next(),label='Custom fit')
        else:
            ax1.plot(m_H_llbb_ElEl_ex[i][:,0],m_H_llbb_ElEl_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i))
    ax1.plot([0, 1100], [0, 1100], ls="--", c=".3",label='Theory')
    ax1.legend(loc='upper left')
    ax1.set_ylim((0,1100))
    ax1.set_xlim((0,1100))
    ax1.set_xlabel('$M_{H}$')
    ax1.set_ylabel('Centroid $M_{llbb}$')
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.scatter(m_H_llbb_MuMu_in[:,0],m_H_llbb_MuMu_in[:,1],alpha=0.5,marker='1',s=150,color='g',label='Points kept')
    ax2.scatter(m_H_llbb_MuMu_out[:,0],m_H_llbb_MuMu_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly+1)))
    for i in range(0,n_poly+1):
        if i==0:
            ax2.plot(m_H_llbb_MuMu_ex[i][:,0],m_H_llbb_MuMu_ex[i][:,1],color=color.next(),label='Custom fit')
        else:
            ax2.plot(m_H_llbb_MuMu_ex[i][:,0],m_H_llbb_MuMu_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i))
    ax2.plot([0, 1100], [0, 1100], ls="--", c=".3",label='Theory')
    ax2.legend(loc='upper left')
    ax2.set_ylim((0,1100))
    ax2.set_xlim((0,1100))
    ax2.set_xlabel('$M_{H}$')
    ax2.set_ylabel('Centroid $M_{llbb}$')
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    plt.savefig('m_llbb.png')

    # Major axis plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121, projection='3d') # ElEl
    ax2 = plt.subplot(122, projection='3d') # MuMu

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.9, wspace=0.2, hspace=0.2)
    plt.tight_layout()

    ax1.plot_trisurf(major_ElEl_in[:,0],major_ElEl_in[:,1],major_ElEl_in[:,2], cmap=cm.coolwarm)
    ax1.scatter(major_ElEl_out[:,0],major_ElEl_out[:,1],major_ElEl_out[:,2],c='k')
    ax1.set_xlim((0,700))
    ax1.set_ylim((0,1100))
    ax1.set_xlabel('$M_{A}$',labelpad=10)
    ax1.set_ylabel('$M_{H}$',labelpad=10)
    ax1.set_zlabel('a',labelpad=15)
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.plot_trisurf(major_MuMu_in[:,0],major_MuMu_in[:,1],major_MuMu_in[:,2], cmap=cm.coolwarm)
    ax2.scatter(major_MuMu_out[:,0],major_MuMu_out[:,1],major_MuMu_out[:,2],c='k')
    ax2.set_xlim((0,700))
    ax2.set_ylim((0,1100))
    ax2.set_xlabel('$M_{A}$',labelpad=10)
    ax2.set_ylabel('$M_{H}$',labelpad=10)
    ax2.set_zlabel('a',labelpad=15)
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    fig.suptitle('Major axis', fontsize=26)

    plt.savefig('major_axis.png')

    # Minor axis plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121, projection='3d') # ElEl
    ax2 = plt.subplot(122, projection='3d') # MuMu

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.9, wspace=0.2, hspace=0.2)
    plt.tight_layout()

    ax1.plot_trisurf(minor_ElEl_in[:,0],minor_ElEl_in[:,1],minor_ElEl_in[:,2], cmap=cm.coolwarm)
    ax1.scatter(minor_ElEl_out[:,0],minor_ElEl_out[:,1],minor_ElEl_out[:,2],c='k')
    ax1.set_xlim((0,700))
    ax1.set_ylim((0,1100))
    ax1.set_xlabel('$M_{A}$',labelpad=10)
    ax1.set_ylabel('$M_{H}$',labelpad=10)
    ax1.set_zlabel('b',labelpad=15)
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.plot_trisurf(minor_MuMu_in[:,0],minor_MuMu_in[:,1],minor_MuMu_in[:,2], cmap=cm.coolwarm)
    ax2.scatter(minor_MuMu_out[:,0],minor_MuMu_out[:,1],minor_MuMu_out[:,2],c='k')
    ax2.set_xlim((0,700))
    ax2.set_ylim((0,1100))
    ax2.set_xlabel('$M_{A}$',labelpad=10)
    ax2.set_ylabel('$M_{H}$',labelpad=10)
    ax2.set_zlabel('b',labelpad=15)
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    fig.suptitle('Minor axis', fontsize=26)

    plt.savefig('minor_axis.png')

    # Tilt angle theta plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121, projection='3d') # ElEl
    ax2 = plt.subplot(122, projection='3d') # MuMu

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.9, wspace=0.2, hspace=0.2)
    plt.tight_layout()

    ax1.plot_trisurf(theta_ElEl_in[:,0],theta_ElEl_in[:,1],theta_ElEl_in[:,2], cmap=cm.coolwarm)
    ax1.scatter(theta_ElEl_out[:,0],theta_ElEl_out[:,1],theta_ElEl_out[:,2],c='k')
    ax1.set_xlim((0,700))
    ax1.set_ylim((0,1100))
    ax1.set_xlabel('$M_{A}$',labelpad=10)
    ax1.set_ylabel('$M_{H}$',labelpad=10)
    ax1.set_zlabel('$\\theta$',labelpad=15)
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.plot_trisurf(theta_MuMu_in[:,0],theta_MuMu_in[:,1],theta_MuMu_in[:,2], cmap=cm.coolwarm)
    ax1.scatter(theta_MuMu_out[:,0],theta_MuMu_out[:,1],theta_MuMu_out[:,2],c='k')
    ax2.set_xlim((0,700))
    ax2.set_ylim((0,1100))
    ax2.set_xlabel('$M_{A}$',labelpad=10)
    ax2.set_ylabel('$M_{H}$',labelpad=10)
    ax2.set_zlabel('$\\theta$',labelpad=15)
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    fig.suptitle('Tilt angle', fontsize=26)


    plt.savefig('theta_tilt.png')

################################################################################################
# RemovePoints #
################################################################################################

def RemovePoints(arr):
    """
    Only keep the points that are higher than the average of the previous mA/mH    
    """
    # Sort according to x #

    y_avg_prev = 0
    inside = np.zeros(arr.shape[0],dtype=bool)
    i = 0
    while True:
        # Get all points with same mA/mH #
        buff = np.zeros((0,2))
        for j in range(i,arr.shape[0]):
            if arr[j,0]==arr[i,0]:
                buff = np.concatenate((buff,arr[j,:].reshape(1,2)),axis=0)
        # Keep points or not #
        new_y_tot = 0
        count = 0
        for j in range(0,buff.shape[0]):
            if buff[j,1]>y_avg_prev and buff[j,1]<buff[j,0] and buff[j,0]<700:
                inside[i+j] = True
                new_y_tot += buff[j,1]
                count += 1
        if count != 0:
            y_avg_prev = new_y_tot/count 
        
        i += buff.shape[0]
        if i == arr.shape[0]:
            break
         
    return inside

################################################################################################
# Extrapolation #
################################################################################################


def Extrapolation(data,n,xmax,save_pol=False,label=''):
    """
    Extrapolate the data with a n order polyfit
    Inputs :
        - data : numpy array [N,2]
            Contains the values of x and y 
        - n : int
            Order of the polynom
        - xmax : float
            maximum value of x to be used for extrapolation
        - label = str
            name to save the pol2 txt file
    Outputs :
        - out :  numpy array [N,2]
            extrapolation of x (from 0 to xmax) = y
    """
    # data = [x,y]
    data_new = np.append(data,np.array([0,0]).reshape((1,2)),axis=0) # added the point (0,0)    
    weight = np.ones(data_new.shape[0])
    weight[-1] = 1000000 # give a very high weight to point (0,0)

    if n>=1:
        coeff = np.polyfit (data_new[:,0],data_new[:,1],n,w=weight)
        x_new = np.arange(0,xmax).reshape(xmax,1)
        p = np.poly1d(coeff)
        y_new = p(x_new)
        out = np.concatenate((x_new,y_new),axis=1)

        if save_pol and n==1:
            np.savetxt('pol'+str(n)+'_fit'+label+'.txt',coeff)
            print ('Saved polynom to %s'%('pol'+str(n)+'_fit'+label+'.txt'))
    elif n==0:
        def function_mA(x,p):
            return (1-(2/(1+np.exp(-x/75))-1)*(1-p))*x 
        def function_mH(x,p):
            return (1-(2/(1+np.exp(-x/175))-1)*(1-p))*x 
        
        x_new = np.arange(0,xmax).reshape(xmax,1)
        if label.find('mA')!=-1:
            popt, pcov = curve_fit(function_mA, data_new[:,0], data_new[:,1])
            popt = np.append(popt,75)
            y_new = function_mA(x_new,popt)
            if save_pol:
                np.savetxt('weird_fit'+label+'.txt',popt)
                print ('Saved polynom to %s'%('weird_fit'+label+'.txt'))
        if label.find('mH')!=-1:
            popt, pcov = curve_fit(function_mH, data_new[:,0], data_new[:,1])
            popt = np.append(popt,175)
            y_new = function_mH(x_new,popt)
            if save_pol:
                np.savetxt('weird_fit'+label+'.txt',popt)
                print ('Saved polynom to %s'%('weird_fit'+label+'.txt'))
        out = np.concatenate((x_new,y_new),axis=1)
        
    return out

if __name__ == '__main__':
    main()
