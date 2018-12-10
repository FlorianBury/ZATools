#! /bin/env python

import sys, os, json
import copy
import numpy as np
import itertools

import matplotlib.pyplot as plt

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
    path_ElEl = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/ellipseParam_ElEl.json"
    path_MuMu = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/ellipseParam_MuMu.json"

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


    for m in ElEl:
        arr1 = np.array([m[5],m[0]]).reshape(1,2) #mA,mbb
        arr2 = np.array([m[6],m[1]]).reshape(1,2) #mH,mllbb
        m_H_llbb_ElEl = np.append(m_H_llbb_ElEl,arr2,axis=0)
        m_A_bb_ElEl = np.append(m_A_bb_ElEl,arr1,axis=0)
    for m in MuMu:
        arr1 = np.array([m[5],m[0]]).reshape(1,2)
        arr2 = np.array([m[6],m[1]]).reshape(1,2)
        m_H_llbb_MuMu = np.append(m_H_llbb_MuMu,arr2,axis=0)
        m_A_bb_MuMu = np.append(m_A_bb_MuMu,arr1,axis=0)

    # m_.... contains [mb,mllbb,mA,mH]

    # Extrapolation #
    m_H_llbb_ElEl_in,m_H_llbb_ElEl_out  = IncreasingPart(m_H_llbb_ElEl)
    m_A_bb_ElEl_in,m_A_bb_ElEl_out  = IncreasingPart(m_A_bb_ElEl)
    m_H_llbb_MuMu_in,m_H_llbb_MuMu_out  = IncreasingPart(m_H_llbb_MuMu)
    m_A_bb_MuMu_in,m_A_bb_MuMu_out  = IncreasingPart(m_A_bb_MuMu)

    m_H_llbb_ElEl_ex = []
    m_A_bb_ElEl_ex = []
    m_H_llbb_MuMu_ex = []
    m_A_bb_MuMu_ex = []
    n_poly = 5
    for i in range(1,n_poly+1):
        m_H_llbb_ElEl_ex.append(Extrapolation(m_H_llbb_ElEl_in,n=i,xmax=1100))
        m_A_bb_ElEl_ex.append(Extrapolation(m_A_bb_ElEl_in,n=i,xmax=1000))
        m_H_llbb_MuMu_ex.append(Extrapolation(m_H_llbb_MuMu_in,n=i,xmax=1100))
        m_A_bb_MuMu_ex.append(Extrapolation(m_A_bb_MuMu_in,n=i,xmax=1000))

    # m_bb plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121) # ElEl
    ax2 = plt.subplot(122) # MuMu

    ax1.scatter(m_A_bb_ElEl_in[:,0],m_A_bb_ElEl_in[:,1],alpha=1,marker='1',s=150,color='g',label='Points kept')
    ax1.scatter(m_A_bb_ElEl_out[:,0],m_A_bb_ElEl_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax1.plot(m_A_bb_ElEl_ex[i][:,0],m_A_bb_ElEl_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i+1))
    ax1.plot([0, 1000], [0, 1000], ls="--", c=".3",label='Theory')
    ax1.legend(loc='upper left')
    ax1.set_ylim((0,1000))
    ax1.set_xlim((0,1000))
    ax1.set_xlabel('$M_{A}$')
    ax1.set_ylabel('Centroid $M_{bb}$')
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.scatter(m_A_bb_MuMu_in[:,0],m_A_bb_MuMu_in[:,1],alpha=1,marker='1',s=150,color='g',label='Points kept')
    ax2.scatter(m_A_bb_MuMu_out[:,0],m_A_bb_MuMu_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax2.plot(m_A_bb_MuMu_ex[i][:,0],m_A_bb_MuMu_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i+1))
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

    ax1.scatter(m_H_llbb_ElEl_in[:,0],m_H_llbb_ElEl_in[:,1],alpha=1,marker='1',s=150,color='g',label='Points kept')
    ax1.scatter(m_H_llbb_ElEl_out[:,0],m_H_llbb_ElEl_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax1.plot(m_H_llbb_ElEl_ex[i][:,0],m_H_llbb_ElEl_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i+1))
    ax1.plot([0, 1100], [0, 1100], ls="--", c=".3",label='Theory')
    ax1.legend(loc='upper left')
    ax1.set_ylim((0,1100))
    ax1.set_xlim((0,1100))
    ax1.set_xlabel('$M_{H}$')
    ax1.set_ylabel('Centroid $M_{llbb}$')
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.scatter(m_H_llbb_MuMu_in[:,0],m_H_llbb_MuMu_in[:,1],alpha=1,marker='1',s=150,color='g',label='Points kept')
    ax2.scatter(m_H_llbb_MuMu_out[:,0],m_H_llbb_MuMu_out[:,1],alpha=0.5,marker='2',s=150,color='r',label='Points removed')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax2.plot(m_H_llbb_MuMu_ex[i][:,0],m_H_llbb_MuMu_ex[i][:,1],color=color.next(),label='Extrapolation order '+str(i+1))
    ax2.plot([0, 1100], [0, 1100], ls="--", c=".3",label='Theory')
    ax2.legend(loc='upper left')
    ax2.set_ylim((0,1100))
    ax2.set_xlim((0,1100))
    ax2.set_xlabel('$M_{H}$')
    ax2.set_ylabel('Centroid $M_{llbb}$')
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    plt.savefig('m_llbb.png')

    
  
def IncreasingPart(arr):
    """
    Separates the increasing part of arr to be used for the extrapolation, and the rest of the points
    Inputs :
        - arr : numpy array [N,2]
            Contains the values of x and y 
    Outputs :
        - inside :  numpy array [N,2]
            Points to be used for extrapolation
        - outside :  numpy array [N,2]
            Points not to be used for extrapolation
    """
    # Sort according to x #
    arr_sorted = arr[np.lexsort((-arr[:,1],arr[:,0]))] # ascending in x, descendin in y

    y_max = 0
    inside = np.zeros((0,2))
    outside= np.zeros((0,2))
    for i in range(0,arr_sorted.shape[0]):
        if arr_sorted[i,1]>y_max:
            inside = np.append(inside,arr_sorted[i,:].reshape(1,2),axis=0)
            y_max = arr_sorted[i,1]
        else: 
            outside = np.append(outside,arr_sorted[i,:].reshape(1,2),axis=0)
    return inside,outside

def Extrapolation(data,n,xmax):
    """
    Extrapolate the data with a n order polyfit
    Inputs :
        - data : numpy array [N,2]
            Contains the values of x and y 
        - n : int
            Order of the polynom
        - xmax : float
            maximum value of x to be used for extrapolation
    Outputs :
        - out :  numpy array [N,2]
            extrapolation of x (from 0 to xmax) = y
    """
    # data = [x,y]
    data_new = np.append(data,np.array([0,0]).reshape((1,2)),axis=0) # added the point (0,0)    
    weight = np.ones(data_new.shape[0])
    weight[-1] = 1000000 # give a very weight to point (0,0)

    coeff = np.polyfit (data_new[:,0],data_new[:,1],n,w=weight)
    x_new = np.arange(0,xmax).reshape(xmax,1)
    p = np.poly1d(coeff)
    y_new = p(x_new)
    out = np.concatenate((x_new,y_new),axis=1)

    return out

if __name__ == '__main__':
    main()
