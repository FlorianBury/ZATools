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

from centroidExtrapolation import *

# Extrapolate the widths as a function of the centroids

#NOTA BENE: MH,MA       = SIMULATED MASSES
#           mllbb, mbb  = RECONSTRUCTED MASSES

def main():
    path_ElEl = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/sigmasFit_ElEl.json"
    path_MuMu = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/sigmasFit_MuMu.json"

    with open(path_ElEl,'r') as f:
        ElEl = json.load(f)
    with open(path_MuMu,'r') as f:
        MuMu = json.load(f)

    # 0 -> mbb
    # 1 -> mllbb
    # 2 -> width_bb
    # 3 -> width_llbb

    w_llbb_ElEl = np.zeros((0,2))
    w_bb_ElEl = np.zeros((0,2))
    w_llbb_MuMu = np.zeros((0,2))
    w_bb_MuMu = np.zeros((0,2))
    # first column is m_(ll)bb 
    # second column is width_(ll)bb

    for m in ElEl:
        arr1 = np.array([m[0],m[2]]).reshape(1,2) # m_bb, width_bb
        arr2 = np.array([m[1],m[3]]).reshape(1,2) # m_llbb, width_llbb
        w_bb_ElEl = np.append(w_bb_ElEl,arr1,axis=0)
        w_llbb_ElEl = np.append(w_llbb_ElEl,arr2,axis=0)
    for m in MuMu:
        arr1 = np.array([m[0],m[2]]).reshape(1,2) # m_bb, width_bb
        arr2 = np.array([m[1],m[3]]).reshape(1,2) # m_llbb, width_llbb
        w_bb_MuMu = np.append(w_bb_MuMu,arr1,axis=0)
        w_llbb_MuMu = np.append(w_llbb_MuMu,arr2,axis=0)


    # m_.... contains [mb,mllbb,mA,mH]

    # Extrapolation #
    #m_H_llbb_ElEl_in,m_H_llbb_ElEl_out  = IncreasingPart(m_H_llbb_ElEl)
    #m_A_bb_ElEl_in,m_A_bb_ElEl_out  = IncreasingPart(m_A_bb_ElEl)
    #m_H_llbb_MuMu_in,m_H_llbb_MuMu_out  = IncreasingPart(m_H_llbb_MuMu)
    #m_A_bb_MuMu_in,m_A_bb_MuMu_out  = IncreasingPart(m_A_bb_MuMu)

    w_llbb_ElEl_ex = []
    w_bb_ElEl_ex = []
    w_llbb_MuMu_ex = []
    w_bb_MuMu_ex = []
    n_poly = 5
    for i in range(1,n_poly+1):
        w_llbb_ElEl_ex.append(Extrapolation(w_llbb_ElEl,n=i,xmax=900))
        w_bb_ElEl_ex.append(Extrapolation(w_bb_ElEl,n=i,xmax=600))
        w_llbb_MuMu_ex.append(Extrapolation(w_llbb_MuMu,n=i,xmax=900))
        w_bb_MuMu_ex.append(Extrapolation(w_bb_MuMu,n=i,xmax=600))

    # m_bb plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121) # ElEl
    ax2 = plt.subplot(122) # MuMu

    ax1.scatter(w_bb_ElEl[:,0],w_bb_ElEl[:,1],alpha=1,marker='2',s=150,color='k',label='Known points')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax1.plot(w_bb_ElEl_ex[i][:,0],w_bb_ElEl_ex[i][:,1],color=next(color),label='Extrapolation order '+str(i+1))
    ax1.legend(loc='upper left')
    ax1.set_ylim((0,130))
    ax1.set_xlim((0,600))
    ax1.set_xlabel('$M_{bb}$')
    ax1.set_ylabel('Width $w_{bb}$')
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.scatter(w_bb_MuMu[:,0],w_bb_MuMu[:,1],alpha=1,marker='2',s=150,color='k',label='Known points')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax2.plot(w_bb_MuMu_ex[i][:,0],w_bb_MuMu_ex[i][:,1],color=next(color),label='Extrapolation order '+str(i+1))
    ax2.legend(loc='upper left')
    ax2.set_ylim((0,130))
    ax2.set_xlim((0,600))
    ax2.set_xlabel('$M_{bb}$')
    ax2.set_ylabel('Width $w_{bb}$')
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    plt.savefig('w_bb.png')

    # m_llbb plot #
    fig = plt.figure(figsize=(16,7))
    ax1 = plt.subplot(121) # ElEl
    ax2 = plt.subplot(122) # MuMu

    ax1.scatter(w_llbb_ElEl[:,0],w_llbb_ElEl[:,1],alpha=1,marker='1',s=150,color='k',label='Known points')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax1.plot(w_llbb_ElEl_ex[i][:,0],w_llbb_ElEl_ex[i][:,1],color=next(color),label='Extrapolation order '+str(i+1))
    ax1.legend(loc='upper left')
    ax1.set_xlim((0,900))
    ax1.set_ylim((0,250))
    ax1.set_xlabel('$M_{llbb}$')
    ax1.set_ylabel('Width $w_{llbb}$')
    ax1.set_title('Z $\\rightarrow$ $e^+$$e^-$',fontsize=26)

    ax2.scatter(w_llbb_MuMu[:,0],w_llbb_MuMu[:,1],alpha=1,marker='1',s=150,color='k',label='Known points')
    color=iter(plt.cm.jet(np.linspace(0.3,1,n_poly)))
    for i in range(0,n_poly):
        ax2.plot(w_llbb_MuMu_ex[i][:,0],w_llbb_MuMu_ex[i][:,1],color=next(color),label='Extrapolation order '+str(i+1))
    ax2.legend(loc='upper left')
    ax2.set_xlim((0,900))
    ax2.set_ylim((0,250))
    ax2.set_xlabel('$M_{llbb}$')
    ax2.set_ylabel('Width $w_{llbb}$')
    ax2.set_title('Z $\\rightarrow$ $\\mu^+$$\\mu^-$',fontsize=26)

    plt.savefig('w_llbb.png')

    
if __name__ == '__main__':
    main()
