#! /bin/env python

import sys, os, json
import copy
import numpy as np

import matplotlib.pyplot as plt

#Compute ellipse parameters for primary map for signal files.

#NOTA BENE: MH,MA       = SIMULATED MASSES
#           mllbb, mbb  = RECONSTRUCTED MASSES

def main():

    path_ElEl = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/ellipseParam_ElEl.json"
    path_MuMu = "/home/ucl/cp3/fbury/scratch/CMSSW_8_0_30/src/cp3_llbb/ZATools/scripts_ZA/ellipsesScripts/ellipseParam_MuMu.json"

    with open(path_ElEl,'r') as f:
        m_ElEl = json.load(f)
    with open(path_MuMu,'r') as f:
        m_MuMu = json.load(f)

    # 0 -> mbb
    # 1 -> mllbb
    # 5 -> mA
    # 6 -> mH

    m_bb_ElEl = []
    m_llbb_ElEl = []
    m_A_ElEl = []
    m_H_ElEl = []

    m_bb_MuMu = []
    m_llbb_MuMu = []
    m_A_MuMu = []
    m_H_MuMu = []

    for m in m_ElEl:
        m_bb_ElEl.append(m[0])
        m_llbb_ElEl.append(m[1])
        m_A_ElEl.append(m[5])
        m_H_ElEl.append(m[6])

    for m in m_MuMu:
        m_bb_MuMu.append(m[0])
        m_llbb_MuMu.append(m[1])
        m_A_MuMu.append(m[5])
        m_H_MuMu.append(m[6])

    fig1 = plt.figure()
    plt.scatter(m_H_ElEl, m_llbb_ElEl, color='b',label='ElEl')
    plt.scatter(m_H_MuMu, m_llbb_MuMu, color='r',label='MuMu')
    plt.plot([0, 1000], [0, 1000], ls="--", c=".3")
    plt.ylim((0,1000))
    plt.xlim((0,1100))
    plt.legend(loc='upper left')
    #plt.show()
    plt.ylabel('$M_{llbb}$')
    plt.xlabel('$M_{H}$')
    plt.savefig("m_llbb.png")
 
    fig2 = plt.figure()
    plt.scatter(m_A_ElEl, m_bb_ElEl, color='b',label='ElEl')
    plt.scatter(m_A_MuMu, m_bb_MuMu, color='r',label='MuMu')
    plt.legend(loc='upper left')
    plt.plot([0, 1000], [0, 1000], ls="--", c=".3")
    plt.ylim((0,800))
    plt.xlim((0,800))
    #plt.show()
    plt.ylabel('$M_{bb}$')
    plt.xlabel('$M_{A}$')
    plt.savefig("m_bb.png")
   

if __name__ == "__main__":
    main()  
