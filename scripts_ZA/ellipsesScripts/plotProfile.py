#! /bin/env python

import sys, os, json
import copy
import numpy as np
import itertools

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

    # Extrapolation #
    m_llbb_ElEl_inc = IncreasingPart(m_H_ElEl,m_llbb_ElEl)
    m_llbb_MuMu_inc = IncreasingPart(m_H_MuMu,m_llbb_MuMu)
    m_bb_ElEl_inc = IncreasingPart(m_A_ElEl,m_bb_ElEl)
    m_bb_MuMu_inc = IncreasingPart(m_A_MuMu,m_bb_MuMu)

    dico_bb = {}
    dico_llbb = {}
    dico_bb['MuMu'] = [[a,b] for a,b in zip(m_A_MuMu,m_bb_MuMu)]
    dico_bb['ElEl'] = [[a,b] for a,b in zip(m_A_ElEl,m_bb_ElEl)]
    dico_bb['MuMu (only increasing)'] = m_bb_MuMu_inc
    dico_bb['ElEl (only increasing)'] = m_bb_ElEl_inc
    dico_llbb['MuMu (only increasing)'] = m_llbb_MuMu_inc
    dico_llbb['ElEl (only increasing)'] = m_llbb_ElEl_inc
    dico_llbb['MuMu'] = [[a,b] for a,b in zip(m_H_MuMu,m_llbb_MuMu)]
    dico_llbb['ElEl'] = [[a,b] for a,b in zip(m_H_ElEl,m_llbb_ElEl)]

    PlotRelation(dico_bb,'m_bb_VS_m_A')
    PlotRelation(dico_llbb,'m_llbb_VS_m_H')

  
def IncreasingPart(x,y):
    """ Only returns the part of the dictribution that is stricly increasing """
    # Sort according to x #
    xy_sort = sorted(((a,b) for a,b in zip(x,y)),key=lambda pair:pair[0])

    y_max = 0
    out = []
    out = []
    for i in range(0,len(xy_sort)):
        if xy_sort[i][1]>y_max:
            if xy_sort[i-1][0]==xy_sort[i][0]:
                out.pop(len(out)-1)
            out.append([xy_sort[i][0],xy_sort[i][1]])
            y_max = xy_sort[i][1]
    return out

def PlotRelation(dico, name):
    """ Different plots (x,y) contained in dictionary, name is the name of the png file to be saved
    key = label
    value = (x,y) list of list
    """
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 
    n = len(list(dico.keys()))
    color=iter(plt.cm.rainbow(np.linspace(0,1,n)))
    fig = plt.figure()
    for key,val in dico.iteritems():
        x = [i[0] for i in val]
        y = [i[1] for i in val]
        plt.scatter(x,y,alpha=0.7,marker=marker.next(),color=color.next(),label=key)

    plt.legend(loc='upper left')
    plt.ylim((0,800))
    plt.xlim((0,800))
    plt.title(name)
    plt.plot([0, 1000], [0, 1000], ls="--", c=".3",label='Theory')
    plt.savefig(name+'.png')


if __name__ == "__main__":
    main()  
