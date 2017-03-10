# -*- coding: utf-8 -*-
"""
@author: sk1712
"""

import math
import numpy as np


def align_matrices(M1, M2, dfun, flag=False):
    """ This function aligns two matrices relative to one another by reordering
        the nodes in M2.  The function uses a version of simulated annealing.

        Inputs:     M1             = first connection matrix (square)
                    M2             = second connection matrix (square)
                    dfun           = distance metric to use for matching:
                                    'absdff' = absolute difference
                                    'sqrdff' = squared difference
                                    'cosang' = cosine of vector angle

                    Mreordered     = reordered connection matrix M2
                    Mindices       = reordered indices
                    cost           = distance between M1 and Mreordered

        Connection matrices can be weighted or binary, directed or undirected.
        They must have the same number of nodes.  M1 can be entered in any
        node ordering.
        
        Original MATLAB version from:
        
            Yusuke Adachi, University of Tokyo, 2010
            Olaf Sporns, Indiana University, 2010
    """

    N = M1.shape[0]

    # define maxcost (greatest possible difference)
    if dfun == 'absdff':
        M2list = M2.flatten().tolist()
        M2list.sort(reverse=True)
        maxcost = np.sum(np.abs(np.sort(M1.flatten())-M2list))
    elif dfun == 'sqrdff':
        M2list = M2.flatten().tolist()
        M2list.sort(reverse=True)
        maxcost = np.sum(np.square(np.sort(M1.flatten())-M2list))
    elif dfun == 'cosang':
        maxcost = np.pi/2

    # initialize lowcost
    if dfun == 'absdff':
        lowcost = 1. * np.sum(np.abs(M1-M2)) / maxcost
    elif dfun == 'sqrdff':
        lowcost = 1. * np.sum(np.square(M1-M2)) / maxcost
    elif dfun == 'cosang':
        lowcost = math.acos(1. * np.dot(M1.flatten(), M2.flatten()) /
                  np.sqrt(np.dot(M1.flatten(), M1.flatten())*np.dot(M2.flatten(), M2.flatten()))) / maxcost

    # initialize 
    mincost = lowcost
    anew = range(N)
    amin = range(N)
    h = 0 
    hcnt = 0

    # set annealing parameters
    # H determines the maximal number of steps
    # Texp determines the steepness of the temperature gradient
    # T0 sets the initial temperature (and scales the energy term)
    # Hbrk sets a break point for the simulation (no further improvement)
    H = 1e06
    Texp = 1-1/H
    T0 = 1e-03
    Hbrk = H/10
    #Texp = 0

    while h<H:
        h += h+1
        hcnt += hcnt+1
        
        # terminate if no new mincost has been found for some time
        if hcnt > Hbrk:
            break
        
        # current temperature
        T = T0 * pow(Texp, h)
        # choose two positions at random and flip them
        atmp = list(anew)
        #r = randperm(N);  % slower
        r = np.floor(N * np.random.uniform(size=2)).astype(int)
        atmp[r[0]] = anew[r[1]]
        atmp[r[1]] = anew[r[0]]
        
        if dfun == 'absdff':
            costnew = 1. * np.sum(np.abs(M1-M2[atmp][:, atmp])) / maxcost
        elif dfun == 'sqrdff':
            costnew = 1. * np.sum(np.square(M1-M2[atmp][:, atmp])) / maxcost
        elif dfun == 'cosang':
            M2atmp = M2[atmp][:, atmp]
            costnew = math.acos(1. * np.dot(M1.flatten(), M2atmp.flatten()) /
                      np.sqrt(np.dot(M1.flatten(), M1.flatten())*
                      np.dot(M2atmp.flatten(), M2atmp.flatten()))) / maxcost

        # annealing step
        if (costnew < lowcost) or (np.random.uniform() < math.exp(-(costnew-lowcost)/T)):
            anew = list(atmp)
            lowcost = costnew
            # is this the absolute best?
            if (lowcost < mincost):
                amin = list(anew)
                mincost = lowcost
                if flag:
                    print('step %d ... current lowest cost = %.6f' % (h, mincost))
                hcnt = 0
            # if the cost is 0 we're done
            if (mincost==0):
                break

    if flag:
        print('step %d ... final lowest cost = %.6f ' % (h, mincost))

    # prepare output
    Mreordered = M2[amin][:, amin]
    Mindices = amin
    cost = mincost
    
    return Mreordered, Mindices, cost
    
    
def random_align(M1, M2, dfun, runs, flag=False):
    N = M2.shape[0]
    mincost = 1
    finalInd = None
    
    for r in range(runs):
        perm = np.random.permutation(N)
        M2temp = M2[perm, :][:, perm]
        Mreord, Mind, align_cost = align_matrices(M1, M2temp, dfun, flag)
        
        if align_cost < mincost:
            mincost = align_cost
            finalInd = perm[Mind]

        if flag:
            print align_cost
        
    return mincost, finalInd