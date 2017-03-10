# -*- coding: utf-8 -*-
"""
@author: sk1712
"""

import sys
from numpy import linalg
from scipy.spatial import distance

from ged_base import GedBase


class GedEdge(GedBase):
    """
    Calculates the edit distance between the edges of two nodes
    A node is regarded as a graph and edges are regarded as nodes
    """
    
    def __init__(self, g1, g2, greedy, vertex1_id, vertex2_id, verbose):
        v1 = g1.nodes()[vertex1_id]
        v2 = g2.nodes()[vertex2_id]
        
        # Initialize base class
        GedBase.__init__(self, v1, v2, greedy, verbose)
        
        # List of edge weights for v1
        self.e1 = [g1.edge[v1][e]['weight'] for e in g1.neighbors(v1)]
        self.hem1 = [g1.node[v]['hemisphere'] for v in g1.neighbors(v1)]
        self.neigh1 = [g1.node[v]['label'] for v in g1.neighbors(v1)]
        self.N = len(self.e1)
        
        # List of edge weights for v2
        self.e2 = [g2.edge[v2][e]['weight'] for e in g2.neighbors(v2)]
        self.hem2 = [g2.node[v]['hemisphere'] for v in g2.neighbors(v2)]
        self.neigh2 = [g2.node[v]['label'] for v in g2.neighbors(v2)]
        self.M = len(self.e2)
        
    def insert_cost(self, i, j):
        cost = 0
        
        if i == j:
            cost = linalg.norm(self.e2[j])
        else:
            cost = sys.maxint
            
        return cost
        
    def delete_cost(self, i, j):
        cost = 0
        
        if i == j:
            cost = linalg.norm(self.e1[i])
        else:
            cost = sys.maxint
            
        return cost
        
    def substitute_cost(self, i, j):
        # Constrain substitutions between nodes from the same hemisphere
        if self.hem1[i] == self.hem2[j]:

            cos_dist = distance.cosine(self.neigh1[i], self.neigh2[j])

            if cos_dist < 1.0:
                # Edge substitution cost
                cost = cos_dist * distance.euclidean(self.e1[i], self.e2[j])
            else:
                cost = sys.maxint
        else:
            cost = sys.maxint
            
        return cost
