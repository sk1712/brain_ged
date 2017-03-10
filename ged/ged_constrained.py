# -*- coding: utf-8 -*-
"""
@author: sk1712
"""

from scipy.spatial import distance
import numpy as np
import sys

from ged_base import GedBase
from ged_edge import GedEdge


class GedConstrained(GedBase):
    """ 
    Base class from Graph Edit Distance
    """
    
    def __init__(self, g1, g2, greedy=False, verbose=False):
        GedBase.__init__(self, g1, g2, greedy, verbose)
                  
        self.N = g1.number_of_nodes()
        self.M = g2.number_of_nodes()
        
        # The node labels
        self.l1 = [g1.node[i]['label'] for i in range(self.N)]
        self.l2 = [g2.node[j]['label'] for j in range(self.M)]
        
        self.hem1 = [g1.node[i]['hemisphere'] for i in range(self.N)]
        self.hem2 = [g2.node[j]['hemisphere'] for j in range(self.M)]
        
    def insert_cost(self, i, j):       
        if i == j:
            # Need to add the edge cost
            v_j = self.g2.nodes()[j]
            edge_list = [self.g2.edge[v_j][e]['weight'] 
                        for e in self.g2.neighbors(v_j)]
            cost = sum(edge_list)
        else:
            cost = sys.maxint
            
        return cost

    def delete_cost(self, i, j):
        if i == j:                     
            # Need to add the edge cost                                            
            v_i = self.g1.nodes()[i]
            edge_list = [self.g1.edge[v_i][e]['weight'] 
                        for e in self.g1.neighbors(v_i)]
            cost = sum(edge_list)
        else:
            cost = sys.maxint
            
        return cost
        
    def substitute_cost(self, i, j):
        if self.hem1[i] == self.hem2[j]:
            cost = 0

            cos_dist = distance.cosine(self.l1[i], self.l2[j])
            if cos_dist < 1.0:
                # If nodes belong to the same hemisphere and have 
                # the same label for at least one vertex then calculate edge edit distance
                
                cost = self.edge_diff(i, j)
                #print("Edit distance for nodes %d, %d is %f" % (i, j, cost))
            else:
                cost = sys.maxint
                
            return cost
        else:
            # else return infinite cost
            return sys.maxint
        
    def edge_diff(self, i, j):      
        edges_i = len(self.g1[self.g1.nodes()[i]])
        edges_j = len(self.g2[self.g2.nodes()[j]])
        
        if edges_i == 0:
            v_j = self.g2.nodes()[j]
            edge_list = [self.g2.edge[v_j][e]['weight'] 
                        for e in self.g2.neighbors(v_j)]
            return sum(edge_list)
            
        elif edges_j == 0:
            v_i = self.g1.nodes()[i]
            edge_list = [self.g1.edge[v_i][e]['weight'] 
                        for e in self.g1.neighbors(v_i)]
            return sum(edge_list)
        
        edge_ged = GedEdge(self.g1, self.g2, self.greedy, i, j, verbose=False)
        edge_distance = edge_ged.distance()
        
        return edge_distance
          
    def distance(self):
        """
        Total distance between the two graphs
        """
        rows, cols, costs = self.calculate_costs()
        print cols[:self.N]
        #M1 = nx.to_numpy_matrix(self.g1)
        #M2 = nx.to_numpy_matrix(self.g2)
        
        self.Mindices = cols[:self.N]
        return np.sum(costs)
