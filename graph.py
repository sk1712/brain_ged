# -*- coding: utf-8 -*-
"""
@author: sk1712
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import read

from nilearn import connectome

conn_kinds = ['correlation', 'partial correlation', 'tangent', 
              'covariance', 'precision']
              

class BrainGraph(object):
    """
    Class representing a brain graph
    """
    
    def __init__(self, subjectID):
        self.subject_ID = subjectID
        self.netx = None
        
        self.parcellation = None

        self.connectivity = {}        
        self.timeseries = {}

        self.vertices = {}        
        self.labels = None
        
        self.features = ''
        
    def set_parcellation(self, parcellation):
        self.parcellation = parcellation

    def conn_log(self):
        weights = np.array(self.connectivity)
        weights[weights > 0] = np.log(weights[weights > 0])
        self.connectivity = weights
        
    def conn_threshold(self, t):
        self.connectivity[self.connectivity < t] = 0
        
    def resample_weights(self, mu, sigma):
        # Get the dimensions of the connectivity matrix
        dim = np.size(self.connectivity, axis=0)
        
        # Get the upper off-diagonal elements of W as a vector
        idx = np.triu_indices(dim, k=1)
        V = self.connectivity[idx]
        # Get the non-zero indices of the vector
        nzI = np.nonzero(V)
        
        #mu = np.mean(V[nzI])
        Vnew = sigma * np.random.randn(np.size(V[nzI])) + mu
        Vnew = np.sort(Vnew)
        
        # Find non-zero indices and order them
        I = np.argsort(V[nzI])
        
        # V(I) returns the ordered weights
        Wnew = np.zeros(np.size(V))
        Wnew[nzI[0][I]] = Vnew
        
        # Save new weights to connectivity matrix
        self.connectivity[idx] = Wnew
        self.connectivity.T[idx] = Wnew
        
    def from_file(self, fileName):
        """
        Load node labels and connectivity matrix from file
        """
        self.connectivity = read.read_network(fileName)

    def to_networkx(self):
        """
        Get a networkx representation of the graph
        """
        
        if self.netx is not None:
            self.netx.clear()
        
        N = self.parcellation.nodes
        # Construct graph G
        G = nx.Graph()
        
        # Add the vertices
        if np.size(self.parcellation.anat_regions) > 0:
            for i in range(N):
                G.add_node(i, label=self.parcellation.anat_regions[i, :],
                           hemisphere=self.parcellation.hemisphere[i, 0])
        else:
            for i in range(N):
                G.add_node(i, label=i)
    
        # Keep and add the non-zero weights
        idx = np.triu_indices(N, 1)
        weights = self.connectivity[idx]
        nnz = np.nonzero(weights)
        idx = np.asarray(idx).T
        
        # Add edges and weights to graph      
        edge_w = weights[nnz]
        edges = np.hstack((idx[nnz], edge_w[..., np.newaxis]))
        edge_list = edges.tolist()    
        edges = [tuple(i) for i in edge_list]
        G.add_weighted_edges_from(edges)
        
        # Inverted weights
        inv_edge_w = 1.0 / edge_w
        inv_edges = np.hstack((idx[nnz], inv_edge_w[..., np.newaxis]))
        inv_edge_list = inv_edges.tolist()    
        inv_edges = [tuple(i) for i in inv_edge_list]
        G.add_weighted_edges_from(inv_edges, weight='inverse')
        
        self.netx = G
        
        return G
        
    def add_features(self, feature_list):
        # Network features added to node labels
        if self.netx is None:
            raise RuntimeError('Class netx attribute is None')
        else:
            all_features = []
            
            # Local graph measures as node attributes
            for f in feature_list:
                if f == 'degree':
                    all_features.append(nx.degree(self.netx))
                elif f == 'strength':
                    all_features.append(nx.degree(self.netx, weight='weight'))
                elif f == 'clustering':
                    all_features.append(nx.clustering(
                                        self.netx, weight='weight'))
                else:
                    raise ValueError('This is an unknown node feature')
            
            # Create a data frame with all the node attributes
            df = pd.DataFrame(all_features)
            
            # Calculate the average and std of the above values for 
            # neighbouring nodes and add as node attributes
            node_features = {}
            for node in self.netx.nodes():
                node_features[node] = df[self.netx.neighbors(node)].mean(axis=1).as_matrix()
                #np.hstack((
                #    df[self.netx.neighbors(node)].mean(axis=1).as_matrix(),
                #    df[self.netx.neighbors(node)].std(axis=1).as_matrix() ))
                
            nx.set_node_attributes(self.netx, 'features', node_features)
            self.features = node_features.values()
        
    def get_features(self, feature_list):
        # Get network features as a feature vector
        if self.netx is None:
            raise RuntimeError('Class netx attribute is None')
        else:
            all_features = []
            
            # Local graph measures as node attributes
            for f in feature_list:
                if f == 'degree':
                    feature = nx.degree(self.netx).values()
                elif f == 'strength':
                    feature = nx.degree(self.netx, weight='weight').values()
                elif f == 'clustering':
                    feature = nx.clustering(self.netx, weight='weight').values()
                    
                all_features.append(np.mean(feature))
                all_features.append(np.std(feature))
                
            return np.asarray(all_features)
    
    def set_paths(self, dataPath, folderName):
        """Set data paths """
        self.data_path = dataPath
        self.folder_name = folderName
        
    def load_timeseries(self, hemisphere=None, edgeType='full', sessionID=None):
        """ Load timeseries """
        
        if self.folder_name == "" or self.data_path == "":
            raise RuntimeError('Data paths need to be specified.')
            
        if hemisphere is None:
            self.edge_type = edgeType
            self.session_ID = sessionID
            self.load_timeseries('L', edgeType, sessionID)
            self.load_timeseries('R', edgeType, sessionID)
            
        else:
            full_path = os.path.join(self.data_path, self.subject_ID,
                                     self.folder_name)
    
            timeseries = read.read_timeseries(full_path, self.subject_ID, 
                                              hemisphere, sessionID)
            self.timeseries[hemisphere] = timeseries

    def load_vertex_coords(self, hemisphere):
        """ Load vertex coordinates """
        
        full_path = os.path.join(self.data_path, self.subject_ID,
                                 self.folder_name)
        
        vertices = read.read_vertex_coords(full_path, self.subject_ID, 
                                           hemisphere)
        self.vertices[hemisphere] = vertices
        
    def get_connectivity(self, parcels):
        # Initialize empty array to store average timeseries for each ROI
        average_timeseries = np.zeros((0, 
                                       np.size(self.timeseries['L'], 
                                               axis=1)))
                                               
        for hem in ['L', 'R']:
            # For each hemisphere
            unique_labels = np.unique(parcels[hem]).tolist()
        
            for l in unique_labels:
                # Retrieve indices of timeseries corresponding to the label l
                binary = (parcels[hem] == l).flatten()
                average_timeseries = np.vstack((average_timeseries, np.mean(
                    self.timeseries[hem][binary, :], axis=0)))
                       
        if self.edge_type == 'full':
            corr = np.corrcoef(average_timeseries)
            np.fill_diagonal(corr, 0)
            # Fisher's z-transform
            self.connectivity = np.arctanh(corr)
        elif self.edge_type in conn_kinds:
            conn_measure = connectome.ConnectivityMeasure(kind=self.edge_type)
            self.connectivity = np.squeeze(
            conn_measure.fit_transform([average_timeseries.T]))
        else:
            raise ValueError('Unknown edge type')
                
        return self.connectivity
            
    def calculate_network(self, method, parcel_centers=False):
        """Calculate connectivity matrix based on functional parcellation
        
        :param method:     Method used for the functional parcellation
        :param hemisphere: Hemisphere to which the functional parcellation 
                           corresponds
        """

        if self.parcellation is None:
            raise RuntimeError('Parcellation needs to be specified.')
        
        self.method = method
        
        # Specify number of parcels
        num_parcels = self.parcellation.nodes

        # Initialize empty array to store average timeseries for each ROI
        average_timeseries = []
        
        for l in range(num_parcels):
            members = self.parcellation.members[l, 0][0]
            hemisphere = self.parcellation.hemisphere[l, 0]
            # Retrieve indices of timeseries corresponding to the label l
            average_timeseries.append(np.mean(
            self.timeseries[hemisphere][members, :], axis=0))
            
        average_timeseries = np.vstack(average_timeseries)
                       
        if self.edge_type == 'full':
            corr = np.corrcoef(average_timeseries)
            np.fill_diagonal(corr, 0)
            # Fisher's z-transform
            self.connectivity = np.arctanh(corr)
        elif self.edge_type in conn_kinds:
            conn_measure = connectome.ConnectivityMeasure(kind=self.edge_type)
            self.connectivity = np.squeeze(
            conn_measure.fit_transform([average_timeseries.T]))
        else:
            raise ValueError('Unknown edge type')
            
        if parcel_centers:  
            # Initialize empty array to store labels for each ROI
            node_labels = np.zeros((self.parcellation.midth_coords))
        
            for l in range(num_parcels):
                # Get parcel center coordinates
                parcel_center = self.get_parcel_center(
                self.timeseries[hemisphere][members, :],
                self.vertices[hemisphere][members, :])
                node_labels = np.vstack((node_labels, parcel_center))
            
            self.labels = node_labels
            
    def get_parcel_center(self, timeseries, vertices):
        """Get the parcel centers as the vertices with the highest average correlation within a parcel

        :param timeseries: Timeseries for all parcel vertices
        :param vertices:   Coordinates for parcel vertices
        """

        corr = np.corrcoef(timeseries)
        corr_mean = np.mean(corr, axis=1)
        # The parcel center is the voxel with the highest average correlation
        center = np.argmax(corr_mean)
        
        return vertices[center, :]
