# -*- coding: utf-8 -*-
"""
@author: sk1712
"""

from scipy import spatial
import numpy as np
import scipy.io as sio

import read


class Parcellation(object):
    """ 
    Class representing a brain parcellation for the whole cortex
    """

    def __init__(self, subjectID):
        """ Initialize parcellation class """
        self.subject_ID = str(subjectID)
        
        self.nodes = 0
        self.hemisphere = []
        self.anat_regions = []
        self.members = []
        self.spherical_coords = []
        self.midth_coords = []
        
        self.parcels = {}

    def from_file(self, infile, hemisphere, idx=None):
        """ Get information about a parcellation from infile """
        
        parc = read.read_parcels(self.subject_ID, infile, hemisphere, idx)
        unique_labels = np.unique(parc).tolist()
        
        desikan_labels = read.read_desikan_labels(self.subject_ID, 
                                                  hemisphere).flatten()
        vertices = read.read_vertex_coords(self.subject_ID, hemisphere)
        midth_vertices = read.read_midth_coords(self.subject_ID, hemisphere)
                                          
        for l in unique_labels:
            binary = (parc == l).flatten()
            
            # Fill in the hemisphere information
            self.hemisphere.append(hemisphere)
            
            # Fill in the top anatomical label appearing in this parcel
            self.anat_regions.append(np.bincount(desikan_labels[binary],
                                                 minlength=36))
            
            # Fill in the parcel members
            self.members.append(np.nonzero(parc == l)[0].T)
            
            # Fill in the spherical coordinates
            parcel_coords = vertices[binary, :]
            
            distance_v = spatial.distance.pdist(parcel_coords)
            distance = spatial.distance.squareform(distance_v)
            parcel_center = np.argmin(np.sum(distance, axis=0))
            self.spherical_coords.append(parcel_coords[parcel_center, :])
            
            midth_coords = midth_vertices[binary, :]
            self.midth_coords.append(midth_coords[parcel_center, :])
            
            self.nodes += 1
            
        print "Done reading file for subject " + self.subject_ID
        
    def from_file_plain(self, infile, hemisphere, idx=None):
        """ Get information about a parcellation from infile """
        
        parc = read.read_parcels(self.subject_ID, infile, hemisphere, idx)           
        self.parcels[hemisphere] = parc
            
        print "Done reading file for subject " + self.subject_ID

    def to_numpy(self):
        self.hemisphere = np.vstack(self.hemisphere)
        members = np.zeros((len(self.members), 1), dtype=object)
        for i in range(len(self.members)):
            members[i, 0] = self.members[i].tolist()
        self.members = members
        self.members = self.members[..., np.newaxis]
        self.spherical_coords = np.vstack(self.spherical_coords)
        self.midth_coords = np.vstack(self.midth_coords)
        
    def load(self, infile):
        """ Load parcellation from file """
        
        parcellation = sio.loadmat(infile)
        
        assert((parcellation['hemisphere'].shape[0] == 
               parcellation['labels'].shape[0]) and 
               (parcellation['labels'].shape[0] == 
               parcellation['members'].shape[1]))
        
        self.nodes = parcellation['members'].shape[1]
        self.hemisphere = parcellation['hemisphere']
        self.anat_regions = parcellation['labels']
        self.members = parcellation['members'].T
        self.spherical_coords = parcellation['sphere_coords']
        self.midth_coords = parcellation['midth_coords']