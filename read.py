# -*- coding: utf-8 -*-
"""
@author: sk1712
"""

import numpy as np
import csv
import os
import struct

import scipy.io as sio
from nibabel.gifti import giftiio

# Useful file paths
RESTRICTED_FILE= '/vol/medic02/users/sk1712/RESTRICTED.csv'
UNRESTRICTED_FILE = '/vol/medic02/users/sk1712/unrestricted.csv'
hundredSubjects = '/vol/medic02/users/sk1712/subjectIDs100.txt'

DATA_PATH = '/vol/dhcp-hcp-data/twins_data'
PROCESSED_PATH = 'processed'

# Name of mat variables for node labels and connectivity
LABEL_VAR = 'labels'
CONN_VAR = 'connectivity'

# Number of grayordinates per hemisphere excluding medial wall
VOXEL_TOTAL_L = 29696
VOXEL_TOTAL_R = 29716


def read_subjects(fileName=hundredSubjects):
    """ Get the list of subjects from a given file
        Each line of the file contains a subject ID 
    """
    subject_list = []
    
    f = open(fileName, 'r')
    for line in f:
        subject_list.append(line.strip())
        
    return subject_list
        
    
def read_twins_csv(fileName):
    """"
    Load IDs of twin pairs from csv file

    :param: fileName, csv file containing pairs of subject IDs
    """
    pairs = []    
    
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.append((row['Twin_ID_1'], row['Twin_ID_2']))
    
    return np.asarray(pairs)
    
    
def read_subject_measure(subjectList, attribute, restricted):
    """Get behavioral data from csv file
    
    :param: subjectList, subjects for which the behavioral data is retrieved
    :param: attribute, the attribute to be retrieved
    :param: restricted, whether attribute is restricted or not
    """ 
    attr_list = {}
        
    if restricted:
        fileName = RESTRICTED_FILE
    else:
        fileName = UNRESTRICTED_FILE
    
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Subject'] in subjectList:
                attr_list[row['Subject']] = row[attribute]
                
    return attr_list
    

def read_labels(labelFile, keepLabels=True):
    """
    Return dictionary with node labels as values
    """
    img = giftiio.read(labelFile)   
    label_dict_orig = img.get_labeltable().get_labels_as_dict()
    
    if keepLabels:
        return label_dict_orig
    else:
        # Fill label dictionary for parcellated regions 
        label_dict = {}
        i = 0
        
        for val in label_dict_orig.values():
            label_dict[i] = val
            i += 1
        
        return label_dict
        
        
def read_pconn(filename):
    """
    Read .pconn.nii file
    """
    f=open(filename,"r");
    b=f.read();f.seek(0)
    c=f.read(540)
    hdr=struct.unpack("i8shh8qddd8dqddddddqq80s24siidddddd4d4d4diii16sc15s",c);
    
    # Move to where the matrix data begins
    f.seek(hdr[23]);
    # Matrix size: number of parcels
    N=hdr[9];
    # Read the matrix
    d=f.read(N*N*4);
    f.close();
    str="%df"%(N*N);
    M=struct.unpack(str,d);
    R=np.reshape(M,[N,N])
    
    # Return the matrix
    return R
    
    
def read_timeseries(folderName, subjectID, hemisphere, sessionID=None):   
    """ Load timeseries for both hemispheres """
    
    # Load timeseries data
    if sessionID:        
        file_timeseries = subjectID + '_dtseries_fix_' + \
        sessionID + '_normalized_corrected_' + hemisphere + '.mat'
        timeseries = sio.loadmat(os.path.join(folderName, file_timeseries))['dtseries' + sessionID]
    else:
        file_timeseries = subjectID + \
        '_dtseries_fix_normalized_corrected_' + hemisphere + '.mat'
        timeseries = sio.loadmat(os.path.join(folderName, file_timeseries))['dtseries']
                                
    return timeseries
    
    
def read_vertex_coords(subjectID, hemisphere):
    """
    Get vertex coordinates on the sphere for a specific subject
    """
    folderName = os.path.join(DATA_PATH, subjectID, PROCESSED_PATH)    
    
    file_vertices = subjectID + '_sphere_vertices_' + hemisphere + '.mat'
    vertices = sio.loadmat(os.path.join(folderName, file_vertices))['vertices']
    mask = sio.loadmat(os.path.join(folderName, subjectID + \
                       '_atlasroi_cdata_' + hemisphere + '.mat'))['cdata']
           
    if (np.size(vertices, axis=0) == np.size(mask, axis=0)):
        vertices = vertices[(mask == 1).flatten()]
    
    return vertices
    
    
def read_midth_coords(subjectID, hemisphere):
    """
    Get midthickness coordinates for a specific subject (for graph visualisation)
    """
    subject_path = os.path.join(DATA_PATH, subjectID, PROCESSED_PATH)

    midthi_path = subjectID + '_midthickness_vertices_'  + hemisphere + '.mat'    
    midthi_verts = sio.loadmat(os.path.join(
                               subject_path, midthi_path))['vertices']
    mask = sio.loadmat(os.path.join(subject_path, subjectID + \
                       '_atlasroi_cdata_' + hemisphere + '.mat'))['cdata']
    
    if (np.size(midthi_verts, axis=0) == np.size(mask, axis=0)):
        midth_coords = midthi_verts[(mask == 1).flatten()]
        
    return midth_coords
    
    
def read_desikan_labels(subjectID, hemisphere):
    """
    Get vertex labels from the Desikan-Killiany atlas
    """

    subject_path = os.path.join(DATA_PATH, subjectID, PROCESSED_PATH)
    filename = subjectID + '_aparc_' + hemisphere + '.mat'
    parc = sio.loadmat(os.path.join(subject_path, filename))['aparc']
    
    # replace -1 with 4
    parc = np.where(parc == -1, 4, parc)

    return parc
    
    
def read_parcels(subjectID, filePath, hemisphere, index=0):
    """
    Get parcel assignments for subject with subjectID
    """

    if '.mat' not in filePath:
        parcels = np.loadtxt(filePath, dtype=int)
        
        if parcels.size == parcels.shape[0]:
            parcels = parcels[np.newaxis, :]
            
        parc = parcels[index, :]
        
        # filter out the medial wall
        subject_path = os.path.join(DATA_PATH, subjectID, PROCESSED_PATH)
        mask = sio.loadmat(os.path.join(subject_path, subjectID + \
                       '_atlasroi_cdata_' + hemisphere + '.mat'))['cdata']
                       
        if (np.size(parc, axis=0) == np.size(mask, axis=0)):
            parc = parc[(mask == 1).flatten()]
    else:
        mat = sio.loadmat(filePath)
        if 'parcels' in mat.keys():
            parc = mat['parcels']
        elif 'aparc' in mat.keys():
            parc = mat['aparc']
        
    return parc
    
    
def read_network(netFile):
    """
    Read network from .mat file and return matrices for node labels
    and connectivity
    """

    try:
        mat_dict = sio.loadmat(netFile)
        connectivity = mat_dict[CONN_VAR]
        return connectivity
    except KeyError:
        return np.array([])
