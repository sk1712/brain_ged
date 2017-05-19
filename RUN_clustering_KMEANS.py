# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:44:10 2016

@author: sa1013
"""

"""
Kmeans clustering to learn a brain parcellation from rest fMRI
====================================================================

We use spatially-constrained Ward-clustering to create a set of
parcels. These parcels are particularly interesting for creating a
'compressed' representation of the data, replacing the data in the fMRI
images by mean on the parcellation.

"""

import warnings
import time
import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import decomposition

warnings.filterwarnings("ignore")

Ks = [25, 35, 50, 75, 100]
#Ks.extend(range(100,260,25))

alpha = 1.0 # just functional

sessions = '12'
hems = 'LR'

mat = loadmat('/vol/medic02/users/sk1712/subjectIDs100_set1.mat')  
subjectIDs = mat['subjectIDs']

# See http://stackoverflow.com/questions/32857029/python-scikit-learn-pca-explained-variance-ratio-cutoff
# for the explanation of variance. Apparently, using k=500 explaines almost 50% of the variance.
### Kmeans ######################################################################

print "Generating parcellations with K-means"
print "Running script for {0} parcels".format(Ks)
for hem in hems:     
    for i, sID in enumerate(subjectIDs):
        subjectID = str(sID[0])
        root = '/vol/vipdata/data/HCP100/'
                
        print "Subject: {0}, ID: {1}".format(i,subjectID)  
        getFrom = root + subjectID + '/processed/'
        #mat = loadmat(getFrom + subjectID + '_N_' + hem + '.mat');
        #neigh = mat['N']  
        #sN = sparse.csr_matrix(neigh)  
        saveTo = getFrom + 'networks/' 
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)   
        for s in sessions: # sessions  
            print "Session: {0}".format(s)
            mat = loadmat(getFrom + subjectID + '_dtseries_fix_' + s + '_normalized_corrected_' + hem + '.mat')
            dtseries = mat['dtseries1'] if s is '1' else mat['dtseries2'] 
            mat = loadmat(getFrom + subjectID + '_midthickness_vertices_' + hem + '.mat')
            coords = mat['vertices']
            mat = loadmat(getFrom + subjectID + '_atlasroi_cdata_' + hem + '.mat')
            cdata = mat['cdata']
            ind = [i for i, j in enumerate(cdata) if j > 0]
            xyz = coords[ind,:]
            # PCA
            pca = decomposition.PCA(n_components=510)
            dtseries_reduced = pca.fit_transform(dtseries)
            print "Explained variance {0:.2f}%".format(100*pca.explained_variance_ratio_.cumsum()[-1])
            merged = dtseries_reduced
            #merged = np.column_stack((alpha*dtseries_reduced,(1-alpha)*xyz))
            print merged.shape

            if hem == 'L':
                sNshape = 29696
            elif hem == 'R':
                sNshape = 29716
		
            wardLabels = np.zeros((sNshape,len(Ks)))
            start = time.time()                    
            for k, K in enumerate(Ks):
                print "{0} -> K: {1}".format(k,K)
                # K-MEANS  
                model = KMeans(n_clusters=K, max_iter=100, 
                               n_init=10, n_jobs=-1)                      
                fitted = model.fit(merged)
                wardLabels[:,k] = fitted.labels_ + 1 
            print "Elapsed time {0} seconds" \
                 .format((time.time() - start))
            saveName = saveTo + subjectID + '_wardLabels_alpha' + str(int(alpha*100)) + '_' + s + '_' + hem
            print "Saved as {0}".format(saveName)
            savemat(saveName + '.mat', {'wardLabels':wardLabels})
            
            
