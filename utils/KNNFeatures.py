from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import os

import numpy as np


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    '''
        This class should implement KNN features extraction 
    '''
    def __init__(self, k_list=[3,8,32], metric='cosine', n_classes=None, n_neighbors=None, n_jobs=os.cpu_count(),eps=1e-6):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric
        
        if n_neighbors is None:
            self.n_neighbors = max(k_list) 
        else:
            self.n_neighbors = n_neighbors
            
        self.eps = eps        
        self.n_classes_ = n_classes
    
    def fit(self, X, y):
        '''
            Set's up the train set and self.NN object
        '''
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 
                                      metric=self.metric, 
                                      n_jobs=1, 
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        self.NN.fit(X)
        
        # Store labels 
        self.y_train = y
        
        # Save how many classes we have
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_
        
        
    def predict(self, X):       
        '''
            Produces KNN features for every object of a dataset X
        '''
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            
            test_feats=[]
            with Pool(processes=self.n_jobs) as pool:
                xs=[X[i:i+1] for i in range(X.shape[0])]
                test_feats=pool.map(self.get_features_for_one,xs)
            '''
                 *Make it parallel*
                     Number of threads should be controlled by `self.n_jobs`  
                     
                     
                     You can use whatever you want to do it
                     For Python 3 the simplest option would be to use 
                     `multiprocessing.Pool` (but don't use `multiprocessing.dummy.Pool` here)
                     You may try use `joblib` but you will most likely encounter an error, 
                     that you will need to google up (and eventually it will work slowly)
                     
                     For Python 2 I also suggest using `multiprocessing.Pool` 
                     You will need to use a hint from this blog 
                     http://qingkaikong.blogspot.ru/2016/12/python-parallel-method-in-class.html
                     I could not get `joblib` working at all for this code 
                     (but in general `joblib` is very convenient)
                     
            '''
            
            # YOUR CODE GOES HERE
            # test_feats =  # YOUR CODE GOES HERE
            # YOUR CODE GOES HERE
            
            # Comment out this line once you implement the code
            #assert False, 'You need to implement it for n_jobs > 1'
            
            
            
        return np.vstack(test_feats)
        
        
    def get_features_for_one(self, x):
        '''
            Computes KNN features for a single object `x`
        '''

        NN_output = self.NN.kneighbors(x)
        
        # Vector of size `n_neighbors`
        # Stores indices of the neighbors
        neighs = NN_output[1][0]
        
        # Vector of size `n_neighbors`
        # Stores distances to corresponding neighbors
        neighs_dist = NN_output[0][0] 

        # Vector of size `n_neighbors`
        # Stores labels of corresponding neighbors
        neighs_y = self.y_train[neighs] 
        
        
        ## ========================================== ##
        ##              YOUR CODE BELOW
        ## ========================================== ##
        
        # We will accumulate the computed features here
        # Eventually it will be a list of lists or np.arrays
        # and we will use np.hstack to concatenate those
        return_list = [] 
        
        
        ''' 
            1. Fraction of objects of every class.
               It is basically a KNNСlassifiers predictions.

               Take a look at `np.bincount` function, it can be very helpful
               Note that the values should sum up to one
        '''
        for k in self.k_list:
            # YOUR CODE GOES HERE
            sorted_neighs_y=neighs_y[np.argsort(neighs_dist)]
            feats=np.bincount(sorted_neighs_y[:k],None,self.n_classes)/k
            assert len(feats) == self.n_classes
            return_list += [feats]
  
        
        '''
            2. Same label streak: the largest number N, 
               such that N nearest neighbors have the same label.
               
               What can help you: `np.where`
        '''
        
        feats = np.where(neighs_y!=neighs_y[0])
        if len(feats[0])==0:
            feats=[len(neighs_y)]
        else:
            feats=[feats[0][0]]
         
        assert len(feats) == 1
        return_list += [feats]
 
        '''
            3. Minimum distance to objects of each class
               Find the first instance of a class and take its distance as features.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.

               `np.where` might be helpful
        '''
        
        
        sorted_dist_args=np.argsort(neighs_dist)
        sorted_class=neighs_y[sorted_dist_args]
        feats = []   
        for cls in range(self.n_classes):
            if cls in sorted_class:
                feats.append(neighs_dist[np.where(sorted_class==cls)[0]][0])
            else:
                feats.append(999)
        assert len(feats) == self.n_classes
        return_list += [feats]
 
        '''
            4. Minimum *normalized* distance to objects of each class
               As 3. but we normalize (divide) the distances
               by the distance to the closest neighbor.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.
               
               Do not forget to add self.eps to denominator.
               
               
        '''
         
        
        feats = [dist/(neighs_dist[sorted_dist_args][0]+self.eps) if dist != 999 else 999 for dist in feats]

        assert len(feats) == self.n_classes
        return_list += [feats]
 
        '''
            5. 
               5.1 Distance to Kth neighbor
                   Think of this as of quantiles of a distribution
               5.2 Distance to Kth neighbor normalized by 
                   distance to the first neighbor
               
               feat_51, feat_52 are answers to 5.1. and 5.2.
               should be scalars
               
               Do not forget to add self.eps to denominator.
        '''
        for k in self.k_list:
            feat_51 = neighs_dist[sorted_dist_args[:k]][-1]
            feat_52 = feat_51/(neighs_dist[sorted_dist_args[:k]][0]+self.eps)
            
            return_list += [[feat_51, feat_52]]
        '''
            6. Mean distance to neighbors of each class for each K from `k_list` 
                   For each class select the neighbors of that class among K nearest neighbors 
                   and compute the average distance to those objects
                   
                   If there are no objects of a certain class among K neighbors, set mean distance to 999
                   
               You can use `np.bincount` with appropriate weights
               Don't forget, that if you divide by something, 
               You need to add `self.eps` to denominator.
        '''
        feats=[]
        for k in self.k_list:
            sorted_neighs_y=neighs_y[np.argsort(neighs_dist)][:k]
            for cls in range(self.n_classes):
                if cls in sorted_neighs_y:
                    feats.append(np.sum(neighs_dist[np.where(sorted_neighs_y==cls)])/len(np.where(sorted_neighs_y==cls)[0]))
                else:
                    feats.append(999)
            
            # YOUR CODE GOES IN HERE
        assert len(feats) == len(self.k_list)*self.n_classes
        return_list += [feats]
        
        # merge
        knn_feats = np.hstack(return_list)
        
        #assert knn_feats.shape == (239,) or knn_feats.shape == (239, 1)
        return knn_feats