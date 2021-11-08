import numpy as np
import scipy as stats
import warnings
from si.data import Dataset
from copy import copy

class VarianceTreshold:

    def __init__(self, treshold=0):
        """
        Variance treshold : baseline for feature selection
        :param treshold: non negative treshold value
        """
        if treshold < 0:
            warnings.warn("The treshold must be a non-negative value")
            threshold = 0
        self.treshold = treshold

    def fit(self, dataset):
        '''
        Calculate var over dataset.X
        '''
        X = dataset.X
        self._var = np.var(X, axis = 0)
        #self.F, self.p = self.score_func(dataset)

    def transform(self, dataset, inline = False):
        '''
        Deletes features with var <= treshold
        '''
        X = dataset.X
        cond = self.var > self.treshold                      # boolean array - if var > treshold then true
        idxs = [i for i in range(len(cond)) if cond[i]]      # i when cond[i] is true
        X_trans = X[:,idx]                                   # select indexes of features that match the condition var > treshold    
        xnames = [dataset.xnames[i] for i in idx]
        if inline: 
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset 
        else:
            from .dataset import dataset
            return Dataset(copy(X_trans), copy(dataset.Y), xnames, copy(dataset.yname))

    
    
    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

class SelectKBest:

    def __init__(self, k:int, score_func):
