import numpy as np
import scipy as stats

class VarianceTreshold

    def __init__(self, treshold=0):
        """
        Variance treshold : baseline for feature selection
        :param treshold: non negative treshold value
        """
        if treshold < 0:
            warning.warn("The treshold must be a non-negative value")
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
        condition = self.var > self.treshold # boolean array
        idxs = [i for i in range(len(condition)) if condition[i]]
        X_trans = X[:,idx]
        xnames = [dataset._xnames[i] for i in idx]
        if inline: 
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset 
        else:


        # index = []
        # for i in range(X.shape[1]):
        #     if self._var[i] > treshold:
        #         index.append(i)

    #def fit_transform():
        #self.fit(dataset)
        #return.self.transform(dataset, inline=inline)


## select K Best

# metodo transform - recebe o dataset e elimina no dataset todas as features que têm uma variância igual ou inferior ao treshold
# fit_transform
