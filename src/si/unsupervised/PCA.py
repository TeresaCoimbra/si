import numpy as np
from si.util.util import StandardScaler

class PCA:

    __init__(self, ncomponents = 2, using = "svd"):
        # ncomponents must be int
        if ncomponents > 0 and isinstance(ncomponents, int):
            self.ncomponents = round(ncomponents)
        else:
            raise Exception("Number of components must be non negative and an integer")

    def fit(self):
        pass
    
    def transform(self, dataset):
        #Scale the features
        scaled = StandardScaler().fit_transform(dataset).X.T
        # using numpy.linalg.svd
        if using.lower()  == "svd": self.u, self.s, self.vh = np.linalg.svd(scaled)
        else:
            self.cov_matrix = np.cov(scaled_feature)   # covariance matrix 
            # s are eigenvalues, u are eigenvectors
			self.s, self.u = np.linalg.eig(self.cov_matrix)  # Compute the eigenvalues and right eigenvectors of a square array.

		self.idxs = np.argsort(self.s)[::-1]  # sort the indexes
        #self.eigen_val =  self.s[self.idxs]   # reorganize by index
		self.eigen_vect = self.u[:, self.idxs]  # reorganize by column index
		self.sub_set_vect = self.eigen_vect[:, :self.n_comp]  # ordered vectors
		return scaled_feature.T.dot(self.sub_set_vect)


    def variance_explained(self):
        # find the explained variance   
    	sum_ = np.sum(self.eigen_val)
		percentage = [i / sum_ * 100 for i in self.eigen_val]
		return np.array(percentage)

	def fit_transform(self, dataset):
		trans = self.transform(dataset)
		exp = self.explained_variances()
		return trans, exp