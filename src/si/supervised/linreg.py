from .model import Model
from ..util.metrics import mse
import numpy as np

class LinearRegression(Model):

    def __init__(self, gd = False, epochs = 1000, lr = 0.001):
        '''Linear regression Model
        epochs: number of epochs 
        lr: learning rate for GD
        '''
        super(LinearRegression,self).__init__()
        self.gd = gd
        slef.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self,dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0],1)), X))  # acrescentar o nosso x só com 1 que corresponde ao termo independente
        self.X = X
        self.Y = Y
        # Closed form or GD
        self.train_gd(X, Y) if self.gd else self.train_closed(X, Y) # implement closed train form (see notes)
        self.is_fitted = True
    
    def train_closed(self, X, Y):
        '''uses closed form linear algebra to fit the model.
        theta=inv(XT*X)*XT*y
        '''
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        
    def train_gd(self,X,Y):
        pass
