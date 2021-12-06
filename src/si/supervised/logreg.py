from numpy.core.numeric import identity
from numpy.lib.function_base import gradient
from .model import Model
from ..util.metrics import mse
import numpy as np

class LogisticRegression(Model):

    def __init__(self, gd = False, epochs = 1000, lr = 0.001):
        '''Linear regression Model
        epochs: number of epochs 
        lr: learning rate for GD
        '''
        super(LogisticRegression,self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self,dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0],1)), X))  # acrescentar o nosso x só com 1 que corresponde ao termo independente
        self.X = X
        self.Y = Y
        # Closed form or GD
        self.train(X,Y)  # implement closed train form (see notes)
        self.is_fitted = True
    
    def train(self,X,Y):
        n = X.shape[1]
        self.hsitory = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            x = np.dot(X, self.theta)
            h = sigmoid(z)
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self,X):
        p = self.probability(X)
        res = 1 if p >=0.5 else 0
        return res

    def cost(self):
        pass