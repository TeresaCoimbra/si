from abc import ABC, abstractmethod
from typing import MutableSequence
from .model import Model
from scipy import signal
import numpy as np

class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1,output_size))
    
    def setWeights(self, weights, bias):
        '''Sets the weights for the neural network. '''
        if(weights.shape!=self.weights.shape):
            raise ValueError(f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        if(bias.shape!=self.bias.shape):
            raise ValueError(f"Shapes mismatch {bias.shape} and {self.bias.shape}")
        self.weights = weights
        self.bias = bias
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward(self, output_error, learning_rate):
        '''Computes dE/dW, dE/dB for a given output erros = dE/dY
        Returns input error = dE/dX to feed the previous layer'''
        # computing the weights error: dE/dW = X.T*dE/dY
        weights_error = np.dot(self.input.T, output_error) 
        # bias error dE/dB=dE/dY
        bias_error = np.sum(output_error, axis=0)
        # error dE/dX to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        # update parameters
        self.weights -= learning_rate*weights_error
        self.bias -= learning_rate*bias_error
        return input_error
    
    def setweights(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
class Activation(Layer):
    
    def __init__(self, activation):
        self.activation = activation
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        '''Passes the error relative to X (the received input from the previous layer)
        Learning rate is not used because there is no learnable parameters'''
        return np.multiply(self.activation.prime(self.input), output_error)

class NN(Model):
    def __init__(self, epochs = 1000, lr=0.001, verbose = True):
        '''Neural network model. 
        Default loss function : MSE.
        :param float lr: The learning rate.'''
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime
        
    def fit(self):
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)
            
            # backward propagation
            error = self.loss_prime(y, output)  # error based on previous predictions
            for layer in reversed(self.layers): # passing the error in an inverse order
                error = layer.backward(error, self.lr)
            
            # calculate average error on all samples
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:  # add parameter to print results in epochs
                print(f"epoch{epoch +1}/{self.epochs} error={err}")
                
        if not self.verbose:
            print(f"error={err}")
        self.is_fitted = True
    
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, input_data):
        assert self.is_fitted, 'Model must be fit'
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predict'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)
        
    
