from abc import abstractmethod
from typing import MutableSequence
from .model import Model
from scipy import signal

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
        return NotImplementedError
    
class Activation(Layer):
    
    def __init__(self, activation):
        self.activation = activation
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return NotImplementedError

class NN(Model):
    def __init__(self, epochs = 1000, lr=0.001, verbose = True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime
        
    def fit(self):
        return NotImplementedError
    
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
        
    
