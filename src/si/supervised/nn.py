from abc import ABC, abstractmethod
from typing import MutableSequence
import numpy as np

from numpy.core.fromnumeric import size, transpose
from scipy.signal.ltisys import LinearTimeInvariant

from .model import Model
from scipy import signal

from si.util.metrics import mse, mse_prime
from si.util.im2col import pad2D, im2col, col2im

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
        '''Computes dE/dW, dE/dB for a given output error = dE/dY
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
    
    def add(self, layer):
        self.layers.append(layer)
        
    def fit(self, dataset):
        X, Y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)
            
            # backward propagation
            error = self.loss_prime(Y, output)  # error based on previous predictions

            for layer in reversed(self.layers): # passing the error in an inverse order
                error = layer.backward(error, self.lr)
            
            # calculate average error on all samples
            err = self.loss(Y, output)
            self.history[epoch] = err
            if self.verbose:  # add parameter to print results in epochs
                print(f"epoch{epoch +1}/{self.epochs} error={err}")
            else:
                print("\r", f"epoch {epoch +1}/{self.epochs} error = {err}")
        self.is_fitted = True
    
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

class Flatten(Layer):

    def forward(self, input):
        self.input_shape= input.shape
        output = input.reshape(input.shape[0],-1)
        return output
    
    def backward(self,output_error, lr):
        return output_error.reshape(self.input_shape)

class Conv2D(Layer):
    def __init__(self, input_shape,kernel_shape, layer_depth, stride = 1, padding = 0):
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride             #step of the convolution
        self.padding = padding
        self.weights = np.random.rand(kernel_shape[0],kernel_shape[1],
                                      self.in_ch, self.out_ch) -0.5

        self.bias = np.zeros((self.out_ch,1))

    def forward(self,input_data):
        s = self.stride
        self.X_shape = input_data.shape
        _, p = pad2D(input_data, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input_data.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2-fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 -fc) / s + 1)

        # convert X and w into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input_data, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)
        return output_data


    def backward(self, output_error, learning_rate):
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        db = np.sum(output_error, axis = (0,1,2))
        db = db.reshape(out_ch,)

        dout_reshaped = output_error.transpose(1,2,3,0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dX_col = W_reshape.T @ dout_reshaped
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, (p,p,p,p),self.stride)

        self.weights -= learning_rate*dW
        self.bias -= learning_rate*db

        return input_error 

    def predict(self, input_data):
        assert self.is_fitted, 'Model must be fit before predicting'
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predicting'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)
        


class Pooling2D(Layer):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def pool(X_col):
        raise NotImplementedError

    def dpool(dX_col,dout_col,pool_cache):
        raise NotImplementedError
    
    def forward(self, input):
        self.X_sahpe = input.shape
        n, h, w, d = input.shape #n imagens, altura, comprimento, n canais
        # calc output size
        h_out = (h-self.size)/self.stride + 1
        w_out = (w-self.size)/self.stride + 1
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension')
        
        h_out,w_out = int(h_out), int(w_out)

        #X_reshaped = input.reshape(n*d,1,h,w)
        X_reshaped = input.reshape()
        self.X_col = im2col(X_reshaped, self.size, self.size, padding=0, stride=self.stride) 
      
        out, self.max_idx = self.pool(self.X_cool)
        out = out.reshape(h_out,w_out,n,d)
        out = out.transpose(2,3,0,1)
        return out

    def backward(self, output_error, learning_rate):
        n, w, h, d = self.Z_shape
        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(2,3,0,1).ravel()
        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = col2im(dX_col, (n*d, 1, h,w), self.size, self.size, padding =0, stride=self.stride)
        dX=dX.reshape(self.X_shape)

        return dX
class MaxPoling(Pooling2D):
    def __init__(self, region_shape):

        self.region_shape = region_shape
        self.region_h, self.region_w = region_shape

    def pool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def dpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)]=dout_col
        return dX_col

    def forward(self,input_data):
        self.X_input = input_data
        _, self.input_h, self.input_w, self.input_f = input_data.shape

        self.out_h = self.input_h // self.region_h
        self.out_w = self.input_w // self.region_w
        output = np.zeros((self.out_h, self.out_w, self.input_f))

        for image, i, j in self.iterate_regions():
            output[i, j] = np.amax(image)
        return output

    def backward(self,output_error, lr):
        n, w, h, d = self.X_shape

        dX_col = np.zeros_like(self.X_shape)
        dout_col = output_error.transpose(2,3,0,1).ravel()

        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = col2im(dX_col, (n*d, 1, h, w), pad=0, stride= self.stride)
        #dX = col2im(dX_col, self.X_shape,(n*d,1,h,w), pad=0, stride = self.stride) # rearrange matrix columns into blocks 
        dX = dX.reshape(self.X_shape)


    def iterate_regions(self):
        for i in range(self.out_h):
            for j in range(self.out_w):
                image = self.X_input[(i * self.region_h): (i * self.region_h + 2), (j * self.region_h):(j * self.region_h + 2)]
                yield image, i, j



