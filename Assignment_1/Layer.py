import numpy as np
from three_layer_neural_network import *

def actFun(z,type):
        if (type.lower() == "tanh"):
            # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            return np.tanh(z)
        elif (type.lower() == "sigmoid"):
            # return 1.0 / ( 1.0 + np.exp(-z))
            return scipy.special.expit(z)
        elif (type.lower() == "relu"):
            return z * (z > 0)
        else:
            raise ValueError("Wrong input type!")


def diff_actFun(z, type):
        if (type.lower() == "tanh"):
            # tanh(x)' = 1 - np.square( (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))
            return 1 - np.power(z, 2)
        elif (type.lower() == "sigmoid"):
            # sigmoid(x)' =  / ( 1 + np.exp(-x)) * (1 - 1. / ( 1 + np.exp(-x)))
            return z * (1 - z)
        elif (type.lower() == "relu"):
            return (z > 0) * 1
        else:
            raise ValueError("wrong input typoe")


class Layer(object):
    def __init__(self, serial, current_dim, input_dim, width):
        self.serial = serial
        self.input_dim = input_dim
        self.current_dim = current_dim
        self.width = width

        self.W = np.random.randn(self.input_dim, self.current_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.current_dim))
        self.dW = None
        self.db = None
        self.dz1= None


    def feedforward(self, X, type):
        self.input_a = X
        self.z = np.dot(X, self.W) + self.b
        self.a = actFun(self.z, type)

        if self.serial == self.width - 1:
            self.probs = np.exp(self.z) / np.sum(np.exp(self.z), axis=1, keepdims=True)
            # self.probs = actFun(self.z, "sigmoid") #<----------fix
            return self.probs

        return self.a


    def backprop(self, X, y, da, type):
    #credit: https://pylessons.com/Neural-network-single-layer-part3/
        if self.serial == self.width - 1:
            num_examples = len(X)
            K = self.probs  # K = an
            K[range(num_examples), y] -= 1  # reference: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
            dz2 = K  # dZn = an - Y
            self.dW = np.dot(self.input_a.T, dz2)
            self.db = np.sum(dz2, axis=0, keepdims=True)
            self.dz1 = np.dot(dz2, self.W.T)

        elif self.serial == 0:
            self.da = da
            dz2 = self.da * diff_actFun(self.a, type)
            self.dW = np.dot(self.input_a.T, dz2)
            self.db = np.sum(dz2, axis=0, keepdims=True)

        else:
            self.da = da
            dz2 = self.da * diff_actFun(self.a, type) # dz = dz * g(z)
            self.dW = np.dot(self.input_a.T, dz2)
            self.db = np.sum(dz2, axis=0, keepdims=True)
            dz_a_prev = self.W
            self.dz1 = np.dot(dz_a_prev, dz2.T).T

        reg_lambda = 0.01
        epsilon = 0.01
        # regularization
        self.dW += reg_lambda * self.W

        # Gradient descent parameter update
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db

        return self.dz1
