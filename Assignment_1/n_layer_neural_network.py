from three_layer_neural_network import *
import numpy as np
from Assignment_1.Layer import Layer
def generate_data():
    # np.random.seed(0)
    # X, y = datasets.make_moons(200, noise=0.20)

    np.random.seed(0)
    X, y = datasets.make_blobs(n_samples=200, centers=2, n_features=2, random_state=0)

    return X, y

class DeepNeuralNetwork(object):
    def __init__(self, dimArray, input_dim, actFun_type, reg_lambda=0.01):
        self.dimArray = dimArray
        self.idimArraynputSize = input_dim
        self.layers = []
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.width = len(self.dimArray)

        # initiate layers
        for i in range(len(dimArray)):
            if i == 0:
                temp_layer = Layer(i, dimArray[i], input_dim, self.width)
            elif i == len(dimArray) - 1:
                temp_layer = Layer(i, dimArray[i], dimArray[i - 1], self.width)
            else:
                temp_layer = Layer(i, dimArray[i], dimArray[i - 1], self.width)
            self.layers.append(temp_layer)


    def feedforward(self, X, type):
        width = len(self.dimArray)
        for i in range(width):
            if(i == 0):
                output = self.layers[i].feedforward(X, self.actFun_type) # input layer
            elif i < width:
                output = self.layers[i].feedforward(output, self.actFun_type)  #hidden
                self.probs = output #record the current a

        return None


    def backprop(self, X, y, type):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        width = len(self.dimArray)
        for i in range(width -1, -1, -1):
            if(i == width - 1): #Last layer
                da = self.layers[i].backprop(X, y, None, self.actFun_type)
            else:
                da = self.layers[i].backprop(X, y, da, self.actFun_type)


    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss
        y_One_Hot = np.zeros((len(y), 2))
        for i in range(0, len(y)):
            if (y[i] == 0):
                y_One_Hot[i, 0] = 1
                y_One_Hot[i, 1] = 0
            else:
                y_One_Hot[i, 0] = 0
                y_One_Hot[i, 1] = 1
        data_loss = - np.sum(y_One_Hot * np.log(self.probs)) / num_examples
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.backprop(X, y, lambda x: self.diff_actFun(x, type=self.actFun_type))

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


    def actFun(self, z, type):
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


    def diff_actFun(self, z, type):
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

    def predict(self, X):
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)

def main():
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
    model = DeepNeuralNetwork([2, 20, 2], 2, actFun_type='sigmoid')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()