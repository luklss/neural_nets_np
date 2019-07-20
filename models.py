import numpy as np

class SimpleNet:
    """ Implements the simplest neural network I could think of.
    A single neuron and bias: y = sigmoid(x) + b. It accepts
    only single inputs and outputs a single number."""



    def __init__(self):

        self.w = np.random.randn()
        self.b = np.random.randn()

    def fit(self,
            x,
            epochs,
            lr = 0.01):

        for epoch in enumerate(epochs):
            print("epoch {} started".format(epoch))


    def predict(self, x):
        return self.w * x + self.b




    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))


