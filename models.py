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
            y,
            epochs,
            lr = 0.01):

        for epoch in enumerate(epochs):
            print("epoch {} started".format(epoch))

            y_hat = self.predict(x)
            error = self.mean_square_error(y, y_hat)


    def predict(self, x):
        return sigmoid(self.w * x + self.b)



    def mean_square_error(y, y_hat):
        assert len(y_hat) == len(y)
        return ((y - y_hat) ^2) / len(x)



    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))


