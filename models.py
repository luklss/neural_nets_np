import numpy as np

class SimpleNet:
    """ Implements the simplest neural network I could think of.
    A single neuron and bias: y = sigmoid(x) + b. It accepts
    only single inputs and outputs a single number."""



    def __init__(self):

        self.w = np.random.randn()
        self.b = np.random.randn()

    def fit(self, x, y, epochs, lr = 0.01):


        for i in range(epochs):
            print("epoch {} started".format(i))

            output_deltas = np.zeros(len(x))

            z = self.w * x + self.b
            y_hat = sigmoid(z)
            error = mean_squared_error(y, y_hat)
            de_dy = mean_squared_error_derivative(y, y_hat)
            dy_dz = sigmoid_derivative(z)
            dz_dw = x

            de_dw = de_dy * dy_dz * dz_dw
            de_db = de_dy * dy_dz

            self.w = self.w - lr * de_dw
            self.b = self.b - lr * de_db


    def predict(self, x):
        return sigmoid(self.w * x + self.b)



def mean_squared_error(y, y_hat):
    assert len(y_hat) == len(y)
    return ((y - y_hat) ** 2) / len(y)


def mean_squared_error_derivative(y, y_hat):
    return y - y_hat



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


