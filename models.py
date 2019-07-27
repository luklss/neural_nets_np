import numpy as np

class SimpleNet:
    """ Implements the simplest neural network I could think of.
    A single neuron and bias: y = sigmoid(x) + b. It accepts
    only single inputs and outputs a single number."""



    def __init__(self, error_f, error_f_d, activation, activation_d):

       self.w = np.random.randn()
       self.b = np.random.randn()

       self.error_f = error_f
       self.error_f_d = error_f_d
       self.activation = activation
       self.activation_d = activation_d

    def fit(self, x, y, epochs, lr = 0.1):


        for i in range(epochs):
            print("epoch {} started".format(i))


            z = self.w * x + self.b
            y_hat = self.activation(z)
            error = self.error_f(y, y_hat)
            de_dy = self.error_f_d(y, y_hat)
            dy_dz = self.activation_d(z)
            dz_dw = x

            de_dw = de_dy * dy_dz * dz_dw
            de_db = de_dy * dy_dz

            self.w = self.w + (lr * de_dw)
            self.b = self.b + (lr * de_db)

            print("error was {}".format(error))
#            print("z was {}".format(z))
#            print("y_hat was {}".format(y_hat))
#            print("de_dy was {}".format(de_dy))
#            print("dy_dz was {}".format(dy_dz))
#            print("dz_dw was {}".format(dz_dw))
#            print("de_dw was {}".format(de_dw))
#            print("new w was {}".format(self.w))
#            print("new b was {}".format(self.b))


    def predict(self, x):
        return sigmoid(self.w * x + self.b)



def mean_squared_error(y, y_hat):
    return ((y - y_hat) ** 2) / 1


def mean_squared_error_derivative(y, y_hat):
    return y - y_hat



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


