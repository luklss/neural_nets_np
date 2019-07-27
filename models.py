import numpy as np

class SimpleNet:
    """ Implements a very simple network for regression, in the form of
    y = sigmoid(x*w1 + b) * w2. The second weight is used so the network
    can output numbers outside of the range(0,1). It accepts only single input and output.
    Not super useful, but it lustrates the concepts well."""



    def __init__(self, error_f, error_f_d, activation, activation_d):

       self.w1 = np.random.randn()
       self.w2 = np.random.randn()
       self.b = np.random.randn()

       self.error_f = error_f
       self.error_f_d = error_f_d
       self.activation = activation
       self.activation_d = activation_d

    def fit(self, x, y, epochs, lr = 0.1):


        for i in range(epochs):
            print("epoch {} started".format(i))


            z = self.w1 * x + self.b
            y_hat = self.activation(z) * self.w2
            error = self.error_f(y, y_hat)
            de_dy = self.error_f_d(y, y_hat)
            dy_dz = self.activation_d(z)
            dy_dw2 = self.activation(z)
            dz_dw1 = x

            de_dw1 = de_dy * dy_dz * dz_dw1
            de_dw2 = de_dy * dy_dw2
            de_db = de_dy * dy_dz

            self.w1 = self.w1 + (lr * de_dw1)
            self.w2 = self.w2 + (lr * de_dw2)
            self.b = self.b + (lr * de_db)

            print("error was {}".format(error))
#            print("z was {}".format(z))
#            print("y_hat was {}".format(y_hat))
#            print("de_dy was {}".format(de_dy))
#            print("dy_dz was {}".format(dy_dz))
#            print("dz_dw1 was {}".format(dz_dw1))
#            print("de_dw1 was {}".format(de_dw1))
#            print("new w was {}".format(self.w1))
#            print("new b was {}".format(self.b))


    def predict(self, x):
        return sigmoid(self.w1 * x + self.b) * self.w2



def mean_squared_error(y, y_hat):
    return ((y - y_hat) ** 2) / 1


def mean_squared_error_derivative(y, y_hat):
    return y - y_hat



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


