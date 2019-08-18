import numpy as np


class FullyConnectedNet:


    def __init__(self, shape, error_f, error_f_d, activation, activation_d):

        self.error_f = error_f
        self.error_f_d = error_f_d
        self.activation = activation
        self.activation_d = activation_d
        self.shape = shape
        self.w = self.initialize_weights()
        self.b = self.initialize_biases()


    def initialize_weights(self):
        weights = []

        for i in range(0, len(self.shape) - 1):
            weights.append(np.random.randn(self.shape[i + 1], self.shape[i]))

        return weights


    def initialize_biases(self):
        biases = []

        for i in range(1, len(self.shape)):
            biases.append(np.random.randn(self.shape[i], 1))

        return biases

    def predict(self, x):

        a = x.T

        for i in range(len(self.w)):
            a = self.activation(np.dot(self.w[i], a) + self.b[i])

        return a.round().T


    def fit(self, x, y, epochs, lr = 0.1):


        for i in range(epochs):

            print("epoch {} started".format(i))

            z = []
            a = []


            de_dw = [np.zeros(w.shape) for w in self.w]
            de_db = [np.zeros(b.shape) for b in self.b]

            # forward pass

            a_previous = x.T
            for i in range(len(self.w)):

                z.append(np.dot(self.w[i], a_previous) + self.b[i])
                a.append(self.activation(z[i]))
                a_previous = a[i]



            # backpropagation
            for i in reversed(range(len(self.w))):

                # let's first get the delta, or de_dz
                if i == len(self.w) - 1: # if it is the output layer
                    de_da = self.error_f_d(y.T, a[i])
                    da_dz = self.activation_d(z[i])
                    de_dz = de_da * da_dz


                else:
                    da_dz = self.activation_d(z[i])
#
                    de_dz = np.dot( self.w[i + 1].T, de_dz) * da_dz


                # now we can calculate the derivatives for w and b
                de_db[i] = de_dz
                dz_dw = a[i - 1]
                de_dw[i] = np.dot(de_dz, dz_dw.T)

            self.w = [w - (lr * de_dw_i) for w, de_dw_i in zip(self.w,de_dw)]
            self.b = [b - (lr * de_db_i) for b, de_db_i in zip(self.b,de_db)]


            error = self.error_f(y.T, a[-1])
            print("error was {}".format(error))





class FlatNet:

    def __init__(self, n_layers, error_f, error_f_d, activation, activation_d):
        self.error_f = error_f
        self.error_f_d = error_f_d
        self.activation = activation
        self.activation_d = activation_d
        self.n_layers = n_layers
        self.w = np.random.randn(self.n_layers) # one weight (neuron) per layer
        self.b = np.random.randn(self.n_layers) # one bias (neuron) per layer



    def predict(self, x):

        a = x

        for i in range(self.n_layers):
            a = self.activation(self.w[i] * a + self.b[i])

        return a

    def fit(self, x, y, epochs, lr = 0.1):


        for i in range(epochs):

            print("epoch {} started".format(i))

            z = []
            a = []

            de_dw = np.zeros(self.n_layers)
            de_db = np.zeros(self.n_layers)

            # forward pass

            a_previous = x
            for i in range(self.n_layers):
                z.append(self.w[i] * a_previous + self.b[i])
                a.append(self.activation(z[i]))
                a_previous = a[i]


            # backpropagation
            for i in reversed(range(self.n_layers)):

                # let's first get the delta, or de_dz
                if i == self.n_layers - 1: # if it is the output layer
                    error = self.error_f(y, a[i])
                    de_da = self.error_f_d(y, a[i])
                    da_dz = self.activation_d(z[i])
                    de_dz = de_da * da_dz


                else:
                    da_dz = self.activation_d(z[i])
                    de_dz = de_dz * self.w[i + 1] * da_dz


                # now we can calculate the derivatives for w and b
                de_db[i] = de_dz
                dz_dw = a[i - 1]
                de_dw[i] = de_dz * dz_dw


            self.w = self.w - (lr * de_dw)
            self.b = self.b - (lr * de_db)




            print("error was {}".format(error))




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

            self.w1 = self.w1 - (lr * de_dw1)
            self.w2 = self.w2 - (lr * de_dw2)
            self.b = self.b - (lr * de_db)

            print("error was {}".format(error))


    def predict(self, x):
        return self.activation(self.w1 * x + self.b) * self.w2



def mean_square_error_1d(y, y_hat):
    return ((y_hat - y) ** 2) / 1


def mean_square_error(y, y_hat):
    n = y.shape[0]
    return np.sum((y_hat - y) ** 2) / n


def mean_square_error_derivative(y, y_hat):
    return y_hat - y



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


