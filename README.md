## Neural net in numpy

This is a simple implementation of three different neural networks in numpy. The ```SimpleNet``` and ```FlatNet``` are milestones for implementing the ```FullyConnectedNet```.

## Requirements

The requirements, including jupyter notebooks are in ```requirements.txt```.

### SimpleNet

The SimpleNet is basically a neuron in the form of ```y = sigmoid(x*w1 + b) * w2```. ```w2``` is just there so the network can work with values outside of the range(0,1) (for regression), restricted by the sigmoid function. I have implemented this as the simplest network I could think of, in order to get the derivatives right without having to worry about many layers or any linear algrebra. Since this network is only a thought exercise, it is just working for a single example. To run it you can do:

```python
simple_net = SimpleNet(mean_square_error_1d, mean_square_error_derivative, sigmoid, sigmoid_derivative) 
simple_net.fit(x = 0.2, y = 0.9, epochs = 1000, lr = 0.09) 
simple_net.predict(0.2) 
```

### FlatNet

FlatNet is similar to the SimpleNet, except that it has ```n``` layers or neurons. This increases in complexity a little, but there is still no linear algebra involved. You can run it with 50 neuros by:
```python
flat_net = FlatNet(50, mean_square_error_1d, mean_square_error_derivative, sigmoid, sigmoid_derivative)
flat_net.fit(x = 0.2, y = 0.9, epochs = 3000, lr = 0.09)
flat_net.predict(0.2)
```

### FullyConnectedNet

This is the real deal, it has as many layers as you want and with as many neurons as well as training examples as you want. You initialize it passing the shape as an array, including the shape of the input and output. For instance, a network with 5 attributes as input, 2 layers of 5 neurons each and 1 output would be [5,2,2,1]. You can also specify a test set. The basic usage is as follows:
```python
data = datasets.make_moons(n_samples = 100)
x = data[0]
y = np.expand_dims(data[1], 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
full_net = FullyConnectedNet([2, 15, 15, 1], mean_square_error, mean_square_error_derivative, sigmoid, sigmoid_derivative)
full_net.fit(x = X_train, y = y_train, epochs = 1000, lr = 0.1, x_test = X_test, y_test = y_test)
```

See the notebook ```usage_example``` for more!

