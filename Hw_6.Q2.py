import numpy as np

# Activation Functions
class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, dA):
        return dA * (self.output > 0)

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, dA):
        return dA * (self.output * (1 - self.output))

# Loss Function
class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

# Layer
class Layer:
    def __init__(self, n_input, n_neurons, activation=None):
        self.weights = np.random.randn(n_input, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation() if activation else None
    
    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.biases
        if self.activation:
            return self.activation.forward(self.output)
        return self.output


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss):
        self.loss = loss
    
    def forward(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
        loss_value = self.loss.forward(x, y)
        return loss_value


nn = NeuralNetwork()
nn.add_layer(Layer(2, 64, ReLU))
nn.add_layer(Layer(64, 64, ReLU))
nn.add_layer(Layer(64, 1))  # Output layer
nn.set_loss(MSELoss())

# Dummy data
X = np.random.randn(10, 2)
Y = np.random.randn(10, 1)

# Forward propagation
loss = nn.forward(X, Y)
print("Loss:", loss)
