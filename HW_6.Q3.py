import numpy as np

# Activation Functions
class ReLU:
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, dA):
        dZ = np.array(dA, copy=True)  # Just converting dz to a correct object.
        dZ[self.input <= 0] = 0  # When z <= 0, you should set dz to 0 as well.
        return dZ

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, dA):
        s = self.output
        return dA * (s * (1 - s))

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

    def backward(self, dA):
        if self.activation:
            dA = self.activation.backward(dA)
        self.dW = np.dot(self.input.T, dA)
        self.db = np.sum(dA, axis=0, keepdims=True)
        self.dInput = np.dot(dA, self.weights.T)
        return self.dInput

# Neural Network
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss):
        self.loss = loss
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true):
        dA = self.loss.backward()
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dW
            layer.biases -= learning_rate * layer.db

# Example Usage
nn = NeuralNetwork()
nn.add_layer(Layer(2, 64, ReLU))
nn.add_layer(Layer(64, 64, ReLU))
nn.add_layer(Layer(64, 1))  # Output layer
nn.set_loss(MSELoss())

# Dummy data
X = np.random.randn(10, 2)
Y = np.random.randn(10, 1)

# Training loop
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    # Forward pass
    y_pred = nn.forward(X)
    loss = nn.loss.forward(y_pred, Y)
    
    # Backward pass
    nn.backward(Y)
    nn.update_weights(learning_rate)
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
