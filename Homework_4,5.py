import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sigmoid_x = Activation.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.output = 0
        self.delta = 0

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = Activation.sigmoid(weighted_sum)
        return self.output

class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def activate(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

class Parameters:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_to_hidden = Layer(input_size, hidden_size)
        self.hidden_to_output = Layer(hidden_size, output_size)

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = Parameters(input_size, hidden_size, output_size)

    def predict(self, inputs):
        hidden_layer_output = self.params.input_to_hidden.activate(inputs)
        output_layer_output = self.params.hidden_to_output.activate(hidden_layer_output)
        return output_layer_output

class ForwardProp:
    @staticmethod
    def forward(model, inputs):
        hidden_layer_output = model.params.input_to_hidden.activate(inputs)
        output_layer_output = model.params.hidden_to_output.activate(hidden_layer_output)
        return hidden_layer_output, output_layer_output

class LossFunction:
    @staticmethod
    def mean_squared_error(predictions, targets):
        return np.mean((predictions - targets) ** 2)

class BackProp:
    @staticmethod
    def backward(model, inputs, targets, hidden_layer_output, output_layer_output, learning_rate):
        output_error = targets - output_layer_output
        output_delta = output_error * Activation.sigmoid_derivative(output_layer_output)

        # Calculate hidden layer error and delta
        hidden_error = np.dot(model.params.hidden_to_output.weights.T, output_delta)
        hidden_delta = hidden_error * Activation.sigmoid_derivative(hidden_layer_output)

        # Update weights and biases in the hidden-to-output layer
        model.params.hidden_to_output.weights += learning_rate * np.outer(hidden_layer_output, output_delta)
        model.params.hidden_to_output.bias += learning_rate * np.sum(output_delta)

        # Update weights and biases in the input-to-hidden layer
        for i in range(len(model.params.input_to_hidden.neurons)):
            model.params.input_to_hidden.neurons[i].weights += learning_rate * hidden_delta[i] * inputs
            model.params.input_to_hidden.neurons[i].bias += learning_rate * hidden_delta[i]





class GradDescent:
    @staticmethod
    def update_weights(model, inputs, targets, learning_rate):
        hidden_layer_output, output_layer_output = ForwardProp.forward(model, inputs)
        BackProp.backward(model, inputs, targets, hidden_layer_output, output_layer_output, learning_rate)

class Training:
    @staticmethod
    def train(model, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                GradDescent.update_weights(model, inputs[i], targets[i], learning_rate)

# Example usage:
input_size = 2
hidden_size = 3
output_size = 1

model = Model(input_size, hidden_size, output_size)

# Example training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Training the model
Training.train(model, inputs, targets, epochs=1000, learning_rate=0.1)

# Testing the model
for input_data in inputs:
    prediction = model.predict(input_data)
    print(f"Input: {input_data}, Prediction: {prediction}")
