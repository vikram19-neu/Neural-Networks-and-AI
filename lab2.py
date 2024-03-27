import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DataLoader:
    @staticmethod
    def load_images_from_folder(folder, label):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img = img.resize((64, 64)).convert('L')  # Resize and convert to grayscale
                images.append(np.asarray(img) / 255.0)  # Normalize pixel values
                labels.append(label)
        return images, labels

class ActivationFunction:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return z > 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)

class Layer:
    def __init__(self, size_input, size_output, activation_function):
        self.W = np.random.randn(size_input, size_output) * np.sqrt(2. / size_input)
        self.b = np.zeros((1, size_output))
        self.activation = activation_function
        self.A = None
        self.Z = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.Z = np.dot(A_prev, self.W) + self.b
        if self.activation == 'relu':
            self.A = ActivationFunction.relu(self.Z)
        elif self.activation == 'sigmoid':
            self.A = ActivationFunction.sigmoid(self.Z)
        return self.A

    def backward(self, dA, A_prev, lambda_reg=0.1, m=1):
        if self.activation == 'relu':
            dZ = dA * ActivationFunction.relu_derivative(self.Z)
        elif self.activation == 'sigmoid':
            dZ = dA * ActivationFunction.sigmoid_derivative(self.Z)
        
        # Ensure the shapes are aligned for the dot product
        self.dW = np.dot(A_prev.T, dZ) / m + (lambda_reg / m) * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.W.T)
        
        return dA_prev

    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions):
        self.layers = []
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(layer_dims[i-1], layer_dims[i], activation_functions[i-1]))

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_cost(self, AL, Y, lambda_reg=0.1):
        m = Y.shape[0]
        cross_entropy_cost = -np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL)) / m
        L2_cost = sum(np.sum(np.square(layer.W)) for layer in self.layers)
        L2_cost = (lambda_reg / (2*m)) * L2_cost
        cost = cross_entropy_cost + L2_cost
        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, X, lambda_reg=0.1):
        m = Y.shape[0]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = X if i == 0 else self.layers[i-1].A  # Correct handling of A_prev for the first layer
            dA = layer.backward(dA, A_prev, lambda_reg, m)

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def train(self, X, Y, learning_rate, num_iterations, print_cost):
        for i in range(num_iterations):
            AL = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            self.backward_propagation(AL, Y, X)
            self.update_parameters(learning_rate)
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost:.6f}")



# Load the dataset

cat_images, cat_labels = DataLoader.load_images_from_folder('Lab1_cat_dataset', 1)


not_cat_images, not_cat_labels = DataLoader.load_images_from_folder('Lab1_not_cat_dataset', 0)

# Prepare the data
X = np.array(cat_images + not_cat_images).reshape(-1, 64*64)  # Flatten the images
y = np.array(cat_labels + not_cat_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define network architecture
layer_dims = [X_train.shape[1], 10, 8, 8, 4, 1]
activation_functions = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

# Initialize the neural network
nn = NeuralNetwork(layer_dims, activation_functions)

# Train the neural network
nn.train(X_train, y_train, learning_rate=0.1, num_iterations=2500, print_cost=True)

# Perform a forward pass on the test set
AL = nn.forward_propagation(X_test)
predictions = AL > 0.5  # Convert probabilities to binary predictions

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

        

