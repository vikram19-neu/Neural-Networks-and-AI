import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def initialize_weights(layer_dims):
    np.random.seed(3)
    weights = {}
    L = len(layer_dims)
    for l in range(1, L):
        weights['W' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        weights['b' + str(l)] = np.zeros((1, layer_dims[l]))
    return weights

def forward_propagation(X, weights):
    caches = []
    A = X
    L = len(weights) // 2
    for l in range(1, L):
        A_prev = A 
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        Z = np.dot(A_prev, W) + b
        A = relu(Z)
        caches.append((A_prev, W, b, Z))
    W = weights['W' + str(L)]
    b = weights['b' + str(L)]
    Z = np.dot(A, W) + b
    AL = sigmoid(Z)
    caches.append((A, W, b, Z))
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL)) / m
    cost = np.squeeze(cost)
    return cost

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[0]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    A_prev, W, b, Z = current_cache
    dZ = dAL * sigmoid_derivative(Z)
    dW = np.dot(A_prev.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev = np.dot(dZ, W.T)
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        A_prev, W, b, Z = current_cache
        dZ = dA_prev * relu_derivative(Z)
        dW = np.dot(A_prev.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        if l > 0:
            dA_prev = np.dot(dZ, W.T)
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    return grads

def update_parameters(weights, grads, learning_rate):
    L = len(weights) // 2
    for l in range(L):
        weights["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        weights["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return weights

def model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    weights = initialize_weights(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, weights)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches)
        weights = update_parameters(weights, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    return weights


dataset_directory = 'directory_path'
cat_images, cat_labels = load_images_from_folder(os.path.join(dataset_directory, 'lab1_cat_dataset'), 1)
not_cat_images, not_cat_labels = load_images_from_folder(os.path.join(dataset_directory, 'lab1_not_cat_dataset'), 0)


X = np.array(cat_images + not_cat_images).reshape(-1, 64*64)  # Flatten the images
y = np.array(cat_labels + not_cat_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


layer_dims = [X_train.shape[1], 10, 8, 8, 4, 1]


trained_weights = model(X_train, y_train, layer_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True)


AL, _ = forward_propagation(X_test, trained_weights)
predictions = AL > 0.5


accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')



   

