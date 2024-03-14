import pickle
import math
import cv2
import numpy as np
import arguments as arg
import math_functions as mf
from matplotlib import pyplot as plt
import glob
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('TkAgg')


class Dataset:
    def __init__(self, address):
        # define the size and classes number
        self.img_height = arg.img_height
        self.img_width = arg.img_width
        self.num_classes = arg.num_classes
        self.address = address
        # create lists to restore img and labels
        self.x = []  # img
        self.y = []  # label

    def loadData(self):
        # get img and label
        for folder in glob.glob(self.address):
            label = folder[-1]
            label = int(label)
            for img_path in glob.glob(folder + '/*.png'):
                img = plt.imread(img_path)
                img = cv2.resize(img, (self.img_height, self.img_width))
                self.x.append(img)
                self.y.append(label)
        # list to numpy
        self.x = np.array(self.x).reshape(len(self.x), -1)
        self.y = one_hot_encode(np.array(self.y), self.num_classes)
        return DataLoader(self.toTorchDataset(), batch_size=arg.batch_size, shuffle=True)

    def toTorchDataset(self):
        x = torch.tensor(self.x)
        y = torch.tensor(self.y)
        return TensorDataset(x, y)


class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = arg.g * torch.randn(num_inputs, 1)
        self.gradients_w = torch.zeros(num_inputs, 1)

    def update(self, learning_rate: float) -> None:
        if arg.normalization is None:
            self.weights -= torch.mul(learning_rate, self.gradients_w)

            # Implement regularization algorithms in your neural network @HW9

        if arg.normalization == 'L2':
            self.weights = torch.mul(1 - learning_rate * arg.lamda, self.weights) - torch.mul(learning_rate,
                                                                                              self.gradients_w)

    def forward(self, inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        neuron_output = torch.mm(inputs, self.weights) + bias
        return neuron_output

    def backward(self, gradient_z: torch.Tensor, inputs: torch.Tensor):
        self.gradients_w = torch.mm(torch.transpose(inputs, 0, 1), gradient_z) / len(gradient_z)

# Implement dropout algorithms in your neural network @HW9

def binomialMask():
    
    binomial = torch.distributions.binomial.Binomial(total_count=1, probs=arg.dropout_rate)
   
    return binomial.sample(torch.zeros(arg.batch_size, 1).shape)


class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation: str, dropout: bool = False):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.inputs = torch.zeros(arg.batch_size, num_neurons)
        self.activation_functions = activation_functions[activation]
        self.activation_derivative = activation_derivatives[activation]
        self.bias = torch.zeros(1)
        self.dropout = dropout
        self.z = torch.zeros(arg.batch_size, num_neurons)
        self.gradients_bias = torch.zeros(num_neurons)

    def forward(self, inputs: torch.Tensor, test: bool = False) -> torch.Tensor:
        if self.dropout:

            # Implement dropout algorithms in your neural network @HW9

            mask = binomialMask()
            self.inputs = torch.mul(mask, inputs)
            if test:
                self.inputs = torch.mul(arg.dropout_rate, self.inputs)
        else:
            self.inputs = inputs
        for i in range(len(self.neurons)):
            self.z[:, i] = self.neurons[i].forward(self.inputs, self.bias).squeeze()
        self.z = torch.Tensor(self.z)
        output = self.activation_functions(self.z)
        return output

    def backward(self, gradients_a: torch.Tensor = None, y_true: torch.Tensor = None) -> torch.Tensor:
        if gradients_a is not None:
            gradients_z = torch.mul(self.activation_derivative(self.z), gradients_a).type(torch.float32)
        elif y_true is not None:
            gradients_z = self.activation_derivative(self.z, y_true).type(torch.float32)
        else:
            raise Exception("No gradient comes in!")
        for i, neuron in enumerate(self.neurons):
            neuron.backward(gradients_z[:, i].reshape(-1, 1), self.inputs)
        weights = self.collectWeight()
        gradients_next_a = torch.mm(gradients_z, torch.transpose(weights, 0, 1))
        return gradients_next_a

    def updateBias(self, learning_rate: float):
        self.bias -= torch.mul(learning_rate, torch.mean(self.gradients_bias))

    def updateWeight(self, learning_rate: float):
        for i in range(len(self.neurons)):
            self.neurons[i].update(learning_rate)

    def collectWeight(self) -> torch.Tensor:
        list_w = [n.weights for n in self.neurons]
        return torch.cat(list_w, dim=1)


class Model:
    def __init__(self, train_loader: DataLoader):
        self.train_loader = train_loader
        self.layers = []
        self.losses = []
        self.init_layers()

    def init_layers(self):
        self.InitLayersInBatch([128], 'relu')
        self.InitLayersInBatch([10], 'softmax')

    def forward(self, inputs: torch.Tensor, y_true: torch.Tensor, test: bool = False):
        layer_output = [inputs]
        if len(self.layers) > 0:
            for i in range(len(self.layers)):
                layer_output.append(self.layers[i].forward(layer_output[i], test))
            loss = mf.cross_entropy(layer_output[-1], y_true)
            # plot_output(layer_output)
            return loss, layer_output[-1]
        else:
            raise Exception("No defined layers!")

    def backward(self, y_true: torch.Tensor):
        gradient = [self.layers[-1].backward(y_true=y_true)]
        for i in range(len(self.layers) - 1):
            gradient.append(self.layers[-i - 2].backward(gradients_a=gradient[i]))

    def update_all(self, learning_rate: float):
        for i in range(len(self.layers)):
            self.layers[i].updateWeight(learning_rate)
            self.layers[i].updateBias(learning_rate)

    def train(self):
        epochs, initial_lr, decay_rate = arg.epochs, arg.initial_lr, arg.decay_rate
        lr_arr = mf.lrArray(epochs, initial_lr, decay_rate)
        for epoch in range(epochs):
            for batch in self.train_loader:
                x, true_labels = batch
                loss, y_true, y_pred, accuracy = self.workLine(x, true_labels)
                self.losses.append(loss)
                self.backward(y_true)
                self.update_all(lr_arr[epoch])
                # Print the accuracy of each batch
                # print(f'Accuracy: {accuracy * 100}%')
                # Print the epoch number and the loss value
                print(f'Epoch {epoch + 1}, Loss: {loss}')

    def test(self, test_loader: DataLoader):
        acc = []
        for batch in test_loader:
            x, true_labels = batch
            _, _, _, accuracy = self.workLine(x, true_labels, test=True)
            acc.append(accuracy)
        # Print the epoch number and the accuracy
        print(f'Test Accuracy: {torch.mean(torch.tensor(acc)) * 100}%')

    def InitLayersInBatch(self, neuron_number_list: list[int], activation: str = 'relu', dropout: bool = False):
        if len(self.layers) == 0:
            latest_output_num = arg.size_input
        else:
            latest_output_num = len(self.layers[-1].neurons)

        neuron_number_list.insert(0, latest_output_num)

        for i in range(len(neuron_number_list) - 1):
            self.layers.append(Layer(neuron_number_list[i + 1], neuron_number_list[i], activation, dropout=dropout))

    def workLine(self, x: torch.Tensor, true_labels: torch.Tensor, test: bool = False):
        x = mf.normalize_input(x)
        y_true = torch.tensor(one_hot_encode(np.array(true_labels), arg.num_classes))
        loss, y_pred = self.forward(x, y_true, test=test)
        predicted_labels = torch.argmax(y_pred, dim=1)
        accuracy = calculateAccuracy(predicted_labels, true_labels)
        return loss, y_true, y_pred, accuracy


# Define the one-hot encoding function
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def plot_loss_curve(model):
    # Plot the loss curve
    losses = model.losses
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def plot_output(output: list[torch.Tensor]):
    length = math.ceil(math.sqrt(len(output))) + 1
    for i in range(len(output)):
        plt.subplot(length, length, i + 1)  # l行l列
        plt.hist(output[i].flatten(), facecolor='g')
        plt.title('layer ' + i.__str__())
        plt.xlim([-100, 100])
        plt.yticks([])
    plt.show()


# save model
def saveModel(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)



def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculateAccuracy(pred_label, true_label):
    acc = 0
    for i in range(len(pred_label)):
        if pred_label[i] == true_label[i]:
            acc += 1
    return acc / len(pred_label)



activation_functions = {
    'linear': mf.linear,
    'sigmoid': mf.sigmoid,
    'relu': mf.relu,
    'tanh': mf.tanh,
    'softmax': mf.softmax
}


activation_derivatives = {
    'linear': mf.linear_derivative,
    'sigmoid': mf.sigmoid_derivative,
    'relu': mf.relu_derivative,
    'tanh': mf.tanh_derivative,
    'softmax': mf.softmax_cross_entropy_derivative
}
