import numpy as np

class Linear:
    def forward(self, x):
        return x
    
    def backward(self, dA, x):
        return dA

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, dA, x):
        dZ = np.array(dA, copy=True)
        dZ[x <= 0] = 0
        return dZ

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dA, x):
        s = self.forward(x)
        return dA * s * (1 - s)

class Tanh:
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, dA, x):
        s = self.forward(x)
        return dA * (1 - s**2)

class Softmax:
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    def backward(self, dA, x):
        s = self.forward(x)
        return dA * s * (1 - s) 


x = np.array([[1.0, 2.0, -1.0], [-3.0, 4.0, 0.5]])

relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()
linear = Linear()
softmax = Softmax()

print("ReLU:", relu.forward(x))
print("Sigmoid:", sigmoid.forward(x))
print("Tanh:", tanh.forward(x))
print("Linear:", linear.forward(x))
print("Softmax:", softmax.forward(x))
