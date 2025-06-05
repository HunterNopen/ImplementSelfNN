import numpy as np

from activation_func import relu, relu_grad, sigmoid, sigmoid_grad

class LinearLayer():
    def __init__(self, input_features, output_size, activation_func = relu, activation_grad = relu_grad):
        self.input_features = input_features
        self.output_size = output_size

        self.activation_func = activation_func
        self.activation_grad = activation_grad

        self.weights = np.random.randn(self.input_features, self.output_size) * np.sqrt(2 / self.input_features)
        self.b = np.zeros((1, self.output_size))

    def forward(self, X):
        self.X = X
        self.z = self.X @ self.weights + self.b
        self.a = self.activation()

        return self.a

    def activation(self):
        return self.activation_func(self.z)
    
    def activation_gradient(self):
        return self.activation_grad(self.z)
        
    def predict(self, X):
        return self.forward(X)
    
    def backprop(self, dA):
        self.dZ = dA * self.activation_gradient()
        self.dX = self.dZ @ self.weights.T

        self.dW = self.X.T @ self.dZ / self.X.shape[0]
        self.dB = np.mean(self.dZ, axis=0, keepdims=True)
        return self.dX