import numpy as np

from utils import check_shapes
from cost_func import MSE, MSE_grad
from activation_func import sigmoid, sigmoid_grad
from data_loader import get_binary_mnist_dataset

class Perceptron():
    def __init__(self, num_features, lr = 0.1, epochs = 100):
        self.num_features = num_features
        self.lr = lr
        self.epochs = epochs
        self.weights = np.random.rand(self.num_features, 1) * 0.01
        self.b = 0.0

    def fit(self, X, y):
        check_shapes(X, y)
        z = self.forward(X)
        pred = sigmoid(z)

        loss = MSE(pred, y)
        print(f'Loss: {loss}')

        grad_loss = MSE_grad(pred, y)
        grad_sigmoid = sigmoid_grad(z)
        grad = grad_loss * grad_sigmoid

        grad_weights = X.T @ grad / len(y)
        grad_b = np.sum(grad) / len(y)

        self.weights -= self.lr * grad_weights
        self.b -= self.lr * grad_b

    def forward(self, X):
        return X @ self.weights + self.b
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f'Current epoch: {epoch}')
            self.fit(X, y)

x, y = get_binary_mnist_dataset()
perceptron = Perceptron(x.shape[1])
perceptron.train(x, y)