import numpy as np

from utils import check_shapes
from cost_func import MSE, MSE_grad
from activation_func import sigmoid, sigmoid_grad, relu, relu_grad, leaky_relu, leaky_relu_grad
from data_loader import get_binary_mnist_dataset
from metrics import Accuracy
class Perceptron():
    def __init__(self, num_features, lr = 0.1, epochs = 100, hidden_dim = 128):
        self.num_features = num_features
        self.lr = lr
        self.epochs = epochs
        self.hidden_dim = hidden_dim

        self.weights1 = np.random.randn(self.num_features, self.hidden_dim) * np.sqrt(2 / self.num_features)
        self.b1 = np.zeros((1, self.hidden_dim))

        self.weights2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2 / self.hidden_dim)
        self.b2 = np.zeros((1, 1))

    def fit(self, X, y):
        check_shapes(X, y)

        z1 = self.forward(X)
        a1 = leaky_relu(z1)

        z2 = self.hidden_forward(a1)
        y_hat = sigmoid(z2)

        loss = MSE(y_hat, y)
        print(f'Loss: {loss}')

        output = (y_hat > 0.5).astype(int)
        print(f'Accuracy: {Accuracy(output, y) * 100}%')

        delta2 = MSE_grad(y_hat, y) * sigmoid_grad(z2)
        dW2 = a1.T @ delta2
        db2 = np.mean(delta2, axis = 0, keepdims=True)

        self.weights2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        da1 = delta2 @ self.weights2.T
        delta1 = da1 * leaky_relu_grad(z1)
        dW1 = X.T @ delta1 / len(y)
        db1 = np.mean(delta1, axis = 0, keepdims=True)

        self.weights1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def forward(self, X):
        return X @ self.weights1 + self.b1

    def hidden_forward(self, X):
        return X @ self.weights2 + self.b2
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f'Current epoch: {epoch}')
            self.fit(X, y)
            print('-----------------------')

x, y = get_binary_mnist_dataset()
perceptron = Perceptron(x.shape[1])
perceptron.train(x, y)