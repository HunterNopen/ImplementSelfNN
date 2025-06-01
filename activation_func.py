import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=1, keepdims = True)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return np.where(z > 0, 1, 0)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)