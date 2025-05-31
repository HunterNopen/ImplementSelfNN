import numpy as np

from utils import check_shapes

def MSE(X, y):
    return np.sum((X-y)**2) / len(y)

def MSE_grad(X, y):
    return 2 * X-y