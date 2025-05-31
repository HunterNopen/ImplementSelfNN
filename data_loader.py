import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.api.datasets import california_housing, mnist

def split_dataset(dataset, val_ratio):
    return train_test_split(dataset, test_size=val_ratio, random_state=5)

def get_regression_price_dataset():
    return california_housing.load_data(version="small", test_split=0.2, seed=113)

def get_mnist_dataset():
    return mnist.load_data()

def get_binary_mnist_dataset():
    (X_train, y_train), (_, _) = mnist.load_data(path="mnist.npz")
    mask = (y_train == 0) | (y_train == 1)
    x, y = X_train[mask], y_train[mask]
    x = x.reshape(x.shape[0], -1) / 255.0
    y = y.reshape(-1, 1)
    return x,y