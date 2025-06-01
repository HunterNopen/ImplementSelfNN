import numpy as np

def Accuracy(pred, labels):
    pred, labels = pred.flatten(), labels.flatten()
    
    return np.mean(pred == labels)