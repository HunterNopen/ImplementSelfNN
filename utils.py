import numpy as np

def check_shapes(array, shape_exp):
    try:
        assert (array.shape[0] == shape_exp.shape[0])
    except:
        raise ValueError("Mismatch in shapes!")