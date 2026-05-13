import numpy as np


def normalize(weights):
    arr = np.array(weights, dtype=float)
    return (arr / arr.sum()).tolist()