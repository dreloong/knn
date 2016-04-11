import numpy as np

def euclidean_distance(X, Y):
    return np.sum((X - Y) ** 2)
