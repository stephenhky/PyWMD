import numpy as np
from scipy.spatial.distance import euclidean

def singleindexing(m, i, j):
    return m*i+j

def unpackindexing(m, k):
    return (k/m, k % m)
