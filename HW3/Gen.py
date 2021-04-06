import numpy as np
from numba import njit

@njit()
# Using central limit theorem
def Gaussian(mean, var):
    uniPoints = np.random.uniform(0.0, 1.0, 12)
    gaussianPoint = (uniPoints.sum() - 6) * var + mean
    return gaussianPoint

@njit()
def Linear(basis, a, W)
    point = np.random.uniform(0.0, a)
    x = np.random.uniform(-1.0, 1.0)
    for i in range(len(W)):
        point += W[i] * (x ** i)
    return point
