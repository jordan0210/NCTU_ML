import numpy
from numba import njit

@njit()
def inverse(A):
    dim = A.shape[0]
    L, U = LUDecomposition(A)
    Y = numpy.eye(dim)
    for i in range(1, dim):
        for j in range(i):
            sum = L[i][j]
            for k in range(j+1, i):
                sum += L[i][k] * Y[k][j]
            Y[i][j] = -sum
    A_inverse = numpy.zeros((dim, dim))
    for i in range(dim-1, -1, -1):
        for j in range(dim):
            sum = 0
            for k in range(i+1, dim):
                sum += U[i][k] * A_inverse[k][j]
            A_inverse[i][j] = (Y[i][j] - sum) / U[i][i]
    return A_inverse

@njit()
def LUDecomposition(A):
    dim = A.shape[0]
    L = numpy.zeros((dim, dim))
    U = numpy.zeros((dim, dim))
    for i in range(dim):
        for j in range(i):
            sum = 0
            for k in range(j):
                sum += L[i][k] * U[k][j]
            L[i][j] = (A[i][j] - sum) / U[j][j]
        L[i][i] = 1
        for j in range(i, dim):
            sum = 0
            for k in range(i):
                sum += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - sum
    return L, U