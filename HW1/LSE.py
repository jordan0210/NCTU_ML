import inverse

import numpy

def doLSE(N, Lambda, Xs, Ys):
    A = numpy.zeros((len(Xs), N))
    b = numpy.array([Ys]).T
    for i in range(len(Xs)):
        for j in range(N):
            A[i][j] = Xs[i]**(N-1-j)
    return inverse.inverse(A.T.dot(A) + Lambda * numpy.eye(N)).dot(A.T).dot(b).T