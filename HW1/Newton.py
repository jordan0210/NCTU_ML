import inverse

import numpy

def doNewtonsMethod(N, Xs, Ys):
    A = numpy.zeros((len(Xs), N))
    b = numpy.array([Ys]).T
    for i in range(len(Xs)):
        for j in range(N):
            A[i][j] = Xs[i]**(N-1-j)
    Hf = 2 * A.T.dot(A)
    x0 = 10 * numpy.random.rand(N, 1) - 5    # Create a x0 in range [-5, 5)

    dx = 1000
    count = 0
    while (dx > 1e-8):
        x1 = x0 - inverse.inverse(Hf).dot(Hf.dot(x0) - 2 * (A.T).dot(b))
        sum = 0
        for i in range(N):
            sum += abs(x1[i] - x0[i])
        if (sum <= dx):
            dx = sum
            x0 = x1
        else:
            break
        count += 1
        if count > 10000:
            break
    return x0.T