import numpy as np
from matplotlib import pyplot

def load(filename):
    file = open(filename, 'r')
    line = file.read()
    points = []
    for point in line.split('\n'):
        points.append([float(i) for i in point.split(' ') if i.strip()])
    points.pop()
    return np.array(points)

def drawPlot(C, points):
    Xs = np.linspace(-60, 60, 1000)
    Ys_mean = np.zeros(1000)
    Ys_var = np.zeros(1000)
    C_inv = np.linalg.inv(C)
    for i in range(1000):
        kT = kernel(points.T[0], Xs[i], 1.0, 1.0)
        Ys_mean[i] = kT @ C_inv @ points.T[1].T
        Ys_var[i] = np.abs((kernel(Xs[i], Xs[i], 1.0, 1.0) + 1/5) - kT @ C_inv @ kT.T) ** 0.5
    print(Ys_var)
    pyplot.plot(points.T[0], points.T[1], 'bo', markersize=5)
    pyplot.plot(Xs, Ys_mean, color='black')
    pyplot.fill_between(Xs, Ys_mean + 2 * Ys_var, Ys_mean - 2 * Ys_var, color='pink')
    pyplot.show()

def Gaussian(points):
    Xs = points.T[0]
    C = np.zeros((34, 34))
    for i in range(34):
        for j in range(34):
            C[i][j] = kernel(Xs[i], Xs[j], 1.0, 1.0)
    C += np.eye(34) / 5
    return C

def kernel(x1, x2, alpha, l):
    return (1 + np.power(x1 - x2, 2) / (2 * alpha * l ** 2)) ** (-alpha)

if __name__ == "__main__":
    points = load("./data/input.data")
    C = Gaussian(points)
    drawPlot(C, points)