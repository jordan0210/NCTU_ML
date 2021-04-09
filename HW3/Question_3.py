import Gen
from inverse import inverse

import numpy as np
import random
from matplotlib import pyplot

def drawPlot(axs, title, W_GT, var, a, points):
    axs.set_title(title)
    axs.set_xlim(-2, 2)
    axs.set_ylim(-15, 20)
    Xs = np.linspace(-2, 2, 1000)
    Ys = Xs * 0
    for i in range(len(W_GT)):
        Ys += W_GT[i] * (Xs ** i)
    axs.plot(Xs, Ys, color='black')

    if len(points) == 0:
        Ys = Ys + a
        axs.plot(Xs, Ys, color='r')
        Ys = Ys - 2 * a
        axs.plot(Xs, Ys, color='r')
    else:
        pointsArray = np.array(points).T
        axs.scatter(pointsArray[0], pointsArray[1], color='blue', s=10)

        Ys_upper = Xs * 0
        Ys_lower = Xs * 0
        for i in range(1000):
            X = np.zeros((1, len(W_GT)))
            for j in range(len(W_GT)):
                X[0][j] = Xs[i] ** j
            Ys_upper[i] = Ys[i] + a + X.dot(var).dot(X.T).item()
            Ys_lower[i] = Ys[i] - a - X.dot(var).dot(X.T).item()
        axs.plot(Xs, Ys_upper, color='r')
        axs.plot(Xs, Ys_lower, color='r')


    pyplot.draw()

if __name__ == "__main__":
    result = open("result.txt", 'w')
    b = float(input("b = "))
    basis = int(input("n = "))
    a = float(input("a = "))
    W_GT = np.zeros((basis, 1), dtype=float)
    for i in range(basis):
        W_GT[i] = float(input(f"W[{i}] = "))

    points = []
    mean = np.zeros((basis, 1), dtype=float)
    var = np.eye(basis, dtype=float) * 1/b
    predic_mean = 0.
    predic_var = 0.
    err = 1
    count = 0

    f, axs = pyplot.subplots(2, 2, figsize=(10, 10))
    drawPlot(axs[0][0], "Ground truth", W_GT, var, a, points)

    while err > 1e-8:
        newPoint = Gen.Linear(basis, a, W_GT.T[0])
        points.append(newPoint)
        count += 1
        result.write(f"Add data point {newPoint}\n\n")

        X = np.zeros((1, basis))
        for i in range(basis):
            X[0][i] = newPoint[0] ** i
        Y = newPoint[1]

        S = inverse(var)
        C = a * X.T.dot(X) + S
        u = inverse(C).dot(a * Y * X.T + S.dot(mean))

        new_mean = X.dot(u).item()
        new_var = a + X.dot(inverse(C)).dot(X.T).item()
        err = abs(new_var - predic_var)
        predic_mean = new_mean
        predic_var = new_var

        mean = u
        var = inverse(C)

        # Output: draw plot and print result
        if count == 10:
            drawPlot(axs[1][0], "After 10 incomes", mean, var, a, points)
        elif count == 50:
            drawPlot(axs[1][1], "After 50 incomes", mean, var, a, points)

        result.write("Posterior mean:\n")
        for i in range(basis):
            result.write(f" {mean[i][0]:12.10f}\n")
        result.write("\nPosterior variance:\n")
        for i in range(basis):
            for j in range(basis):
                result.write(f" {var[i][j]:13.10f}, ")
            result.write("\n")

        result.write(f"\nPredictive distribution ~ N({predic_mean:7f}, {predic_var:7f})\n")
        result.write("--------------------------------------------------\n")

    drawPlot(axs[0][1], "Predict result", mean, var, a, points)
    pyplot.show()