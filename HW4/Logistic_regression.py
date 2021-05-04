from Gen import Gaussian

import numpy as np
import math
from matplotlib import pyplot

def drawPlot(axs, title, N, points, labels):
    axs.set_title(title)

    for i in range(N):
        if labels[i] == 0:
            axs.scatter(points[i][0], points[i][1], color='red', s=10)
        else:
            axs.scatter(points[i][0], points[i][1], color='blue', s=10)
    pyplot.draw()

def printResult(W, labels, predict):
    print("----------------------------------------")
    print("w:")
    for w in W:
        print(f"{w:>15.10f}")

    CM = np.zeros((2, 2), dtype=int)
    for i in range(len(labels)):
        CM[labels[i]][predict[i]] += 1

    print("\nConfusion Matrix:")
    print("{0:>48}".format("Predict cluster 1 Predict cluster 2"))
    print(f"Is cluster 1 {CM[0][0]:^17} {CM[0][1]:^17}")
    print(f"Is cluster 2 {CM[1][0]:^17} {CM[1][1]:^17}\n")
    print(f"Sensitivity (Successfully predict cluster 1): {CM[0][0]/sum(CM[0]):7.5f}")
    print(f"Specificity (Successfully predict cluster 2): {CM[1][1]/sum(CM[1]):7.5f}\n")

def Gradient_descent(X, labels, learning_rate):
    W = np.random.rand(3, 1)
    dW = np.ones((3, 1))

    count = 0
    while (np.linalg.norm(dW) > 1e-2):
        count+=1
        dW = X.T @ (labels - 1 / (1 + np.exp(-X @ W)))
        W = W + learning_rate * dW
        if count == 10000:
            break
    return W

def Newton(X, labels, learning_rate):
    W = np.random.rand(3, 1)
    dW = np.ones((3, 1))

    count = 0
    while (np.linalg.norm(dW) > 1e-2):
        count+=1
        gradient = X.T @ (labels - 1 / (1 + np.exp(-X @ W)))
        try:
            parameter = (np.exp(-X @ W) / ((1 + np.exp(-X @ W))**2)).reshape(X.shape[0])
            D = np.diag(parameter)
            Hessian_inv = np.linalg.inv(X.T @ D @ X)
            dW = Hessian_inv @ gradient
        except np.linalg.LinAlgError:
            dW = gradient
        W = W + learning_rate * dW

        if count == 10000:
            break

    return W

def doPredict(W, points):
    predict = []
    for point in points:
        X = np.array([[1, point[0], point[1]]])
        if (1 / (1 + np.exp(-(X @ W)))) >= 0.5:
            predict.append(1)
        else:
            predict.append(0)

    return np.array([predict], dtype=int).T

if __name__ == "__main__":
    learning_rate = float(input("learning rate = "))
    N = int(input("number of data points = "))
    mx = [float(input("mx1 = ")), float(input("mx2 = "))]
    my = [float(input("my1 = ")), float(input("my2 = "))]
    vx = [float(input("vx1 = ")), float(input("vx2 = "))]
    vy = [float(input("vy1 = ")), float(input("vy2 = "))]

    points = []
    for label in range(2):
        for i in range(N):
            point = [Gaussian(mx[label], vx[label]), Gaussian(my[label], vy[label])]
            points.append(point)

    X = np.ones((len(points), 3))
    X[:, 1:] = np.vstack(points)
    labels = np.zeros((2*N, 1), dtype=int)
    labels[N:] = np.ones((N, 1), dtype=int)

    f, axs = pyplot.subplots(1, 3, figsize=(10, 10))
    drawPlot(axs[0], "Ground truth", 2*N, points, labels.flatten())

    W = Gradient_descent(X, labels, learning_rate)
    predict = doPredict(W, points)
    printResult(W.flatten(), labels.flatten(), predict.flatten())
    drawPlot(axs[1], "Gradient descent", 2*N, points, predict.flatten())

    W = Newton(X, labels, learning_rate)
    predict = doPredict(W, points)
    printResult(W.flatten(), labels.flatten(), predict.flatten())
    drawPlot(axs[2], "Newton's method", 2*N, points, predict.flatten())

    pyplot.show()