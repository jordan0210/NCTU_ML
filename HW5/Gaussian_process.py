import numpy as np
from matplotlib import pyplot
from scipy.optimize import minimize

def load(filename):
    file = open(filename, 'r')
    line = file.read()
    points = []
    for point in line.split('\n'):
        points.append([float(i) for i in point.split(' ') if i.strip()])
    points.pop()
    return np.array(points)

def drawPlot(axs, title, points, C, beta, sigma, alpha, l):
    axs.set_title(title)
    axs.set_xlim(-60, 60)

    Xs = np.linspace(-60, 60, 1000)
    Ys_mean = np.zeros(1000)
    Ys_var = np.zeros(1000)

    C_inv = np.linalg.inv(C)
    for i in range(1000):
        kT = kernel(points.T[0], Xs[i], sigma, alpha, l)
        Ys_mean[i] = kT @ C_inv @ points.T[1].T
        Ys_var[i] = np.abs((kernel(Xs[i], Xs[i], sigma, alpha, l) + 1/beta) - kT @ C_inv @ kT.T) ** 0.5

    axs.plot(points.T[0], points.T[1], 'bo', markersize=5)
    axs.plot(Xs, Ys_mean, color='black')
    axs.fill_between(Xs, Ys_mean + 2 * Ys_var, Ys_mean - 2 * Ys_var, color='pink')
    pyplot.draw()

def Gaussian(points, beta, sigma, alpha, l):
    Xs = points.T[0].reshape(-1, 1)
    C = kernel(Xs, Xs.T, sigma, alpha, l)
    C += np.eye(len(Xs)) / beta
    return C

def kernel(x1, x2, sigma, alpha, l):
    return sigma * (1 + np.power(x1 - x2, 2) / (2 * alpha * l ** 2)) ** (-alpha)

def object_function(theta, points, beta):
    theta = theta.ravel()
    C = Gaussian(points, beta, theta[0], theta[1], theta[2])
    Ys = points.T[1].reshape(-1, 1)
    target = 0.5 * np.log(2*np.pi) * len(points)
    target += 0.5 * Ys.T @ np.linalg.inv(C) @ Ys
    target += 0.5 * np.log(np.linalg.det(C))
    return target.ravel()


if __name__ == "__main__":
    points = load("./data/input.data")

    beta = 5
    sigma = 1
    alpha = 1
    l = 1

    C = Gaussian(points, beta, sigma, alpha, l)

    opt = minimize(object_function, [sigma, alpha, l],
                    bounds=((1e-8, 1e6), (1e-8, 1e6), (1e-8, 1e6)),
                    args=(points, beta))
    sigma_opt = opt.x[0]
    alpha_opt = opt.x[1]
    l_opt = opt.x[2]

    C_opt = Gaussian(points, beta, sigma, alpha, l)

    f, axs = pyplot.subplots(1, 2, figsize=(10, 5))
    drawPlot(axs[0], "normal" + "\n" + f"sigma = {sigma:.2f}, alpha = {alpha:.2f}, l = {l:.2f}", points, C, beta, sigma, alpha, l)
    drawPlot(axs[1], "optimize" + "\n" + f"sigma = {sigma_opt:.2f}, alpha = {alpha_opt:.2f}, l = {l_opt:.2f}", points, C, beta, sigma_opt, alpha_opt, l_opt)
    pyplot.show()