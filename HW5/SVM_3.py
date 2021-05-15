from load import load_label

import numpy as np
from matplotlib import pyplot
from scipy.spatial.distance import cdist
from libsvm.svmutil import *

def load_data(filename):
    file = open(filename, 'r')
    lines = file.read()
    data = []
    for line in lines.split('\n'):
        pixels = line.split(',')
        data.append([float(pixels[i]) for i in range(len(pixels)) if pixels[i].strip()])
    data.pop()
    return data

def new_kernel(x1, x2, gamma):
    data = np.zeros((len(x1), len(x2) + 1))
    linear = x1 @ x2.T
    RBF = np.exp(- gamma * cdist(x1, x2, 'sqeuclidean'))
    data[:, 1:] = linear + RBF
    data[:, :1] = np.arange(len(x1))[:, np.newaxis]+1
    return data

if __name__ == "__main__":
    # set parameter
    log2c = log2g = [i-10 for i in range(0, 21, 2)]

    # load data
    X_train = load_data("./data/X_train.csv")
    Y_train = load_label("./data/Y_train.csv")
    X_test = load_data("./data/X_test.csv")
    Y_test = load_label("./data/Y_test.csv")

    # do grid search to see the difference between different parameters
    best = 0.0
    best_log2c = 0
    best_log2g = 0

    # search best params
    acc = np.zeros((len(log2g), len(log2c)), dtype=float)
    for i in range(len(log2g)):
        for j in range(len(log2c)):
            param = f"-q -t 4 -v 3 -c {2**log2c[j]} -g {2**log2g[i]}"
            data = new_kernel(np.array(X_train), np.array(X_train), 2**log2g[i])
            model = svm_train(Y_train, [list(row) for row in data], param)
            acc[i][j] = round(model, 2)
            if (best < model):
                best = model
                best_log2g = log2g[i]
                best_log2c = log2c[j]
    # test
    param = f"-q -t 4 -c {2**best_log2c} -g {2**best_log2g}"
    best_train_data = new_kernel(np.array(X_train), np.array(X_train), 2**best_log2g)
    model = svm_train(Y_train, [list(row) for row in best_train_data], param)
    test_data = new_kernel(np.array(X_test), np.array(X_train), 2**best_log2g)
    _, p_acc, _ = svm_predict(Y_test, [list(row) for row in test_data], model)

    # show results
    print(f"best:       {best}")
    print(f"best_log2c: {best_log2c}")
    print(f"best_log2g: {best_log2g}")
    print(f"p_acc:      {p_acc[0]}%")
    print(f"acc:\n{acc}")

    _, axs = pyplot.subplots(1, 1, figsize=(6, 6))
    axs.axis('tight')
    axs.axis('off')
    axs.set_title("linear + RBF")
    table = axs.table(cellText=acc, cellLoc="center",
            colLabels=[f"log2c=\n{log2c[i]}" for i in range(len(log2c))],
            rowLabels=[f"log2g={log2g[i]}" for i in range(len(log2g))], loc="center")
    table.scale(1, 2)
    pyplot.show()
