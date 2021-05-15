from load import *

import numpy as np
from matplotlib import pyplot
from libsvm.svmutil import *

def svm_grid_search(log2c, log2g, train, kernel, Y_test, X_test):
    best = 0.0
    best_log2c = 0
    best_log2g = 0

    # search best params
    acc = np.zeros((len(log2g), len(log2c)), dtype=float)
    for i in range(len(log2g)):
        for j in range(len(log2c)):
            param = f"-q -t {kernel} -v 3 -c {2**log2c[j]}"
            if (kernel != 0):
                param += f" -g {2**log2g[i]}"
            model = svm_train(train, param)
            acc[i][j] = round(model, 2)
            if (best < model):
                best = model
                best_log2g = log2g[i]
                best_log2c = log2c[j]
    # test
    param = f"-q -t {kernel} -c {2**best_log2c}"
    if (kernel != 0):
        param += f" -g {2**best_log2g}"
    model = svm_train(train, param)
    _, p_acc, _ = svm_predict(Y_test, X_test, model)
    return best, best_log2c, best_log2g, acc, p_acc[0]

def drawTable(size, title, kernel, acc, log2c, log2g):
    _, axs = pyplot.subplots(1, 1, figsize=size)
    axs.axis('tight')
    axs.axis('off')
    axs.set_title(title)
    if (kernel == 0):
        table = axs.table(cellText=acc, cellLoc="center",
                    colLabels=[f"log2c=\n{log2c[i]}" for i in range(len(log2c))],
                    loc="center")
        table.scale(1, 2)
    else:
        table = axs.table(cellText=acc, cellLoc="center",
                colLabels=[f"log2c=\n{log2c[i]}" for i in range(len(log2c))],
                rowLabels=[f"log2g={log2g[i]}" for i in range(len(log2g))], loc="center")
        table.scale(1, 2)
    pyplot.show()

if __name__ == "__main__":
    # set parameters
    linear = 0
    polynomial = 1
    RBF = 2
    log2c = log2g = [i-10 for i in range(0, 21, 2)]

    # load data
    X_train = load_data("./data/X_train.csv")
    Y_train = load_label("./data/Y_train.csv")
    X_test = load_data("./data/X_test.csv")
    Y_test = load_label("./data/Y_test.csv")

    resultFile = open(f"result.txt", 'w')

    # create grid search result
    train = svm_problem(Y_train, X_train)

    best, best_log2c, _, acc, p_acc = svm_grid_search(log2c, [0], train, linear, Y_test, X_test)
    resultFile.write("============ linear kernel ============\n")
    resultFile.write(f"best:       {best}\n")
    resultFile.write(f"best_log2c: {best_log2c}\n")
    resultFile.write(f"p_acc:      {p_acc}%\n")
    resultFile.write(f"acc:\n{acc}\n")
    resultFile.write("=======================================\n\n")

    drawTable((6, 3), "linear", linear, acc, log2c, log2g)

    best, best_log2c, best_log2g, acc, p_acc = svm_grid_search(log2c, log2g, train, polynomial, Y_test, X_test)
    resultFile.write("========== polynomial kernel ==========\n")
    resultFile.write(f"best:       {best}\n")
    resultFile.write(f"best_log2c: {best_log2c}\n")
    resultFile.write(f"best_log2g: {best_log2g}\n")
    resultFile.write(f"p_acc:      {p_acc}%\n")
    resultFile.write(f"acc:\n{acc}\n")
    resultFile.write("=======================================\n\n")

    drawTable((6, 6), "polynomial", polynomial, acc, log2c, log2g)

    best, best_log2c, best_log2g, acc, p_acc = svm_grid_search(log2c, log2g, train, RBF, Y_test, X_test)
    resultFile.write("============= RBF  kernel =============\n")
    resultFile.write(f"best:       {best}\n")
    resultFile.write(f"best_log2c: {best_log2c}\n")
    resultFile.write(f"best_log2g: {best_log2g}\n")
    resultFile.write(f"p_acc:      {p_acc}%\n")
    resultFile.write(f"acc:\n{acc}\n")
    resultFile.write("=======================================")

    drawTable((6, 6), "RBF", RBF, acc, log2c, log2g)